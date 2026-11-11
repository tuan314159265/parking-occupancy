"""Synthetic evidence checks only: these tests never run the real test dataset."""
import json
from pathlib import Path
import tempfile
from typing import TypedDict
import unittest

from PIL import Image
import final_test_module as flow
import parking_module as parking


class SyntheticFrame(TypedDict):
    sample_id: str
    camera: str
    weather: str
    date: str
    image_path: str
    geometry_id: str
    labels: dict[str, int]


class FinalTestWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.md = self.root / "manifests/within_camera"
        self.val = self.root / "validation"
        self.out = self.root / "test"
        self.data = self.root / "data"
        self.data.mkdir()
        self.records: list[SyntheticFrame] = [
            SyntheticFrame(sample_id=sid, camera="CAM", weather="sunny", date="2026-01-01",
                           image_path=f"{sid}.jpg", geometry_id="layout", labels={"1": 0, "2": 1})
            for sid in ("A", "B", "C")
        ]
        self.md.mkdir(parents=True)
        (self.md / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.records))
        layout = dict(width=120, height=100, spaces=[
            dict(space_id="1", polygon=[[5, 40], [50, 40], [50, 90], [5, 90]]),
            dict(space_id="2", polygon=[[65, 40], [115, 40], [115, 90], [65, 90]])])
        flow._write(self.md.parent / "slot_geometries.json", dict(layouts={"layout": layout}))
        lock = dict(complete=True, selected_on="val", manifest_sha256={
            "test.jsonl": flow._sha(self.md / "test.jsonl"),
            "../slot_geometries.json": flow._sha(self.md.parent / "slot_geometries.json")})
        flow._write(self.val / "locked_config.json", lock)
        self.lock_sha = flow._sha(self.val / "locked_config.json")
        for r in self.records:
            Image.new("RGB", (120, 100), "gray").save(self.data / r["image_path"])
        predictions = [dict(sample_id=sid, space_id=slot, state=state)
                       for sid, states in (("A", (0, 1)), ("B", (1, 1)), ("C", (None, 1)))
                       for slot, state in zip(("1", "2"), states)]
        timings = [dict(sample_id=r["sample_id"], preprocess_ms=1, model_ms=2,
                        postprocess_ms=1, total_ms=5, inference_calls=1) for r in self.records]
        summary = {}
        for method in flow.METHODS:
            meta = dict(split="test", complete=True, selected_on="val",
                        no_test_parameter_selection=True, validation_lock_sha256=self.lock_sha)
            report = parking.evaluate_occupancy(self.records, predictions, timings,
                                                missing_policy="error", method=method, metadata=meta)
            parking.export_evaluation(report, self.out / method)
            flow._write(self.out / method / "pixel_budget.json", [
                dict(sample_id=r["sample_id"], input_pixels=100, inference_calls=1) for r in self.records])
            summary[method] = report["overall"]
        flow._write(self.out / "test_summary.json", summary)
        flow._write(self.out / "workflow_status.json", dict(status="complete", synthetic=True))

    def test_tables_include_groups_confusion_and_full_image_timing(self):
        rows = flow.export_tables(self.out, self.md, self.val)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["total_ms_mean"], 5)
        self.assertEqual(rows[0]["timing_coverage"], 1)
        self.assertIsNone(rows[0]["input_pixels_mean"])
        self.assertEqual(rows[1]["input_pixels_mean"], 100)
        self.assertTrue((self.out / "hog_svm/predictions.jsonl").is_file())
        self.assertTrue((self.out / "hog_svm/timings.jsonl").is_file())
        for file in ("comparison.csv", "by_camera.csv", "by_weather.csv", "by_camera_weather.csv",
                     "confusion_matrices.csv", "timings_per_frame.csv", "README.md"):
            self.assertTrue((self.out / "tables" / file).is_file())

    def test_tampered_prediction_rejected_before_reporting(self):
        path = self.out / "hog_svm/aligned_predictions.jsonl"
        rows = flow._rows(path)
        rows[0]["y_pred"] = 1
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        with self.assertRaisesRegex(ValueError, "disagree with metrics"):
            flow.export_tables(self.out, self.md, self.val)

    def test_missing_prediction_rejected(self):
        path = self.out / "hog_svm/aligned_predictions.jsonl"
        path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
        with self.assertRaisesRegex(ValueError, "slot predictions"):
            flow.verify_saved_test(self.out, self.md, self.val)

    def test_same_mean_but_different_per_image_pixels_rejected(self):
        path = self.out / "yolo_fullframe_highres/pixel_budget.json"
        rows = flow._json(path)
        rows[0]["input_pixels"] = 50
        rows[1]["input_pixels"] = 150
        flow._write(path, rows)
        with self.assertRaisesRegex(ValueError, "differ per frame"):
            flow.verify_saved_test(self.out, self.md, self.val)

    def test_validation_results_cannot_be_presented_as_test(self):
        path = self.out / "hog_svm/metrics.json"
        report = flow._json(path)
        report["metadata"]["split"] = "val"
        flow._write(path, report)
        with self.assertRaisesRegex(ValueError, "Not complete test evidence"):
            flow.export_tables(self.out, self.md, self.val)

    def test_examples_use_saved_predictions_and_disclose_selection(self):
        selected = flow.export_examples(self.out, self.md, self.val, self.data)
        self.assertEqual(len(selected), 12)
        for method in flow.METHODS:
            by_category = {r["category"]: r["sample_id"] for r in selected if r["method"] == method}
            self.assertEqual(by_category, {"correct": "A", "error": "B", "unknown_only": "C"})
        self.assertTrue(all((self.out / "examples" / r["figure"]).is_file() for r in selected))
        with self.assertRaises(FileExistsError):
            flow.export_examples(self.out, self.md, self.val, self.data)
        flow.export_tables(self.out, self.md, self.val)
        self.assertTrue(flow.seal_evidence(self.out).is_file())

    def test_refuse_overwriting_existing_run_without_inference(self):
        with self.assertRaises(FileExistsError):
            flow.run_final_test(self.data, self.md, self.val, self.root / "not_needed.pt", self.out)

    def test_changed_manifest_rejected(self):
        with (self.md / "test.jsonl").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "Changed locked split"):
            flow.verify_saved_test(self.out, self.md, self.val)


if __name__ == "__main__":
    unittest.main()
