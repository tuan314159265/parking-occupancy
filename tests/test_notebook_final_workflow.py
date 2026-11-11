"""Execute the unified notebook control flow without any real test inference."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import parking_module as parking


ROOT = Path(__file__).resolve().parents[1]


class UnifiedNotebookTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        notebook = json.loads((ROOT / "notebooks/01_pklot_run_all.ipynb").read_text())
        self.cells = {c.get("id"): "".join(c["source"]) for c in notebook["cells"]}
        self.env = {}
        exec(self.cells["environment-imports"], self.env)
        exec(self.cells["pklot-1"], self.env)
        self.env.update(WORK_DIR=self.root, OUTPUT_DIR=self.root / "outputs/full", parking=parking,
                        DATASET_DIR=self.root / "data/raw/PKLot", MANIFEST_DIR=self.root / "manifests/within_camera")
        self.val = self.root / "outputs/validation_v1"
        self.val.mkdir(parents=True)
        (self.val / "locked_config.json").write_text(json.dumps(dict(complete=True, selected_on="val")))
        (self.val / "preregistered_plan.json").write_text(json.dumps(dict(plan=dict(
            protocol="within_camera", checkpoint=dict(filename="yolo11n.pt")))))
        (self.val / "validation_summary.json").write_text(json.dumps(dict(complete=True, methods={})))
        (self.root / "models").mkdir()
        (self.root / "models/yolo11n.pt").write_bytes(b"synthetic placeholder, never loaded")

    def test_default_reads_existing_validation_without_search_or_network(self):
        self.assertFalse(self.env["RUN_VALIDATION_SEARCH"])
        self.assertFalse(self.env["RUN_FINAL_TEST"])
        with patch.object(parking, "run_validation_search", side_effect=AssertionError("must not search")):
            with patch("urllib.request.urlopen", side_effect=AssertionError("must not download")):
                exec(self.cells["bounded-validation-code"], self.env)
        self.assertEqual(self.env["TASK12_DIR"], self.val)
        self.assertTrue(self.env["TASK12_SUMMARY"]["complete"])
        self.assertEqual(self.env["TASK12_CHECKPOINT"], self.root / "models/yolo11n.pt")

    def test_explicit_artifact_directory_preserved(self):
        self.env["VALIDATION_ARTIFACT_DIR"] = str(self.val)
        exec(self.cells["bounded-validation-code"], self.env)
        self.assertEqual(self.env["TASK12_DIR"], self.val)

    def test_local_code_preferred_without_overwriting_repo_source(self):
        source = self.root / "parking_module.py"
        source.write_text("VALUE = 456\n")
        with patch.dict(self.env["sys"].modules):
            with patch("urllib.request.urlopen", side_effect=AssertionError("must not download")):
                exec(self.cells["pklot-3"], self.env)
        self.assertEqual(self.env["parking"].VALUE, 456)
        self.assertEqual(source.read_text(), "VALUE = 456\n")

    def test_local_split_reused_without_fetching(self):
        split = self.root / "data/manifests/pklot_v1"
        split.mkdir(parents=True)
        (split / "split_lock.json").write_text("{}")
        data = dict(geometries={}, lock={}, protocols=dict(within_camera=dict(train=[{}], val=[{}], test=[{}])))
        with patch.object(parking, "load_project_splits", return_value=data) as load:
            with patch.object(parking, "fetch_project_splits", side_effect=AssertionError("must not fetch")):
                with patch.object(parking, "read_project_frame", return_value=SimpleNamespace(sample_id="synthetic")):
                    exec(self.cells["data-split-code"], self.env)
        load.assert_called_once_with(split)
        self.assertEqual(self.env["MANIFEST_DIR"], split / "within_camera")

    def test_disabled_test_safe_before_setup_and_does_not_infer(self):
        env = dict(RUN_FINAL_TEST=False)
        exec(self.cells["final-test-code"], env)
        self.assertEqual(env["TEST_REPORTS"], {})

    def test_completed_test_is_read_without_calling_model(self):
        out = self.root / "existing_test"
        out.mkdir()
        (out / "test_summary.json").write_text("{}")
        flow = Mock()
        flow.verify_saved_test.return_value = {"hog_svm": "synthetic report"}
        self.env.update(FINAL_TEST_DIR=out, TASK12_DIR=self.val, final_test=flow)
        exec(self.cells["final-test-code"], self.env)
        flow.run_final_test.assert_not_called()
        flow.verify_saved_test.assert_called_once()
        self.assertEqual(self.env["TEST_REPORTS"], {"hog_svm": "synthetic report"})

    def test_enabled_test_uses_same_notebook_paths_and_frozen_runner(self):
        flow = Mock()
        flow.run_final_test.return_value = {"hog_svm": "synthetic report"}
        out = self.root / "new_test"
        self.env.update(RUN_FINAL_TEST=True, FINAL_TEST_INPUTS=dict(test_frames=270),
                        FINAL_TEST_DIR=out, TASK12_DIR=self.val,
                        TASK12_CHECKPOINT=self.root / "models/yolo11n.pt", final_test=flow)
        exec(self.cells["final-test-code"], self.env)
        flow.run_final_test.assert_called_once_with(self.env["DATASET_DIR"], self.env["MANIFEST_DIR"],
            self.val, self.root / "models/yolo11n.pt", out, device="auto")

    def test_existing_example_selection_is_not_regenerated(self):
        out = self.root / "existing_test"
        (out / "examples").mkdir(parents=True)
        selected = [dict(category="error", sample_id="synthetic")]
        (out / "examples/selection.json").write_text(json.dumps(dict(selected=selected)))
        flow = Mock()
        self.env.update(TEST_REPORTS={"hog_svm": {}}, FINAL_TEST_DIR=out, final_test=flow)
        exec(self.cells["final-test-images-code"], self.env)
        flow.export_examples.assert_not_called()
        self.assertEqual(self.env["FINAL_TEST_EXAMPLES"], selected)

    def test_no_test_means_no_tables_images_or_seal(self):
        flow = Mock()
        self.env.update(TEST_REPORTS={}, final_test=flow)
        for name in ("final-test-tables-code", "final-test-images-code", "final-test-seal-code"):
            exec(self.cells[name], self.env)
        flow.export_tables.assert_not_called()
        flow.export_examples.assert_not_called()
        flow.seal_evidence.assert_not_called()

    def test_validation_results_display_without_legacy_demo_variables(self):
        metrics = dict(precision=0.8, recall=0.6, f1=0.6857, accuracy=0.7,
                       coverage=0.9, false_vacancy_rate=0.2, mae_vacant_count=1.5,
                       timing=dict(total_ms_mean=25.0))
        self.env.update(TASK12_SUMMARY=dict(methods=dict(hog_svm=metrics)), TASK12_DIR=self.val)
        # The notebook must also work without the optional pandas display package.
        with patch.dict(self.env["sys"].modules, {"pandas": None}):
            exec(self.cells["results-code"], self.env)
        self.assertEqual(self.env["validation_rows"][0]["split"], "val")
        self.assertEqual(self.env["validation_rows"][0]["ms_per_image"], 25.0)


if __name__ == "__main__":
    unittest.main()
