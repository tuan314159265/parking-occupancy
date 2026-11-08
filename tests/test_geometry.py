"""Parking-line geometry behavior and the module-to-notebook experiment flow."""
import builtins
import copy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import parking_module as parking

HAS_GEOMETRY = all(importlib.util.find_spec(name) is not None for name in ("cv2", "numpy"))
if HAS_GEOMETRY:
    import cv2
    import numpy as np

REPO = Path(__file__).resolve().parents[1]


class GeometryDependencyTests(unittest.TestCase):
    def test_data_tools_import_without_geometry_or_plotting_dependencies(self):
        script = """
import builtins
real_import = builtins.__import__
def without_geometry(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy', 'cv2', 'matplotlib'}:
        raise AssertionError('Unexpected optional dependency: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = without_geometry
import parking_module as parking
assert parking.occupancy_metrics([1], [1])['f1'] == 1
assert callable(parking.read_frame)
"""
        subprocess.run([sys.executable, "-c", script], cwd=REPO, check=True,
                       capture_output=True, text=True)

    def test_missing_geometry_dependency_has_installation_instructions(self):
        real_import = builtins.__import__

        def missing_opencv(name, *args, **kwargs):
            if name == "cv2":
                raise ModuleNotFoundError("No module named cv2")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=missing_opencv):
            with self.assertRaisesRegex(ImportError, r"pip install -e \."):
                parking.analyze_parking_geometry(None, {})


@unittest.skipUnless(HAS_GEOMETRY, "Install project dependencies to run image-processing tests")
class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.layout = {
            "camera": "UFPR04", "width": 320, "height": 240,
            "spaces": [{"space_id": "1", "polygon": [[50, 50], [150, 50], [150, 150], [50, 150]]}],
        }
        self.blank = np.zeros((240, 320, 3), dtype=np.uint8)
        self.painted = self.blank.copy()
        cv2.rectangle(self.painted, (50, 50), (150, 150), (255, 255, 255), 3)

    def test_detects_painted_boundaries_from_pixels(self):
        result = parking.analyze_parking_geometry(self.painted, self.layout)
        self.assertGreaterEqual(len(result["segments"]), 4)
        self.assertGreater(result["slots"][0]["boundary_support"], .85)
        self.assertEqual(result["slots"][0]["quality_flag"], "supported")
        for line in result["segments"]:
            self.assertTrue(all(0 <= x < 320 for x in line["xyxy"][::2]))
            self.assertTrue(all(0 <= y < 240 for y in line["xyxy"][1::2]))

    def test_blank_has_no_artificial_roi_edges_or_occupancy(self):
        result = parking.analyze_parking_geometry(self.blank, self.layout)
        self.assertEqual(result["segments"], [])
        self.assertEqual(result["slots"][0]["quality_flag"], "no_support")
        self.assertEqual(np.count_nonzero(result["images"]["roi_edges"]), 0)
        compact = {key: value for key, value in result.items() if key != "images"}
        self.assertNotIn("occupied", json.dumps(compact))
        self.assertNotIn("vacant", json.dumps(compact))

    def test_rejects_wrong_direction_and_segments_crossing_outside_roi(self):
        raw = np.array([[[50, 50, 150, 150]], [[0, 100, 319, 100]],
                        [[50, 50, 150, 50]]], dtype=np.int32)
        with patch.object(cv2, "HoughLinesP", return_value=raw):
            result = parking.analyze_parking_geometry(self.blank, self.layout)
        self.assertEqual([s["xyxy"] for s in result["segments"]], [[50, 50, 150, 50]])

    def test_duplicate_segments_do_not_inflate_support_or_extend_infinitely(self):
        short = np.array([[[50, 50, 80, 50]]], dtype=np.int32)
        with patch.object(cv2, "HoughLinesP", return_value=short):
            one = parking.analyze_parking_geometry(self.blank, self.layout)
        with patch.object(cv2, "HoughLinesP", return_value=np.repeat(short, 10, axis=0)):
            repeated = parking.analyze_parking_geometry(self.blank, self.layout)
        self.assertEqual(one["slots"], repeated["slots"])
        self.assertLess(one["slots"][0]["edge_support"][0], .4)
        self.assertEqual(one["slots"][0]["edge_support"][2], 0)
        self.assertEqual(one["slots"][0]["quality_flag"], "limited_support")

    def test_undirected_angles_wrap_at_180_degrees(self):
        layout = copy.deepcopy(self.layout)
        layout["spaces"][0]["polygon"] = [[50, 50], [150, 52], [150, 150], [50, 150]]
        # Top edge is about +1 degree; a reversed candidate is about 179 degrees.
        raw = np.array([[[150, 50, 50, 52]]], dtype=np.int32)
        with patch.object(cv2, "HoughLinesP", return_value=raw):
            result = parking.analyze_parking_geometry(
                self.blank, layout, {"angle_tolerance_deg": 3.0})
        self.assertEqual(len(result["segments"]), 1)
        self.assertGreater(result["slots"][0]["edge_support"][0], .95)

    def test_uses_selected_layout_and_rejects_resized_image(self):
        alternate = copy.deepcopy(self.layout)
        alternate["spaces"][0]["polygon"] = [[220, 50], [300, 50], [300, 150], [220, 150]]
        self.assertEqual(parking.analyze_parking_geometry(self.painted, alternate)["segments"], [])
        with self.assertRaisesRegex(ValueError, "Image size"):
            parking.analyze_parking_geometry(self.painted[:100], self.layout)

    def test_all_frozen_project_layouts_are_accepted(self):
        layouts = json.loads((REPO / "data/manifests/pklot_v1/slot_geometries.json").read_text())["layouts"]
        for geometry_id, layout in layouts.items():
            with self.subTest(geometry_id=geometry_id):
                image = np.zeros((layout["height"], layout["width"], 3), dtype=np.uint8)
                result = parking.analyze_parking_geometry(image, layout)
                self.assertEqual(len(result["slots"]), len(layout["spaces"]))
                self.assertEqual(result["quality_counts"]["no_support"], len(layout["spaces"]))

    def test_config_is_copied_and_defaults_are_not_modified(self):
        before = dict(parking.DEFAULT_GEOMETRY_CONFIG)
        overrides = {"min_line_length": 30}
        result = parking.analyze_parking_geometry(self.painted, self.layout, overrides)
        self.assertEqual(result["config"]["min_line_length"], 30)
        result["config"]["min_line_length"] = 90
        self.assertEqual(overrides, {"min_line_length": 30})
        self.assertEqual(parking.DEFAULT_GEOMETRY_CONFIG, before)

    def test_invalid_config_images_and_polygons(self):
        for config in ({"gaussian_kernel": 4}, {"canny_low": 200},
                       {"boundary_distance": float("nan")}, {"min_roi_fraction": 1.1},
                       {"min_line_length": 1.5}, {"unknown": 1}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                parking.analyze_parking_geometry(self.blank, self.layout, config)
        for image in (None, self.blank.astype(float), self.blank[:, :, 0]):
            with self.assertRaises(ValueError):
                parking.analyze_parking_geometry(image, self.layout)
        bad = copy.deepcopy(self.layout)
        bad["spaces"][0]["polygon"][1] = [50, 50]
        with self.assertRaisesRegex(ValueError, "polygon"):
            parking.analyze_parking_geometry(self.blank, bad)

    def test_roi_unions_overlapping_spaces_and_accepts_image_boundary(self):
        overlapping = copy.deepcopy(self.layout)
        overlapping["spaces"].append({"space_id": "2", "polygon": [[100, 100], [320, 100],
                                                                                [320, 240], [100, 240]]})
        result = parking.analyze_parking_geometry(self.blank, overlapping)
        self.assertEqual(int(result["images"]["roi"][120, 120]), 255)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Matplotlib needed for notebook execution")
    def test_notebook_loads_standalone_module_and_exports_real_loader_results(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        notebook = json.loads((REPO / "notebooks/01_pklot_run_all.ipynb").read_text())
        cells = {cell.get("id"): "".join(cell["source"]) for cell in notebook["cells"]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Match the notebook's download/import lifecycle without network access.
            downloaded = root / "parking_module.py"
            downloaded.write_bytes((REPO / "parking_module.py").read_bytes())
            spec = importlib.util.spec_from_file_location("geometry_downloaded_parking", downloaded)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            self.addCleanup(sys.modules.pop, spec.name, None)
            spec.loader.exec_module(module)

            xml = root / "dataset/UFPR04/Sunny/2013-01-01/frame.xml"
            xml.parent.mkdir(parents=True)
            xml.write_text('<parking><space id="1" occupied="0"><contour>'
                           '<point x="50" y="50"/><point x="150" y="50"/>'
                           '<point x="150" y="150"/><point x="50" y="150"/>'
                           '</contour></space></parking>')
            cv2.imwrite(str(xml.with_suffix(".png")), self.painted)
            frame = module.read_frame(xml, root / "dataset")
            geometry_id, layout = module._geometry_record(asdict(frame))
            record = dict(sample_id=frame.sample_id, image_path=frame.image_path,
                          xml_path=frame.xml_path, geometry_id=geometry_id, labels={"1": 0},
                          camera=frame.camera, weather=frame.weather, split="train")
            namespace = dict(parking=module, TRAIN_FRAMES=[record],
                             SLOT_GEOMETRIES={geometry_id: layout},
                             DATASET_DIR=root / "dataset", WORK_DIR=root,
                             OUTPUT_DIR=root / "outputs/full", PROTOCOL="within_camera")
            with patch.object(plt, "show"):
                for name in ("environment-imports", "pklot-2", "geometry-config", "geometry-visualization", "geometry-demo"):
                    exec(compile(cells[name], name, "exec"), namespace)
            exported = json.loads((root / "outputs/full/within_camera/geometry/geometry_results.json").read_text())
            self.assertEqual(exported["results"][0]["geometry_id"], geometry_id)
            self.assertEqual(exported["results"][0]["slots"][0]["quality_flag"], "supported")
            self.assertNotIn("images", exported["results"][0])
            self.assertEqual(list(namespace["GEOMETRY_RESULTS"]), [frame.sample_id])
            self.assertIs(namespace["parking"], module)
            figure = namespace["show_parking_geometry"](
                self.painted, layout, module.analyze_parking_geometry(self.painted, layout),
                "Synthetic geometry integration check")
            self.assertEqual(len(figure.axes), 6)
            plt.close(figure)
            namespace["TRAIN_FRAMES"] = [dict(record, split="test")]
            with self.assertRaisesRegex(ValueError, "training frames"):
                exec(cells["geometry-demo"], namespace)
            namespace["parking"] = object()
            with self.assertRaisesRegex(RuntimeError, "OWNER/REPO/REF"):
                exec(cells["geometry-config"], namespace)


if __name__ == "__main__":
    unittest.main()
