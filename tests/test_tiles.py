"""Color ROI coverage, context retention and coordinate transforms."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import parking_module as parking

REPO = Path(__file__).resolve().parents[1]


def space(sid, left, top, right, bottom):
    return {"space_id": str(sid), "polygon": [[left, top], [right, top],
                                               [right, bottom], [left, bottom]]}


class ParkingTileTests(unittest.TestCase):
    def setUp(self):
        y, x = np.indices((400, 600))
        self.image = np.stack([x % 256, y % 256, (x+y) % 256], axis=-1).astype(np.uint8)
        self.layout = {"camera": "UFPR04", "width": 600, "height": 400,
                       "spaces": [space(1, 0, 0, 40, 40), space(2, 200, 100, 400, 250),
                                  space(3, 560, 360, 600, 400)]}

    def test_native_tiles_preserve_exact_color_pixels_and_context(self):
        result = parking.make_parking_tiles(self.image, self.layout, padding=16, min_tile_size=1)
        self.assertEqual(len(result["tiles"]), 4)
        self.assertTrue(result["all_spaces_covered"])
        self.assertTrue(result["all_context_covered"])
        for tile in result["tiles"]:
            metadata = tile["metadata"]
            left, top, right, bottom = metadata["bounds_xyxy"]
            np.testing.assert_array_equal(tile["image"], self.image[top:bottom, left:right])
            self.assertEqual(metadata["scale_xy"], [1.0, 1.0])
            self.assertEqual(metadata["padding_ltrb"], [0, 0, 0, 0])
            # A returned image is independent of the source array.
            tile["image"][:] = 0
            self.assertGreater(np.count_nonzero(self.image[top:bottom, left:right]), 0)
        self.assertTrue(all(item["context_tile_ids"] for item in result["coverage"].values()))

    def test_increases_overlap_for_a_space_crossing_the_seams(self):
        result = parking.make_parking_tiles(self.image, self.layout, padding=0,
                                             overlap=.05, max_overlap=.6, min_tile_size=1)
        self.assertGreater(result["effective_overlap"], .05)
        self.assertIsNone(result["fallback_reason"])
        self.assertEqual(result["grid"], [2, 2])
        self.assertTrue(result["coverage"]["2"]["tile_ids"])
        # Pixel union of all crops covers the complete ROI.
        mask = np.zeros(self.image.shape[:2], dtype=bool)
        for tile in result["tiles"]:
            left, top, right, bottom = tile["metadata"]["bounds_xyxy"]
            mask[top:bottom, left:right] = True
        self.assertTrue(mask.all())

    def test_large_space_falls_back_to_one_full_roi(self):
        layout = dict(self.layout, spaces=[space(1, 0, 0, 600, 400)])
        result = parking.make_parking_tiles(self.image, layout, min_tile_size=1)
        self.assertEqual(result["grid"], [1, 1])
        self.assertEqual(result["fallback_reason"], "context_exceeds_max_overlap")
        self.assertEqual(result["effective_overlap"], 0)
        np.testing.assert_array_equal(result["tiles"][0]["image"], self.image)

    def test_padding_roi_and_image_boundary_clipping(self):
        layout = dict(self.layout, spaces=[space(1, 12.5, 20.5, 90.5, 100.5)])
        result = parking.make_parking_tiles(self.image, layout, padding=32)
        self.assertEqual(result["roi_bounds_xyxy"], [0, 0, 123, 133])
        self.assertTrue(result["all_context_covered"])
        self.assertEqual(len(result["tiles"]), 1)

    def test_requested_rectangular_grids_and_small_axes(self):
        for grid in ((1, 2), (2, 1), (1, 1)):
            with self.subTest(grid=grid):
                result = parking.make_parking_tiles(self.image, self.layout, grid=grid, min_tile_size=1)
                self.assertEqual(result["grid"], list(grid))
                self.assertEqual(len(result["tiles"]), grid[0]*grid[1])
        small = parking.make_parking_tiles(self.image, self.layout, min_tile_size=400)
        self.assertEqual(small["grid"], [1, 1])

    def test_letterbox_round_trip_uses_actual_rounded_scales_and_padding(self):
        result = parking.make_parking_tiles(self.image, self.layout, grid=(1, 1), output_size=(301, 301))
        metadata = result["tiles"][0]["metadata"]
        self.assertEqual(result["tiles"][0]["image"].shape, (301, 301, 3))
        self.assertGreater(metadata["padding_ltrb"][1], 0)
        # Rounding means x and y scale may differ slightly.
        points = np.array([[0, 0], [600, 400], [123.4, 99.3]])
        mapped = parking.image_points_to_tile(points, metadata)
        np.testing.assert_allclose(parking.tile_points_to_image(mapped, metadata), points, atol=1e-10)
        left, top, right, bottom = metadata["padding_ltrb"]
        np.testing.assert_allclose(mapped[0], [left, top])
        np.testing.assert_allclose(mapped[1], [301-right, 301-bottom])
        original_boxes = np.array([[12.5, 24.5, 555.5, 350.5]])
        tile_boxes = parking.image_points_to_tile(original_boxes.reshape(-1, 2), metadata).reshape(-1, 4)
        np.testing.assert_allclose(parking.tile_boxes_to_image(tile_boxes, metadata), original_boxes)
        clipped = parking.tile_boxes_to_image([[0, 0, 301, 301]], metadata, clip=True)
        np.testing.assert_allclose(clipped, [[0, 0, 600, 400]])
        pad_pixel = result["tiles"][0]["image"][0, 0]
        np.testing.assert_array_equal(pad_pixel, [114, 114, 114])

    def test_offsets_and_empty_detection_arrays(self):
        layout = dict(self.layout, spaces=[space(1, 150, 100, 350, 250)])
        result = parking.make_parking_tiles(self.image, layout, padding=10, output_size=(640, 480))
        metadata = result["tiles"][0]["metadata"]
        corner = parking.image_points_to_tile([140, 90], metadata)
        np.testing.assert_allclose(parking.tile_points_to_image(corner, metadata), [140, 90])
        self.assertEqual(parking.tile_boxes_to_image(np.empty((0, 4)), metadata).shape, (0, 4))
        json.dumps({k: v for k, v in result.items() if k != "tiles"})
        json.dumps(metadata)

    def test_all_frozen_layouts_retain_every_space_and_context(self):
        layouts = json.loads((REPO / "data/manifests/pklot_v1/slot_geometries.json").read_text())["layouts"]
        for geometry_id, layout in layouts.items():
            with self.subTest(geometry_id=geometry_id):
                image = np.zeros((layout["height"], layout["width"], 3), dtype=np.uint8)
                result = parking.make_parking_tiles(image, layout)
                self.assertTrue(result["all_spaces_covered"])
                self.assertTrue(result["all_context_covered"])
                self.assertLessEqual(len(result["tiles"]), 4)
                self.assertEqual(len(result["coverage"]), len(layout["spaces"]))

    def test_rejects_invalid_inputs(self):
        for config in ({"padding": -1}, {"grid": (3, 2)}, {"overlap": .7, "max_overlap": .6},
                       {"overlap": float("nan")}, {"min_tile_size": 0}, {"output_size": (0, 640)}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                parking.make_parking_tiles(self.image, self.layout, **config)
        with self.assertRaises(ValueError):
            parking.make_parking_tiles(self.image[:100], self.layout)
        with self.assertRaises(ValueError):
            parking.make_parking_tiles(self.image, dict(self.layout, spaces=[]))
        with self.assertRaises(ValueError):
            parking.make_parking_tiles(self.image, dict(self.layout, spaces=[space(1, -2, 0, 20, 20)]))
        metadata = parking.make_parking_tiles(self.image, self.layout)["tiles"][0]["metadata"]
        for points in ([1, 2, 3], [float("nan"), 0]):
            with self.assertRaises(ValueError): parking.tile_points_to_image(points, metadata)
        with self.assertRaises(ValueError): parking.tile_boxes_to_image([4, 3, 2, 1], metadata)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Matplotlib needed for notebook execution")
    def test_notebook_demo_reads_frame_creates_color_tiles_and_plots(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        notebook = json.loads((REPO / "notebooks/01_pklot_run_all.ipynb").read_text())
        cells = {c.get("id"): "".join(c["source"]) for c in notebook["cells"]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            xml = root / "UFPR04/Sunny/2013-01-01/frame.xml"
            xml.parent.mkdir(parents=True)
            xml.write_text('<parking><space id="1" occupied="1"><contour>'
                           '<point x="50" y="50"/><point x="250" y="50"/>'
                           '<point x="250" y="200"/><point x="50" y="200"/>'
                           '</contour></space></parking>')
            cv2.imwrite(str(xml.with_suffix(".png")), self.image)
            frame = parking.read_frame(xml, root)
            geometry_id, layout = parking._geometry_record(asdict(frame))
            record = dict(sample_id=frame.sample_id, xml_path=frame.xml_path, geometry_id=geometry_id,
                          labels={"1": 1}, camera=frame.camera, weather=frame.weather, split="train")
            namespace = dict(parking=parking, TRAIN_FRAMES=[record], SLOT_GEOMETRIES={geometry_id: layout},
                             DATASET_DIR=root)
            with patch.object(plt, "show"):
                for name in ("environment-imports", "pklot-2", "roi-tiles-code", "roi-tiles-visualization", "roi-tiles-demo"):
                    exec(compile(cells[name], name, "exec"), namespace)
            result = namespace["TILE_RESULTS"][frame.sample_id]
            self.assertTrue(result["all_context_covered"])
            self.assertEqual(result["tiles"][0]["image"].shape[-1], 3)
            figure = namespace["show_parking_tiles"](self.image, layout, result, "Synthetic color ROI check")
            figure.savefig("/tmp/parking-tiles-preview.png", dpi=100)
            plt.close(figure)
            namespace["TRAIN_FRAMES"] = [dict(record, split="test")]
            with self.assertRaisesRegex(ValueError, "training frames"):
                exec(cells["roi-tiles-demo"], namespace)


if __name__ == "__main__":
    unittest.main()
