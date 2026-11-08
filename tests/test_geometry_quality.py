"""Weak-geometry flags relative to a per-layout reference, and the vacancy policy."""
import unittest

import numpy as np
import cv2

import parking_module as parking


def lot_image(shift_x=0, brightness=1.0, seed=0):
    """Gray asphalt, painted separators every 50 px and a static curb strip."""
    rng = np.random.default_rng(seed)
    image = np.full((240, 320, 3), 90, np.uint8)
    cv2.rectangle(image, (0, 8), (319, 28), (170, 170, 170), -1)
    for x in range(40, 291, 50):  # Separators at both sides of all 5 spaces.
        cv2.line(image, (x, 60), (x, 200), (235, 235, 235), 3)
    image = np.clip(image + rng.normal(0, 3, image.shape), 0, 255)
    image = np.roll(image * brightness, shift_x, axis=1)
    return image.astype(np.uint8)


LAYOUT = {"width": 320, "height": 240, "spaces": [
    {"space_id": str(i), "polygon": [[40 + 50 * i, 60], [90 + 50 * i, 60],
                                    [90 + 50 * i, 200], [40 + 50 * i, 200]]}
    for i in range(5)]}


class GeometryQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = parking.build_geometry_reference(
            [lot_image(seed=s) for s in range(3)], LAYOUT)

    def assess(self, image, reference="default"):
        geometry = parking.analyze_parking_geometry(image, LAYOUT)
        return parking.assess_geometry_quality(
            geometry, self.reference if reference == "default" else reference)

    def test_reference_summarizes_layout_without_labels(self):
        ref = self.reference
        self.assertEqual(ref["frames"], 3)
        self.assertEqual(ref["gray"].shape, (240, 320))
        self.assertEqual(set(ref["slots"]), {str(i) for i in range(5)})
        self.assertTrue(all(s["supported_rate"] == 1.0 for s in ref["slots"].values()))
        with self.assertRaises(ValueError):
            parking.build_geometry_reference([], LAYOUT)

    def test_normal_frame_has_no_flags(self):
        quality = self.assess(lot_image(seed=7))
        self.assertTrue(quality["reference_available"])
        self.assertEqual(quality["frame_flags"], [])
        self.assertEqual(quality["weak_slot_ids"], [])
        self.assertLess(abs(quality["camera_shift"]["dx"]), 1.5)

    def test_camera_shift_is_flagged(self):
        quality = self.assess(lot_image(shift_x=12, seed=7))
        self.assertIn("layout_shift_suspected", quality["frame_flags"])
        self.assertAlmostEqual(quality["camera_shift"]["dx"], 12, delta=1.5)

    def test_degraded_frame_loses_segments_and_lines(self):
        quality = self.assess(np.full((240, 320, 3), 90, np.uint8))
        self.assertIn("few_line_segments", quality["frame_flags"])
        self.assertEqual(quality["weak_slot_ids"], [str(i) for i in range(5)])

    def test_space_that_never_shows_lines_is_not_flagged(self):
        reference = dict(self.reference, slots=dict(self.reference["slots"]))
        reference["slots"]["0"] = {"supported_rate": 0.0, "median_support": 0.0}
        quality = self.assess(np.full((240, 320, 3), 90, np.uint8), reference)
        self.assertNotIn("0", quality["weak_slot_ids"])

    def test_without_reference_nothing_is_flagged(self):
        quality = self.assess(np.full((240, 320, 3), 90, np.uint8), reference=None)
        self.assertFalse(quality["reference_available"])
        self.assertEqual((quality["frame_flags"], quality["weak_slot_ids"]), ([], []))

    def test_invalid_quality_config_is_rejected(self):
        geometry = parking.analyze_parking_geometry(lot_image(), LAYOUT)
        for config in ({"unknown": 1}, {"lost_support_ratio": 2}, {"reference_padding": -1}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                parking.assess_geometry_quality(geometry, self.reference, config)


class GeometryPolicyTests(unittest.TestCase):
    PREDICTIONS = [
        {"sample_id": "f", "slot_id": "a", "state": "vacant", "score": 0.0},
        {"sample_id": "f", "slot_id": "b", "state": "occupied", "score": 0.9},
        {"sample_id": "f", "space_id": "c", "state": 0},
        {"sample_id": "f", "slot_id": "d", "state": "unknown", "score": 0.2},
    ]

    def quality(self, frame_flags=(), slot_flags=None):
        return {"frame_flags": list(frame_flags), "slot_flags": slot_flags or {}}

    def test_weak_space_turns_vacant_into_unknown_only(self):
        result = parking.apply_geometry_policy(
            self.PREDICTIONS, self.quality(slot_flags={"a": ["boundary_lost"], "b": ["boundary_lost"]}))
        states = [p["state"] for p in result["predictions"]]
        self.assertEqual(states, ["unknown", "occupied", 0, "unknown"])
        self.assertEqual(result["changed_space_ids"], ["a"])
        self.assertEqual(result["predictions"][1]["geometry_flags"], ["boundary_lost"])

    def test_frame_flag_applies_to_every_vacant_space(self):
        result = parking.apply_geometry_policy(self.PREDICTIONS, self.quality(["layout_shift_suspected"]))
        self.assertEqual([p["state"] for p in result["predictions"]],
                         ["unknown", "occupied", "unknown", "unknown"])
        self.assertEqual(result["changed_space_ids"], ["a", "c"])

    def test_report_only_keeps_states_and_inputs_are_not_modified(self):
        before = [dict(p) for p in self.PREDICTIONS]
        result = parking.apply_geometry_policy(
            self.PREDICTIONS, self.quality(["few_line_segments"]), policy="report_only")
        self.assertEqual([p["state"] for p in result["predictions"]], [p["state"] for p in before])
        self.assertEqual(result["changed_space_ids"], [])
        self.assertEqual(self.PREDICTIONS, before)
        with self.assertRaises(ValueError):
            parking.apply_geometry_policy(self.PREDICTIONS, self.quality(), policy="vacant")

    def test_policy_output_is_accepted_by_evaluator(self):
        result = parking.apply_geometry_policy(self.PREDICTIONS, self.quality(["layout_shift_suspected"]))
        manifest = [{"sample_id": "f", "camera": "UFPR04", "weather": "sunny",
                     "labels": {"a": 0, "b": 1, "c": 0, "d": 1}}]
        report = parking.evaluate_occupancy(manifest, result["predictions"], missing_policy="error")
        self.assertEqual(report["overall"]["unknown_slots"], 3)
        self.assertEqual(report["overall"]["tp"], 1)


if __name__ == "__main__":
    unittest.main()
