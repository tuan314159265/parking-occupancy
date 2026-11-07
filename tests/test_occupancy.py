"""Vehicle-to-space geometry in original image coordinates."""
import copy
import math
import unittest

import parking_module as parking


class VehicleSpaceScoringTests(unittest.TestCase):
    def setUp(self):
        self.layout = dict(spaces=[
            dict(space_id="a", polygon=[[0, 10], [10, 10], [10, 20], [0, 20]]),
            dict(space_id="b", polygon=[[10, 10], [20, 10], [20, 20], [10, 20]]),
        ])

    def score(self, box=(0, 0, 10, 20), confidence=.8, fraction=.5):
        return parking.score_vehicle_spaces(
            [parking.VehicleDetection(box, confidence, "car")], self.layout, fraction)

    def test_lower_box_is_used_and_scores_are_not_binary_predictions(self):
        rows = self.score()
        self.assertEqual(rows, [
            dict(detection_index=0, space_id="a", overlap=1., score=.8),
            dict(detection_index=0, space_id="b", overlap=0., score=0.),
        ])
        self.assertAlmostEqual(self.score(fraction=1)[0]["score"], .4)
        self.assertNotIn("state", rows[0])

    def test_box_straddling_spaces_splits_overlap(self):
        rows = self.score(box=(5, 0, 15, 20))
        self.assertEqual([r["overlap"] for r in rows], [.5, .5])
        self.assertEqual([r["score"] for r in rows], [.4, .4])

    def test_polygon_intersection_not_bounding_rectangle(self):
        self.layout["spaces"] = [dict(space_id="diamond", polygon=[
            [5, 10], [10, 15], [5, 20], [0, 15]])]
        self.assertAlmostEqual(self.score()[0]["overlap"], .5)
        self.layout["spaces"][0]["polygon"].reverse()
        self.assertAlmostEqual(self.score()[0]["overlap"], .5)

    def test_outside_box_and_empty_inputs(self):
        self.assertEqual([r["score"] for r in self.score(box=(30, 0, 40, 20))], [0., 0.])
        self.assertEqual(parking.score_vehicle_spaces([], self.layout), [])
        self.layout["spaces"] = []
        self.assertEqual(self.score(), [])

    def test_labels_are_ignored_and_inputs_unchanged(self):
        expected = self.score()
        for space in self.layout["spaces"]:
            space["occupied"] = object()
        before = [dict(s) for s in self.layout["spaces"]]
        self.assertEqual(self.score(), expected)
        self.assertEqual(self.layout["spaces"], before)

    def test_invalid_boxes_confidence_and_fraction_are_rejected(self):
        for box in [(0, 0, 0, 10), (10, 0, 0, 10), (0, 0, 10, math.inf), (0, 1)]:
            with self.subTest(box=box), self.assertRaises(ValueError):
                self.score(box=box)
        for confidence in [-.1, 1.1, math.nan, True]:
            with self.subTest(confidence=confidence), self.assertRaises(ValueError):
                self.score(confidence=confidence)
        for fraction in [0, -1, 2, math.nan, True, "0.5"]:
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                self.score(fraction=fraction)

    def test_invalid_or_duplicate_polygons_are_rejected(self):
        for polygon in [[], [[0, 0], [1, 1], [2, 2]],
                        [[0, 0], [10, 0], [5, 5], [10, 10], [0, 10]],
                        [[0, 0], [10, 0], [math.nan, 10]]]:
            with self.subTest(polygon=polygon), self.assertRaises(ValueError):
                parking.score_vehicle_spaces([], dict(spaces=[dict(space_id="a", polygon=polygon)]))
        self.layout["spaces"].append(copy.deepcopy(self.layout["spaces"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.score()


if __name__ == "__main__":
    unittest.main()
