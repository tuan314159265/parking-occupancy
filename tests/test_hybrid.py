"""Execute the hybrid notebook with a deterministic detector and real image IO."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
HAS_MATPLOTLIB = importlib.util.find_spec('matplotlib') is not None
if HAS_MATPLOTLIB:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
import numpy as np
import parking_module as parking

REPO = Path(__file__).resolve().parents[1]


class EmptyDetector:
    def __init__(self):
        self.config = {'checkpoint': 'test'}
        self.model = SimpleNamespace(names={2: 'car', 7: 'truck'})
        self.calls = []

    def _sync(self):
        pass

    def warmup(self):
        pass

    def _infer(self, image):
        self.calls.append(image.copy())
        return SimpleNamespace(boxes=None, speed={})


class ArrayResult:
    def __init__(self, values):
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class ScriptedDetector(EmptyDetector):
    def __init__(self, batches):
        super().__init__()
        self.batches = iter(batches)

    def _infer(self, image):
        self.calls.append(image.copy())
        boxes, scores, classes = next(self.batches)
        return SimpleNamespace(
            boxes=SimpleNamespace(xyxy=ArrayResult(boxes), conf=ArrayResult(scores),
                                  cls=ArrayResult(classes)), speed={})


@unittest.skipUnless(HAS_MATPLOTLIB, 'Matplotlib needed for notebook execution')
class HybridNotebookTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / 'dataset'
        xml = self.dataset / 'UFPR04/Sunny/2013-01-01/frame.xml'
        xml.parent.mkdir(parents=True)
        xml.write_text('<parking><space id="1" occupied="1"><contour>'
                       '<point x="0" y="0"/><point x="600" y="0"/>'
                       '<point x="600" y="400"/><point x="0" y="400"/>'
                       '</contour></space></parking>')
        image = np.zeros((400, 600, 3), dtype=np.uint8)
        image[:, :, 0] = 80
        cv2.imwrite(str(xml.with_suffix('.png')), image)
        self.frame = parking.read_frame(xml, self.dataset)
        geometry_id, _ = parking._geometry_record(asdict(self.frame))
        self.record = dict(sample_id=self.frame.sample_id, xml_path=self.frame.xml_path,
                           geometry_id=geometry_id, labels={'1': 1})
        notebook = json.loads((REPO / 'notebooks/01_pklot_run_all.ipynb').read_text())
        self.cells = {c.get('id'): ''.join(c['source']) for c in notebook['cells']}
        self.addCleanup(plt.close, 'all')

    def run_hybrid(self, detector, overrides=None):
        namespace = dict(parking=parking, FULLFRAME_DETECTOR=detector, DATASET_DIR=self.dataset,
                         OUTPUT_DIR=self.root / 'outputs/full', PROTOCOL='within_camera',
                         PROJECT_DATASET_VERSION='pklot_v1', VAL_FRAMES=[self.record],
                         YOLO_FULL_VAL={self.frame.sample_id: []}, checkpoint_sha256='test',
                         run_info={'versions': {}, 'val_manifest_sha256': 'test'}, asdict=asdict)
        exec(self.cells['environment-imports'], namespace)
        exec(self.cells['pklot-2'], namespace)
        exec(self.cells['roi-tiles-code'], namespace)
        namespace['TILE_CONFIG'].update(overrides or {})
        with patch.object(plt, 'show'):
            exec(compile(self.cells['hybrid-code'], 'hybrid-code', 'exec'), namespace)
        return namespace

    def test_shared_tiler_and_fallback_are_used_by_actual_validation_loop(self):
        detector = EmptyDetector()
        with patch.object(parking, 'make_parking_tiles', wraps=parking.make_parking_tiles) as tiler:
            result = self.run_hybrid(detector)
        tiler.assert_called_once()
        self.assertEqual(len(detector.calls), 1)
        metadata = result['HYBRID_METADATA'][self.frame.sample_id]
        self.assertTrue(metadata['tiling']['all_spaces_covered'])
        self.assertTrue(metadata['tiling']['all_context_covered'])
        self.assertEqual(metadata['tiling']['fallback_reason'], 'context_exceeds_max_overlap')
        self.assertEqual(result['HYBRID_TIMINGS'][0]['inference_calls'], 1)
        exported = [json.loads(line) for line in (result['HYBRID_OUTPUT_DIR'] / 'val_detections.jsonl').read_text().splitlines()]
        self.assertEqual(exported[0]['metadata'], metadata)
        self.assertEqual(exported[0]['detections'], [])


    def set_spaces(self, polygons):
        xml = self.dataset / self.frame.xml_path
        spaces = []
        for index, polygon in enumerate(polygons, 1):
            contour = ''.join(f'<point x="{x}" y="{y}"/>' for x, y in polygon)
            spaces.append(f'<space id="{index}" occupied="1"><contour>{contour}</contour></space>')
        xml.write_text('<parking>' + ''.join(spaces) + '</parking>')
        self.frame = parking.read_frame(xml, self.dataset)
        geometry_id, _ = parking._geometry_record(asdict(self.frame))
        self.record.update(geometry_id=geometry_id, labels={s.space_id: 1 for s in self.frame.spaces})

    def test_resized_letterbox_maps_once_and_discards_padding_only_boxes(self):
        self.set_spaces([[[100, 50], [400, 50], [400, 250], [100, 250]]])
        # ROI [68,18,432,282] -> 301x501: width 301, height 218, top padding 141.
        scale_x, scale_y = 301/364, 218/264
        original = np.array([120., 90., 300., 210.])
        tile_box = [(original[0]-68)*scale_x, (original[1]-18)*scale_y+141,
                    (original[2]-68)*scale_x, (original[3]-18)*scale_y+141]
        detector = ScriptedDetector([([tile_box, [0, 0, 10, 10]], [.8, .99], [2, 2])])
        result = self.run_hybrid(detector, {'output_size': (301, 501)})
        detections = result['HYBRID_VAL'][self.frame.sample_id]
        self.assertEqual(len(detections), 1)
        np.testing.assert_allclose(detections[0].xyxy, original, atol=1e-8)
        self.assertEqual(detector.calls[0].shape, (501, 301, 3))
        metadata = result['HYBRID_METADATA'][self.frame.sample_id]
        self.assertEqual(metadata['tiles'][0]['padding_ltrb'], [0, 141, 0, 142])
        self.assertFalse(metadata['kept_box_sources'][0]['touches_crop_edge'])

    def test_overlapping_tiles_deduplicate_original_boxes_across_vehicle_classes(self):
        self.set_spaces([[[0, 0], [40, 0], [40, 40], [0, 40]],
                         [[220, 150], [380, 150], [380, 250], [220, 250]],
                         [[560, 360], [600, 360], [600, 400], [560, 400]]])
        # Shared tiling with 32px context requires [0,0,423,282], etc.
        batches = []
        for index, (left, top) in enumerate(((0, 0), (177, 0), (0, 118), (177, 118))):
            # Same vehicle extends past the right edge in the left-hand crops.
            box = [400-left, 180-top, 440-left, 220-top]
            batches.append(([box], [.99 if index % 2 == 0 else .65], [2 if index % 2 == 0 else 7]))
        detector = ScriptedDetector(batches)
        result = self.run_hybrid(detector, {'min_tile_size': 1})
        detections = result['HYBRID_VAL'][self.frame.sample_id]
        self.assertEqual(len(detector.calls), 4)
        self.assertEqual(len(detections), 1)
        np.testing.assert_allclose(detections[0].xyxy, [400, 180, 440, 220])
        self.assertEqual(detections[0].class_name, 'truck')
        self.assertEqual(detections[0].confidence, .65)
        metadata = result['HYBRID_METADATA'][self.frame.sample_id]
        self.assertEqual(metadata['boxes_before_nms'], 4)
        self.assertEqual(metadata['boxes_after_nms'], 1)
        self.assertFalse(metadata['kept_box_sources'][0]['touches_crop_edge'])
        self.assertTrue(metadata['tiling']['all_context_covered'])


if __name__ == '__main__':
    unittest.main()
