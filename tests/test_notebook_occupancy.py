"""Run occupancy notebook cells with real images and deterministic detections."""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
HAS_MATPLOTLIB = importlib.util.find_spec('matplotlib') is not None
if HAS_MATPLOTLIB:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

import parking_module as parking

REPO = Path(__file__).resolve().parents[1]


@unittest.skipUnless(HAS_MATPLOTLIB, 'Matplotlib needed for notebook execution')
class OccupancyNotebookTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        cv2.imwrite(str(self.root / 'frame.png'), np.zeros((40, 40, 3), dtype=np.uint8))
        self.record = dict(sample_id='frame', image_path='frame.png', geometry_id='layout',
                           camera='UFPR04', weather='sunny', labels={'a': 1, 'b': 0})
        self.layout = dict(spaces=[
            dict(space_id='a', polygon=[[0, 10], [10, 10], [10, 20], [0, 20]]),
            dict(space_id='b', polygon=[[10, 10], [20, 10], [20, 20], [10, 20]]),
        ])
        detections = [parking.VehicleDetection((0, 0, 10, 20), .4, 'car')]
        self.namespace = dict(parking=parking, WORK_DIR=self.root, DATASET_DIR=self.root,
                              PROTOCOL='example', VAL_FRAMES=[self.record],
                              SLOT_GEOMETRIES={'layout': self.layout},
                              YOLO_FULL_VAL={'frame': detections}, HYBRID_VAL={'frame': detections},
                              cv2=cv2, np=np, plt=plt, Line2D=Line2D, json=json)
        (self.root / 'val.jsonl').write_text(json.dumps(self.record) + '\n')
        self.namespace.update(hashlib=hashlib, MANIFEST_DIR=self.root,
                              PROJECT_DATASET_VERSION='test', provenance={'sha256': 'test'},
                              FULLFRAME_DETECTOR=SimpleNamespace(config={'checkpoint': 'fake'}),
                              HYBRID_CONFIG={'tiling': {}}, TEST_FRAMES=None)
        notebook = json.loads((REPO / 'notebooks/01_pklot_run_all.ipynb').read_text())
        self.cells = {c.get('id'): ''.join(c['source']) for c in notebook['cells']}
        self.addCleanup(plt.close, 'all')

    def execute(self, cell):
        with patch.object(plt, 'show'):
            exec(compile(self.cells[cell], cell, 'exec'), self.namespace)

    def test_both_methods_export_complete_evaluator_compatible_predictions(self):
        self.execute('occupancy-code')
        results = self.namespace['OCCUPANCY_PREVIEW']
        self.assertEqual(results['yolo_fullframe'], results['yolo_hybrid'])
        for method in results:
            path = self.root / 'outputs/occupancy/example/preview' / method / 'predictions.jsonl'
            report = parking.evaluate_occupancy([self.record], path, missing_policy='error')
            self.assertEqual(report['overall']['accuracy'], 1)
            self.assertEqual(report['overall']['mae_vacant_count'], 0)
        self.assertEqual(len(plt.gcf().axes), 2)

    def test_missing_detector_frame_is_not_treated_as_empty_detections(self):
        self.namespace['HYBRID_VAL'] = {}
        with self.assertRaisesRegex(ValueError, 'exactly'):
            self.execute('occupancy-code')
        self.assertFalse((self.root / 'outputs/occupancy/example/preview').exists())

    def test_validation_selects_one_shared_config_and_exports_reports(self):
        self.namespace['HYBRID_VAL']['frame'] = [
            parking.VehicleDetection((3, 0, 13, 20), .7, 'car')]
        self.execute('occupancy-code')
        self.execute('validation-code')
        directory = self.root / 'outputs/occupancy/example/validation'
        frozen = json.loads((directory / 'frozen_config.json').read_text())
        search = json.loads((directory / 'search.json').read_text())
        self.assertEqual(len(search), 18)
        winner = max(search, key=lambda row: (row['mean_f1'], row['mean_coverage'],
                                             -row['mean_mae_vacant_count']))
        self.assertEqual(frozen['config'], winner['config'])
        self.assertEqual(frozen['selected_on'], 'val')
        self.assertEqual(frozen['val_manifest_sha256'],
                         hashlib.sha256((self.root / 'val.jsonl').read_bytes()).hexdigest())
        self.assertNotEqual(frozen['detection_sha256']['yolo_fullframe'],
                            frozen['detection_sha256']['yolo_hybrid'])
        for method, frames in self.namespace['OCCUPANCY_VAL'].items():
            self.assertEqual(frames['frame']['config'], frozen['config'])
            report = parking.evaluate_occupancy([self.record], directory / method / 'predictions.jsonl')
            self.assertEqual(report['overall']['f1'], winner['methods'][method]['f1'])
            self.assertTrue(any((directory / method / 'evaluation').iterdir()))
        self.assertIsNone(self.namespace['TEST_FRAMES'])


if __name__ == '__main__':
    unittest.main()
