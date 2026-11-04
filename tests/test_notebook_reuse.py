"""Execute notebook setup from typical local kernel working directories."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
import parking_module as parking

REPO = Path(__file__).resolve().parents[1]
XML = ('<parking><space id="1" occupied="1"><contour>'
       '<point x="1" y="1"/><point x="8" y="1"/><point x="8" y="8"/>'
       '</contour></space></parking>')


class NotebookReuseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.repo = Path(temporary.name) / 'project'
        (self.repo / 'configs').mkdir(parents=True)
        (self.repo / 'configs/pklot_v1.json').write_text('{}')
        (self.repo / 'parking_module.py').write_text('# repository source must remain untouched\n')
        (self.repo / 'notebooks').mkdir()
        root = self.repo / 'data/raw/PKLot'
        xml = root / 'UFPR04/Sunny/2013-01-01/frame.xml'
        xml.parent.mkdir(parents=True)
        xml.write_text(XML)
        Image.new('RGB', (10, 10)).save(xml.with_suffix('.jpg'))
        parking.audit(root, self.repo / 'outputs/full')
        notebook = json.loads((REPO / 'notebooks/01_pklot_run_all.ipynb').read_text())
        self.cells = {c.get('id'): ''.join(c['source']) for c in notebook['cells']}
        self.namespace = dict(parking=parking, USE_GOOGLE_DRIVE=False,
                              DRIVE_PROJECT_DIR='parking-occupancy', SHARED_DATASET_DIR='',
                              SHARED_AUDIT_DIR='', DOWNLOAD_DATA=True, FORCE_AUDIT=False)
        exec(self.cells['environment-imports'], self.namespace)

    def test_root_and_notebook_directory_reuse_same_dataset_and_audit(self):
        for directory in (self.repo, self.repo / 'notebooks'):
            with self.subTest(directory=directory), contextlib.chdir(directory):
                with patch.object(parking, 'prepare', side_effect=AssertionError('must not redownload')):
                    with patch.object(parking, 'audit', side_effect=AssertionError('must reuse audit')):
                        exec(self.cells['pklot-drive'], self.namespace)
                        exec(self.cells['pklot-5'], self.namespace)
                self.assertEqual(self.namespace['WORK_DIR'], self.repo)
                self.assertEqual(self.namespace['summary']['valid_frames'], 1)
        self.assertFalse((self.repo / 'notebooks/data').exists())

    def test_shared_dataset_selection_is_preserved(self):
        self.namespace['SHARED_DATASET_DIR'] = str(self.repo / 'data/raw/PKLot')
        with contextlib.chdir(self.repo / 'notebooks'):
            exec(self.cells['pklot-drive'], self.namespace)
        self.assertEqual(self.namespace['DATASET_DIR'], self.repo / 'data/raw/PKLot')

    def test_module_download_does_not_overwrite_source_or_add_notebook_files(self):
        original = (self.repo / 'parking_module.py').read_bytes()
        self.namespace.update(OWNER='owner', REPO='repo', REF='branch', MODULE_PATH='parking_module.py')
        download = b'VALUE = 123\n'
        with contextlib.chdir(self.repo / 'notebooks'):
            exec(self.cells['pklot-drive'], self.namespace)
            # Keep importlib's process-global module registration isolated during this test.
            with patch.dict(self.namespace['sys'].modules):
                with patch('urllib.request.urlopen', return_value=io.BytesIO(download)):
                    exec(self.cells['pklot-3'], self.namespace)
        self.assertEqual(self.namespace['parking'].VALUE, 123)
        self.assertEqual((self.repo / 'parking_module.py').read_bytes(), original)
        cache = self.repo / 'outputs/runtime'
        self.assertEqual((cache / 'parking_module.py').read_bytes(), download)
        self.assertEqual(json.loads((cache / 'module_source.json').read_text())['ref'], 'branch')
        self.assertFalse((self.repo / 'notebooks/parking_module.py').exists())


if __name__ == '__main__':
    unittest.main()
