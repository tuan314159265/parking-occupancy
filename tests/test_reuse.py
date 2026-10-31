import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
import parking_module as parking

XML = '<parking><space id="1" occupied="1"><contour><point x="1" y="1"/><point x="8" y="1"/><point x="8" y="8"/></contour></space></parking>'

class ReuseTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.raw = self.base/'raw'
        self.root = self.raw/'PKLot'
        self.output = self.base/'personal-output'
        self.xml = self.root/'UFPR04/Sunny/2012-10-25/frame.xml'
        self.xml.parent.mkdir(parents=True)
        self.xml.write_text(XML)
        Image.new('RGB',(10,10)).save(self.xml.with_suffix('.jpg'))

    def run_reuse(self, **kwargs):
        return parking.ensure_dataset(self.raw,self.output,**kwargs)

    def test_existing_xml_without_provenance_skips_prepare(self):
        with patch.object(parking,'prepare',side_effect=AssertionError('must not download')):
            self.assertEqual(self.run_reuse()['valid_frames'],1)
        self.assertFalse((self.output/'provenance.json').exists())

    def test_valid_cache_skips_prepare_and_audit(self):
        parking.audit(self.root,self.output)
        with patch.object(parking,'prepare',side_effect=AssertionError),patch.object(parking,'audit',side_effect=AssertionError):
            self.assertEqual(self.run_reuse()['valid_frames'],1)

    def test_force_audit_rechecks(self):
        parking.audit(self.root,self.output)
        with patch.object(parking,'audit',wraps=parking.audit) as audit:
            self.run_reuse(force_audit=True)
            audit.assert_called_once()

    def test_changed_sample_invalidates_cache(self):
        parking.audit(self.root,self.output)
        self.xml.write_text(XML.replace('occupied="1"','occupied="0"'))
        with patch.object(parking,'audit',wraps=parking.audit) as audit:
            self.assertEqual(self.run_reuse()['counts']['vacant'],1)
            audit.assert_called_once()

    def test_corrupt_cache_falls_back_to_audit(self):
        parking.audit(self.root,self.output)
        (self.output/'audit_summary.json').write_text('bad JSON')
        self.assertEqual(self.run_reuse()['valid_frames'],1)

    def test_shared_manifest_rebased_without_writes_to_source(self):
        owner_output=self.base/'owner-output'
        parking.audit(self.root,owner_output)
        # Another user's shortcut has a different mount path but identical relative paths.
        shortcut=self.base/'shortcut'
        shutil.copytree(self.root,shortcut)
        before={str(p.relative_to(self.base)):p.read_bytes() for d in [self.root,owner_output] for p in d.rglob('*') if p.is_file()}
        with patch.object(parking,'prepare',side_effect=AssertionError),patch.object(parking,'audit',side_effect=AssertionError):
            result=self.run_reuse(dataset_root=shortcut,cached_audit_dir=owner_output)
        self.assertEqual(result['root'],str(shortcut))
        self.assertTrue((self.output/'valid_samples.jsonl').is_file())
        after={str(p.relative_to(self.base)):p.read_bytes() for d in [self.root,owner_output] for p in d.rglob('*') if p.is_file()}
        self.assertEqual(before,after)

    def test_shared_without_cache_only_writes_personal_output(self):
        with patch.object(parking,'prepare',side_effect=AssertionError):
            self.run_reuse(dataset_root=self.root)
        self.assertTrue((self.output/'audit_summary.json').is_file())
        self.assertFalse((self.root/'audit_summary.json').exists())

    def test_bad_shared_path_never_downloads(self):
        with patch.object(parking,'prepare',side_effect=AssertionError),self.assertRaisesRegex(ValueError,'shared dataset path'):
            self.run_reuse(dataset_root=self.base/'typo')
        self.assertFalse(self.output.exists())

    def test_shared_output_must_be_separate(self):
        with self.assertRaises(ValueError):
            parking.ensure_dataset(self.raw,self.root/'outputs',dataset_root=self.root)

    def test_interrupted_extraction_is_not_reused(self):
        (self.raw/'.pklot-extraction-in-progress').touch()
        with patch.object(parking,'prepare',return_value={'valid_frames':1}) as prepare:
            self.run_reuse()
            prepare.assert_called_once()
        with self.assertRaisesRegex(ValueError,'incomplete'):
            self.run_reuse(dataset_root=self.root)

    def test_complete_archive_skips_network(self):
        import io
        import tarfile
        raw = self.base/'fresh'
        raw.mkdir()
        archive = raw/'PKLot.tar.gz'
        with tarfile.open(archive,'w:gz') as tar:
            for path in (self.xml,self.xml.with_suffix('.jpg')):
                content = path.read_bytes()
                member = tarfile.TarInfo('PKLot/PKLot/'+path.relative_to(self.root).as_posix())
                member.size=len(content)
                tar.addfile(member,io.BytesIO(content))
        with patch.object(parking,'EXPECTED_BYTES',archive.stat().st_size),patch.object(parking.subprocess,'run',side_effect=AssertionError('must not use network')):
            result=parking.ensure_dataset(raw,self.output)
        self.assertEqual(result['valid_frames'],1)
        self.assertFalse((raw/'.pklot-extraction-in-progress').exists())

    def test_interrupted_audit_invalidates_old_summary(self):
        parking.audit(self.root,self.output)
        with patch.object(parking,'read_frame',side_effect=KeyboardInterrupt),self.assertRaises(KeyboardInterrupt):
            parking.audit(self.root,self.output)
        self.assertFalse((self.output/'audit_summary.json').exists())

if __name__ == '__main__':unittest.main()
