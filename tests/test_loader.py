import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from parking_occupancy.pklot import audit, read_frame, split_for
from parking_occupancy.evaluation import occupancy_metrics

XML = '<parking><space id="1" occupied="1"><contour><point x="1" y="1"/><point x="8" y="1"/><point x="8" y="8"/><point x="1" y="8"/></contour></space></parking>'

class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.xml = self.root/'PUC/Cloudy/2012-10-25/frame.xml'
        self.xml.parent.mkdir(parents=True)
        self.xml.write_text(XML)
        Image.new('RGB',(10,10)).save(self.xml.with_suffix('.jpg'))

    def test_metadata_and_polygon(self):
        f = read_frame(self.xml,self.root)
        self.assertEqual((f.camera,f.weather,f.date),('PUC','overcast','2012-10-25'))
        self.assertTrue(f.spaces[0].occupied)
        self.assertEqual(len(f.spaces[0].polygon),4)

    def test_invalid_labels(self):
        for label in ['-1','2','', 'true']:
            self.xml.write_text(XML.replace('occupied="1"',f'occupied="{label}"'))
            with self.assertRaises(ValueError):read_frame(self.xml,self.root)

    def test_bad_geometry(self):
        for value in ['nan','-5','50']:
            self.xml.write_text(XML.replace('x="8"',f'x="{value}"'))
            with self.assertRaises(ValueError):read_frame(self.xml,self.root)

    def test_missing_image(self):
        self.xml.with_suffix('.jpg').unlink()
        with self.assertRaises(ValueError):read_frame(self.xml,self.root)

    def test_audit_orphans_and_rejections(self):
        self.xml.write_text(XML.replace('occupied="1"','occupied="-1"'))
        Image.new('RGB',(10,10)).save(self.xml.parent/'orphan.jpg')
        result = audit(self.root,self.root/'audit')
        self.assertEqual(result['rejected_frames'],1)
        self.assertEqual(result['images_without_xml'],1)
        self.assertEqual(result['valid_frames'],0)

    def test_corrupt_image(self):
        self.xml.with_suffix('.jpg').write_bytes(b'bad image')
        with self.assertRaises(OSError):read_frame(self.xml,self.root)

    def test_same_day_split(self):
        from dataclasses import replace
        f=read_frame(self.xml,self.root)
        self.assertEqual(split_for(f),split_for(replace(f,camera='UFPR04',weather='rainy')))

    def test_metrics(self):
        m=occupancy_metrics([1,1,0,0],[1,0,1,0])
        self.assertEqual(m['f1'],.5)
        self.assertEqual(m['accuracy'],.5)
        with self.assertRaises(ValueError):occupancy_metrics([],[])

if __name__ == '__main__':unittest.main()
