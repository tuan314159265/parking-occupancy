import copy
from dataclasses import asdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import parking_module as parking


class ProjectSplitTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base=Path(tmp.name)
        self.manifest=self.base/'inventory/valid_samples.jsonl'
        self.manifest.parent.mkdir()
        self.config=json.loads(Path(__file__).resolve().parents[1].joinpath('configs/pklot_v1.json').read_text())
        self.config['max_frames_per_camera_weather']={'train':2,'val':1,'test':1}
        self.config['day_assignment_trials']=16
        self.rows=[]
        for camera,month in [('UFPR04',1),('UFPR05',2),('PUCPR',3)]:
            for day in range(1,21):
                for weather in ('sunny','overcast','rainy'):
                    date=f'2013-{month:02d}-{day:02d}'
                    name=f'{camera}/{weather}/{date}/{date}_12_00_00'
                    self.rows.append(dict(sample_id=name,image_path=name+'.jpg',xml_path=name+'.xml',
                        camera=camera,weather=weather,date=date,directory_date=date,date_source='filename',
                        width=20,height=20,spaces=[dict(space_id='1',occupied=bool(day%2),polygon=[[1,1],[9,1],[9,9]])]))
        self.write_inventory()

    def write_inventory(self):
        self.manifest.write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        self.manifest.with_name('audit_summary.json').write_text(json.dumps(dict(audit_version=parking.AUDIT_VERSION,valid_frames=len(self.rows))))

    def build(self, name='frozen'):
        out=self.base/name
        lock=parking.build_project_splits(self.manifest,out,self.config)
        return out,lock

    def test_day_disjoint_budget_and_camera_transfer(self):
        out,lock=self.build()
        loaded=parking.load_project_splits(out)
        for name,manifests in loaded['protocols'].items():
            for a,b in [('train','val'),('train','test'),('val','test')]:
                self.assertFalse({r['date'] for r in manifests[a]} & {r['date'] for r in manifests[b]})
            for split,rows in manifests.items():
                from collections import Counter
                counts=Counter((r['camera'],r['weather']) for r in rows)
                self.assertTrue(all(n<=self.config['max_frames_per_camera_weather'][split] for n in counts.values()))
        transfer=loaded['protocols']['transfer_ufpr04_to_ufpr05']
        self.assertEqual({r['camera'] for r in transfer['test']},{'UFPR05'})
        self.assertEqual({r['camera'] for s in ['train','val'] for r in transfer[s]},{'UFPR04'})

    def test_deterministic_inventory_order_and_config_key_order(self):
        _,first=self.build('a')
        self.rows.reverse();self.write_inventory()
        self.config['protocols']['within_camera']['day_ratios']={'test':.15,'val':.15,'train':.7}
        _,second=self.build('b')
        self.assertEqual(first['artifacts_sha256'],second['artifacts_sha256'])

    def test_transfer_reserves_target_days_across_source_camera(self):
        changed=[]
        for row in self.rows:
            if row['camera']=='UFPR04' and row['date']=='2013-01-01':
                row['date']='2013-02-01'
                changed.append(row['sample_id'])
        self.write_inventory()
        out,_=self.build()
        transfer=parking.load_project_splits(out)['protocols']['transfer_ufpr04_to_ufpr05']
        self.assertFalse(set(changed) & {r['sample_id'] for split in ['train','val'] for r in transfer[split]})

    def test_locked_artifacts_and_configuration(self):
        out,first=self.build()
        self.assertEqual(first,parking.build_project_splits(self.manifest,out,self.config))
        self.config['seed']=99
        with self.assertRaisesRegex(ValueError,'locked'):
            parking.build_project_splits(self.manifest,out,self.config)

    def test_modified_manifest_is_rejected(self):
        out,_=self.build()
        (out/'within_camera/test.jsonl').write_text('')
        with self.assertRaisesRegex(ValueError,'checksum'):
            parking.load_project_splits(out)

    def test_geometry_variants_are_preserved(self):
        for row in self.rows:
            if row['date'].endswith(('01','02','03','04','05','06','07','08','09','10')):
                row['spaces'][0]['polygon']=[[2,2],[9,2],[9,9]]
        self.write_inventory()
        out,_=self.build()
        layouts=parking.load_project_splits(out)['geometries']
        self.assertGreater(len(layouts),3)

    def test_validator_rejects_day_leakage(self):
        out,_=self.build()
        loaded=parking.load_project_splits(out)
        manifests=loaded['protocols']['within_camera']
        manifests['test'][0]['date']=manifests['train'][0]['date']
        with self.assertRaisesRegex(ValueError,'leakage'):
            parking.validate_project_splits(manifests,loaded['geometries'])

    def test_legacy_audit_requires_refresh(self):
        self.manifest.with_name('audit_summary.json').write_text(json.dumps(dict(valid_frames=len(self.rows))))
        with self.assertRaisesRegex(ValueError,'current loader'):
            self.build()

    def test_fetch_frozen_metadata_and_reuse_downloaded_files(self):
        out,lock=self.build()
        def fake_open(request,timeout):
            name=request.full_url.split('/data/manifests/pklot_v1/')[1].split('?')[0]
            return BytesIO((out/name).read_bytes())
        dest=self.base/'downloaded'
        with patch('urllib.request.urlopen',side_effect=fake_open) as network:
            parking.fetch_project_splits('owner','repo','main',dest)
            self.assertEqual(network.call_count,len(lock['artifacts_sha256'])+1)
        with patch('urllib.request.urlopen',side_effect=fake_open) as network:
            parking.fetch_project_splits('owner','repo','main',dest)
            self.assertEqual(network.call_count,1)

    def test_uppercase_points_and_filename_dates(self):
        from PIL import Image
        path=self.base/'dataset/UFPR05/Sunny/2013-14-16/2013-04-16_06_30_00.xml'
        path.parent.mkdir(parents=True)
        path.write_text('<parking><space id="1" occupied="0"><contour><Point x="1" y="1"/><Point x="8" y="1"/><Point x="8" y="8"/></contour></space></parking>')
        Image.new('RGB',(10,10)).save(path.with_suffix('.jpg'))
        frame=parking.read_frame(path,self.base/'dataset')
        self.assertEqual(frame.date,'2013-04-16')
        self.assertEqual(frame.directory_date,'2013-14-16')
        self.assertEqual(frame.date_source,'filename')
        geometry_id,_=parking._geometry_record(asdict(frame))
        record=dict(sample_id=frame.sample_id,xml_path=frame.xml_path,
                    geometry_id=geometry_id,labels={'1':0})
        self.assertEqual(parking.read_project_frame(record,self.base/'dataset'),frame)
        path.write_text(path.read_text().replace('occupied="0"','occupied="1"'))
        with self.assertRaisesRegex(ValueError,'differs from the frozen'):
            parking.read_project_frame(record,self.base/'dataset')

if __name__=='__main__':unittest.main()
