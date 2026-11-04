import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import parking_module as p


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.truth=[dict(sample_id='A',camera='UFPR04',weather='sunny',labels={'1':1,'2':1,'3':0,'4':0}),
                    dict(sample_id='B',camera='UFPR05',weather='rainy',labels={'1':1,'2':0,'3':1})]
        self.pred=[dict(sample_id=sample,space_id=str(i+1),state=state)
                   for sample,states in [('A',[1,0,0,None]),('B',[None,1,1])] for i,state in enumerate(states)]
        self.times=[dict(sample_id='A',preprocess_ms=10,model_ms=20,postprocess_ms=5,total_ms=40,inference_calls=1),
                    dict(sample_id='B',preprocess_ms=20,model_ms=80,postprocess_ms=10,total_ms=120,inference_calls=4)]

    def test_hand_calculated_seven_slots(self):
        report=p.evaluate_occupancy(self.truth,list(reversed(self.pred)),self.times)
        m=report['overall']
        self.assertEqual(m['confusion_matrix']['values'],[[1,1,1],[1,2,1]])
        self.assertEqual(m['confusion_matrix_covered'],[[1,1],[1,2]])
        for key,expected in [('coverage',5/7),('accuracy',3/7),('precision',2/3),('recall',.5),
                            ('f1',4/7),('false_vacancy_rate',.25),('reported_vacancy_error_rate',.5),
                            ('mae_vacant_count',.5),('mae_occupied_count',.5)]:
            self.assertAlmostEqual(m[key],expected)
        self.assertAlmostEqual(m['covered_only']['accuracy'],3/5)
        self.assertEqual(m['timing']['total_ms_mean'],80)
        self.assertEqual(m['timing']['inference_calls_sum'],5)
        self.assertEqual(m['timing']['model_ms_mean'],50)
        self.assertAlmostEqual(m['timing']['total_ms_p95'],116)
        self.assertEqual(report['per_frame'][0]['vacant_count_upper'],3)
        self.assertEqual(report['by_camera']['UFPR04']['frames'],1)
        self.assertEqual(report['by_weather']['rainy']['slots'],3)
        self.assertEqual(report['by_camera_weather']['UFPR05/rainy']['unknown_occupied'],1)

    def test_missing_predictions_are_unknown_not_dropped(self):
        m=p.evaluate_occupancy(self.truth,self.pred[1:])['overall']
        self.assertEqual(m['missing_predictions'],1)
        self.assertEqual(m['unknown_slots'],3)
        self.assertAlmostEqual(m['accuracy'],2/7)
        with self.assertRaisesRegex(ValueError,'Missing predictions'):
            p.evaluate_occupancy(self.truth,self.pred[1:],missing_policy='error')

    def test_all_unknown_and_missing_timing(self):
        m=p.evaluate_occupancy(self.truth,[])['overall']
        for name in ['accuracy','coverage','precision','recall','f1']:
            self.assertEqual(m[name],0)
        self.assertEqual(m['missing_predictions'],7)
        self.assertEqual(m['unknown_occupied'],4)
        self.assertEqual(m['unknown_vacant'],3)
        self.assertEqual(m['mae_vacant_count'],1.5)
        self.assertIsNone(m['reported_vacancy_error_rate'])
        self.assertIsNone(m['timing']['total_ms_mean'])
        self.assertEqual(m['timing']['coverage'],0)

    def test_perfect_binary_matches_legacy_and_no_undefined_claims(self):
        preds=[dict(sample_id=r['sample_id'],space_id=k,state=v) for r in self.truth for k,v in r['labels'].items()]
        m=p.evaluate_occupancy(self.truth,preds)['overall']
        self.assertEqual(m['accuracy'],1)
        self.assertEqual(m['f1'],1)
        self.assertEqual(m['mae_vacant_count'],0)
        truth=[dict(sample_id='C',camera='cam',weather='sunny',labels={'1':0})]
        m=p.evaluate_occupancy(truth,[dict(sample_id='C',space_id='1',state=0)])['overall']
        self.assertIsNone(m['false_vacancy_rate'])
        self.assertEqual(m['precision'],0)
        self.assertEqual(m['recall'],0)
        self.assertEqual(m['reported_vacancy_error_rate'],0)

    def test_count_mae_is_per_frame_not_per_slot(self):
        truth=[dict(sample_id='A',camera='cam',weather='sunny',labels={'1':0}),
               dict(sample_id='B',camera='cam',weather='sunny',labels={str(i):0 for i in range(3)})]
        preds=[dict(sample_id=r['sample_id'],space_id=k,state=1) for r in truth for k in r['labels']]
        self.assertEqual(p.evaluate_occupancy(truth,preds)['overall']['mae_vacant_count'],2)

    def test_duplicate_or_unexpected_keys_rejected(self):
        for preds in [self.pred+self.pred[:1],self.pred+[dict(sample_id='OTHER',space_id='1',state=1)],
                      self.pred+[dict(sample_id='A',space_id='unlabelled',state=1)]]:
            with self.assertRaises(ValueError):p.evaluate_occupancy(self.truth,preds)
        with self.assertRaises(ValueError):p.evaluate_occupancy(self.truth+self.truth[:1],self.pred)
        with self.assertRaises(ValueError):p.evaluate_occupancy([],[])

    def test_scores_threshold_and_dataclass(self):
        truth=[dict(sample_id='A',camera='cam',weather='sunny',labels={'1':1,'2':0})]
        preds=[p.OccupancyPrediction('A','1',.5),p.OccupancyPrediction('A','2',.49)]
        self.assertEqual(p.evaluate_occupancy(truth,preds)['overall']['accuracy'],1)
        self.assertEqual(p.evaluate_occupancy(truth,preds,threshold=.6)['overall']['false_vacancy_rate'],1)
        preds=[dict(sample_id='A',slot_id='1',state='unknown',occupied_score=.9),dict(sample_id='A',slot_id='2',state='vacant')]
        self.assertEqual(p.evaluate_occupancy(truth,preds)['overall']['unknown_slots'],1)

    def test_invalid_scores_states_truth_and_aliases(self):
        for score in [float('nan'),float('inf'),-.1,1.1,'0.5',True]:
            with self.subTest(score=score),self.assertRaises(ValueError):
                p.evaluate_occupancy(self.truth,[dict(sample_id='A',space_id='1',occupied_score=score)])
        for record in [dict(state='maybe'),dict(),dict(state=0,occupied_score=.9),dict(state=1,slot_id='2')]:
            prediction=dict(sample_id='A',space_id='1',**record)
            with self.assertRaises(ValueError):p.evaluate_occupancy(self.truth,[prediction])
        invalid=[dict(sample_id='A',camera='cam',weather='sunny',labels={'1':None})]
        with self.assertRaises(ValueError):p.evaluate_occupancy(invalid,[])
        with self.assertRaises(ValueError):p.evaluate_occupancy(self.truth,self.pred,threshold=float('nan'))

    def test_partial_timing_coverage_and_strict_timing_validation(self):
        timing=p.evaluate_occupancy(self.truth,self.pred,self.times[:1])['overall']['timing']
        self.assertEqual(timing['coverage'],.5)
        self.assertEqual(timing['total_ms_mean'],40)
        self.assertEqual(timing['missing_sample_ids'],['B'])
        for update in [dict(total_ms=1),dict(total_ms=float('nan')),dict(model_ms=-1),dict(inference_calls=1.5)]:
            records=[dict(self.times[0],**update)]
            with self.assertRaises(ValueError):p.evaluate_occupancy(self.truth,self.pred,records)
        for records in [self.times+self.times[:1],[dict(self.times[0],sample_id='X')]]:
            with self.assertRaises(ValueError):p.evaluate_occupancy(self.truth,self.pred,records)

    def test_timer_accumulates_tiles_and_original_image_total(self):
        # Total 11 ms; preprocess 1, two model calls 2+3, postprocess 1 ms.
        clock=[0,.001,.002,.003,.005,.006,.009,.009,.010,.011]
        calls=[]
        with patch.object(p.time,'perf_counter',side_effect=clock):
            with p.FrameTimer('A',synchronize=lambda:calls.append('sync')) as timer:
                with timer.stage('preprocess'):pass
                with timer.stage('model'):pass
                with timer.stage('model'):pass
                with timer.stage('postprocess'):pass
            record=timer.record()
        self.assertAlmostEqual(record['total_ms'],11)
        self.assertAlmostEqual(record['model_ms'],5)
        self.assertEqual(record['inference_calls'],2)
        self.assertEqual(len(calls),10)
        p.evaluate_occupancy(self.truth,self.pred,[record])

    def test_failed_or_incomplete_timer_cannot_be_reported(self):
        timer=p.FrameTimer('A')
        with self.assertRaises(RuntimeError):timer.record()
        with self.assertRaises(ValueError):
            with timer:
                with timer.stage('model'):raise ValueError('model failed')
        with self.assertRaises(RuntimeError):timer.record()
        with p.FrameTimer('B') as incomplete:
            with incomplete.stage('model'):pass
        with self.assertRaises(RuntimeError):incomplete.record()
        with self.assertRaises(RuntimeError):
            with p.FrameTimer('C') as nested:
                with nested.stage('model'):
                    with nested.stage('model'):pass

    def test_export_json_csv_roundtrip(self):
        report=p.evaluate_occupancy(self.truth,self.pred,self.times,method='fixture')
        with tempfile.TemporaryDirectory() as temp:
            output=p.export_evaluation(report,temp)
            data=json.loads((output/'metrics.json').read_text())
            self.assertEqual(data['method'],'fixture')
            self.assertNotIn('per_slot',data)
            with (output/'per_slot.csv').open() as stream:slots=list(csv.DictReader(stream))
            with (output/'per_frame.csv').open() as stream:frames=list(csv.DictReader(stream))
            self.assertEqual(len(slots),7)
            self.assertEqual(len(frames),2)
            self.assertEqual(frames[1]['total_ms'],'120.0')
            self.assertEqual(sum(s['y_pred']=='' for s in slots),2)
            self.assertEqual(len((output/'aligned_predictions.jsonl').read_text().splitlines()),7)

    def test_frame_ground_truth_and_invalid_input_types(self):
        frame=p.Frame('A','image.jpg','image.xml','cam','sunny','2026-11-04',10,10,
                      (p.ParkingSpace('1',True,((1,1),(8,1),(8,8))),))
        result=p.evaluate_occupancy([frame],[p.OccupancyPrediction('A','1',.8)])
        self.assertEqual(result['overall']['accuracy'],1)
        with self.assertRaises(ValueError):p.evaluate_occupancy([[]],[])
        with self.assertRaises(ValueError):p.evaluate_occupancy(self.truth,self.pred,timings=[[]])
        with self.assertRaises(ValueError):p.evaluate_occupancy(self.truth,self.pred,metadata=[])

    def test_standalone_cli_exports_and_records_input_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            for name,rows in [('truth',self.truth),('predictions',self.pred),('timings',self.times)]:
                (directory/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            module=Path(p.__file__).resolve()
            command=[sys.executable,str(module),'evaluate','--manifest',str(directory/'truth.jsonl'),
                     '--predictions',str(directory/'predictions.jsonl'),'--timings',str(directory/'timings.jsonl'),
                     '--method','synthetic_cli_check','--output',str(directory/'result')]
            result=subprocess.run(command,cwd=temp,check=True,capture_output=True,text=True)
            self.assertAlmostEqual(json.loads(result.stdout)['accuracy'],3/7)
            output=json.loads((directory/'result/metrics.json').read_text())
            self.assertEqual(len(output['metadata']['manifest_sha256']),64)
            self.assertEqual(len(output['metadata']['timings_sha256']),64)

    def test_built_in_hand_check(self):
        self.assertTrue(p.evaluator_self_check()['passed'])

if __name__=='__main__':unittest.main()
