"""Validation safeguards and exact cached/full geometric scoring equivalence."""
import copy
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import parking_module as p

class ValidationTests(unittest.TestCase):
    def test_exact_pixel_budgets_and_stride_checks(self):
        for n in [1,2,4]:
            shape=p.validation_pixel_shape(n,640)
            self.assertEqual(shape[0]*shape[1],n*640*640)
        self.assertEqual(p.validation_pixel_shape(2,640),[640,1280])
        for n,size in [(3,640),(4,641),(1,0)]:
            with self.assertRaises(ValueError):p.validation_pixel_shape(n,size)

    def test_cached_positive_pair_scores_preserve_all_grid_decisions(self):
        layout=dict(spaces=[dict(space_id='a',polygon=[[0,10],[10,10],[10,20],[0,20]]),
                            dict(space_id='b',polygon=[[10,10],[20,10],[20,20],[10,20]])])
        boxes=[p.VehicleDetection((3,0,13,20),.7,'car'),p.VehicleDetection((3,0,13,20),.3,'truck')]
        kept=None
        for fraction,overlap,margin in itertools.product([.33,.5,.67],[.2,.3,.4],[0,.05]):
            cfg=dict(p.DEFAULT_OCCUPANCY_CONFIG,lower_fraction=fraction,min_overlap=overlap,ambiguity_margin=margin)
            expected=p.infer_parking_occupancy('sample',boxes,layout,cfg)
            rows=[r for r in p.score_vehicle_spaces(boxes,layout,fraction) if r['overlap']>0]
            actual=p.infer_parking_occupancy('sample',boxes,layout,cfg,_scored_pairs=rows,_kept_indices=kept)
            kept=actual['kept_detection_indices']
            self.assertEqual(expected,actual)

    def test_edge_first_nms_is_class_agnostic(self):
        candidates=[dict(edge=True,detection=p.VehicleDetection((0,0,10,10),.9,'car')),
                    dict(edge=False,detection=p.VehicleDetection((0,0,10,10),.6,'truck'))]
        result=p._validation_nms(candidates,.5)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0].class_name,'truck')

    def test_fake_detector_exact_hybrid_and_fullframe_tensor_pixels(self):
        class Detector:
            def warmup(self,shape):pass
            def predict(self,image,shape):return [],shape[0]*shape[1]
        layout=dict(width=640,height=640,spaces=[
            dict(space_id='a',polygon=[[20,20],[60,20],[60,60],[20,60]]),
            dict(space_id='b',polygon=[[580,580],[620,580],[620,620],[580,620]])])
        image=np.zeros((640,640,3),dtype=np.uint8)
        plan=dict(detector=dict(base_imgsz=640),hybrid=dict(min_box_size_px=2,edge_margin_px=4,global_nms_iou=.5))
        variant=dict(kind='hybrid',tiling=dict(padding=0,overlap=.2,grid=[2,2],max_overlap=.6,min_tile_size=1,output_size=None))
        boxes,budget=p._validation_detect(image,layout,Detector(),variant,plan)
        shape=p.validation_pixel_shape(budget['inference_calls'],640)
        _,control=p._validation_detect(image,layout,Detector(),dict(kind='full',geometry_id='g',shapes={'g':shape}),plan)
        self.assertEqual(boxes,[])
        self.assertEqual(budget['input_pixels'],control['input_pixels'])
        self.assertEqual(control['inference_calls'],1)

    def test_raw_svm_margins_become_states_not_probability_scores(self):
        rows=[dict(sample_id='frame',camera='cam',weather='sunny',labels={'a':0,'b':1})]
        quality={'frame':dict(frame_flags=[],slot_flags={})}
        preds=p._validation_hog_predictions(rows,np.asarray([-4.,8.]),0,quality,'unknown')
        self.assertNotIn('occupied_score',preds[0])
        self.assertEqual(p.evaluate_occupancy(rows,preds,missing_policy='error')['overall']['accuracy'],1)

    def test_lock_checks_each_artifact_and_rejects_partial_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);md=root/'split';md.mkdir()
            for name in ['train.jsonl','val.jsonl','test.jsonl']:(md/name).write_text(name)
            checkpoint=root/'model.pt';checkpoint.write_bytes(b'model')
            hog=root/'hog_svm.joblib';hog.write_bytes(b'hog')
            refs=root/'geometry_references.joblib';refs.write_bytes(b'geometry')
            lock=dict(complete=True,selected_on='val',manifest_sha256={n:p._validation_sha(md/n) for n in ['train.jsonl','val.jsonl','test.jsonl']},
                      checkpoint_sha256=p._validation_sha(checkpoint),hog_model_sha256=p._validation_sha(hog),
                      geometry_reference_sha256=p._validation_sha(refs),engine_sha256=p.validation_engine_sha256())
            path=root/'locked_config.json';p._validation_write(path,lock)
            self.assertEqual(p.verify_validation_lock(path,md,checkpoint),lock)
            for target in [checkpoint,hog,refs,md/'train.jsonl',md/'val.jsonl',md/'test.jsonl']:
                data=target.read_bytes();target.write_bytes(data+b'changed')
                with self.assertRaises(ValueError):p.verify_validation_lock(path,md,checkpoint)
                target.write_bytes(data)
            lock['complete']=False;p._validation_write(path,lock)
            with self.assertRaisesRegex(ValueError,'complete'):p.verify_validation_lock(path,md,checkpoint)

    def test_test_entrypoint_checks_lock_before_model_or_test_images(self):
        with patch.object(p,'verify_validation_lock',side_effect=ValueError('checkpoint mismatch')):
            with patch.object(p,'_ValidationYOLO') as detector,patch.object(p,'_validation_benchmark') as infer:
                with self.assertRaisesRegex(ValueError,'checkpoint mismatch'):
                    p.run_locked_test('data','manifests','validation','checkpoint','output')
                detector.assert_not_called();infer.assert_not_called()

    def test_warmup_pixel_observations_are_reset_before_real_batch(self):
        from types import SimpleNamespace
        detector=object.__new__(p._ValidationYOLO)
        detector._pixels=[640*640]  # A backend GPU warm-up forward.
        model=SimpleNamespace(_forward_pre_hooks={0:detector._observe_input})
        p._validation_reset_pixel_observers(SimpleNamespace(model=SimpleNamespace(model=model)))
        self.assertEqual(detector._pixels,[])
        tensor=np.zeros((1,3,640,1280),dtype=np.float32)
        detector._observe_input(None,(tensor,))
        self.assertEqual(sum(detector._pixels),640*1280)

    def test_validation_mode_skips_duplicate_training_and_default_never_runs_test(self):
        notebook=json.loads(Path(__file__).resolve().parents[1].joinpath('notebooks/01_pklot_run_all.ipynb').read_text())
        cells={c.get('id'):''.join(c['source']) for c in notebook['cells']}
        namespace=dict(VALIDATION_ONLY_MODE=True)
        for cell in ['hog-svm-code','full-frame-code','hybrid-code','occupancy-code','pipeline-code','validation-code','validation-geometry-code']:
            exec(cells[cell],namespace)
        self.assertNotIn('FULLFRAME_DETECTOR',namespace)
        with patch.object(p,'run_locked_test') as inference:
            exec(cells['final-test-code'],dict(parking=p,RUN_FINAL_TEST=False))
            inference.assert_not_called()

    def test_locked_test_uses_frozen_params_without_any_search(self):
        import joblib
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);md=root/'manifests'/'within_camera';md.mkdir(parents=True)
            record=dict(sample_id='fixture',geometry_id='g')
            (md/'test.jsonl').write_text(json.dumps(record)+'\n')
            layout=dict(width=640,height=640,spaces=[dict(space_id='a',polygon=[[20,20],[60,20],[60,60],[20,60]])])
            (md.parent/'slot_geometries.json').write_text(json.dumps(dict(layouts={'g':layout})))
            valdir=root/'validation';valdir.mkdir()
            for name in ['geometry_references','hog_svm']:joblib.dump({},valdir/(name+'.joblib'))
            plan=dict(cpu_threads=1,detector=dict(base_imgsz=640))
            p._validation_write(valdir/'preregistered_plan.json',dict(plan=plan))
            selected=dict(tiling=dict(grid=[1,1],padding=0),highres_shapes={'g':[640,640]},
                          occupancy_config=dict(p.DEFAULT_OCCUPANCY_CONFIG),confidence=.25)
            lock=dict(selected=selected,plan_sha256=p._validation_hash(plan),protocol='within_camera')
            p._validation_write(valdir/'locked_config.json',lock)
            def benchmark(*args):return dict(metadata={},overall=dict(f1=.7)),[]
            with patch.object(p,'verify_validation_lock',return_value=lock),patch.object(p,'_ValidationYOLO'), \
                 patch.object(p,'_validation_benchmark',side_effect=benchmark) as infer,patch.object(p,'export_evaluation'), \
                 patch.object(p,'run_validation_search',side_effect=AssertionError('test must not search')):
                reports=p.run_locked_test(root,md,valdir,root/'checkpoint',root/'test-output')
            self.assertEqual(infer.call_count,4)
            for call in infer.call_args_list:
                self.assertEqual(call.args[7],selected['occupancy_config'])
                self.assertEqual(call.args[8],.25)
            self.assertTrue(all(r['metadata']['split']=='test' and r['metadata']['no_test_parameter_selection'] for r in reports.values()))

    def test_plan_contains_bounded_grid_default_and_checkpoint_digest(self):
        plan=json.loads(Path(__file__).resolve().parents[1].joinpath('configs/validation_v1.json').read_text())
        self.assertEqual(len(plan['checkpoint']['sha256']),64)
        n=len(plan['hybrid']['padding'])*len(plan['hybrid']['overlap'])*len(plan['detector']['confidence'])*len(plan['matching']['lower_fraction'])*len(plan['matching']['min_overlap'])*len(plan['matching']['ambiguity_margin'])
        self.assertEqual(n,144)
        self.assertIn(plan['default']['padding'],plan['hybrid']['padding'])
        self.assertIn(plan['default']['overlap'],plan['hybrid']['overlap'])
        self.assertFalse(plan['detector']['rect'])

if __name__=='__main__':unittest.main()
