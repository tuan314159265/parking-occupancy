"""Export compact, reviewable evidence after a complete task-12 run."""
import argparse
import csv
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import parking_module as parking


def write_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='outputs/validation_v1')
    parser.add_argument('--output',default='reports/validation_v1')
    args=parser.parse_args()
    source,out=Path(args.run),Path(args.output)
    summary=json.loads((source/'validation_summary.json').read_text())
    lock=json.loads((source/'locked_config.json').read_text())
    if not summary['complete'] or not lock['complete'] or summary['selected_on']!='val':
        raise ValueError('Publish only a complete, locked validation run')
    registration=json.loads((source/'preregistered_plan.json').read_text())
    plan=registration['plan']
    parking.verify_validation_lock(source/'locked_config.json',REPO/'data/manifests'/plan['dataset_version']/plan['protocol'],REPO/'models'/plan['checkpoint']['filename'])
    hog_search=json.loads((source/'hog_search.json').read_text())
    chosen_hog=next(item for item in hog_search if item['C']==lock['selected']['hog_C'] and item['threshold']==lock['selected']['hog_threshold'])
    if any(not math.isclose(summary['methods']['hog_svm'][k],chosen_hog['metrics'][k],rel_tol=0,abs_tol=1e-12) for k in ('f1','accuracy','coverage','mae_vacant_count')):
        raise ValueError('Fresh HOG benchmark does not reproduce search predictions')
    for method,expected in summary['selected']['methods'].items():
        actual=summary['methods'][method]
        if any(not math.isclose(actual[k],expected[k],rel_tol=0,abs_tol=1e-12) for k in ('f1','accuracy','coverage','mae_vacant_count')):
            raise ValueError('Fresh benchmark does not reproduce search predictions')
    out.mkdir(parents=True,exist_ok=True)
    for name in ('locked_config.json','preregistered_plan.json','tile_plans.json','comparison.csv','validation_summary.json'):
        (out/name).write_text((source/name).read_text())
    fields=('f1','accuracy','precision','recall','coverage','false_vacancy_rate','reported_vacancy_error_rate','mae_vacant_count')
    ledger=json.loads((source/'detector_search.json').read_text())
    rows=[]
    for item in ledger:
        row=dict(candidate_id=item['candidate_id'],padding=item['tiling']['padding'],requested_overlap=item['tiling']['overlap'],
                 merged_confidence=item['confidence'],**item['matching'],mean_f1=item['mean_f1'],mean_coverage=item['mean_coverage'],
                 mean_mae_vacant_count=item['mean_mae_vacant_count'],hybrid_mean_input_pixels=item['hybrid_mean_input_pixels'],
                 delta_mean_f1_from_default=item['delta_mean_f1_from_default'],improved_over_default=item['improved_over_default'],selected=item['selected'])
        for method,metrics in item['methods'].items():
            row.update({method+'_'+key:metrics[key] for key in fields})
        rows.append(row)
    write_csv(out/'detector_search.csv',rows)
    hog=json.loads((source/'hog_search.json').read_text())
    default=next(h for h in hog if h['C']==1 and h['threshold']==0)
    write_csv(out/'hog_search.csv',[dict(C=h['C'],threshold=h['threshold'],
              **{key:h['metrics'][key] for key in fields},delta_f1_from_C1_threshold0=h['metrics']['f1']-default['metrics']['f1'],
              selected=(h['C']==lock['selected']['hog_C'] and h['threshold']==lock['selected']['hog_threshold'])) for h in hog])
    for group in ('by_camera','by_weather','by_camera_weather'):
        rows=[]
        for method in summary['methods']:
            metrics=json.loads((source/'selected'/method/'metrics.json').read_text())
            for name,values in metrics[group].items():
                rows.append(dict(method=method,group=name,frames=values['frames'],slots=values['slots'],
                                 occupied_support=values['occupied_support'],**{key:values[key] for key in fields},
                                 ms_per_original_image=values['timing']['total_ms_mean']))
        write_csv(out/(group+'.csv'),rows)
    budgets=[]
    for method in summary['methods']:
        for row in json.loads((source/'selected'/method/'pixel_budget.json').read_text()):
            budgets.append(dict(method=method,sample_id=row['sample_id'],inference_calls=row['inference_calls'],
                                yolo_tensor_pixels=None if method=='hog_svm' else row['input_pixels']))
    write_csv(out/'pixel_budget.csv',budgets)
    versions={}
    for name in ('Pillow','numpy','opencv-python','opencv-python-headless','torch','torchvision','ultralytics',
                 'scikit-image','scikit-learn','scipy','joblib','threadpoolctl'):
        versions[name]=importlib.metadata.version(name)
    cpu='unknown'
    if Path('/proc/cpuinfo').is_file():
        cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                  if line.startswith('model name')),'unknown')
    provenance=dict(environment=summary['environment'],packages=versions,cpu=cpu,platform=platform.platform(),
        torch_cpu_threads=plan['cpu_threads'],svm_fit_blas_thread_limit=plan['cpu_threads'],opencv_threads=1,
        other_blas_threads='not separately constrained during HOG feature/decision inference',checkpoint_sha256=lock['checkpoint_sha256'],engine_sha256=lock['engine_sha256'],
        legacy_test_outputs_present_at_start=True,reviewed_main_commit='293893df',
        new_test_inference_performed=False,validation_only=True,
        raw_search_sha256=parking._validation_sha(source/'detector_search.json'),
        raw_hog_search_sha256=parking._validation_sha(source/'hog_search.json'))
    parking._validation_write(out/'provenance.json',provenance)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods=['hog_svm','yolo_fullframe','yolo_hybrid','yolo_fullframe_highres']
    names=['HOG + SVM','YOLO full 640','Hybrid 640','Full pixel-matched']
    colors=['#656D78','#3974B5','#DA8546','#4A986C']
    metrics=[summary['methods'][name] for name in methods]
    pixels={method:sum(row['input_pixels'] for row in json.loads((source/'selected'/method/'pixel_budget.json').read_text()))/len(json.loads((source/'selected'/method/'pixel_budget.json').read_text()))
            for method in methods}
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    axes[0].bar(names,[m['f1'] for m in metrics],color=colors)
    axes[0].set_ylim(0,1.05);axes[0].set_ylabel('Occupied F1 (unknown penalized)')
    for i,m in enumerate(metrics):axes[0].text(i,m['f1']+.02,f"{m['f1']:.3f}",ha='center')
    axes[1].bar(names,[m['timing']['total_ms_mean'] for m in metrics],color=colors)
    axes[1].set_ylabel('Full pipeline ms / original image (CPU)')
    axes[2].bar(names[1:],[pixels[m]/1e6 for m in methods[1:]],color=colors[1:])
    axes[2].set_ylabel('Mean YOLO tensor megapixels / image')
    for ax in axes:ax.tick_params(axis='x',rotation=25);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Validation only — parameters selected on this split; not final-test results')
    fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    selected=lock['selected']
    lines=['# Kết quả validation v1','',
           'Đã chạy đủ 1.182 ảnh train và 270 validation (15.120 ô); đây là số liệu dùng chọn tham số, không phải test độc lập.',
           'YOLO11n cùng checkpoint ở tất cả nhánh. Tập test không được chạy lại trong task này.','',
           '| Phương pháp | F1 | Accuracy | Coverage | MAE ô trống | ms/ảnh CPU |',
           '|---|---:|---:|---:|---:|---:|']
    for name,m in zip(names,metrics):
        lines.append(f"| {name} | {m['f1']:.4f} | {m['accuracy']:.4f} | {m['coverage']:.4f} | {m['mae_vacant_count']:.3f} | {m['timing']['total_ms_mean']:.1f} |")
    full,hybrid,high=(summary['methods'][m] for m in ('yolo_fullframe','yolo_hybrid','yolo_fullframe_highres'))
    lines+=['','## Cấu hình đã khóa','',f"- HOG/SVM: C={selected['hog_C']}, ngưỡng margin={selected['hog_threshold']}.",
            f"- Hybrid: padding={selected['tiling']['padding']} px; overlap yêu cầu={selected['tiling']['overlap']}.",
            f"- Confidence hậu xử lý dùng chung={selected['confidence']}; YOLO confidence floor cố định=0,15.",
            '- Quy tắc ghép: `'+json.dumps(selected['occupancy_config'],sort_keys=True)+'`.',
            '- Checkpoint SHA256: `'+lock['checkpoint_sha256']+'`.',
            '- Toàn bộ 9 cấu hình HOG và 144 tổ hợp joint có trong hog_search.csv và detector_search.csv, gồm cả cấu hình không cải thiện.','',
            '## Đối chiếu pixel và diễn giải','',
            f"Hybrid so với full 640: ΔF1={hybrid['f1']-full['f1']:+.4f}; so với toàn ảnh khớp pixel: ΔF1={hybrid['f1']-high['f1']:+.4f}.",
            'Pixel tensor của hybrid và đối chứng bằng nhau trên từng ảnh; padding/overlap hiệu dụng và fallback nằm trong tile_plans.json. Pixel bằng nhau không đồng nghĩa FLOPs/latency bằng nhau.',
            'Không quy toàn bộ lợi ích so với full 640 cho ROI; cần đọc cùng kết quả full pixel-matched và thời gian. Ngân sách pixel không áp dụng cho HOG/SVM.',
            'Timing là một lượt đo sau warm-up trên CPU local, có thể chịu tải của máy; không so trực tiếp với timing GPU Colab. Kết quả validation có thiên lệch chọn tham số.','',
            '## Khóa và lịch sử test','',
            'locked_config.json giữ checkpoint, split, model, tham chiếu train, engine và phiên bản; model HOG/geometry reference cụ thể nằm trong outputs/validation_v1/ và cần giữ để chạy test đúng khóa.',
            'Notebook cũ đã lưu output test tại main 293893d. Task này không dùng điểm test để xếp hạng nhưng không gọi test v1 là holdout chưa quan sát; xem docs/validation.md về lịch sử.',
            'Ảnh/feature/checkpoint/bundle lớn không được đưa vào bản báo cáo Git này. Bằng chứng chi tiết per-slot/per-frame ở outputs/validation_v1/selected/.','']
    (out/'README.md').write_text('\n'.join(lines))
    hashes={path.name:parking._validation_sha(path) for path in out.iterdir() if path.is_file() and path.name!='artifact_hashes.json'}
    parking._validation_write(out/'artifact_hashes.json',hashes)
    print('Published compact validation evidence:',out)

if __name__=='__main__':main()
