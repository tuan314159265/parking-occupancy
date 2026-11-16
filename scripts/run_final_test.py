"""Run the locked final test from the command line (same steps as notebook section 6.3).

1. check_inputs: lock, checkpoint, HOG model, geometry references, library versions, test images.
2. run_final_test: frozen parking_module runner; refuses a non-empty output directory.
3. export_tables / export_examples / seal_evidence: tables, example figures, SHA256 of evidence.

Nothing here selects parameters or trains on test data. Use --check-only to verify
inputs without inference.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import final_test_module as flow


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/raw/PKLot",
                        help="Thư mục chứa UFPR04/UFPR05/PUCPR (ảnh toàn cảnh + XML)")
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "data/manifests/pklot_v1/within_camera")
    parser.add_argument("--validation-dir", type=Path, default=ROOT / "outputs/validation_v1",
                        help="Chứa locked_config.json, preregistered_plan.json, hog_svm.joblib, geometry_references.joblib")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/yolo11n.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/locked_test_v1",
                        help="Thư mục MỚI cho lượt test; không ghi đè bằng chứng cũ")
    parser.add_argument("--device", default="cpu", help="cpu, 0 (GPU đầu tiên) hoặc auto")
    parser.add_argument("--examples-per-category", type=int, default=1)
    parser.add_argument("--check-only", action="store_true", help="Chỉ kiểm tra khóa và dữ liệu, không suy luận")
    args = parser.parse_args(argv)

    info = flow.check_inputs(args.dataset_root, args.manifest_dir, args.validation_dir, args.checkpoint)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    if args.check_only:
        return
    flow.run_final_test(args.dataset_root, args.manifest_dir, args.validation_dir, args.checkpoint,
                        args.output, device=args.device)
    overall = flow.export_tables(args.output, args.manifest_dir, args.validation_dir)
    flow.export_examples(args.output, args.manifest_dir, args.validation_dir, args.dataset_root,
                         per_category=args.examples_per_category)
    sealed = flow.seal_evidence(args.output)
    for row in overall:
        print(f"{flow.LABELS[row['method']]:<24} F1={row['f1']:.4f} acc={row['accuracy']:.4f} "
              f"coverage={row['coverage']:.4f} MAE={row['mae_vacant_count']:.3f} {row['total_ms_mean']:.1f} ms/ảnh")
    print("Bảng:", args.output / "tables")
    print("Hash bằng chứng:", sealed)


if __name__ == "__main__":
    main()
