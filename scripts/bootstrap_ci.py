"""Bootstrap theo ngày cho F1 và chênh lệch F1 giữa các phương pháp (chỉ đọc dự đoán đã lưu).

Không gọi model, không chọn tham số. Đơn vị lấy mẫu là cặp (camera, ngày), vì các ảnh trong cùng
ngày không độc lập. Lấy mẫu có hoàn lại *trong từng camera* để giữ nguyên cơ cấu camera
(PUCPR chiếm 59,5% số ô). F1 tính đúng như parking_module: 2TP / (2TP + FP + FN + unknown_occupied).

Dùng:  python scripts/bootstrap_ci.py reports/final_test_v1 [--output bootstrap_ci.csv]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

METHODS = ("hog_svm", "yolo_fullframe", "yolo_hybrid", "yolo_fullframe_highres")
PAIRS = (("hog_svm", "yolo_hybrid"), ("yolo_hybrid", "yolo_fullframe_highres"),
         ("yolo_hybrid", "yolo_fullframe"), ("hog_svm", "yolo_fullframe_highres"))


def load(test_dir):
    """Return {method: {(camera, date): (tp, fp, fn_plus_unknown_occ)}} from aligned_predictions."""
    counts, keys = {}, set()
    for method in METHODS:
        per_day = {}
        with (Path(test_dir) / method / "aligned_predictions.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                r = json.loads(line)
                key = (r["camera"], r["date"])
                per_day.setdefault(key, [0, 0, 0])
                if r["y_true"] == 1 and r["y_pred"] == 1:
                    per_day[key][0] += 1
                elif r["y_true"] == 0 and r["y_pred"] == 1:
                    per_day[key][1] += 1
                elif r["y_true"] == 1:  # vacant hoặc unknown: đều bị phạt ở recall/F1 chính
                    per_day[key][2] += 1
        counts[method] = per_day
        keys |= set(per_day)
    return counts, sorted(keys)


def f1(tp, fp, fn):
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("test_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None, help="Mặc định: <test_dir>/tables/bootstrap_ci.csv")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    counts, days = load(args.test_dir)
    cameras = sorted({c for c, _ in days})
    index = {k: i for i, k in enumerate(days)}
    stats = {m: np.array([counts[m].get(k, [0, 0, 0]) for k in days], dtype=np.int64) for m in METHODS}
    groups = [np.array([index[k] for k in days if k[0] == cam]) for cam in cameras]
    rng = np.random.default_rng(args.seed)

    draws = np.empty((args.resamples, len(days)), dtype=np.int64)  # số lần mỗi ngày được chọn
    for b in range(args.resamples):
        weights = np.zeros(len(days), dtype=np.int64)
        for group in groups:
            np.add.at(weights, rng.choice(group, size=len(group), replace=True), 1)
        draws[b] = weights

    def boot_f1(method):
        tot = draws @ stats[method]
        return np.array([f1(*row) for row in tot])

    point = {m: f1(*stats[m].sum(axis=0)) for m in METHODS}
    boot = {m: boot_f1(m) for m in METHODS}
    rows = []
    for m in METHODS:
        lo, hi = np.percentile(boot[m], [2.5, 97.5])
        rows.append(dict(kind="f1", a=m, b="", estimate=round(point[m], 4), ci_low=round(lo, 4),
                         ci_high=round(hi, 4), p_a_not_better=""))
    for a, b in PAIRS:
        diff = boot[a] - boot[b]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        rows.append(dict(kind="f1_diff", a=a, b=b, estimate=round(point[a] - point[b], 4),
                         ci_low=round(lo, 4), ci_high=round(hi, 4),
                         p_a_not_better=round(float((diff <= 0).mean()), 4)))
    output = args.output or Path(args.test_dir) / "tables" / "bootstrap_ci.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(days)} cụm (camera, ngày): " + ", ".join(f"{c}={len(g)}" for c, g in zip(cameras, groups)))
    for r in rows:
        label = r["a"] if r["kind"] == "f1" else f"{r['a']} - {r['b']}"
        print(f"{r['kind']:<8} {label:<45} {r['estimate']:>8} [{r['ci_low']}, {r['ci_high']}] {r['p_a_not_better']}")
    print("Đã ghi:", output)


if __name__ == "__main__":
    main()
