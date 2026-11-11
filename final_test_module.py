"""Step-by-step final-test helpers. Never train or select parameters on test.

Inference remains in the frozen parking_module engine. This file only checks
inputs, invokes its locked runner and exports evidence from saved predictions.
"""
from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import json


METHODS = ("hog_svm", "yolo_fullframe", "yolo_hybrid", "yolo_fullframe_highres")
LABELS = {"hog_svm": "HOG + SVM", "yolo_fullframe": "YOLO full 640",
          "yolo_hybrid": "Hybrid", "yolo_fullframe_highres": "YOLO full pixel-matched"}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot export an empty table")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_inputs(dataset_root, manifest_dir, validation_dir, checkpoint, check_dataset=True):
    """Verify the frozen lock and optionally all selected XML/image paths.

    This reads metadata, not test predictions; it performs no model inference.
    Missing models/references must be restored, never rebuilt using test data.
    """
    import parking_module as parking

    root, md, val = map(Path, (dataset_root, manifest_dir, validation_dir))
    required = [Path(checkpoint), val / "locked_config.json", val / "preregistered_plan.json",
                val / "hog_svm.joblib", val / "geometry_references.joblib", md / "test.jsonl"]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen artifacts:\n" + "\n".join(missing))
    lock = parking.verify_validation_lock(val / "locked_config.json", md, checkpoint)
    plan = _json(val / "preregistered_plan.json")["plan"]
    plan_sha = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if plan_sha != lock["plan_sha256"]:
        raise ValueError("Validation plan differs from the frozen lock")
    records = _rows(md / "test.jsonl")
    if not records or len({r["sample_id"] for r in records}) != len(records):
        raise ValueError("Test manifest must be nonempty and have unique sample IDs")
    if check_dataset:
        resolved_root = root.resolve()
        for index, row in enumerate(records, 1):
            image = (resolved_root / row["image_path"]).resolve()
            if not image.is_relative_to(resolved_root) or not image.is_file():
                raise FileNotFoundError(f"Missing or invalid image path: {image}")
            parking.read_project_frame(row, root)
            if index % 50 == 0 or index == len(records):
                print(f"Checked frozen dataset: {index}/{len(records)} frames", flush=True)
    return dict(protocol=lock["protocol"], test_frames=len(records),
                test_slots=sum(len(r["labels"]) for r in records),
                selected_on=lock["selected_on"], selected=lock["selected"],
                validation_lock_sha256=_sha(val / "locked_config.json"),
                dataset_checked=check_dataset, test_inference_performed=False)


def run_final_test(dataset_root, manifest_dir, validation_dir, checkpoint, output, device="cpu"):
    """Explicit costly step. Refuse existing output to preserve previous evidence."""
    import parking_module as parking

    out, val = Path(output), Path(validation_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output already contains files: {out}. Use a NEW run directory; do not overwrite evidence.")
    # Prevent an output directory from replacing any part of the frozen inputs.
    for protected in (val, Path(manifest_dir), Path(checkpoint).parent, Path(dataset_root)):
        a, b = out.resolve(), protected.resolve()
        if a == b or a.is_relative_to(b) or b.is_relative_to(a):
            raise ValueError(f"Final-test output must be separate from input artifacts: {protected}")
    info = check_inputs(dataset_root, manifest_dir, validation_dir, checkpoint)
    request = dict(status="running", device=str(device), inputs=info,
                   parameter_selection=False, legacy_test_previously_observed=True)
    _write(out / "workflow_status.json", request)
    try:
        reports = parking.run_locked_test(dataset_root, manifest_dir, validation_dir,
                                          checkpoint, out, device=device)
        verify_saved_test(out, manifest_dir, validation_dir)
    except Exception as exc:
        _write(out / "workflow_status.json", dict(request, status="failed", error=str(exc)))
        raise
    _write(out / "workflow_status.json", dict(request, status="complete", methods=list(reports)))
    return reports


def verify_saved_test(test_dir, manifest_dir, validation_dir):
    """Recompute metrics from saved evidence, verify complete coverage and pixel pairing.

    No model is called. Predictions are evaluated as stored states; geometric
    evidence scores are never reinterpreted as occupancy probabilities.
    """
    import parking_module as parking

    out, md, val = map(Path, (test_dir, manifest_dir, validation_dir))
    records = _rows(md / "test.jsonl")
    lock = _json(val / "locked_config.json")
    if not lock.get("complete") or lock.get("selected_on") != "val":
        raise ValueError("Only complete validation locks can accompany final-test evidence")
    for name, digest in lock["manifest_sha256"].items():
        if _sha(md / name) != digest:
            raise ValueError(f"Changed locked split artifact: {name}")
    expected_frames = {r["sample_id"] for r in records}
    expected_keys = {(r["sample_id"], str(slot)) for r in records for slot in r["labels"]}
    reports, pixels = {}, {}
    summary = _json(out / "test_summary.json")
    for method in METHODS:
        base = out / method
        stored = _json(base / "metrics.json")
        meta = stored["metadata"]
        if (stored["method"] != method or meta.get("split") != "test"
                or not meta.get("complete") or meta.get("selected_on") != "val"
                or meta.get("no_test_parameter_selection") is not True
                or meta.get("validation_lock_sha256") != _sha(val / "locked_config.json")):
            raise ValueError(f"Not complete test evidence for the supplied lock: {method}")
        aligned = _rows(base / "aligned_predictions.jsonl")
        if (len(aligned) != len(expected_keys)
                or {(r["sample_id"], r["space_id"]) for r in aligned} != expected_keys):
            raise ValueError(f"Missing/extra/duplicate slot predictions: {method}")
        with (base / "per_frame.csv").open(encoding="utf-8", newline="") as stream:
            frame_rows = list(csv.DictReader(stream))
        if len(frame_rows) != len(expected_frames) or {r["sample_id"] for r in frame_rows} != expected_frames:
            raise ValueError(f"Wrong frame coverage: {method}")
        timings = [dict(sample_id=r["sample_id"], inference_calls=int(r["inference_calls"]),
                        **{k: float(r[k]) for k in ("preprocess_ms", "model_ms", "postprocess_ms", "total_ms")})
                   for r in frame_rows]
        predictions = [dict(sample_id=r["sample_id"], space_id=r["space_id"], state=r["y_pred"]) for r in aligned]
        fresh = parking.evaluate_occupancy(records, predictions, timings, missing_policy="error",
                                            method=method, metadata=meta)
        for key in ("overall", "by_camera", "by_weather", "by_camera_weather"):
            if fresh[key] != stored[key]:
                raise ValueError(f"Saved predictions/timings disagree with metrics: {method}/{key}")
        if summary[method] != fresh["overall"]:
            raise ValueError(f"test_summary differs from evidence: {method}")
        budgets = _json(base / "pixel_budget.json")
        if len(budgets) != len(expected_frames) or {r["sample_id"] for r in budgets} != expected_frames:
            raise ValueError(f"Wrong pixel budget frame coverage: {method}")
        pixels[method] = {r["sample_id"]: r["input_pixels"] for r in budgets}
        fresh["per_frame"] = sorted(fresh["per_frame"], key=lambda r: r["sample_id"])
        reports[method] = fresh
    if pixels["yolo_hybrid"] != pixels["yolo_fullframe_highres"]:
        raise ValueError("Hybrid and high-resolution test pixels differ per frame")
    return reports


def _metric_row(method, group, values):
    keys = ("frames", "slots", "precision", "recall", "f1", "accuracy", "coverage",
            "false_vacancy_rate", "reported_vacancy_error_rate", "mae_vacant_count")
    row = dict(method=method, group=group, **{k: values[k] for k in keys})
    for k in ("preprocess_ms_mean", "model_ms_mean", "postprocess_ms_mean", "total_ms_mean",
              "total_ms_p50", "total_ms_p95", "inference_calls_mean"):
        row[k] = values["timing"][k]
    row["timing_coverage"] = values["timing"]["coverage"]
    return row


def export_tables(test_dir, manifest_dir, validation_dir):
    """Export overall/group CSVs and real stage timing, never parameter search."""
    out = Path(test_dir)
    reports = verify_saved_test(out, manifest_dir, validation_dir)
    tables = out / "tables"
    overall = [_metric_row(name, "all", r["overall"]) for name, r in reports.items()]
    for row in overall:
        budgets = _json(out / row["method"] / "pixel_budget.json")
        row["input_pixels_mean"] = (None if row["method"] == "hog_svm" else
                                     sum(r["input_pixels"] for r in budgets) / len(budgets))
    _csv(tables / "comparison.csv", overall)
    for method, report in reports.items():
        # Plain inputs for the common evaluator, separate from aligned truth evidence.
        predictions = [dict(sample_id=r["sample_id"], space_id=r["space_id"], state=r["y_pred"])
                       for r in report["per_slot"]]
        timings = [dict(sample_id=r["sample_id"], **r["timing"]) for r in report["per_frame"]]
        for name, records in (("predictions.jsonl", predictions), ("timings.jsonl", timings)):
            (out / method / name).write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n"
                                                     for r in records), encoding="utf-8")
    for grouping in ("by_camera", "by_weather", "by_camera_weather"):
        _csv(tables / f"{grouping}.csv", [_metric_row(name, group, metric)
             for name, report in reports.items() for group, metric in report[grouping].items()])
    matrix = [dict(method=name, true_label=true, predicted_label=predicted, count=report["overall"]["confusion_matrix"]["values"][i][j])
              for name, report in reports.items() for i, true in enumerate(("vacant", "occupied"))
              for j, predicted in enumerate(("vacant", "occupied", "unknown"))]
    _csv(tables / "confusion_matrices.csv", matrix)
    timing = [dict(method=name, **{key: row[key] for key in ("sample_id", "camera", "weather", "date")},
                   **row["timing"]) for name, report in reports.items() for row in report["per_frame"]]
    _csv(tables / "timings_per_frame.csv", timing)
    lines = ["# Final-test results", "", "Configuration selected on validation; no test parameter search.", "",
             "The legacy notebook had already exposed the test split; this is not an unseen blind holdout.", "",
             "| Method | F1 | Accuracy | Coverage | Vacant MAE | ms/image |", "|---|---:|---:|---:|---:|---:|"]
    for row in overall:
        lines.append(f"| {LABELS[row['method']]} | {row['f1']:.4f} | {row['accuracy']:.4f} | {row['coverage']:.4f} | {row['mae_vacant_count']:.3f} | {row['total_ms_mean']:.1f} |")
    lines += ["", "Timing is the original-image pipeline, including all tiles. The model stage includes Ultralytics internal preprocessing/NMS and hybrid mapping, not just neural forward time.",
              "", "Read the CPU/GPU execution metadata before comparing these times with validation."]
    (tables / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return overall


def export_examples(test_dir, manifest_dir, validation_dir, dataset_root, per_category=1):
    """Choose deterministic correct/error/unknown examples per camera/weather.

    Selection is descriptive, after evaluation, never feedback into parameters.
    'Correct' means ALL slots known and correct; 'error' means a known wrong
    prediction; 'unknown_only' has abstentions but no known wrong prediction.
    Some groups may have no example in a category; do not invent one.
    """
    from PIL import Image, ImageDraw

    if isinstance(per_category, bool) or not isinstance(per_category, int) or per_category < 1:
        raise ValueError("per_category must be a positive integer")
    out, md, root = map(Path, (test_dir, manifest_dir, dataset_root))
    reports = verify_saved_test(out, md, validation_dir)
    records = {r["sample_id"]: r for r in _rows(md / "test.jsonl")}
    layouts = _json(md.parent / "slot_geometries.json")["layouts"]
    examples = out / "examples"
    if examples.exists() and any(examples.iterdir()):
        raise FileExistsError("Examples already exist; preserve them or choose another test run directory")
    selection, availability = [], []
    colors = {0: (20, 190, 50), 1: (230, 50, 40), None: (255, 170, 0)}
    for method, report in reports.items():
        slots_by_frame = defaultdict(list)
        for slot in report["per_slot"]:
            slots_by_frame[slot["sample_id"]].append(slot)
        grouped = defaultdict(list)
        for sid, slots in slots_by_frame.items():
            wrong = any(s["y_pred"] is not None and s["y_pred"] != s["y_true"] for s in slots)
            unknown = any(s["y_pred"] is None for s in slots)
            category = "error" if wrong else "unknown_only" if unknown else "correct"
            row = records[sid]
            grouped[(row["camera"], row["weather"], category)].append(sid)
        groups = sorted({(r["camera"], r["weather"]) for r in records.values()})
        for camera, weather in groups:
            for category in ("correct", "error", "unknown_only"):
                candidates = sorted(grouped[(camera, weather, category)])
                availability.append(dict(method=method, camera=camera, weather=weather,
                                         category=category, available_frames=len(candidates),
                                         selected_frames=min(len(candidates), per_category)))
                for sid in candidates[:per_category]:
                    record = records[sid]
                    layout = layouts[record["geometry_id"]]
                    relative = Path(method) / camera / weather / category / (hashlib.sha256(sid.encode()).hexdigest()[:16] + ".jpg")
                    target = examples / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with Image.open(root / record["image_path"]) as source:
                        image = source.convert("RGB")
                    draw = ImageDraw.Draw(image)
                    by_slot = {s["space_id"]: s for s in slots_by_frame[sid]}
                    for space in layout["spaces"]:
                        slot = by_slot[str(space["space_id"])]; points = [tuple(p) for p in space["polygon"]]
                        color = colors[slot["y_pred"]]
                        draw.line(points + [points[0]], fill=color, width=3)
                        # English/ASCII labels work with Pillow's default font everywhere.
                        pred = "?" if slot["y_pred"] is None else str(slot["y_pred"])
                        label = f"{slot['space_id']} T{slot['y_true']} P{pred}"
                        x, y = points[0]
                        draw.text((x, y), label, fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))
                    draw.rectangle((0, 0, image.width, 38), fill=(0, 0, 0))
                    draw.text((5, 4), f"{method} | {camera}/{weather} | {category}", fill="white")
                    draw.text((5, 21), "Green=vacant Red=occupied Orange=unknown; T=true P=prediction", fill="white")
                    image.save(target, quality=92)
                    selection.append(dict(method=method, camera=camera, weather=weather, category=category,
                                          sample_id=sid, source_image=record["image_path"], figure=str(relative)))
    _write(examples / "selection.json", dict(rule="First sample_id in sorted order per method/camera/weather/category; no parameter changes",
                                             categories=availability, selected=selection))
    _csv(examples / "availability.csv", availability)
    return selection


def seal_evidence(test_dir):
    """Hash current evidence after tables/images; this is integrity, not scientific approval."""
    root = Path(test_dir)
    if _json(root / "workflow_status.json").get("status") != "complete":
        raise ValueError("Only a complete test run may be sealed")
    for required in (root / "tables/comparison.csv", root / "examples/selection.json"):
        if not required.is_file():
            raise FileNotFoundError(required)
    artifact = root / "evidence_sha256.json"
    hashes = {p.relative_to(root).as_posix(): _sha(p) for p in sorted(root.rglob("*")) if p.is_file() and p != artifact}
    _write(artifact, dict(algorithm="sha256", files=hashes))
    return artifact
