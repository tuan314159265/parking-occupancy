"""Standalone PKLot module for GitHub → Colab.

Requires Python >=3.10 and Pillow; geometry analysis additionally uses NumPy/OpenCV.
Import directly or run:
  python parking_module.py prepare
  python parking_module.py audit --root data/raw/PKLot
No repository package installation required.
"""

from dataclasses import dataclass
from typing import Protocol

@dataclass(frozen=True)
class ParkingSpace:
    space_id: str
    occupied: bool
    polygon: tuple[tuple[float, float], ...]
    # Parking-space geometry, never vehicle ground-truth boxes.

@dataclass(frozen=True)
class Frame:
    sample_id: str
    image_path: str
    xml_path: str
    camera: str
    weather: str
    date: str
    width: int
    height: int
    spaces: tuple[ParkingSpace, ...]
    directory_date: str = ""
    date_source: str = "directory"

@dataclass(frozen=True)
class VehicleDetection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_name: str

class VehicleDetector(Protocol):
    def predict(self, image_path: str) -> list[VehicleDetection]: ...

@dataclass(frozen=True)
class OccupancyPrediction:
    sample_id: str
    space_id: str
    occupied_score: float | None


"""Strict full-frame PKLot loader and reproducible audit CLI."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image

WEATHER = {"sunny": "sunny", "rainy": "rainy", "cloudy": "overcast", "overcast": "overcast"}
AUDIT_VERSION = 2

CAMERAS = {"UFPR04", "UFPR05", "PUCPR", "PUC", "parking1a", "parking1b", "parking2"}

def metadata(path):
    parts = path.parts
    for i, part in enumerate(parts):
        if part.lower() in WEATHER and i > 0 and i + 1 < len(parts):
            camera, day = parts[i-1], parts[i+1]
            if camera not in CAMERAS:
                raise ValueError(f"unknown camera: {camera}")
            # Some PKLot directory dates are mistyped. Prefer the acquisition
            # timestamp in the file name; retain the directory value in Frame.
            import re
            timestamp = re.fullmatch(r"(\d{4}-\d{2}-\d{2})_\d{2}_\d{2}_\d{2}", path.stem)
            if timestamp:
                acquisition_day = date.fromisoformat(timestamp.group(1)).isoformat()
            else:
                acquisition_day = date.fromisoformat(day).isoformat()
            return camera, WEATHER[part.lower()], acquisition_day
    raise ValueError("expected camera/weather/YYYY-MM-DD/image.xml")

def read_frame(xml_path, root):
    xml_path, root = Path(xml_path), Path(root)
    candidates = [p for p in xml_path.parent.iterdir()
                  if p.stem == xml_path.stem and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if len(candidates) != 1:
        raise ValueError(f"expected one matching image, found {len(candidates)}")
    image_path = candidates[0]
    with Image.open(image_path) as image:
        width, height = image.size
        image.verify()
    with Image.open(image_path) as image:
        image.load()  # decode pixels, also catches truncated JPEGs
    camera, weather, day = metadata(xml_path.relative_to(root))
    tree = ET.parse(xml_path).getroot()
    if tree.tag != "parking":
        raise ValueError(f"unexpected XML root: {tree.tag}")
    spaces, ids = [], set()
    for element in tree.findall("space"):
        sid, label = element.get("id"), element.get("occupied")
        if not sid or sid in ids or label not in {"0", "1"}:
            raise ValueError(f"invalid/duplicate space id or occupancy: {sid}/{label}")
        ids.add(sid)
        contour = element.find("contour")
        points = [] if contour is None else [p for p in contour if p.tag.lower() == "point"]
        polygon = tuple((float(p.attrib["x"]), float(p.attrib["y"])) for p in points)
        if len(polygon) < 3 or len(set(polygon)) != len(polygon):
            raise ValueError(f"invalid contour: {sid}")
        if any(not math.isfinite(x) or not math.isfinite(y) or
               not 0 <= x <= width or not 0 <= y <= height for x, y in polygon):
            raise ValueError(f"out-of-image/nonfinite contour: {sid}")
        area2 = abs(sum(x*v-u*y for (x,y),(u,v) in zip(polygon, polygon[1:]+polygon[:1])))
        if area2 <= 0:
            raise ValueError(f"zero-area contour: {sid}")
        spaces.append(ParkingSpace(sid, label == "1", polygon))
    if not spaces:
        raise ValueError("no annotated spaces")
    return Frame(xml_path.relative_to(root).with_suffix("").as_posix(),
                 image_path.relative_to(root).as_posix(), xml_path.relative_to(root).as_posix(),
                 camera, weather, day, width, height, tuple(spaces),
                 xml_path.parent.name, "filename" if xml_path.stem.startswith(day+"_") else "directory")

def split_for(frame, seed=42):
    # Use date globally: even simultaneous UFPR camera views stay in one split.
    bucket = int(hashlib.sha256(f"{seed}:{frame.date}".encode()).hexdigest()[:16], 16) / 2**64
    return "train" if bucket < .70 else "val" if bucket < .85 else "test"

def audit(root, output, seed=42):
    root, output = Path(root).resolve(), Path(output)
    if not root.is_dir():
        raise ValueError(f"dataset root does not exist: {root}")
    xmls = sorted(root.rglob("*.xml"))
    if not xmls:
        raise ValueError(f"no full-frame XML files under {root}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit_summary.json").unlink(missing_ok=True)  # interrupted audit must not leave a usable cache
    stats, rejected, count = Counter(), 0, 0
    with (output/"valid_samples.jsonl").open("w") as valid, (output/"issues.jsonl").open("w") as issues:
        for path in xmls:
            try:
                frame = read_frame(path, root)
                row = asdict(frame)
                row["split"] = split_for(frame, seed)
                valid.write(json.dumps(row)+"\n")
                count += 1
                stats[f"camera/{frame.camera}"] += 1
                stats[f"weather/{frame.weather}"] += 1
                stats[f"split/{row['split']}"] += 1
                if frame.directory_date != frame.date:
                    stats["date_directory_mismatch"] += 1
                for s in frame.spaces:
                    stats["occupied" if s.occupied else "vacant"] += 1
            except (ValueError, OSError, ET.ParseError, KeyError) as exc:
                rejected += 1
                issues.write(json.dumps({"path":path.relative_to(root).as_posix(), "error":str(exc)})+"\n")
        orphan = 0
        for image in sorted(root.rglob("*")):
            if image.suffix.lower() in {".jpg", ".jpeg", ".png"} and not image.with_suffix(".xml").exists():
                orphan += 1
                issues.write(json.dumps({"path":image.relative_to(root).as_posix(), "error":"missing XML"})+"\n")
    summary = dict(audit_version=AUDIT_VERSION, root=str(root), seed=seed, xml_count=len(xmls), valid_frames=count,
                   rejected_frames=rejected, images_without_xml=orphan, counts=dict(stats))
    (output/"audit_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    return summary

def audit_main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--output", default="outputs")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    result = audit(args.root, args.output, args.seed)
    print(json.dumps(result, indent=2))
    if result["valid_frames"] == 0:
        raise SystemExit("No valid frames; inspect issues.jsonl")



"""Binary occupancy metrics: occupied=1. Vehicle mAP is not defined by PKLot XML."""
def occupancy_metrics(truth, predicted):
    truth, predicted = list(truth), list(predicted)
    if not truth or len(truth) != len(predicted):
        raise ValueError("expected equal nonempty lists")
    if any(type(x) not in (bool, int) or x not in (0, 1) for x in truth+predicted):
        raise ValueError("labels must be binary")
    tp = sum(t == 1 and p == 1 for t,p in zip(truth,predicted))
    tn = sum(t == 0 and p == 0 for t,p in zip(truth,predicted))
    fp = sum(t == 0 and p == 1 for t,p in zip(truth,predicted))
    fn = sum(t == 1 and p == 0 for t,p in zip(truth,predicted))
    div = lambda a,b: a/b if b else 0.0
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, accuracy=div(tp+tn,len(truth)),
                precision=div(tp,tp+fp), recall=div(tp,tp+fn), f1=div(2*tp,2*tp+fp+fn))


# Shared occupancy evaluator: no detector-specific dependencies.
from contextlib import contextmanager
from numbers import Integral, Real
import time


def _eval_number(value, field, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    value = float(value)
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{field} outside allowed range")
    return value


def _eval_id(value, field):
    if isinstance(value, bool) or not isinstance(value, (str, Integral)) or not str(value).strip():
        raise ValueError(f"{field} must be a nonempty string or integer")
    return str(value)


def _eval_records(records):
    return _read_jsonl(records) if isinstance(records, (str, Path)) else list(records)


def _eval_state(value):
    if value is None or (isinstance(value, str) and value == "unknown"):
        return None
    if isinstance(value, (bool, Integral)) and value in (0, 1):
        return int(value)
    if isinstance(value, str) and value in ("vacant", "occupied"):
        return int(value == "occupied")
    raise ValueError("state must be 0/vacant, 1/occupied or None/unknown")


def _prediction_record(record, threshold):
    record = asdict(record) if isinstance(record, OccupancyPrediction) else record
    if not isinstance(record, dict):
        raise ValueError("Each prediction must be a dictionary or OccupancyPrediction")
    sample = _eval_id(record.get("sample_id"), "sample_id")
    space = _eval_id(record.get("space_id", record.get("slot_id")), "space_id")
    if "slot_id" in record and _eval_id(record["slot_id"], "slot_id") != space:
        raise ValueError("slot_id and space_id disagree")
    if "occupied_score" not in record and "state" not in record:
        raise ValueError("Prediction must contain occupied_score or state")
    score = record.get("occupied_score")
    if score is not None:
        score = _eval_number(score, "occupied_score", maximum=1)
    if "state" in record:
        predicted = _eval_state(record["state"])
        if predicted is not None and score is not None and predicted != int(score >= threshold):
            raise ValueError("state disagrees with occupied_score at the evaluation threshold")
    else:
        predicted = None if score is None else int(score >= threshold)
    return (sample, space), dict(y_pred=predicted, occupied_score=score)


def _classification_summary(slots):
    # Rows: true vacant/occupied. Columns: predicted vacant/occupied/unknown.
    matrix = [[0, 0, 0], [0, 0, 0]]
    for row in slots:
        matrix[row["y_true"]][2 if row["y_pred"] is None else row["y_pred"]] += 1
    tn, fp, u_vacant = matrix[0]
    fn, tp, u_occupied = matrix[1]
    count = len(slots)
    known = tp + tn + fp + fn
    div = lambda a,b: a/b if b else 0.0
    covered = dict(accuracy=div(tp+tn,known),precision=div(tp,tp+fp),
                   recall=div(tp,tp+fn),f1=div(2*tp,2*tp+fp+fn))
    return dict(slots=count,covered_slots=known,unknown_slots=count-known,
                missing_predictions=sum(r["missing_prediction"] for r in slots),
                occupied_support=tp+fn+u_occupied,vacant_support=tn+fp+u_vacant,
                tp=tp,tn=tn,fp=fp,fn=fn,unknown_occupied=u_occupied,unknown_vacant=u_vacant,
                coverage=div(known,count),accuracy=div(tp+tn,count),
                precision=div(tp,tp+fp),recall=div(tp,tp+fn+u_occupied),
                f1=div(2*tp,2*tp+fp+fn+u_occupied),
                false_vacancy_rate=fn/(tp+fn+u_occupied) if tp+fn+u_occupied else None,
                reported_vacancy_error_rate=fn/(tn+fn) if tn+fn else None,
                covered_only=covered,
                confusion_matrix=dict(true_labels=["vacant","occupied"],
                    predicted_labels=["vacant","occupied","unknown"],values=matrix),
                confusion_matrix_covered=[[tn,fp],[fn,tp]])


def _eval_percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    position = (len(values)-1)*q
    low = int(position)
    high = min(low+1,len(values)-1)
    return values[low]+(values[high]-values[low])*(position-low)


def _timing_summary(frames):
    measured = [row for row in frames if row.get("timing") is not None]
    result = dict(measured_frames=len(measured),total_frames=len(frames),
                  coverage=len(measured)/len(frames) if frames else 0.0,
                  missing_sample_ids=[r["sample_id"] for r in frames if r.get("timing") is None])
    for name in ("preprocess_ms","model_ms","postprocess_ms","total_ms"):
        result[name+"_mean"] = sum(r["timing"][name] for r in measured)/len(measured) if measured else None
    totals = [r["timing"]["total_ms"] for r in measured]
    result.update(total_ms_p50=_eval_percentile(totals,.5),total_ms_p95=_eval_percentile(totals,.95),
                  total_ms_sum=sum(totals) if measured else None,
                  inference_calls_sum=sum(r["timing"]["inference_calls"] for r in measured) if measured else None,
                  inference_calls_mean=sum(r["timing"]["inference_calls"] for r in measured)/len(measured) if measured else None)
    return result


def _evaluation_group(slots, frames):
    result = _classification_summary(slots)
    result.update(frames=len(frames),
                  mae_vacant_count=sum(r["absolute_vacant_count_error"] for r in frames)/len(frames),
                  mae_occupied_count=sum(r["absolute_occupied_count_error"] for r in frames)/len(frames),
                  complete_frame_coverage=sum(r["unknown_count"]==0 for r in frames)/len(frames),
                  timing=_timing_summary(frames))
    return result


def evaluate_occupancy(manifest, predictions, timings=None, threshold=.5,
                       missing_policy="unknown", method=None, metadata=None):
    """Evaluate predictions by (sample_id, space_id), never list position.

    Ground truth: frozen manifest rows (labels), audit rows (spaces), or Frame.
    Prediction: occupied_score in [0,1] or state=0/1/None (unknown).
    Missing rows are unknown by default; duplicates/unexpected keys always fail.
    Main recall/F1 penalize unknown occupied slots; accuracy penalizes all unknowns.
    """
    threshold = _eval_number(threshold,"threshold",maximum=1)
    if missing_policy not in ("unknown","error"):
        raise ValueError("missing_policy must be unknown or error")
    if metadata is not None and not isinstance(metadata,dict):
        raise ValueError("metadata must be a JSON-compatible dictionary")
    truth, expected = {}, {}
    for source in _eval_records(manifest):
        source = asdict(source) if isinstance(source, Frame) else source
        if not isinstance(source,dict):
            raise ValueError("Ground-truth rows must be dictionaries or Frame objects")
        sample = _eval_id(source.get("sample_id"),"sample_id")
        if sample in truth:
            raise ValueError(f"Duplicate ground-truth sample: {sample}")
        camera, weather = source.get("camera"), source.get("weather")
        if not isinstance(camera,str) or not camera or not isinstance(weather,str) or not weather:
            raise ValueError("Ground truth requires camera and weather")
        labels = source.get("labels")
        if labels is None:
            spaces = source.get("spaces",[])
            labels = {}
            for space in spaces:
                key = _eval_id(space.get("space_id"),"space_id")
                if key in labels:
                    raise ValueError("Duplicate ground-truth space id")
                labels[key] = space.get("occupied")
        if not isinstance(labels,dict) or not labels:
            raise ValueError("Ground truth must contain nonempty labels or spaces")
        truth[sample] = dict(sample_id=sample,camera=camera,weather=weather,date=source.get("date"))
        for space,label in labels.items():
            space = _eval_id(space,"space_id")
            if not isinstance(label,(bool,Integral)) or label not in (0,1):
                raise ValueError("Ground-truth labels must be binary; unknown truth is not allowed")
            key = sample,space
            if key in expected:
                raise ValueError("Duplicate normalized ground-truth key")
            expected[key] = int(label)
    if not truth:
        raise ValueError("Cannot evaluate an empty manifest")
    predicted = {}
    for record in _eval_records(predictions):
        key,normalized = _prediction_record(record,threshold)
        if key in predicted:
            raise ValueError(f"Duplicate prediction: {key}")
        if key not in expected:
            raise ValueError(f"Prediction key not present in selected manifest: {key}")
        predicted[key] = normalized
    missing = set(expected)-set(predicted)
    if missing and missing_policy == "error":
        raise ValueError(f"Missing predictions for {len(missing)} slots")
    time_by_sample = {}
    if timings is not None:
        for record in _eval_records(timings):
            if not isinstance(record,dict):
                raise ValueError("Timing records must be dictionaries")
            sample = _eval_id(record.get("sample_id"),"sample_id")
            if sample not in truth or sample in time_by_sample:
                raise ValueError(f"Unexpected or duplicate timing sample: {sample}")
            timing = {name:_eval_number(record.get(name),name) for name in
                      ("preprocess_ms","model_ms","postprocess_ms","total_ms")}
            if timing["total_ms"]+1e-6 < sum(timing[name] for name in ("preprocess_ms","model_ms","postprocess_ms")):
                raise ValueError("total_ms must include preprocessing, all model calls and postprocessing")
            calls = record.get("inference_calls")
            if isinstance(calls,bool) or not isinstance(calls,Integral) or calls < 0:
                raise ValueError("inference_calls must be a nonnegative integer")
            timing["inference_calls"] = int(calls)
            time_by_sample[sample] = timing
    slots, by_frame = [], {sample:[] for sample in truth}
    for key,label in sorted(expected.items()):
        sample,space = key
        prediction = predicted.get(key,dict(y_pred=None,occupied_score=None))
        row = dict(truth[sample],space_id=space,y_true=label,**prediction,missing_prediction=key in missing)
        slots.append(row)
        by_frame[sample].append(row)
    frames = []
    for sample,rows in sorted(by_frame.items()):
        actual_vacant = sum(r["y_true"]==0 for r in rows)
        actual_occupied = len(rows)-actual_vacant
        pred_vacant = sum(r["y_pred"]==0 for r in rows)
        pred_occupied = sum(r["y_pred"]==1 for r in rows)
        unknown = sum(r["y_pred"] is None for r in rows)
        frames.append(dict(truth[sample],slots=len(rows),true_vacant_count=actual_vacant,
            true_occupied_count=actual_occupied,predicted_vacant_count=pred_vacant,
            predicted_occupied_count=pred_occupied,unknown_count=unknown,
            vacant_count_lower=pred_vacant,vacant_count_upper=pred_vacant+unknown,
            absolute_vacant_count_error=abs(pred_vacant-actual_vacant),
            absolute_occupied_count_error=abs(pred_occupied-actual_occupied),timing=time_by_sample.get(sample)))
    report = dict(schema_version=1,method=method,threshold=threshold,missing_policy=missing_policy,
                  metadata={} if metadata is None else metadata,
                  overall=_evaluation_group(slots,frames),per_frame=frames,per_slot=slots)
    for name,fields in (("by_camera",("camera",)),("by_weather",("weather",)),("by_camera_weather",("camera","weather"))):
        groups = {}
        for values in sorted({tuple(r[f] for f in fields) for r in frames}):
            group_frames = [r for r in frames if tuple(r[f] for f in fields)==values]
            group_slots = [r for r in slots if tuple(r[f] for f in fields)==values]
            groups["/".join(values)] = _evaluation_group(group_slots,group_frames)
        report[name] = groups
    return report


class FrameTimer:
    """Measure one original image, including repeated model stages for all tiles.

    Supply a synchronize callback for asynchronous accelerators. Loading weights
    and warm-up belong outside this timer; image loading belongs in preprocessing.
    """
    def __init__(self, sample_id, synchronize=None):
        self.sample_id = _eval_id(sample_id,"sample_id")
        self.synchronize = synchronize or (lambda:None)
        self._started = self._finished = self._failed = self._active_stage = False
        self._durations = {name:0.0 for name in ("preprocess","model","postprocess")}
        self._calls = Counter()

    def __enter__(self):
        if self._started:
            raise RuntimeError("Create a fresh FrameTimer for each original image")
        self.synchronize()
        self._start = time.perf_counter()
        self._started = True
        return self

    @contextmanager
    def stage(self, name):
        if name not in self._durations or not self._started or self._finished or self._active_stage:
            raise RuntimeError("Unknown, nested or inactive timing stage")
        self._active_stage = True
        self.synchronize()
        start = time.perf_counter()
        try:
            yield
            self.synchronize()
        except BaseException:
            self._failed = True
            raise
        finally:
            self._durations[name] += (time.perf_counter()-start)*1000
            self._calls[name] += 1
            self._active_stage = False

    def __exit__(self, exc_type, exc, tb):
        self.synchronize()
        self._total = (time.perf_counter()-self._start)*1000
        self._finished = True
        self._failed = self._failed or exc_type is not None
        return False

    def record(self, inference_calls=None):
        if not self._finished or self._failed or any(self._calls[name]==0 for name in self._durations):
            raise RuntimeError("Need one completed, successful frame and all three stages")
        calls = self._calls["model"] if inference_calls is None else inference_calls
        if isinstance(calls,bool) or not isinstance(calls,Integral) or calls < 0:
            raise ValueError("inference_calls must be a nonnegative integer")
        return dict(sample_id=self.sample_id,total_ms=self._total,inference_calls=int(calls),
                    **{name+"_ms":value for name,value in self._durations.items()})


def export_evaluation(report, output):
    """Save aggregate JSON, per-frame CSV and aligned per-slot CSV/JSONL."""
    import csv
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    aggregate = {key:value for key,value in report.items() if key not in ("per_frame","per_slot")}
    (output/"metrics.json").write_text(json.dumps(aggregate,ensure_ascii=False,allow_nan=False,indent=2)+"\n")
    frames = []
    for row in report["per_frame"]:
        record = {key:value for key,value in row.items() if key != "timing"}
        for key in ("preprocess_ms","model_ms","postprocess_ms","total_ms","inference_calls"):
            record[key] = None if row["timing"] is None else row["timing"][key]
        frames.append(record)
    for filename,rows in (("per_frame.csv",frames),("per_slot.csv",report["per_slot"])):
        with (output/filename).open("w",newline="") as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (output/"aligned_predictions.jsonl").open("w") as stream:
        for row in report["per_slot"]:
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
    return output


def evaluator_self_check():
    """Two hand-calculated frames, including both types of unknown prediction."""
    manifest = [dict(sample_id="A",camera="UFPR04",weather="sunny",labels={"1":1,"2":1,"3":0,"4":0}),
                dict(sample_id="B",camera="UFPR05",weather="rainy",labels={"1":1,"2":0,"3":1})]
    states = {"A":[1,0,0,None],"B":[None,1,1]}
    predictions = [dict(sample_id=sample,space_id=str(i+1),state=state)
                   for sample,values in states.items() for i,state in enumerate(values)]
    timings = [dict(sample_id="A",preprocess_ms=10,model_ms=20,postprocess_ms=5,total_ms=40,inference_calls=1),
               dict(sample_id="B",preprocess_ms=20,model_ms=80,postprocess_ms=10,total_ms=120,inference_calls=4)]
    result = evaluate_occupancy(manifest,predictions,timings,method="synthetic_self_check")
    metrics = result["overall"]
    expected = dict(accuracy=3/7,coverage=5/7,precision=2/3,recall=1/2,f1=4/7,
                    false_vacancy_rate=1/4,reported_vacancy_error_rate=1/2,mae_vacant_count=1/2)
    if metrics["confusion_matrix"]["values"] != [[1,1,1],[1,2,1]]:
        raise AssertionError("Hand-calculated confusion matrix does not match")
    for key,value in expected.items():
        if not math.isclose(metrics[key],value,rel_tol=1e-12):
            raise AssertionError(f"Hand-calculated {key} does not match")
    if metrics["timing"]["total_ms_mean"] != 80:
        raise AssertionError("Timing must be per original frame, not per tile/model call")
    return dict(passed=True,synthetic=True,expected=expected,confusion_matrix=[[1,1,1],[1,2,1]],total_ms_mean=80)


"""Download/resume official archive; safely extract full frames only; audit."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

URL = "https://www.inf.ufpr.br/vri/databases/PKLot.tar.gz"
EXPECTED_BYTES = 4898276304  # UFPR HEAD on 2026-10-28

def prepare(raw, output, download=True):
    raw = Path(raw).resolve()
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw/"PKLot.tar.gz"
    if download and (not archive.exists() or archive.stat().st_size != EXPECTED_BYTES):
        subprocess.run(["curl", "-fL", "--retry", "3", "--connect-timeout", "30",
                        "-C", "-", "-o", str(archive), URL], check=True)
    if not archive.exists() or archive.stat().st_size != EXPECTED_BYTES:
        raise ValueError("archive incomplete or upstream size changed; do not audit as complete")
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(chunk)
    target = raw/"PKLot"
    pending = raw/".pklot-extraction-in-progress"
    pending.touch()
    extracted = 0
    # Read through EOF: gzip/tar corruption raises; never execute links or external paths.
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            parts = Path(member.name).parts
            if len(parts) < 3 or parts[:2] != ("PKLot", "PKLot"):
                continue
            if not member.isfile() or ".." in parts or Path(member.name).is_absolute():
                continue
            if Path(member.name).suffix.lower() not in {".xml", ".jpg", ".jpeg", ".png"}:
                continue
            dest = target.joinpath(*parts[2:])
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src, dest.open("wb") as dst:
                for chunk in iter(lambda: src.read(1024*1024), b""):
                    dst.write(chunk)
            extracted += 1
    pending.unlink()
    result = audit(target, output)
    Path(output, "provenance.json").write_text(json.dumps(dict(url=URL, bytes=archive.stat().st_size,
        sha256=digest.hexdigest(), extracted_files=extracted,
        checksum_note="Locally calculated fingerprint; no publisher checksum available."), indent=2))
    if not result["valid_frames"]:
        raise ValueError("no valid full frames")
    return result

def _cached_audit(root, source, seed, allow_rebase=False):
    """Fast cache check, not a new full audit. Dataset must remain unchanged."""
    try:
        summary = json.loads((source / "audit_summary.json").read_text())
        if (summary.get("audit_version") != AUDIT_VERSION or summary.get("seed") != seed
                or summary.get("valid_frames", 0) <= 0):
            return None
        if not allow_rebase and Path(summary["root"]).resolve() != root:
            return None
        if not (source / "issues.jsonl").is_file():
            return None
        with (source / "valid_samples.jsonl").open() as stream:
            sample = json.loads(next(stream))
        # Verify one actual frame against the cached record, including labels and geometry.
        # Relative paths allow a shared manifest to work under another user's Drive shortcut.
        xml = (root / sample["xml_path"]).resolve()
        image = (root / sample["image_path"]).resolve()
        if not xml.is_relative_to(root) or not image.is_relative_to(root):
            return None
        frame = read_frame(xml, root)
        expected = json.loads(json.dumps(asdict(frame)))
        if any(sample.get(key) != value for key, value in expected.items()):
            return None
        if sample.get("split") != split_for(frame, seed):
            return None
        return dict(summary, root=str(root))
    except (OSError, ValueError, KeyError, TypeError, StopIteration, ET.ParseError):
        return None


def ensure_dataset(raw, output, dataset_root=None, cached_audit_dir=None,
                   download=True, force_audit=False, seed=42):
    """Reuse extracted data and audit, or prepare missing private data.

    dataset_root explicitly selects read-only existing/shared data. A bad shared
    path raises instead of downloading into or modifying the shared location.
    force_audit revalidates an unchanged cache after dataset edits.
    """
    import shutil
    raw, output = Path(raw).resolve(), Path(output).resolve()
    shared = dataset_root is not None
    root = Path(dataset_root).resolve() if shared else raw / "PKLot"
    if shared and (output.is_relative_to(root) or root.is_relative_to(output)):
        raise ValueError("Keep personal output and shared dataset in separate directories")
    if cached_audit_dir is not None:
        cache = Path(cached_audit_dir).resolve()
        if cache != output and (output.is_relative_to(cache) or cache.is_relative_to(output)):
            raise ValueError("Keep shared audit cache separate from personal outputs")
        if shared and cache == output:
            raise ValueError("Shared audit cache must not be the personal output directory")
    pending = root.parent / ".pklot-extraction-in-progress"
    if pending.exists():
        if shared:
            raise ValueError("Shared dataset extraction is incomplete; ask owner to finish it")
        print("Previous extraction was interrupted; resume preparation.")
        summary = prepare(raw, output, download)
        if seed != 42:
            summary = audit(root, output, seed)
        return summary
    if not root.is_dir() or next(root.rglob("*.xml"), None) is None:
        if shared:
            raise ValueError(f"No full-frame XML at shared dataset path: {root}. Check Drive shortcut/path.")
        print("No extracted dataset; download/resume and extract.")
        summary = prepare(raw, output, download)
        if seed != 42:
            summary = audit(root, output, seed)
        return summary

    print(f"Existing dataset: {root}. Skip download and extraction.")
    if not force_audit:
        sources = [(output, False)]
        if cached_audit_dir is not None:
            sources.append((Path(cached_audit_dir).resolve(), True))
        for source, allow_rebase in sources:
            summary = _cached_audit(root, source, seed, allow_rebase)
            if summary is not None:
                if source != output:
                    output.mkdir(parents=True, exist_ok=True)
                    (output / "audit_summary.json").unlink(missing_ok=True)
                    for name in ("valid_samples.jsonl", "issues.jsonl"):
                        shutil.copyfile(source / name, output / name)
                    (output / "audit_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
                print("Reuse audit/manifest; skip full audit (checked one sample).")
                return summary
    print("No compatible audit cache or FORCE_AUDIT=True; audit existing dataset once.")
    summary = audit(root, output, seed)
    if not summary["valid_frames"]:
        raise ValueError("No valid frames; inspect personal outputs/issues.jsonl")
    return summary


# Project-level manifests: this is separate from split_for(), the audit preview.
LOT_BY_CAMERA = {"UFPR04": "UFPR", "UFPR05": "UFPR", "PUCPR": "PUCPR", "PUC": "PUCPR"}
WEATHERS = ("sunny", "overcast", "rainy")


def _read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _stable_key(seed, value):
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _geometry_record(row):
    spaces = sorted((dict(space_id=s["space_id"], polygon=s["polygon"]) for s in row["spaces"]),
                    key=lambda s: s["space_id"])
    record = dict(camera=row["camera"], lot=LOT_BY_CAMERA[row["camera"]],
                  width=row["width"], height=row["height"], spaces=spaces)
    digest = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return f"{row['camera']}_{digest[:16]}", record


def _assign_project_days(rows, ratios, caps, seed, trials):
    """Choose day groups using only camera/weather availability, never labels/scores."""
    days = sorted({r["date"] for r in rows})
    names = [name for name in ("train","val","test") if name in ratios]
    if len(days) < len(names):
        raise ValueError("Not enough acquisition days for the requested splits")
    if any(v <= 0 for v in ratios.values()) or abs(sum(ratios.values())-1) > 1e-9:
        raise ValueError("Day split ratios must be positive and sum to 1")
    n = len(days)
    sizes = {name: int(n*ratios[name]) for name in names}
    for name in sorted(names, key=lambda k: (-(n*ratios[k]-sizes[k]), k))[:n-sum(sizes.values())]:
        sizes[name] += 1
    if any(v == 0 for v in sizes.values()):
        raise ValueError("Split ratios leave an empty day partition")
    strata = sorted({(r["camera"],r["weather"]) for r in rows})
    day_counts = {}
    totals = Counter((r["camera"],r["weather"]) for r in rows)
    for row in rows:
        day_counts.setdefault(row["date"], Counter())[(row["camera"],row["weather"])] += 1
    best = None
    for trial in range(trials):
        shuffled = sorted(days, key=lambda day: _stable_key(f"{seed}:{trial}",day))
        assignments, counts, cursor = {}, {}, 0
        for name in names:
            selected = shuffled[cursor:cursor+sizes[name]]
            cursor += sizes[name]
            counts[name] = Counter()
            for day in selected:
                assignments[day] = name
                counts[name].update(day_counts[day])
        missing = sum(counts[name][stratum] == 0 for name in names for stratum in strata)
        deficit = sum(max(0,caps[name]-counts[name][stratum])/caps[name]
                      for name in names for stratum in strata)
        balance = sum(abs(counts[name][stratum]/totals[stratum]-ratios[name])
                      for name in names for stratum in strata)
        objective = (missing, round(deficit,12), round(balance,12), trial)
        if best is None or objective < best[0]:
            best = objective, assignments
    if best[0][0]:
        raise ValueError("Cannot cover camera/weather strata in every split; revise subset budget or day ratios")
    return best[1]


def _sample_project_frames(rows, cap, seed):
    """Round-robin over days so a long day's captures do not consume the budget."""
    grouped = {}
    for row in rows:
        grouped.setdefault(row["date"], []).append(row)
    for day in grouped:
        grouped[day].sort(key=lambda row: _stable_key(seed,row["sample_id"]))
    day_order = sorted(grouped, key=lambda day: _stable_key(seed,day))
    result, index = [], 0
    while len(result) < min(cap,len(rows)):
        for day in day_order:
            if index < len(grouped[day]):
                result.append(grouped[day][index])
                if len(result) == cap:
                    break
        index += 1
    return sorted(result, key=lambda row: row["sample_id"])


def validate_project_splits(manifests, geometries, heldout_camera=None):
    """Check leakage, unique keys and exact label/geometry correspondence."""
    day_sets, id_sets = {}, {}
    for name in ("train", "val", "test"):
        rows = manifests[name]
        if not rows:
            raise ValueError(f"Empty {name} manifest")
        ids = [r["sample_id"] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate sample ids in {name}")
        day_sets[name], id_sets[name] = {r["date"] for r in rows}, set(ids)
        for row in rows:
            if row["split"] != name:
                raise ValueError("Incorrect split field")
            geometry = geometries[row["geometry_id"]]
            if geometry["camera"] != row["camera"]:
                raise ValueError("Geometry belongs to a different camera")
            if set(row["labels"]) != {s["space_id"] for s in geometry["spaces"]}:
                raise ValueError("Slot ids and labels differ from geometry")
            if any(type(label) is not int or label not in (0,1) for label in row["labels"].values()):
                raise ValueError("Occupancy labels must be 0 or 1")
            if heldout_camera and ((name == "test") != (row["camera"] == heldout_camera)):
                raise ValueError("Held-out camera leaked into training/validation or test contains another camera")
    for a,b in (("train","val"),("train","test"),("val","test")):
        if day_sets[a] & day_sets[b] or id_sets[a] & id_sets[b]:
            raise ValueError(f"Day/sample leakage between {a} and {b}")
    return {name: sorted(values) for name,values in day_sets.items()}


def build_project_splits(manifest, output, config, overwrite=False):
    """Freeze a shared metadata-based subset and polygon layouts for every method."""
    manifest, output = Path(manifest), Path(output)
    if isinstance(config, (str,Path)):
        config = json.loads(Path(config).read_text())
    summary = json.loads((manifest.parent/"audit_summary.json").read_text())
    if summary.get("audit_version") != AUDIT_VERSION:
        raise ValueError("Run audit with the current loader before freezing project splits")
    source_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    lock_file = output/"split_lock.json"
    if lock_file.exists() and not overwrite:
        lock = json.loads(lock_file.read_text())
        if lock["source_manifest_sha256"] != source_sha or lock["config_sha256"] != config_sha:
            raise ValueError("Existing split is locked to another inventory/config. Use a new dataset_version/output directory")
        for filename,digest in lock["artifacts_sha256"].items():
            if hashlib.sha256((output/filename).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Frozen split artifact was modified: {filename}")
        print("Reuse locked project splits:",output)
        return lock
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise ValueError("Output is nonempty without a valid lock; choose a new output directory")
    rows = _read_jsonl(manifest)
    if len(rows) != summary["valid_frames"] or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Inventory count mismatch or duplicate sample ids")
    cameras = set(config["cameras"])
    rows = [r for r in rows if r["camera"] in cameras]
    if {r["camera"] for r in rows} != cameras:
        raise ValueError("Configured camera missing from inventory")
    for row in rows:
        date.fromisoformat(row["date"])
        if row["weather"] not in WEATHERS or not row["spaces"]:
            raise ValueError("Invalid weather or empty space list")
    geometries, artifacts, reports = {}, {}, {}
    audit_record = {key:summary[key] for key in ("audit_version","xml_count","valid_frames","rejected_frames","images_without_xml") if key in summary}
    audit_record["counts"] = {key:value for key,value in summary.get("counts",{}).items() if not key.startswith("split/")}
    artifacts["inventory_audit.json"] = json.dumps(audit_record,sort_keys=True,indent=2)+"\n"
    seed, caps = config["seed"], config["max_frames_per_camera_weather"]
    if config["day_assignment_trials"] < 1:
        raise ValueError("At least one metadata-only day assignment trial is required")
    if any(type(caps[k]) is not int or caps[k] <= 0 for k in ("train","val","test")):
        raise ValueError("Frame budgets must be positive integers")
    for protocol, spec in config["protocols"].items():
        if spec["kind"] == "within_camera":
            pool = rows
            day_map = _assign_project_days(pool,spec["day_ratios"],caps,seed,config["day_assignment_trials"])
            buckets = {name: [r for r in pool if day_map[r["date"]] == name] for name in caps}
            heldout = None
        elif spec["kind"] == "camera_transfer":
            heldout = spec["target_camera"]
            targets = [r for r in rows if r["camera"] == heldout]
            target_days = {r["date"] for r in targets}
            sources = [r for r in rows if r["camera"] in spec["source_cameras"] and r["date"] not in target_days]
            day_map = _assign_project_days(sources,spec["source_day_ratios"],caps,seed,config["day_assignment_trials"])
            day_map.update({day:"test" for day in target_days})
            buckets = {name: [r for r in sources if day_map[r["date"]] == name] for name in ("train","val")}
            buckets["test"] = targets
        else:
            raise ValueError(f"Unknown protocol: {spec['kind']}")
        manifests, stats, deficits = {}, {}, {}
        for name, bucket in buckets.items():
            selected = []
            strata = sorted({(r["camera"],r["weather"]) for r in bucket})
            for camera,weather in strata:
                candidates = [r for r in bucket if (r["camera"],r["weather"]) == (camera,weather)]
                chosen = _sample_project_frames(candidates,caps[name],seed)
                selected.extend(chosen)
                if len(chosen) < caps[name]:
                    deficits[f"{name}/{camera}/{weather}"] = caps[name]-len(chosen)
            packed = []
            for row in sorted(selected, key=lambda r:r["sample_id"]):
                gid,geometry = _geometry_record(row)
                geometries[gid] = geometry
                record = {key:row[key] for key in ("sample_id","image_path","xml_path","camera","weather","date",
                                                    "directory_date","date_source","width","height")}
                record.update(lot=LOT_BY_CAMERA[row["camera"]],split=name,geometry_id=gid,
                              labels={s["space_id"]:int(s["occupied"]) for s in row["spaces"]})
                packed.append(record)
            manifests[name] = packed
            stats[name] = dict(frames=len(packed), slots=sum(len(r["labels"]) for r in packed),
                               occupied=sum(sum(r["labels"].values()) for r in packed),
                               by_camera_weather=dict(Counter(f"{r['camera']}/{r['weather']}" for r in packed)))
            artifacts[f"{protocol}/{name}.jsonl"] = "".join(json.dumps(r,sort_keys=True,separators=(",", ":"))+"\n" for r in packed)
        selected_days = validate_project_splits(manifests,geometries,heldout)
        reports[protocol] = dict(stats=stats,shortfall_against_caps=deficits,selected_days=selected_days,
                                 all_eligible_day_assignments=dict(sorted(day_map.items())),
                                 heldout_camera=heldout)
    artifacts["slot_geometries.json"] = json.dumps(dict(coordinates="original image pixels; x right, y down",
        note="Parking-space polygons; never vehicle ground truth. Geometry variants are preserved per frame.",
        layouts=geometries),sort_keys=True,indent=2)+"\n"
    lock = dict(dataset_version=config["dataset_version"],schema_version=1,audit_version=AUDIT_VERSION,
                seed=seed,config=config,config_sha256=config_sha,source_manifest_sha256=source_sha,
                eligible_frames=len(rows),protocols=reports,
                artifacts_sha256={name:hashlib.sha256(content.encode()).hexdigest() for name,content in artifacts.items()})
    output.mkdir(parents=True,exist_ok=True)
    for name,content in artifacts.items():
        path=output/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(content)
    lock_file.write_text(json.dumps(lock,sort_keys=True,indent=2)+"\n")
    return lock


def load_project_splits(directory):
    """Load and verify the exact versioned manifests distributed to the team."""
    directory = Path(directory)
    lock = json.loads((directory/"split_lock.json").read_text())
    for name,digest in lock["artifacts_sha256"].items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Artifact path escapes manifest directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen artifact checksum mismatch: {name}")
    geometry = json.loads((directory/"slot_geometries.json").read_text())["layouts"]
    protocols = {}
    for name,report in lock["protocols"].items():
        manifests = {split:_read_jsonl(directory/name/f"{split}.jsonl") for split in ("train","val","test")}
        validate_project_splits(manifests,geometry,report["heldout_camera"])
        protocols[name] = manifests
    return dict(lock=lock,geometries=geometry,protocols=protocols)


def fetch_project_splits(owner, repo, ref, output, dataset_version="pklot_v1"):
    """Fetch frozen metadata only; no dataset images or per-user random resplitting."""
    import urllib.request
    from urllib.parse import quote
    output = Path(output)
    base = (f"https://raw.githubusercontent.com/{quote(owner,safe='')}/{quote(repo,safe='')}/"
            f"{quote(ref,safe='')}/data/manifests/{quote(dataset_version,safe='')}")
    def fetch(name):
        request = urllib.request.Request(f"{base}/{quote(name,safe='/')}?refresh={time.time_ns()}",
                                         headers={"Cache-Control":"no-cache"})
        with urllib.request.urlopen(request,timeout=60) as response:
            return response.read()
    lock_bytes = fetch("split_lock.json")
    lock = json.loads(lock_bytes)
    if lock["dataset_version"] != dataset_version:
        raise ValueError("Unexpected split version")
    pending = {}
    for name,digest in lock["artifacts_sha256"].items():
        path = (output/name).resolve()
        if not path.is_relative_to(output.resolve()):
            raise ValueError("Artifact path escapes manifest directory")
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
            continue
        content = fetch(name)
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("GitHub metadata changed while downloading; retry or pin REF to a commit")
        pending[name] = content
    output.mkdir(parents=True,exist_ok=True)
    for name,content in pending.items():
        path=output/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(content)
    (output/"split_lock.json").write_bytes(lock_bytes)
    return load_project_splits(output)


def read_project_frame(record, dataset_root):
    """Read raw pixels/labels and reject changes from the frozen selected frame."""
    root = Path(dataset_root).resolve()
    xml = (root/record["xml_path"]).resolve()
    if not xml.is_relative_to(root):
        raise ValueError("XML path escapes dataset root")
    frame = read_frame(xml,root)
    row = json.loads(json.dumps(asdict(frame)))
    gid,_ = _geometry_record(row)
    labels = {s.space_id:int(s.occupied) for s in frame.spaces}
    if gid != record["geometry_id"] or labels != record["labels"] or frame.sample_id != record["sample_id"]:
        raise ValueError("Dataset frame differs from the frozen manifest; do not silently modify the shared split")
    return frame


def score_vehicle_spaces(detections, layout, lower_fraction=.5):
    """Score original-image vehicle boxes against known convex space polygons.

    The overlap is intersection area / lower-box area. Multiplying it by
    detector confidence gives a ranking score, not an occupancy probability.
    Labels are never read. No assignment or vacancy decision is made here.
    NumPy/OpenCV remain lazy imports so the standalone data tools stay usable.
    """
    import cv2
    import numpy as np

    if isinstance(lower_fraction, bool) or not isinstance(lower_fraction, Real):
        raise ValueError("lower_fraction must be a number in (0, 1]")
    if not math.isfinite(lower_fraction) or not 0 < lower_fraction <= 1:
        raise ValueError("lower_fraction must be a number in (0, 1]")
    polygons = []
    space_ids = []
    for space in layout["spaces"]:
        space_id = _eval_id(space["space_id"], "space_id")
        if space_id in space_ids:
            raise ValueError(f"Duplicate space_id: {space_id}")
        polygon = np.asarray(space["polygon"], dtype=np.float32)
        if (polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3
                or not np.isfinite(polygon).all()
                or not cv2.isContourConvex(polygon) or cv2.contourArea(polygon) <= 0):
            raise ValueError(f"Space {space_id} must have a finite convex polygon")
        space_ids.append(space_id)
        polygons.append(polygon)

    rows = []
    for detection_index, detection in enumerate(detections):
        box = np.asarray(detection.xyxy, dtype=np.float64)
        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError("Vehicle boxes must contain four finite coordinates")
        x1, y1, x2, y2 = map(float, box)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Vehicle boxes must have positive width and height")
        confidence = _eval_number(detection.confidence, "confidence", maximum=1)
        bottom_y = y2 - (y2 - y1) * lower_fraction
        lower_box = np.array([[x1, bottom_y], [x2, bottom_y], [x2, y2], [x1, y2]],
                             dtype=np.float32)
        lower_area = cv2.contourArea(lower_box)
        if not np.isfinite(lower_box).all() or lower_area <= 0:
            raise ValueError("Vehicle lower box cannot be represented at this precision")
        for space_id, polygon in zip(space_ids, polygons):
            area, _ = cv2.intersectConvexConvex(lower_box, polygon)
            overlap = min(1., max(0., float(area) / lower_area))
            rows.append(dict(detection_index=detection_index, space_id=space_id,
                             overlap=overlap, score=overlap * confidence))
    return rows


DEFAULT_OCCUPANCY_CONFIG = dict(
    lower_fraction=.5, min_overlap=.3, min_score=.15,
    ambiguity_margin=.05, nms_iou=.5,
)


def infer_parking_occupancy(sample_id, detections, layout, config=None):
    """Apply identical geometric matching rules to either detector's output.

    Class-agnostic NMS precedes matching. Each vehicle proposes its best space;
    near-ties abstain. Competing proposals are resolved by descending score,
    never forcing a vehicle into its second-choice space. Each vehicle and
    space can participate in at most one accepted match.

    Unassigned spaces with eligible overlap are unknown; others are vacant.
    Vacancy assumes the detector did not miss a vehicle, not proof of emptiness.
    `score` is geometric evidence, while `state` is the evaluator input.
    Defaults are provisional and must be selected on validation before testing.
    """
    sample_id = _eval_id(sample_id, "sample_id")
    cfg = dict(DEFAULT_OCCUPANCY_CONFIG)
    if config is not None:
        if set(config) - set(cfg):
            raise ValueError("Unknown occupancy configuration keys")
        cfg.update(config)
    for key, value in cfg.items():
        cfg[key] = _eval_number(value, key, maximum=1)
    if any(cfg[key] == 0 for key in ("lower_fraction", "min_overlap", "min_score")):
        raise ValueError("lower_fraction, min_overlap and min_score must be positive")
    detections = list(detections)
    rows = score_vehicle_spaces(detections, layout, cfg["lower_fraction"])

    def iou(first, second):
        a, b = first.xyxy, second.xyxy
        intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(
            0., min(a[3], b[3]) - max(a[1], b[1]))
        union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
        return intersection / union

    # Coordinate/class tie-breaks make the decision independent of input order.
    order = sorted(range(len(detections)), key=lambda i: (
        -detections[i].confidence, tuple(detections[i].xyxy), detections[i].class_name))
    kept = []
    for index in order:
        if not any(iou(detections[index], detections[j]) > cfg["nms_iou"] for j in kept):
            kept.append(index)
    by_detection = {index: [] for index in kept}
    evidence = {str(s["space_id"]): 0. for s in layout["spaces"]}
    uncertain = set()
    for row in rows:
        if row["detection_index"] not in by_detection:
            continue
        space_id = row["space_id"]
        evidence[space_id] = max(evidence[space_id], row["score"])
        if row["overlap"] >= cfg["min_overlap"] and row["score"] > 0:
            uncertain.add(space_id)
            by_detection[row["detection_index"]].append(row)
    proposals = []
    for index in kept:
        candidates = sorted(by_detection[index], key=lambda r: (-r["score"], r["space_id"]))
        if not candidates or candidates[0]["score"] < cfg["min_score"]:
            continue
        if len(candidates) > 1 and candidates[0]["score"] - candidates[1]["score"] <= cfg["ambiguity_margin"]:
            continue
        proposals.append(candidates[0])
    matches = {}
    for row in sorted(proposals, key=lambda r: (-r["score"], kept.index(r["detection_index"]))):
        matches.setdefault(row["space_id"], row)
    predictions = []
    for space in layout["spaces"]:
        space_id = str(space["space_id"])
        match = matches.get(space_id)
        state = "occupied" if match else "unknown" if space_id in uncertain else "vacant"
        predictions.append(dict(
            sample_id=sample_id, slot_id=space_id, score=evidence[space_id], state=state,
            detection_index=match["detection_index"] if match else None,
            match_score=match["score"] if match else None,
        ))
    return dict(predictions=predictions, config=cfg, kept_detection_indices=kept,
                counts={state: sum(p["state"] == state for p in predictions)
                        for state in ("occupied", "vacant", "unknown")})


# Parking-line geometry. NumPy/OpenCV are imported when needed.
# Pixel parameters target original 1280x720 PKLot frames; tune on train/val.
DEFAULT_GEOMETRY_CONFIG = {
    "gaussian_kernel": 5,
    "gaussian_sigma": 1.2,
    "canny_low": 50,
    "canny_high": 150,
    "hough_votes": 20,
    "min_line_length": 20,
    "max_line_gap": 10,
    "roi_margin": 8,
    "min_roi_fraction": 0.9,
    "angle_tolerance_deg": 12.0,
    "boundary_distance": 6.0,
    "min_edge_support": 0.25,
    "min_supported_edges": 2,
}


def _geometry_angle_difference(first, second):
    """Undirected angle difference in degrees, in [0, 90]."""
    import numpy as np

    return np.abs((np.asarray(first) - np.asarray(second) + 90.0) % 180.0 - 90.0)


def analyze_parking_geometry(image_bgr, layout, config=None):
    """Detect parking lines and measure boundary support, never occupancy.

    image_bgr is a uint8 BGR array in original image coordinates.
    layout is the entry selected by a frame's geometry_id, containing
    width, height and spaces (space_id, polygon). Occupancy labels are
    neither required nor read. config overrides DEFAULT_GEOMETRY_CONFIG;
    distances are original-image pixels. Defaults are uncalibrated.

    Returns JSON-compatible segments, per-slot boundary/edge support,
    quality counts, effective config, and an images dict of NumPy arrays.
    Exclude images when serializing. Boundary support unions finite line
    evidence sampled at <=2 pixels and weights edges by length. The flags
    supported, limited_support and no_support describe evidence only;
    they are not calibrated correctness scores or occupancy predictions.

    NumPy/OpenCV are imported on demand so dataset tools need only Pillow.
    """
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "Geometry analysis requires NumPy and OpenCV. Install the project dependencies: "
            "pip install -e ., "
            "or pip install numpy opencv-python-headless for standalone use."
        ) from exc
    cfg = dict(DEFAULT_GEOMETRY_CONFIG)
    if config is not None:
        if set(config) - set(cfg):
            raise ValueError("Unknown geometry configuration key")
        cfg.update(config)
    if (not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8
            or image_bgr.ndim != 3 or image_bgr.shape[2] != 3):
        raise ValueError("Expected a uint8 BGR image with three channels")
    height, width = image_bgr.shape[:2]
    if (width, height) != (layout["width"], layout["height"]):
        raise ValueError("Image size differs from layout; use original pixel coordinates")
    integer_keys = ("gaussian_kernel", "hough_votes", "min_line_length",
                    "max_line_gap", "roi_margin", "min_supported_edges")
    if any(type(cfg[key]) is not int for key in integer_keys):
        raise ValueError("Kernel, Hough counts, pixel lengths and edge count must be integers")
    if (cfg["gaussian_kernel"] < 1 or cfg["gaussian_kernel"] % 2 != 1
            or cfg["gaussian_sigma"] < 0
            or not 0 <= cfg["canny_low"] < cfg["canny_high"] <= 255
            or cfg["hough_votes"] < 1 or cfg["min_line_length"] < 1
            or cfg["max_line_gap"] < 0 or cfg["roi_margin"] < 0
            or not 0 < cfg["min_roi_fraction"] <= 1
            or not 0 < cfg["angle_tolerance_deg"] < 90
            or cfg["boundary_distance"] <= 0
            or not 0 < cfg["min_edge_support"] <= 1
            or cfg["min_supported_edges"] < 1
            or not all(np.isfinite(value) for value in cfg.values())):
        raise ValueError("Invalid geometry configuration")

    polygons, slot_ids = [], []
    for space in layout["spaces"]:
        polygon = np.asarray(space["polygon"], dtype=np.float64)
        if (polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3
                or not np.isfinite(polygon).all()
                or np.any(polygon < 0) or np.any(polygon > [width, height])
                or cv2.contourArea(polygon.astype(np.float32)) <= 0
                or np.any(np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1) == 0)):
            raise ValueError("Invalid parking-space polygon")
        polygons.append(polygon)
        slot_ids.append(str(space["space_id"]))
    if not polygons or len(set(slot_ids)) != len(slot_ids):
        raise ValueError("Expected nonempty layout with unique space IDs")

    roi = np.zeros((height, width), dtype=np.uint8)
    # Fill separately: overlapping spaces must form a union, never holes.
    for polygon in polygons:
        cv2.fillPoly(roi, [np.rint(polygon).astype(np.int32)], 255)
    margin = cfg["roi_margin"]
    if margin:
        roi = cv2.dilate(roi, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1)))
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cfg["gaussian_kernel"]
    blurred = cv2.GaussianBlur(gray, (kernel, kernel), cfg["gaussian_sigma"])
    edges = cv2.Canny(blurred, cfg["canny_low"], cfg["canny_high"], L2gradient=True)
    roi_edges = cv2.bitwise_and(edges, roi)
    vectors = np.concatenate([np.roll(polygon, -1, axis=0) - polygon for polygon in polygons])
    expected_angles = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0])) % 180
    cv2.setRNGSeed(42)
    raw = cv2.HoughLinesP(roi_edges, rho=1, theta=np.pi / 180,
                          threshold=cfg["hough_votes"], minLineLength=cfg["min_line_length"],
                          maxLineGap=cfg["max_line_gap"])
    raw_lines = [] if raw is None else raw.reshape(-1, 4).tolist()
    segments = []
    for x1, y1, x2, y2 in sorted(raw_lines):
        start = np.array([x1, y1], dtype=float)
        end = np.array([x2, y2], dtype=float)
        length = float(np.linalg.norm(end - start))
        if length < cfg["min_line_length"]:
            continue
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180)
        if np.min(_geometry_angle_difference(angle, expected_angles)) > cfg["angle_tolerance_deg"]:
            continue
        samples = np.rint(np.linspace(start, end, max(2, int(np.ceil(length)) + 1))).astype(int)
        if np.mean(roi[samples[:, 1], samples[:, 0]] > 0) < cfg["min_roi_fraction"]:
            continue
        segments.append({"xyxy": [x1, y1, x2, y2], "length_px": length,
                         "angle_deg": angle, "supports_boundary": False})

    # Distances to finite segments, not infinite Hough lines.
    segment_xy = np.asarray([line["xyxy"] for line in segments], dtype=float).reshape(-1, 4)
    starts, ends = segment_xy[:, :2], segment_xy[:, 2:]
    directions = ends - starts
    lengths_squared = np.sum(directions ** 2, axis=1)
    angles = np.asarray([line["angle_deg"] for line in segments])
    slots = []
    for sid, polygon in zip(slot_ids, polygons):
        edge_support, edge_lengths = [], []
        for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
            vector = end - start
            length = float(np.linalg.norm(vector))
            angle = float(np.degrees(np.arctan2(vector[1], vector[0])) % 180)
            # Midpoint samples represent equal-length bins on this finite polygon edge.
            bins = max(1, int(np.ceil(length / 2.0)))
            points = start + ((np.arange(bins) + 0.5) / bins)[:, None] * vector
            supported = np.zeros(bins, dtype=bool)
            aligned = np.flatnonzero(_geometry_angle_difference(angles, angle)
                                     <= cfg["angle_tolerance_deg"])
            for index in aligned:
                delta = points - starts[index]
                t = np.clip((delta @ directions[index]) / lengths_squared[index], 0, 1)
                nearest = starts[index] + t[:, None] * directions[index]
                close = np.linalg.norm(points - nearest, axis=1) <= cfg["boundary_distance"]
                supported |= close
                if close.any():
                    segments[index]["supports_boundary"] = True
            edge_support.append(float(np.mean(supported)))
            edge_lengths.append(length)
        count = sum(value >= cfg["min_edge_support"] for value in edge_support)
        coverage = float(np.average(edge_support, weights=edge_lengths))
        flag = ("supported" if count >= cfg["min_supported_edges"] else
                "limited_support" if coverage > 0 else "no_support")
        slots.append({"space_id": sid, "boundary_support": coverage,
                      "edge_support": edge_support, "supported_edges": count,
                      "quality_flag": flag})
    return {"config": cfg, "raw_line_count": len(raw_lines), "segments": segments,
            "slots": slots, "quality_counts": {
                flag: sum(slot["quality_flag"] == flag for slot in slots)
                for flag in ("supported", "limited_support", "no_support")},
            "images": {"gray": gray, "blurred": blurred, "edges": edges,
                       "roi": roi, "roi_edges": roi_edges}}


def make_parking_tiles(image_bgr, layout, padding=32, grid=(2, 2), overlap=0.2,
                       max_overlap=0.6, min_tile_size=256, output_size=None):
    """Crop color parking tiles from the selected known polygon layout.

    grid is (rows, columns), each at most two. Padding is original-image
    pixels of context around each space, clipped at image edges. Overlap is
    the shared strip as a fraction of the ROI dimension, increased as needed
    up to max_overlap. Small axes use one tile; if full-space context cannot
    fit under the overlap cap, return the full ROI as one tile.

    output_size optionally specifies (width, height) for aspect-preserving
    letterboxing; otherwise preserve native pixels. Bounds are xyxy image
    edges, with right/bottom exclusive for slicing. Each tile contains its
    BGR image and metadata for mapping coordinates, including actual rounded
    resize scales and letterbox padding. No occupancy labels are read.
    """
    import cv2
    import numpy as np

    if (not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8
            or image_bgr.ndim != 3 or image_bgr.shape[2] != 3):
        raise ValueError("Expected a uint8 BGR image with three channels")
    height, width = image_bgr.shape[:2]
    if (width, height) != (layout["width"], layout["height"]):
        raise ValueError("Image size differs from layout")
    if (type(padding) is not int or padding < 0 or type(min_tile_size) is not int
            or min_tile_size < 1 or len(grid) != 2
            or any(type(n) is not int or n not in (1, 2) for n in grid)
            or not 0 <= overlap <= max_overlap < 1):
        raise ValueError("Invalid padding, grid, overlap or minimum tile size")
    if output_size is not None and (len(output_size) != 2 or any(
            type(n) is not int or n < 1 for n in output_size)):
        raise ValueError("output_size must be positive integer (width, height)")
    boxes, space_ids = [], []
    for space in layout["spaces"]:
        polygon = np.asarray(space["polygon"], dtype=float)
        if (polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3
                or not np.isfinite(polygon).all() or np.any(polygon < 0)
                or np.any(polygon > [width, height])
                or cv2.contourArea(polygon.astype(np.float32)) <= 0):
            raise ValueError("Invalid parking-space polygon")
        lo, hi = polygon.min(axis=0), polygon.max(axis=0)
        boxes.append([*lo, *hi])
        space_ids.append(str(space["space_id"]))
    if not boxes or len(set(space_ids)) != len(space_ids):
        raise ValueError("Expected nonempty layout with unique space IDs")
    boxes = np.asarray(boxes)
    context = boxes.copy()
    context[:, :2] = np.maximum(0, boxes[:, :2] - padding)
    context[:, 2:] = np.minimum([width, height], boxes[:, 2:] + padding)
    roi = [int(np.floor(context[:, 0].min())), int(np.floor(context[:, 1].min())),
           int(np.ceil(context[:, 2].max())), int(np.ceil(context[:, 3].max()))]
    x0, y0, x1, y1 = roi
    rows = grid[0] if y1 - y0 >= 2 * min_tile_size else 1
    columns = grid[1] if x1 - x0 >= 2 * min_tile_size else 1
    effective_overlap = float(overlap)
    # For two end-anchored crops of length t, [a,b] fits if t >= min(b, L-a).
    # This finds the overlap needed to retain each full space plus its context.
    for axis, count, origin, length in ((0, columns, x0, x1 - x0),
                                        (1, rows, y0, y1 - y0)):
        if count == 2:
            a = context[:, axis] - origin
            b = context[:, axis + 2] - origin
            required = float(np.max(2 * np.minimum(b, length - a) / length - 1))
            effective_overlap = max(effective_overlap, required)
    fallback_reason = None
    if effective_overlap > max_overlap + 1e-12:
        rows, columns = 1, 1
        effective_overlap = 0.0
        fallback_reason = "context_exceeds_max_overlap"
    elif rows == columns == 1:
        effective_overlap = 0.0
    else:
        effective_overlap = min(effective_overlap, float(max_overlap))

    def intervals(start, end, count):
        if count == 1:
            return [(start, end)]
        length = int(np.ceil((end - start) * (1 + effective_overlap) / 2))
        return [(start, start + length), (end - length, end)]

    def contained(bounds, candidates):
        return ((candidates[:, 0] >= bounds[0]) & (candidates[:, 1] >= bounds[1])
                & (candidates[:, 2] <= bounds[2]) & (candidates[:, 3] <= bounds[3]))

    tiles = []
    for row, (top, bottom) in enumerate(intervals(y0, y1, rows)):
        for column, (left, right) in enumerate(intervals(x0, x1, columns)):
            bounds = [left, top, right, bottom]
            crop = image_bgr[top:bottom, left:right].copy()
            crop_height, crop_width = crop.shape[:2]
            scale_xy, pad = [1.0, 1.0], [0, 0, 0, 0]
            if output_size is not None:
                out_width, out_height = output_size
                scale = min(out_width / crop_width, out_height / crop_height)
                resized_width = min(out_width, max(1, int(round(crop_width * scale))))
                resized_height = min(out_height, max(1, int(round(crop_height * scale))))
                resized = cv2.resize(crop, (resized_width, resized_height),
                                     interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
                pad_left = (out_width - resized_width) // 2
                pad_top = (out_height - resized_height) // 2
                pad = [pad_left, pad_top, out_width - resized_width - pad_left,
                       out_height - resized_height - pad_top]
                crop = cv2.copyMakeBorder(resized, pad[1], pad[3], pad[0], pad[2],
                                          cv2.BORDER_CONSTANT, value=(114, 114, 114))
                scale_xy = [resized_width / crop_width, resized_height / crop_height]
            metadata = {
                "tile_id": f"r{row}c{column}", "bounds_xyxy": bounds,
                "offset_xy": [left, top], "source_size": [crop_width, crop_height],
                "output_size": [crop.shape[1], crop.shape[0]], "scale_xy": scale_xy,
                "padding_ltrb": pad,
                "space_ids": [sid for sid, ok in zip(space_ids, contained(bounds, boxes)) if ok],
                "context_space_ids": [sid for sid, ok in zip(space_ids, contained(bounds, context)) if ok],
            }
            tiles.append({"image": crop, "metadata": metadata})
    coverage = {
        sid: {"tile_ids": [t["metadata"]["tile_id"] for t in tiles if sid in t["metadata"]["space_ids"]],
              "context_tile_ids": [t["metadata"]["tile_id"] for t in tiles if sid in t["metadata"]["context_space_ids"]]}
        for sid in space_ids
    }
    all_covered = all(item["tile_ids"] for item in coverage.values())
    all_context_covered = all(item["context_tile_ids"] for item in coverage.values())
    if not all_covered or not all_context_covered:
        raise RuntimeError("Tile layout does not cover every full parking space and its context")
    return {"roi_bounds_xyxy": roi, "requested_grid": list(grid), "grid": [rows, columns],
            "requested_overlap": float(overlap), "effective_overlap": effective_overlap,
            "context_padding_px": padding, "fallback_reason": fallback_reason,
            "all_spaces_covered": all_covered, "all_context_covered": all_context_covered,
            "coverage": coverage, "tiles": tiles}


def _transform_tile_points(points, metadata, to_image):
    import numpy as np

    points = np.asarray(points, dtype=float)
    scale = np.asarray(metadata["scale_xy"], dtype=float)
    offset = np.asarray(metadata["offset_xy"], dtype=float)
    pad = np.asarray(metadata["padding_ltrb"], dtype=float)
    if (points.ndim < 1 or points.shape[-1] != 2 or not np.isfinite(points).all()
            or scale.shape != (2,) or offset.shape != (2,) or pad.shape != (4,)
            or not np.isfinite(scale).all() or not np.isfinite(offset).all()
            or not np.isfinite(pad).all() or np.any(scale <= 0) or np.any(pad < 0)):
        raise ValueError("Expected finite (...,2) points and valid tile transform metadata")
    if to_image:
        return (points - pad[:2]) / scale + offset
    return (points - offset) * scale + pad[:2]


def tile_points_to_image(points, metadata):
    """Map (...,2) tile coordinates to original image edges, without clipping."""
    return _transform_tile_points(points, metadata, to_image=True)


def image_points_to_tile(points, metadata):
    """Map (...,2) original image coordinates to tile edges, without clipping."""
    return _transform_tile_points(points, metadata, to_image=False)


def tile_boxes_to_image(boxes, metadata, clip=False):
    """Map (...,4) xyxy boxes to the original image using the tile metadata.

    Optional clipping removes letterbox overflow at the source crop bounds.
    A box entirely in padding can become zero-area; filter those downstream.
    """
    import numpy as np

    boxes = np.asarray(boxes, dtype=float)
    if (boxes.ndim < 1 or boxes.shape[-1] != 4 or not np.isfinite(boxes).all()
            or np.any(boxes[..., 2:] < boxes[..., :2])):
        raise ValueError("Expected finite ordered (...,4) xyxy boxes")
    mapped = tile_points_to_image(boxes.reshape(-1, 2), metadata).reshape(boxes.shape)
    if clip:
        bounds = metadata["bounds_xyxy"]
        mapped[..., [0, 2]] = np.clip(mapped[..., [0, 2]], bounds[0], bounds[2])
        mapped[..., [1, 3]] = np.clip(mapped[..., [1, 3]], bounds[1], bounds[3])
    return mapped


def prepare_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--output", default="outputs/full")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--wait-complete", action="store_true", help="Wait up to 24h for an already running download")
    args = parser.parse_args(argv)
    if args.wait_complete:
        archive = Path(args.raw)/"PKLot.tar.gz"
        status = Path(args.output).parent/"pipeline_status.json"
        status.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + 24*3600
        while not archive.exists() or archive.stat().st_size < EXPECTED_BYTES:
            status.write_text(json.dumps(dict(stage="downloading", bytes=archive.stat().st_size if archive.exists() else 0, expected_bytes=EXPECTED_BYTES)))
            if time.monotonic() >= deadline:
                raise TimeoutError("Download not complete after 24 hours; resume prepare manually")
            time.sleep(15)
        args.no_download = True
        status.write_text(json.dumps(dict(stage="extracting_and_auditing")))
    result = prepare(args.raw, args.output, not args.no_download)
    print(json.dumps(result, indent=2))
    if args.wait_complete:
        status.write_text(json.dumps(dict(stage="complete", summary=result), indent=2))



def project_splits_main(argv=None):
    parser = argparse.ArgumentParser(description="Freeze shared PKLot subsets and slot geometry")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="data/manifests/pklot_v1")
    parser.add_argument("--config", default="configs/pklot_v1.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    lock = build_project_splits(args.manifest,args.output,args.config,args.overwrite)
    print(json.dumps({name:report["stats"] for name,report in lock["protocols"].items()},indent=2))


def evaluation_main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate parking occupancy by frame and slot keys")
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--predictions",required=True,help="JSONL occupied_score or state records")
    parser.add_argument("--timings",help="Optional JSONL timing per original image")
    parser.add_argument("--threshold",type=float,default=.5)
    parser.add_argument("--missing-policy",choices=["unknown","error"],default="unknown")
    parser.add_argument("--method",default="unspecified")
    parser.add_argument("--metadata",help="Optional JSON object with protocol, split, device and environment")
    parser.add_argument("--output",default="outputs/evaluation")
    args = parser.parse_args(argv)
    metadata = json.loads(Path(args.metadata).read_text()) if args.metadata else {}
    if not isinstance(metadata,dict):
        raise ValueError("Metadata JSON must be an object")
    metadata.update(manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                    predictions_sha256=hashlib.sha256(Path(args.predictions).read_bytes()).hexdigest())
    if args.timings:
        metadata["timings_sha256"] = hashlib.sha256(Path(args.timings).read_bytes()).hexdigest()
    report = evaluate_occupancy(args.manifest,args.predictions,args.timings,args.threshold,
                                args.missing_policy,args.method,metadata)
    export_evaluation(report,args.output)
    print(json.dumps(report["overall"],indent=2,allow_nan=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description="PKLot single-file pipeline")
    parser.add_argument("command", choices=["prepare", "audit", "splits", "evaluate"])
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv[:1])
    {"prepare":prepare_main,"audit":audit_main,"splits":project_splits_main,"evaluate":evaluation_main}[args.command](argv[1:])

if __name__ == "__main__":
    main()
