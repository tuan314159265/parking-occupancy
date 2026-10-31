"""Standalone PKLot module for GitHub → Colab.

Requires Python >=3.10 and Pillow. Import directly or run:
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
    occupied_score: float


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


"""Download/resume official archive; safely extract full frames only; audit."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
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


def main(argv=None):
    parser = argparse.ArgumentParser(description="PKLot single-file pipeline")
    parser.add_argument("command", choices=["prepare", "audit", "splits"])
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv[:1])
    {"prepare":prepare_main,"audit":audit_main,"splits":project_splits_main}[args.command](argv[1:])

if __name__ == "__main__":
    main()
