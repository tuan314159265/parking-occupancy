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
AUDIT_VERSION = 1

CAMERAS = {"UFPR04", "UFPR05", "PUCPR", "PUC", "parking1a", "parking1b", "parking2"}

def metadata(path):
    parts = path.parts
    for i, part in enumerate(parts):
        if part.lower() in WEATHER and i > 0 and i + 1 < len(parts):
            camera, day = parts[i-1], parts[i+1]
            if camera not in CAMERAS:
                raise ValueError(f"unknown camera: {camera}")
            date.fromisoformat(day)
            return camera, WEATHER[part.lower()], day
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
        polygon = tuple((float(p.attrib["x"]), float(p.attrib["y"]))
                        for p in element.findall("contour/point"))
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
                 camera, weather, day, width, height, tuple(spaces))

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



def main(argv=None):
    parser = argparse.ArgumentParser(description="PKLot single-file pipeline")
    parser.add_argument("command", choices=["prepare", "audit"])
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv[:1])
    (prepare_main if args.command == "prepare" else audit_main)(argv[1:])

if __name__ == "__main__":
    main()
