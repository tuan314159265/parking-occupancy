"""Build a small local handoff ZIP: frozen models/configs/code, never dataset.

Run from the repository with the locked validation Python environment.
The output is ignored by Git and is not uploaded anywhere by this script.
"""
import argparse
import json
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import final_test_module as flow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/parking-final-test-bundle.zip")
    args = parser.parse_args()
    val = ROOT / "outputs/validation_v1"
    md = ROOT / "data/manifests/pklot_v1/within_camera"
    info = flow.check_inputs(ROOT / "data/raw/PKLot", md, val, ROOT / "models/yolo11n.pt", check_dataset=False)
    paths = ["parking_module.py", "final_test_module.py", "notebooks/01_pklot_run_all.ipynb",
             "models/yolo11n.pt",
             "outputs/validation_v1/locked_config.json", "outputs/validation_v1/preregistered_plan.json",
             "outputs/validation_v1/hog_svm.joblib", "outputs/validation_v1/geometry_references.joblib",
             "reports/validation_v1/provenance.json", "configs/validation_v1.json", "configs/pklot_v1.json",
             "data/manifests/pklot_v1/slot_geometries.json", "data/manifests/pklot_v1/split_lock.json"]
    # The common loader verifies every artifact listed in split_lock, including
    # protocols other than the one used by the selected final-test experiment.
    split_lock = flow._json(md.parent / "split_lock.json")
    paths += [f"data/manifests/pklot_v1/{name}" for name in split_lock["artifacts_sha256"]]
    paths += ["outputs/validation_v1/validation_summary.json", "outputs/validation_v1/comparison.csv",
              "outputs/validation_v1/hog_search.json", "outputs/validation_v1/detector_search.json",
              "outputs/validation_v1/tile_plans.json"]
    paths = list(dict.fromkeys(paths))
    for rel in paths:
        if not (ROOT / rel).is_file():
            raise FileNotFoundError(ROOT / rel)
    packages = flow._json(ROOT / "reports/validation_v1/provenance.json")["packages"]
    requirements = "\n".join(f"{name}=={version}" for name, version in sorted(packages.items()) if name != "opencv-python-headless") + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.output, "w", compression=ZIP_DEFLATED) as archive:
        for rel in paths:
            archive.write(ROOT / rel, rel)
        archive.writestr("requirements-locked-reference.txt", requirements)
        archive.writestr("bundle_manifest.json", json.dumps(dict(
            inputs=info, files={rel: flow._sha(ROOT / rel) for rel in paths},
            python_reference="3.14.7", dataset_included=False, final_test_performed=False,
            environment_note="Requirements record the local CPU validation environment; wheel availability on Colab is not guaranteed. Do not bypass the lock or silently change versions."
        ), ensure_ascii=False, indent=2) + "\n")
    print(f"Bundle ready: {args.output} ({args.output.stat().st_size / 1024**2:.2f} MiB)")
    print("Contains frozen code/models/metadata. Contains no PKLot images and performs no test inference.")


if __name__ == "__main__":
    main()
