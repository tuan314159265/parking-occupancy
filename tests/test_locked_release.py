"""Guard the frozen release: code, splits, plan and requirements must match the validation lock.

These checks need no dataset, model or network. If one fails, the change would make
verify_validation_lock() refuse the final test, so revert it or start a new validation run.
"""
import hashlib
import json
from pathlib import Path
import re
import unittest
from unittest.mock import patch

import parking_module as parking

ROOT = Path(__file__).resolve().parents[1]
LOCK = json.loads((ROOT / "reports/validation_v1/locked_config.json").read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class LockedReleaseTests(unittest.TestCase):
    def test_engine_source_matches_lock(self):
        # Hash only the source code; ultralytics is not needed for this check.
        with patch.object(parking, "prepare_validation_runtime", lambda: False):
            self.assertEqual(parking.validation_engine_sha256(), LOCK["engine_sha256"])

    def test_frozen_manifests_match_lock(self):
        manifest_dir = ROOT / "data/manifests/pklot_v1" / LOCK["protocol"]
        for name, digest in LOCK["manifest_sha256"].items():
            self.assertEqual(sha256(manifest_dir / name), digest, name)

    def test_published_plan_and_checkpoint_match_lock(self):
        plan = json.loads((ROOT / "reports/validation_v1/preregistered_plan.json").read_text(encoding="utf-8"))["plan"]
        self.assertEqual(parking._validation_hash(plan), LOCK["plan_sha256"])
        config = json.loads((ROOT / "configs/validation_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(config["checkpoint"]["sha256"], LOCK["checkpoint_sha256"])
        self.assertEqual(config["checkpoint"]["url"], LOCK["checkpoint_source"])
        self.assertTrue(LOCK["complete"])
        self.assertEqual(LOCK["selected_on"], "val")

    def test_published_report_hashes(self):
        hashes = json.loads((ROOT / "reports/validation_v1/artifact_hashes.json").read_text(encoding="utf-8"))
        for name, digest in hashes.items():
            self.assertEqual(sha256(ROOT / "reports/validation_v1" / name), digest, name)

    def test_requirements_pin_locked_library_versions(self):
        text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        pins = dict(re.findall(r"^([A-Za-z0-9_.-]+)==([^\s#]+)", text, flags=re.M))
        packages = dict(numpy="numpy", opencv="opencv-python", torch="torch", ultralytics="ultralytics",
                        skimage="scikit-image", sklearn="scikit-learn")
        for key, package in packages.items():
            locked = LOCK["environment"][key].split("+")[0]
            self.assertIn(package, pins, package)
            # opencv wheels carry a build suffix (4.14.0.94) on top of cv2.__version__ (4.14.0).
            self.assertTrue(pins[package] == locked or pins[package].startswith(locked + "."), package)


if __name__ == "__main__":
    unittest.main()
