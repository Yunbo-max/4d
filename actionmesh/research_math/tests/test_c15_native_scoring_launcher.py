"""Single-owner/stable-claim checks for the C15 controller launcher."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import launch_c15_native_scoring as launcher


class C15NativeScoringLauncherTests(unittest.TestCase):
    def consumption(self):
        return {
            "authorization_ref": {"path": "inputs/c15/auth.json",
                                  "sha256": "a" * 64},
            "run_id": "c15-native-scoring-001",
            "task_id": "c15-six-role-official-scoring",
            "native_plan_ref": {"path": "plans/c15/native.json",
                                "sha256": "b" * 64},
            "native_plan_digest": "c" * 64,
            "harness_plan_ref": {"path": "plans/c15/harness.json",
                                 "sha256": "d" * 64},
        }

    def test_claim_is_single_owner_and_reused_after_interruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "inputs/c15/authorization-consumption/receipt.json"
            path.parent.mkdir(parents=True)
            value = self.consumption()
            path.write_text(json.dumps(value))
            first_lock, first_ref = launcher.claim_launch(
                root, path, value, expected_harness_digest="e" * 64)
            try:
                with self.assertRaisesRegex(RuntimeError, "live controller owner"):
                    launcher.claim_launch(
                        root, path, value, expected_harness_digest="e" * 64)
            finally:
                first_lock.close()
            second_lock, second_ref = launcher.claim_launch(
                root, path, value, expected_harness_digest="e" * 64)
            try:
                self.assertEqual(first_ref, second_ref)
                claim = json.loads((root / second_ref["path"]).read_text())
                self.assertEqual(claim["kind"], "c15-launch-claim")
                self.assertEqual(claim["task_id"], "c15-six-role-official-scoring")
            finally:
                second_lock.close()

    def test_launch_requires_complete_installed_harness_before_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            consumption = root / "receipt.json"; consumption.write_text("{}")
            skill = root / "skill"; skill.mkdir()
            with self.assertRaisesRegex(ValueError, "research-autopilot"):
                launcher.launch(
                    root, consumption, skill_dir=skill,
                    approved_plan_digest="f" * 64)


if __name__ == "__main__":
    unittest.main()
