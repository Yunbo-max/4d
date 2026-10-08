"""Local-only non-regression checks for the isolated C04 scoring profile."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import launch_c04_native_scoring as launcher
import prepare_c04_native_scoring as plan
from research_math import c04_native_scoring as scoring
from research_math import c14_native_scoring as c14_scoring


class C04NativeScoringProfileTests(unittest.TestCase):
    def test_profile_isolated_from_c14_defaults(self):
        self.assertEqual(c14_scoring.REQUEST_KIND, "c14-native-scoring-request")
        self.assertEqual(c14_scoring.CANDIDATE_ROLE, "corotational_residual")
        self.assertEqual(scoring.REQUEST_KIND, "c04-native-scoring-request")
        self.assertEqual(scoring.CANDIDATE_ROLE, "robust_conic_protection")
        self.assertEqual(scoring.CONTROL_ROLES, (
            "b0", "b_star", "deterministic_protection", "strength_matched_repair"))
        self.assertEqual(
            scoring.CONTRAST_DIRECTION,
            "robust conic protection minus control; lower is better")

    def test_plan_and_launcher_have_only_c04_identity(self):
        self.assertEqual(plan.TASK_ID, "c04-five-role-official-scoring")
        self.assertEqual(plan.PRIMARY_METRIC, "cd_3d")
        self.assertEqual(plan.GUARDRAIL_METRICS, ("cd_4d", "cd_motion"))
        self.assertEqual(plan.AUTHORIZATION_SCOPE, "single_c04_scoring_attempt")
        self.assertEqual(plan.AUTHORIZATION_KIND, "c04-gpu-resume-authorization")
        self.assertEqual(launcher.CLAIM_KIND, "c04-launch-claim")
        self.assertEqual(launcher.ENV_PREFIX, "C04")
        self.assertIn("actionmesh/research_math/robust_motion_candidate.py",
                      plan.EXTRA_CODE_SOURCES)
        self.assertIn("actionmesh/prepare_c14_native_scoring.py",
                      plan.EXTRA_CODE_SOURCES)
        self.assertIn("actionmesh/launch_c14_native_scoring.py",
                      plan.EXTRA_CODE_SOURCES)


    def test_complete_source_closure_imports_in_an_independent_staging_root(self):
        # The separate interpreter cannot import omitted project files from the
        # live checkout: -I discards PYTHONPATH and only staged code is inserted.
        root = Path(__file__).resolve().parents[3]
        refs = set(plan.EXTRA_CODE_SOURCES) | {
            plan.COMPARISON_SOURCE, plan.SCORING_SOURCE, plan.LAUNCHER_SOURCE,
            "actionmesh/research_math/__init__.py",
            "actionmesh/official_actionbench_adapter.py",
        }
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory)
            for relative in refs:
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / relative, destination)
            script = (
                "import sys,json;sys.path.insert(0,sys.argv[1]);"
                "import launch_c04_native_scoring as launch;"
                "import prepare_c04_native_scoring as plan;"
                "from research_math import c04_native_scoring as score;"
                "assert plan.CANDIDATE_ID=='4d-math-20261006-c04';"
                "assert score.CANDIDATE_ROLE=='robust_conic_protection';"
                "print(json.dumps({'launcher':launch.__file__,'planner':plan.__file__,"
                "'scorer':score.__file__}))")
            result = subprocess.run(
                [sys.executable, "-I", "-c", script, str(staging / "actionmesh")],
                cwd=staging, capture_output=True, text=True, check=True,
                env={key: value for key, value in os.environ.items()
                     if key != "PYTHONPATH"})
            observed = json.loads(result.stdout)
            for path in observed.values():
                Path(path).resolve().relative_to(staging)



if __name__ == "__main__":
    unittest.main()
