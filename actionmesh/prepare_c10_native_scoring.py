"""C10 profile for the authorization-gated official-scoring plan builder."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from research_math import c10_native_scoring as scoring


_path = Path(__file__).with_name("prepare_c14_native_scoring.py")
_spec = importlib.util.spec_from_file_location("_c10_shared_plan_builder", _path)
if _spec is None or _spec.loader is None:
    raise ImportError("Unable to load shared native scoring plan builder")
_shared = importlib.util.module_from_spec(_spec)
_shared.scoring = scoring
_spec.loader.exec_module(_shared)

_shared.scoring = scoring
_shared.CANDIDATE_ID = "4d-math-20261006-c10"
_shared.ROLE_FOR_CONTRACT_ARM = {
    "treatment": "pinned_integrable_solve", "baseline": "b_star", "b0": "b0",
    "direct_common_lift": "direct_common_lift",
    "independent_local_repair": "independent_local_repair",
}
_shared.PROFILE = "c10"
_shared.TASK_ID = "c10-five-role-official-scoring"
_shared.AUTHORIZATION_SCOPE = "single_c10_scoring_attempt"
_shared.INPUT_NAMESPACE = "c10"
_shared.OUTPUT_DIRECTORY = "c10-scoring-output"
_shared.SCORING_MODULE = "research_math.c10_native_scoring"
_shared.COMPARISON_SOURCE = "actionmesh/research_math/c10_native_comparison.py"
_shared.SCORING_SOURCE = "actionmesh/research_math/c10_native_scoring.py"
_shared.LAUNCHER_SOURCE = "actionmesh/launch_c10_native_scoring.py"
_shared.EXTRA_CODE_SOURCES = (
    "actionmesh/prepare_c10_native_scoring.py",
    "actionmesh/prepare_c14_native_scoring.py",
    "actionmesh/launch_c14_native_scoring.py",
    "actionmesh/research_math/c14_native_scoring.py",
    "actionmesh/research_math/integrable_gradient_candidate.py",
    "actionmesh/prepare_integrable_gradient_candidate.py",
)
_shared.LAUNCH_TICKET_KIND = "c10-staged-launch-ticket"
_shared.AUTHORIZATION_KIND = "c10-gpu-resume-authorization"
_shared.RESERVATION_KIND = "c10-gpu-resume-authorization-reservation"
_shared.CONSUMPTION_KIND = "c10-gpu-resume-authorization-consumption"
_shared.FAMILY_SPLIT_KIND = "c10-family-split"
_shared.FAMILY_ASSIGNMENTS_KIND = "c10-family-assignments"
_shared.FAMILY_DERIVATION_KIND = "c10-family-derivation"
_shared.FAMILY_REVIEW_KIND = "c10-family-derivation-review"
_shared.CRITERIA_KIND = "c10-outcome-criteria"
_shared.ANALYSIS_KIND = "c10-g01-analysis-plan"
_shared.ADMISSION_KIND = "c10-scientific-dispatch-admission"
_shared.CONTRACT_CONTRASTS = {
    "treatment": "pinned_integrable_solve", "baseline": "b_star",
    "controls": ["b0", "direct_common_lift", "independent_local_repair"],
}
_shared.CONTRACT_ARM_NAMES = dict(_shared.ROLE_FOR_CONTRACT_ARM)
_shared.PRIMARY_METRIC = "cd_3d"
_shared.GUARDRAIL_METRICS = ("cd_4d", "cd_motion")






def __getattr__(name):
    return getattr(_shared, name)


def main() -> int:
    return _shared.main()


if __name__ == "__main__":
    raise SystemExit(main())
