"""C11 profile for the authorization-gated official-scoring plan builder."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from research_math import c11_native_scoring as scoring


_path = Path(__file__).with_name("prepare_c14_native_scoring.py")
_spec = importlib.util.spec_from_file_location("_c11_shared_plan_builder", _path)
if _spec is None or _spec.loader is None:
    raise ImportError("Unable to load shared native scoring plan builder")
_shared = importlib.util.module_from_spec(_spec)
_shared.scoring = scoring
_spec.loader.exec_module(_shared)

_shared.scoring = scoring
_shared.CANDIDATE_ID = "4d-math-20261006-c11"
_shared.ROLE_FOR_CONTRACT_ARM = {
    "treatment": "rotation_preserving_stretch_projection",
    "baseline": "b_star", "b0": "b0",
    "arap_repair": "arap_repair", "elastic_repair": "elastic_repair",
}
_shared.PROFILE = "c11"
_shared.TASK_ID = "c11-five-role-official-scoring"
_shared.AUTHORIZATION_SCOPE = "single_c11_scoring_attempt"
_shared.INPUT_NAMESPACE = "c11"
_shared.OUTPUT_DIRECTORY = "c11-scoring-output"
_shared.SCORING_MODULE = "research_math.c11_native_scoring"
_shared.COMPARISON_SOURCE = "actionmesh/research_math/c11_native_comparison.py"
_shared.SCORING_SOURCE = "actionmesh/research_math/c11_native_scoring.py"
_shared.LAUNCHER_SOURCE = "actionmesh/launch_c11_native_scoring.py"
_shared.EXTRA_CODE_SOURCES = (
    "actionmesh/prepare_c11_native_scoring.py",
    "actionmesh/prepare_c14_native_scoring.py",
    "actionmesh/launch_c14_native_scoring.py",
    "actionmesh/research_math/c14_native_scoring.py",
    "actionmesh/research_math/strain_projection_candidate.py",
    "actionmesh/research_math/integrable_gradient_candidate.py",
    "actionmesh/research_math/corotational_residual_candidate.py",
    "actionmesh/prepare_strain_projection_candidate.py",
)
_shared.LAUNCH_TICKET_KIND = "c11-staged-launch-ticket"
_shared.AUTHORIZATION_KIND = "c11-gpu-resume-authorization"
_shared.RESERVATION_KIND = "c11-gpu-resume-authorization-reservation"
_shared.CONSUMPTION_KIND = "c11-gpu-resume-authorization-consumption"
_shared.FAMILY_SPLIT_KIND = "c11-family-split"
_shared.FAMILY_ASSIGNMENTS_KIND = "c11-family-assignments"
_shared.FAMILY_DERIVATION_KIND = "c11-family-derivation"
_shared.FAMILY_REVIEW_KIND = "c11-family-derivation-review"
_shared.CRITERIA_KIND = "c11-outcome-criteria"
_shared.ANALYSIS_KIND = "c11-g01-analysis-plan"
_shared.ADMISSION_KIND = "c11-scientific-dispatch-admission"
_shared.CONTRACT_CONTRASTS = {
    "treatment": "rotation_preserving_stretch_projection",
    "baseline": "b_star", "controls": ["b0", "arap_repair", "elastic_repair"],
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
