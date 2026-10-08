"""C07 profile for the authorization-gated official-scoring plan builder."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from research_math import c07_native_scoring as scoring


_path = Path(__file__).with_name("prepare_c14_native_scoring.py")
_spec = importlib.util.spec_from_file_location("_c07_shared_plan_builder", _path)
if _spec is None or _spec.loader is None:
    raise ImportError("Unable to load shared native scoring plan builder")
_shared = importlib.util.module_from_spec(_spec)
_shared.scoring = scoring
_spec.loader.exec_module(_shared)

_shared.scoring = scoring
_shared.CANDIDATE_ID = "4d-math-20261006-c07"
_shared.ROLE_FOR_CONTRACT_ARM = {
    "treatment": "partial_mass_native_fallback", "baseline": "b_star", "b0": "b0",
    "confidence_threshold_fallback": "confidence_threshold_fallback",
    "full_mass_transport": "full_mass_transport",
}
_shared.PROFILE = "c07"
_shared.TASK_ID = "c07-five-role-official-scoring"
_shared.AUTHORIZATION_SCOPE = "single_c07_scoring_attempt"
_shared.INPUT_NAMESPACE = "c07"
_shared.OUTPUT_DIRECTORY = "c07-scoring-output"
_shared.SCORING_MODULE = "research_math.c07_native_scoring"
_shared.COMPARISON_SOURCE = "actionmesh/research_math/c07_native_comparison.py"
_shared.SCORING_SOURCE = "actionmesh/research_math/c07_native_scoring.py"
_shared.LAUNCHER_SOURCE = "actionmesh/launch_c07_native_scoring.py"
_shared.EXTRA_CODE_SOURCES = (
    "actionmesh/prepare_c07_native_scoring.py",
    "actionmesh/prepare_c14_native_scoring.py",
    "actionmesh/launch_c14_native_scoring.py",
    "actionmesh/research_math/c14_native_scoring.py",
    "actionmesh/research_math/c11_native_comparison.py",
    "actionmesh/research_math/partial_transport_candidate.py",
    "actionmesh/research_math/protected_geometry_candidate.py",
    "actionmesh/prepare_partial_transport_candidate.py",
    "actionmesh/research_ten/__init__.py",
    "actionmesh/research_ten/m01_elasticity.py",
)
_shared.LAUNCH_TICKET_KIND = "c07-staged-launch-ticket"
_shared.AUTHORIZATION_KIND = "c07-gpu-resume-authorization"
_shared.RESERVATION_KIND = "c07-gpu-resume-authorization-reservation"
_shared.CONSUMPTION_KIND = "c07-gpu-resume-authorization-consumption"
_shared.FAMILY_SPLIT_KIND = "c07-family-split"
_shared.FAMILY_ASSIGNMENTS_KIND = "c07-family-assignments"
_shared.FAMILY_DERIVATION_KIND = "c07-family-derivation"
_shared.FAMILY_REVIEW_KIND = "c07-family-derivation-review"
_shared.CRITERIA_KIND = "c07-outcome-criteria"
_shared.ANALYSIS_KIND = "c07-g01-analysis-plan"
_shared.ADMISSION_KIND = "c07-scientific-dispatch-admission"
_shared.CONTRACT_CONTRASTS = {
    "treatment": "partial_mass_native_fallback", "baseline": "b_star",
    "controls": ["b0", "confidence_threshold_fallback", "full_mass_transport"],
}
_shared.CONTRACT_ARM_NAMES = dict(_shared.ROLE_FOR_CONTRACT_ARM)
_shared.PRIMARY_METRIC = "cd_motion"
_shared.GUARDRAIL_METRICS = ("cd_3d", "cd_4d")


def __getattr__(name):
    return getattr(_shared, name)


def main() -> int:
    return _shared.main()


if __name__ == "__main__":
    raise SystemExit(main())
