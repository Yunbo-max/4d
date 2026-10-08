"""C06 profile for the authorization-gated official-scoring plan builder."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from research_math import c06_native_scoring as scoring


_path = Path(__file__).with_name("prepare_c14_native_scoring.py")
_spec = importlib.util.spec_from_file_location("_c06_shared_plan_builder", _path)
if _spec is None or _spec.loader is None:
    raise ImportError("Unable to load shared native scoring plan builder")
_shared = importlib.util.module_from_spec(_spec)
_shared.scoring = scoring
_spec.loader.exec_module(_shared)

_shared.scoring = scoring
_shared.CANDIDATE_ID = "4d-math-20261006-c06"
_shared.ROLE_FOR_CONTRACT_ARM = {
    "treatment": "area_marginal_transport", "baseline": "b_star", "b0": "b0",
    "row_softmax": "row_softmax",
    "vertex_density_transport": "vertex_density_transport",
}
_shared.PROFILE = "c06"
_shared.TASK_ID = "c06-five-role-official-scoring"
_shared.AUTHORIZATION_SCOPE = "single_c06_scoring_attempt"
_shared.INPUT_NAMESPACE = "c06"
_shared.OUTPUT_DIRECTORY = "c06-scoring-output"
_shared.SCORING_MODULE = "research_math.c06_native_scoring"
_shared.COMPARISON_SOURCE = "actionmesh/research_math/c06_native_comparison.py"
_shared.SCORING_SOURCE = "actionmesh/research_math/c06_native_scoring.py"
_shared.LAUNCHER_SOURCE = "actionmesh/launch_c06_native_scoring.py"
_shared.EXTRA_CODE_SOURCES = (
    "actionmesh/prepare_c06_native_scoring.py",
    "actionmesh/prepare_c14_native_scoring.py",
    "actionmesh/launch_c14_native_scoring.py",
    "actionmesh/research_math/c14_native_scoring.py",
    "actionmesh/research_math/c11_native_comparison.py",
    "actionmesh/research_math/area_transport_candidate.py",
    "actionmesh/research_math/protected_geometry_candidate.py",
    "actionmesh/prepare_area_transport_candidate.py",
    "actionmesh/research_ten/__init__.py",
    "actionmesh/research_ten/m01_elasticity.py",
)
_shared.LAUNCH_TICKET_KIND = "c06-staged-launch-ticket"
_shared.AUTHORIZATION_KIND = "c06-gpu-resume-authorization"
_shared.RESERVATION_KIND = "c06-gpu-resume-authorization-reservation"
_shared.CONSUMPTION_KIND = "c06-gpu-resume-authorization-consumption"
_shared.FAMILY_SPLIT_KIND = "c06-family-split"
_shared.FAMILY_ASSIGNMENTS_KIND = "c06-family-assignments"
_shared.FAMILY_DERIVATION_KIND = "c06-family-derivation"
_shared.FAMILY_REVIEW_KIND = "c06-family-derivation-review"
_shared.CRITERIA_KIND = "c06-outcome-criteria"
_shared.ANALYSIS_KIND = "c06-g01-analysis-plan"
_shared.ADMISSION_KIND = "c06-scientific-dispatch-admission"
_shared.CONTRACT_CONTRASTS = {
    "treatment": "area_marginal_transport", "baseline": "b_star",
    "controls": ["b0", "row_softmax", "vertex_density_transport"],
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
