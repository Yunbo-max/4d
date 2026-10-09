"""C20 profile for the authorization-gated official-scoring plan builder."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from research_math import c20_native_scoring as scoring


_path = Path(__file__).with_name("prepare_c14_native_scoring.py")
_spec = importlib.util.spec_from_file_location("_c20_shared_plan_builder", _path)
if _spec is None or _spec.loader is None:
    raise ImportError("Unable to load shared native scoring plan builder")
_shared = importlib.util.module_from_spec(_spec)
_shared.scoring = scoring
_spec.loader.exec_module(_shared)

_shared.scoring = scoring
_shared.CANDIDATE_ID = "4d-math-20261006-c20"
_shared.ROLE_FOR_CONTRACT_ARM = {
    "treatment": "joint_monotone_phase_amplitude", "baseline": "b_star",
    "b0": "b0", "phase_only": "phase_only",
    "amplitude_only": "amplitude_only", "simple_lag": "simple_lag",
}
# C20 uses the same retained native-context B0 identity contract as C11.
_shared.PROFILE = "c11"
_shared.TASK_ID = "c20-six-role-official-scoring"
_shared.AUTHORIZATION_SCOPE = "single_c20_scoring_attempt"
_shared.INPUT_NAMESPACE = "c20"
_shared.OUTPUT_DIRECTORY = "c20-scoring-output"
_shared.SCORING_MODULE = "research_math.c20_native_scoring"
_shared.COMPARISON_SOURCE = "actionmesh/research_math/c20_native_comparison.py"
_shared.SCORING_SOURCE = "actionmesh/research_math/c20_native_scoring.py"
_shared.LAUNCHER_SOURCE = "actionmesh/launch_c20_native_scoring.py"
_shared.EXTRA_CODE_SOURCES = (
    "actionmesh/prepare_c20_native_scoring.py",
    "actionmesh/prepare_c14_native_scoring.py",
    "actionmesh/launch_c14_native_scoring.py",
    "actionmesh/research_math/c14_native_scoring.py",
    "actionmesh/research_math/c11_native_comparison.py",
    "actionmesh/research_math/phase_amplitude_candidate.py",
    "actionmesh/research_math/phase_amplitude_artifacts.py",
    "actionmesh/research_math/c20_consensus_target.py",
    "actionmesh/prepare_c20_consensus_target.py",
    "actionmesh/prepare_phase_amplitude_candidate.py",
)
_shared.LAUNCH_TICKET_KIND = "c20-staged-launch-ticket"
_shared.AUTHORIZATION_KIND = "c20-gpu-resume-authorization"
_shared.RESERVATION_KIND = "c20-gpu-resume-authorization-reservation"
_shared.CONSUMPTION_KIND = "c20-gpu-resume-authorization-consumption"
_shared.FAMILY_SPLIT_KIND = "c20-family-split"
_shared.FAMILY_ASSIGNMENTS_KIND = "c20-family-assignments"
_shared.FAMILY_DERIVATION_KIND = "c20-family-derivation"
_shared.FAMILY_REVIEW_KIND = "c20-family-derivation-review"
_shared.CRITERIA_KIND = "c20-outcome-criteria"
_shared.ANALYSIS_KIND = "c20-g01-analysis-plan"
_shared.ADMISSION_KIND = "c20-scientific-dispatch-admission"
_shared.CONTRACT_CONTRASTS = {
    "treatment": "joint_monotone_phase_amplitude", "baseline": "b_star",
    "controls": ["b0", "phase_only", "amplitude_only", "simple_lag"],
}
_shared.CONTRACT_ARM_NAMES = dict(_shared.ROLE_FOR_CONTRACT_ARM)
_shared.PRIMARY_METRIC = "cd_motion"
_shared.GUARDRAIL_METRICS = ("cd_3d", "cd_4d")
_shared.ALLOW_G01_STAGES = True


def validate_family_binding(root: Path, request: dict, family: dict) -> None:
    comparison = scoring.read_json(scoring.resolve_ref(root, request["comparison_ref"]))
    stage = comparison.get("application_stage")
    ids = {"d1": family.get("d1_ids"), "d2": family.get("d2_ids"),
           "confirmation": family.get("confirmation_ids")}
    if (stage != request.get("application_stage") or stage not in ids
            or request.get("uid") not in (ids[stage] or [])
            or comparison.get("g01_family_split_digest") != family.get("split_digest")
            or scoring.read_json(scoring.resolve_ref(
                root, comparison.get("g01_family_split_ref"))).get("split_digest") !=
               family.get("split_digest")):
        raise ValueError("C20 scoring request is not bound to its frozen G01 stage")


_shared.validate_family_binding = validate_family_binding


def __getattr__(name):
    return getattr(_shared, name)


def main() -> int:
    return _shared.main()


if __name__ == "__main__":
    raise SystemExit(main())
