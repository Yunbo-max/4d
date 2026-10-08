"""C01 profile for the authorization-gated official-scoring plan builder.

The shared builder supplies schema verification, source-derived family closure,
single-use authorization reservation, immutable launch ticket and finalized
native/harness plan consumption.  This profile supplies C01-specific identities,
roles, metrics and paths without sharing method construction.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from research_math import c01_native_scoring as scoring


_path = Path(__file__).with_name("prepare_c14_native_scoring.py")
_spec = importlib.util.spec_from_file_location("_c01_shared_plan_builder", _path)
if _spec is None or _spec.loader is None:
    raise ImportError("Unable to load shared native scoring plan builder")
_shared = importlib.util.module_from_spec(_spec)
_shared.scoring = scoring
_spec.loader.exec_module(_shared)

_shared.scoring = scoring
_shared.CANDIDATE_ID = "4d-math-20261006-c01"
_shared.ROLE_FOR_CONTRACT_ARM = {
    "treatment": "self_map_subtraction", "baseline": "b_star", "b0": "b0",
    "raw_uncorrected": "raw_uncorrected",
    "mean_bias": "mean_bias",
}
_shared.PROFILE = "c01"
_shared.TASK_ID = "c01-five-role-official-scoring"
_shared.AUTHORIZATION_SCOPE = "single_c01_scoring_attempt"
_shared.INPUT_NAMESPACE = "c01"
_shared.OUTPUT_DIRECTORY = "c01-scoring-output"
_shared.SCORING_MODULE = "research_math.c01_native_scoring"
_shared.COMPARISON_SOURCE = "actionmesh/research_math/c01_native_comparison.py"
_shared.SCORING_SOURCE = "actionmesh/research_math/c01_native_scoring.py"
_shared.LAUNCHER_SOURCE = "actionmesh/launch_c01_native_scoring.py"
_shared.EXTRA_CODE_SOURCES = (
    "actionmesh/prepare_c01_native_scoring.py",
    "actionmesh/prepare_c14_native_scoring.py",
    "actionmesh/launch_c14_native_scoring.py",
    "actionmesh/research_math/c14_native_scoring.py",
    "actionmesh/research_math/self_map_candidate.py",
    "actionmesh/research_math/native_context_delivery.py",
    "actionmesh/research_math/native_context_runner.py",
    "actionmesh/research_math/pipeline_decoder_observer.py",
    "actionmesh/research_math/decoder_observer.py",
    "actionmesh/research_math/complete_unit_export.py",
    "actionmesh/research_math/deterministic_knn.py",
)
_shared.LAUNCH_TICKET_KIND = "c01-staged-launch-ticket"
_shared.AUTHORIZATION_KIND = "c01-gpu-resume-authorization"
_shared.RESERVATION_KIND = "c01-gpu-resume-authorization-reservation"
_shared.CONSUMPTION_KIND = "c01-gpu-resume-authorization-consumption"
_shared.FAMILY_SPLIT_KIND = "c01-family-split"
_shared.FAMILY_ASSIGNMENTS_KIND = "c01-family-assignments"
_shared.FAMILY_DERIVATION_KIND = "c01-family-derivation"
_shared.FAMILY_REVIEW_KIND = "c01-family-derivation-review"
_shared.CRITERIA_KIND = "c01-outcome-criteria"
_shared.ANALYSIS_KIND = "c01-g01-analysis-plan"
_shared.ADMISSION_KIND = "c01-scientific-dispatch-admission"
_shared.CONTRACT_CONTRASTS = {
    "treatment": "self_map_subtraction", "baseline": "b_star",
    "controls": ["b0", "raw_uncorrected", "mean_bias"],
}
_shared.CONTRACT_ARM_NAMES = {
    "treatment": "self_map_subtraction", "baseline": "b_star", "b0": "b0",
    "raw_uncorrected": "raw_uncorrected",
    "mean_bias": "mean_bias",
}
_shared.PRIMARY_METRIC = "cd_motion"
_shared.GUARDRAIL_METRICS = ("cd_3d", "cd_4d")


_bind_standard_arms = _shared.bind_contract_arms

def _bind_c01_arms(contract: dict, comparison: dict, root: Path | None = None):
    # Native official-generation B0 predates the candidate report schema. Its
    # exact implementation is bound by the receipt-retained generation identity.
    refs = _bind_standard_arms(contract, comparison, root=None)
    if root is not None:
        by_role = {row["role"]: row for row in comparison["roles"]}
        for row in comparison["roles"]:
            identity_row = by_role[row["alias_of"]] if row.get("alias_of") else row
            if identity_row["role"] == "b0":
                identity = scoring.read_json(scoring.resolve_ref(
                    root, identity_row["generation_identity_ref"]))
                actual = identity.get("instrument_code_sha256", {}).get(
                    "research_math/native_context_runner.py")
            else:
                report = scoring.read_json(scoring.resolve_ref(
                    root, identity_row["report_ref"]))
                actual = report.get("implementation_sha256")
            if actual != identity_row["implementation_sha256"]:
                raise ValueError("Frozen C01 arm/report implementation mismatch")
    return refs

_shared.bind_contract_arms = _bind_c01_arms


def __getattr__(name):
    return getattr(_shared, name)


def main() -> int:
    return _shared.main()


if __name__ == "__main__":
    raise SystemExit(main())
