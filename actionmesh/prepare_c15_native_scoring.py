"""C15 profile for the authorization-gated official-scoring plan builder."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from research_math import c15_native_scoring as scoring


_path = Path(__file__).with_name("prepare_c14_native_scoring.py")
_spec = importlib.util.spec_from_file_location("_c15_shared_plan_builder", _path)
if _spec is None or _spec.loader is None:
    raise ImportError("Unable to load shared native scoring plan builder")
_shared = importlib.util.module_from_spec(_spec)
_shared.scoring = scoring
_spec.loader.exec_module(_shared)

_shared.scoring = scoring
_shared.CANDIDATE_ID = "4d-math-20261006-c15"
_shared.ROLE_FOR_CONTRACT_ARM = {
    "treatment": "protected_residual_svt", "baseline": "b_star", "b0": "b0",
    "gaussian": "gaussian", "unprotected_svt": "unprotected_svt",
    "rank_matched_tsvd": "rank_matched_tsvd",
}
_shared.PROFILE = "c15"
_shared.TASK_ID = "c15-six-role-official-scoring"
_shared.AUTHORIZATION_SCOPE = "single_c15_scoring_attempt"
_shared.INPUT_NAMESPACE = "c15"
_shared.OUTPUT_DIRECTORY = "c15-scoring-output"
_shared.SCORING_MODULE = "research_math.c15_native_scoring"
_shared.COMPARISON_SOURCE = "actionmesh/research_math/c15_native_comparison.py"
_shared.SCORING_SOURCE = "actionmesh/research_math/c15_native_scoring.py"
_shared.LAUNCHER_SOURCE = "actionmesh/launch_c15_native_scoring.py"
_shared.EXTRA_CODE_SOURCES = (
    "actionmesh/prepare_c15_native_scoring.py",
    "actionmesh/prepare_c14_native_scoring.py",
    "actionmesh/launch_c14_native_scoring.py",
    "actionmesh/research_math/c14_native_scoring.py",
    "actionmesh/research_math/protected_lowrank_candidate.py",
    "actionmesh/prepare_protected_lowrank_candidate.py",
)
_shared.LAUNCH_TICKET_KIND = "c15-staged-launch-ticket"
_shared.AUTHORIZATION_KIND = "c15-gpu-resume-authorization"
_shared.RESERVATION_KIND = "c15-gpu-resume-authorization-reservation"
_shared.CONSUMPTION_KIND = "c15-gpu-resume-authorization-consumption"
_shared.FAMILY_SPLIT_KIND = "c15-family-split"
_shared.FAMILY_ASSIGNMENTS_KIND = "c15-family-assignments"
_shared.FAMILY_DERIVATION_KIND = "c15-family-derivation"
_shared.FAMILY_REVIEW_KIND = "c15-family-derivation-review"
_shared.CRITERIA_KIND = "c15-outcome-criteria"
_shared.ANALYSIS_KIND = "c15-g01-analysis-plan"
_shared.ADMISSION_KIND = "c15-scientific-dispatch-admission"
_shared.CONTRACT_CONTRASTS = {
    "treatment": "protected_residual_svt", "baseline": "b_star",
    "controls": ["b0", "gaussian", "unprotected_svt", "rank_matched_tsvd"],
}
_shared.CONTRACT_ARM_NAMES = dict(_shared.ROLE_FOR_CONTRACT_ARM)
_shared.PRIMARY_METRIC = "cd_motion"
_shared.GUARDRAIL_METRICS = ("cd_3d", "cd_4d")


def bind_contract_arms(contract: dict, comparison: dict,
                       root: Path | None = None) -> list[dict]:
    """Bind six contract roles to exact frozen implementations, including B0."""
    rows = comparison.get("roles")
    by_role = {row.get("role"): row for row in rows or []
               if isinstance(row, dict)}
    if set(by_role) != set(_shared.ROLE_FOR_CONTRACT_ARM.values()):
        raise ValueError("Frozen comparison lacks exact six logical C15 roles")
    arms = contract["arm_requirements"]
    refs = []
    for arm_role, comparison_role in _shared.ROLE_FOR_CONTRACT_ARM.items():
        arm = arms[arm_role]
        row = by_role[comparison_role]
        identity = by_role[row["alias_of"]] if row.get("alias_of") else row
        implementation_refs = arm.get("implementation_refs")
        if (arm.get("name") != comparison_role
                or arm.get("revision") != row.get("method_id")
                or not isinstance(implementation_refs, list)
                or not implementation_refs):
            raise ValueError("Contract arm differs from frozen method identity: "
                             + arm_role)
        for ref in implementation_refs:
            if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
                raise ValueError("Exact arm implementation reference required")
            refs.append(ref)
        implementation_sha = identity.get("implementation_sha256")
        if (not isinstance(implementation_sha, str)
                or implementation_sha not in {
                    ref["sha256"] for ref in implementation_refs}):
            raise ValueError("Contract implementation differs from frozen arm: "
                             + arm_role)
        if root is not None and identity.get("report_ref") is not None:
            report = scoring.read_json(scoring.resolve_ref(
                root, identity["report_ref"]))
            reported = report.get("implementation_sha256")
            if identity.get("role") == "b0":
                source_ref = identity.get("implementation_ref")
                generator_ref = identity.get("generation_identity_ref")
                pinned = comparison.get("input_refs", [])
                if (source_ref not in implementation_refs
                        or source_ref not in pinned
                        or source_ref.get("sha256") != implementation_sha
                        or generator_ref not in pinned
                        or (reported is not None and reported != implementation_sha)):
                    raise ValueError(
                        "C15 B0 implementation is not pinned by source/generation closure")
                scoring.resolve_ref(root, source_ref)
                generation = scoring.read_json(scoring.resolve_ref(
                    root, generator_ref))
                if (generation.get("kind") != "native-context-generation-identity"
                        or generation.get("uid") != comparison.get("uid")
                        or generation.get("instrument_code_sha256", {}).get(
                            "research_math/native_context_runner.py")
                        != implementation_sha):
                    raise ValueError("C15 B0 native producer identity mismatch")
            elif reported != implementation_sha:
                raise ValueError("Frozen C15 arm/report implementation mismatch: "
                                 + arm_role)
    alias = by_role["b_star"].get("alias_of")
    if alias is not None:
        target_arm = next(key for key, role in
                          _shared.ROLE_FOR_CONTRACT_ARM.items() if role == alias)
        if (arms["baseline"]["implementation_refs"]
                != arms[target_arm]["implementation_refs"]):
            raise ValueError("Aliased B* must bind the target implementation")
    return list(_shared._unique_refs(refs, "arm implementation").values())


def require_c15_contract(contract: dict, *, benchmark_revision: str) -> None:
    arms = contract.get("arm_requirements")
    if (contract.get("benchmark_id") != "facebook/actionbench"
            or contract.get("benchmark_revision") != benchmark_revision
            or contract.get("primary_metric") != _shared.PRIMARY_METRIC
            or [row.get("name") for row in contract.get("metrics", [])]
            != list(scoring.METRICS)
            or contract.get("contrasts") != _shared.CONTRACT_CONTRASTS
            or not isinstance(arms, dict)
            or set(arms) != set(_shared.CONTRACT_ARM_NAMES)
            or any(arms[role].get("name") != name
                   for role, name in _shared.CONTRACT_ARM_NAMES.items())):
        raise ValueError("Exact C15 six-role native contract required")
    scorer = contract.get("scorer")
    if not isinstance(scorer, dict) or scorer.get("kind") != "official":
        raise ValueError("C15 requires the official ActionBench scorer")


_shared.bind_contract_arms = bind_contract_arms
_shared.require_c14_contract = require_c15_contract


def __getattr__(name):
    return getattr(_shared, name)


def main() -> int:
    return _shared.main()


if __name__ == "__main__":
    raise SystemExit(main())
