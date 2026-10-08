"""C08 profile for the authorization-gated official-scoring plan builder."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from research_math import c08_native_scoring as scoring


_path = Path(__file__).with_name("prepare_c14_native_scoring.py")
_spec = importlib.util.spec_from_file_location("_c08_shared_plan_builder", _path)
if _spec is None or _spec.loader is None:
    raise ImportError("Unable to load shared native scoring plan builder")
_shared = importlib.util.module_from_spec(_spec)
_shared.scoring = scoring
_spec.loader.exec_module(_shared)

_shared.scoring = scoring
_shared.CANDIDATE_ID = "4d-math-20261006-c08"
_shared.ROLE_FOR_CONTRACT_ARM = {
    "treatment": "endpoint_bridge", "baseline": "b_star", "b0": "b0",
    "local_transition_tracker": "local_transition_tracker",
    "coordinate_smoother": "coordinate_smoother",
    "geometry_cycle_control": "geometry_cycle_control"}
_shared.PROFILE = "c08"
_shared.TASK_ID = "c08-six-role-official-scoring"
_shared.AUTHORIZATION_SCOPE = "single_c08_scoring_attempt"
_shared.INPUT_NAMESPACE = "c08"
_shared.OUTPUT_DIRECTORY = "c08-scoring-output"
_shared.SCORING_MODULE = "research_math.c08_native_scoring"
_shared.COMPARISON_SOURCE = "actionmesh/research_math/c08_native_comparison.py"
_shared.SCORING_SOURCE = "actionmesh/research_math/c08_native_scoring.py"
_shared.LAUNCHER_SOURCE = "actionmesh/launch_c08_native_scoring.py"
_shared.EXTRA_CODE_SOURCES = (
    "actionmesh/prepare_c08_native_scoring.py",
    "actionmesh/prepare_c14_native_scoring.py",
    "actionmesh/launch_c14_native_scoring.py",
    "actionmesh/research_math/c14_native_scoring.py",
    "actionmesh/research_math/c11_native_comparison.py",
    "actionmesh/research_math/trajectory_bridge_candidate.py",
    "actionmesh/prepare_trajectory_bridge_candidate.py")
_shared.LAUNCH_TICKET_KIND = "c08-staged-launch-ticket"
_shared.AUTHORIZATION_KIND = "c08-gpu-resume-authorization"
_shared.RESERVATION_KIND = "c08-gpu-resume-authorization-reservation"
_shared.CONSUMPTION_KIND = "c08-gpu-resume-authorization-consumption"
_shared.FAMILY_SPLIT_KIND = "c08-family-split"
_shared.FAMILY_ASSIGNMENTS_KIND = "c08-family-assignments"
_shared.FAMILY_DERIVATION_KIND = "c08-family-derivation"
_shared.FAMILY_REVIEW_KIND = "c08-family-derivation-review"
_shared.CRITERIA_KIND = "c08-outcome-criteria"
_shared.ANALYSIS_KIND = "c08-g01-analysis-plan"
_shared.ADMISSION_KIND = "c08-scientific-dispatch-admission"
_shared.CONTRACT_CONTRASTS = {
    "treatment": "endpoint_bridge", "baseline": "b_star",
    "controls": ["b0", "local_transition_tracker", "coordinate_smoother",
                 "geometry_cycle_control"]}
_shared.CONTRACT_ARM_NAMES = dict(_shared.ROLE_FOR_CONTRACT_ARM)
_shared.PRIMARY_METRIC = "cd_motion"
_shared.GUARDRAIL_METRICS = ("cd_3d", "cd_4d")


def bind_contract_arms(contract: dict, comparison: dict,
                       root: Path | None = None) -> list[dict]:
    """Bind all six frozen C08 roles to exact implementations."""
    rows = comparison.get("roles")
    by_role = {row.get("role"): row for row in rows or [] if isinstance(row, dict)}
    if set(by_role) != set(_shared.ROLE_FOR_CONTRACT_ARM.values()):
        raise ValueError("Frozen comparison lacks exact six logical C08 roles")
    arms = contract["arm_requirements"]
    refs = []
    for arm_role, comparison_role in _shared.ROLE_FOR_CONTRACT_ARM.items():
        arm, row = arms[arm_role], by_role[comparison_role]
        identity = by_role[row["alias_of"]] if row.get("alias_of") else row
        implementation_refs = arm.get("implementation_refs")
        if (arm.get("name") != comparison_role
                or arm.get("revision") != row.get("method_id")
                or not isinstance(implementation_refs, list)
                or not implementation_refs):
            raise ValueError("Contract arm differs from frozen method identity: " + arm_role)
        if any(not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
               for ref in implementation_refs):
            raise ValueError("Exact arm implementation reference required")
        refs.extend(implementation_refs)
        implementation_sha = identity.get("implementation_sha256")
        if not isinstance(implementation_sha, str) or implementation_sha not in {
                ref["sha256"] for ref in implementation_refs}:
            raise ValueError("Contract implementation differs from frozen arm: " + arm_role)
        if root is not None and identity.get("report_ref") is not None:
            report = scoring.read_json(scoring.resolve_ref(root, identity["report_ref"]))
            reported = report.get("implementation_sha256")
            if identity.get("role") == "b0":
                source_ref = identity.get("implementation_ref")
                generator_ref = identity.get("generation_identity_ref")
                pinned = comparison.get("input_refs", [])
                if (source_ref not in implementation_refs or source_ref not in pinned
                        or source_ref.get("sha256") != implementation_sha
                        or generator_ref not in pinned
                        or (reported is not None and reported != implementation_sha)):
                    raise ValueError("C08 B0 implementation is not pinned")
                generation = scoring.read_json(scoring.resolve_ref(root, generator_ref))
                if (generation.get("kind") != "native-context-generation-identity"
                        or generation.get("uid") != comparison.get("uid")
                        or generation.get("instrument_code_sha256", {}).get(
                            "research_math/native_context_runner.py") != implementation_sha):
                    raise ValueError("C08 B0 native producer identity mismatch")
            elif reported != implementation_sha:
                raise ValueError("Frozen C08 arm/report implementation mismatch: " + arm_role)
    alias = by_role["b_star"].get("alias_of")
    if alias is not None:
        target = next(key for key, role in _shared.ROLE_FOR_CONTRACT_ARM.items()
                      if role == alias)
        if arms["baseline"]["implementation_refs"] != arms[target]["implementation_refs"]:
            raise ValueError("Aliased B* must bind the target implementation")
    return list(_shared._unique_refs(refs, "arm implementation").values())


def require_c08_contract(contract: dict, *, benchmark_revision: str) -> None:
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
        raise ValueError("Exact C08 six-role native contract required")
    if not isinstance(contract.get("scorer"), dict) or contract["scorer"].get("kind") != "official":
        raise ValueError("C08 requires the official ActionBench scorer")


_shared.bind_contract_arms = bind_contract_arms
_shared.require_c14_contract = require_c08_contract


def __getattr__(name):
    return getattr(_shared, name)


def main() -> int:
    return _shared.main()


if __name__ == "__main__":
    raise SystemExit(main())
