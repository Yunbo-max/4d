"""C05 profile for the authorization-gated official-scoring plan builder."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from research_math import c05_native_scoring as scoring


_path = Path(__file__).with_name("prepare_c14_native_scoring.py")
_spec = importlib.util.spec_from_file_location("_c05_shared_plan_builder", _path)
if _spec is None or _spec.loader is None:
    raise ImportError("Unable to load shared native scoring plan builder")
_shared = importlib.util.module_from_spec(_spec)
_shared.scoring = scoring
_spec.loader.exec_module(_shared)

_shared.scoring = scoring
_shared.CANDIDATE_ID = "4d-math-20261006-c05"
_shared.ROLE_FOR_CONTRACT_ARM = {
    "treatment": "joint_spatial_labels", "baseline": "b_star", "b0": "b0",
    "localized_mean": "localized_mean",
    "temperature_matched_mean": "temperature_matched_mean",
    "surface_projected_mean": "surface_projected_mean",
    "independent_top1": "independent_top1",
}
_shared.PROFILE = "c05"
_shared.TASK_ID = "c05-seven-role-official-scoring"
_shared.AUTHORIZATION_SCOPE = "single_c05_scoring_attempt"
_shared.INPUT_NAMESPACE = "c05"
_shared.OUTPUT_DIRECTORY = "c05-scoring-output"
_shared.SCORING_MODULE = "research_math.c05_native_scoring"
_shared.COMPARISON_SOURCE = "actionmesh/research_math/c05_native_comparison.py"
_shared.SCORING_SOURCE = "actionmesh/research_math/c05_native_scoring.py"
_shared.LAUNCHER_SOURCE = "actionmesh/launch_c05_native_scoring.py"
_shared.EXTRA_CODE_SOURCES = (
    "actionmesh/prepare_c05_native_scoring.py",
    "actionmesh/prepare_c14_native_scoring.py",
    "actionmesh/launch_c14_native_scoring.py",
    "actionmesh/research_math/c14_native_scoring.py",
    "actionmesh/research_math/spatial_mode_candidate.py",
    "actionmesh/research_math/c05_candidate_artifacts.py",
    "actionmesh/research_math/c05_mode_bank.py",
    "actionmesh/prepare_c05_mode_bank.py",
    "actionmesh/prepare_spatial_mode_candidate.py",
)
_shared.LAUNCH_TICKET_KIND = "c05-staged-launch-ticket"
_shared.AUTHORIZATION_KIND = "c05-gpu-resume-authorization"
_shared.RESERVATION_KIND = "c05-gpu-resume-authorization-reservation"
_shared.CONSUMPTION_KIND = "c05-gpu-resume-authorization-consumption"
_shared.FAMILY_SPLIT_KIND = "c05-family-split"
_shared.FAMILY_ASSIGNMENTS_KIND = "c05-family-assignments"
_shared.FAMILY_DERIVATION_KIND = "c05-family-derivation"
_shared.FAMILY_REVIEW_KIND = "c05-family-derivation-review"
_shared.CRITERIA_KIND = "c05-outcome-criteria"
_shared.ANALYSIS_KIND = "c05-g01-analysis-plan"
_shared.ADMISSION_KIND = "c05-scientific-dispatch-admission"
_shared.CONTRACT_CONTRASTS = {
    "treatment": "joint_spatial_labels", "baseline": "b_star",
    "controls": ["b0", "localized_mean", "temperature_matched_mean",
                 "surface_projected_mean", "independent_top1"],
}
_shared.CONTRACT_ARM_NAMES = dict(_shared.ROLE_FOR_CONTRACT_ARM)
_shared.PRIMARY_METRIC = "cd_3d"
_shared.GUARDRAIL_METRICS = ("cd_4d", "cd_motion")


def bind_contract_arms(contract: dict, comparison: dict,
                       root: Path | None = None) -> list[dict]:
    """Bind all C05 roles without inventing C11-style report fields.

    C05 candidate-role reports retain ``implementation_ref`` plus the shared
    JSON construction/solver certificates.  The comparison request promotes
    that exact source reference to each logical row; it does not rewrite the
    immutable role report with a synthetic ``implementation_sha256`` field.
    """
    rows = comparison.get("roles")
    by_role = {row.get("role"): row for row in rows or [] if isinstance(row, dict)}
    if set(by_role) != set(_shared.ROLE_FOR_CONTRACT_ARM.values()):
        raise ValueError("Frozen comparison lacks exact seven logical C05 roles")
    arms = contract["arm_requirements"]
    refs = []
    for arm_role, comparison_role in _shared.ROLE_FOR_CONTRACT_ARM.items():
        arm, row = arms[arm_role], by_role[comparison_role]
        # Only the prospective B* logical alias borrows another role's method
        # identity.  A completed role may also have ``alias_of`` solely because
        # its sequence bytes equal an earlier physical case; that must not
        # replace the role's own implementation identity.
        identity = (by_role[row["alias_of"]]
                    if row.get("alias_of") and row.get("implementation_ref") is None
                    else row)
        implementation_refs = arm.get("implementation_refs")
        if (arm.get("name") != comparison_role
                or arm.get("revision") != row.get("method_id")
                or not isinstance(implementation_refs, list)
                or not implementation_refs
                or any(not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
                       for ref in implementation_refs)):
            raise ValueError("Contract arm differs from frozen C05 identity: " + arm_role)
        refs.extend(implementation_refs)
        source_ref = identity.get("implementation_ref")
        implementation_sha = identity.get("implementation_sha256")
        if (not isinstance(source_ref, dict)
                or set(source_ref) != {"path", "sha256"}
                or implementation_sha != source_ref["sha256"]
                or source_ref not in implementation_refs):
            raise ValueError("Contract implementation differs from frozen C05 arm")
        if root is not None:
            scoring.resolve_ref(root, source_ref)
            pinned = comparison.get("input_refs", [])
            if source_ref not in pinned:
                raise ValueError("C05 frozen implementation is not pinned")
        if root is not None and identity.get("report_ref") is not None:
            report = scoring.read_json(scoring.resolve_ref(root, identity["report_ref"]))
            if identity.get("role") == "b0":
                generation_ref = identity.get("generation_identity_ref")
                if generation_ref not in pinned:
                    raise ValueError("C05 B0 implementation is not pinned")
                generation = scoring.read_json(scoring.resolve_ref(root, generation_ref))
                if (generation.get("kind") != "native-context-generation-identity"
                        or generation.get("uid") != comparison.get("uid")
                        or generation.get("generation", {}).get("seed") !=
                           comparison.get("inference_seed")
                        or generation.get("instrument_code_sha256", {}).get(
                            "research_math/native_context_runner.py") != implementation_sha):
                    raise ValueError("C05 B0 producer identity differs")
            elif identity.get("role") in scoring.comparison_module.METHOD_ROLES:
                plain = scoring.comparison_module._plain_ref
                if (report.get("kind") != "c05-spatial-mode-role-report"
                        or report.get("candidate_arm") != identity["role"]
                        or report.get("method_id") != identity["method_id"]
                        or report.get("uid") != comparison.get("uid")
                        or report.get("outer_seed") != comparison.get("inference_seed")
                        or plain(report.get("implementation_ref", {})) != source_ref
                        or plain(report.get("certificate_ref", {})) !=
                           identity.get("shared_certificate_ref")
                        or plain(report.get("solver_certificate_ref", {})) !=
                           identity.get("shared_solver_certificate_ref")):
                    raise ValueError("Frozen C05 artifact role/report identity mismatch")
            elif report.get("implementation_sha256") != implementation_sha:
                raise ValueError("Frozen physical C05 B* implementation mismatch")
    alias = by_role["b_star"].get("alias_of")
    if alias is not None and by_role["b_star"].get("implementation_ref") is None:
        target = next(key for key, role in _shared.ROLE_FOR_CONTRACT_ARM.items()
                      if role == alias)
        if arms["baseline"]["implementation_refs"] != arms[target]["implementation_refs"]:
            raise ValueError("Aliased C05 B* must bind target implementation")
    return list(_shared._unique_refs(refs, "arm implementation").values())


def require_c05_contract(contract: dict, *, benchmark_revision: str) -> None:
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
        raise ValueError("Exact C05 seven-role native contract required")
    if contract.get("scorer", {}).get("kind") != "official":
        raise ValueError("C05 requires official ActionBench scorer")


_shared.bind_contract_arms = bind_contract_arms
_shared.require_c14_contract = require_c05_contract


def __getattr__(name):
    return getattr(_shared, name)


def main() -> int:
    return _shared.main()


if __name__ == "__main__":
    raise SystemExit(main())
