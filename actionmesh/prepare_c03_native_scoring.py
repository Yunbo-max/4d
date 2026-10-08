"""C03 profile for the authorization-gated official-scoring plan builder."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from research_math import c03_native_scoring as scoring


_path = Path(__file__).with_name("prepare_c14_native_scoring.py")
_spec = importlib.util.spec_from_file_location("_c03_shared_plan_builder", _path)
if _spec is None or _spec.loader is None:
    raise ImportError("Unable to load shared native scoring plan builder")
_shared = importlib.util.module_from_spec(_spec)
_shared.scoring = scoring
_spec.loader.exec_module(_shared)

_shared.scoring = scoring
_shared.CANDIDATE_ID = "4d-math-20261006-c03"
_shared.ROLE_FOR_CONTRACT_ARM = {
    "treatment": "full_smoothed_unsquared", "baseline": "b_star", "b0": "b0",
    "intercept_squared": "intercept_squared",
    "intercept_unsquared": "intercept_unsquared", "unit_C01": "unit_C01",
    "diagonal_squared": "diagonal_squared",
    "diagonal_unsquared": "diagonal_unsquared", "full_squared": "full_squared"}
_shared.PROFILE = "c03"
_shared.TASK_ID = "c03-nine-role-official-scoring"
_shared.AUTHORIZATION_SCOPE = "single_c03_scoring_attempt"
_shared.INPUT_NAMESPACE = "c03"
_shared.OUTPUT_DIRECTORY = "c03-scoring-output"
_shared.SCORING_MODULE = "research_math.c03_native_scoring"
_shared.COMPARISON_SOURCE = "actionmesh/research_math/c03_native_comparison.py"
_shared.SCORING_SOURCE = "actionmesh/research_math/c03_native_scoring.py"
_shared.LAUNCHER_SOURCE = "actionmesh/launch_c03_native_scoring.py"
_shared.EXTRA_CODE_SOURCES = (
    "actionmesh/prepare_c03_native_scoring.py",
    "actionmesh/prepare_c14_native_scoring.py",
    "actionmesh/launch_c14_native_scoring.py",
    "actionmesh/research_math/c14_native_scoring.py",
    "actionmesh/research_math/c11_native_comparison.py",
    "actionmesh/research_math/c03_label_bank.py",
    "actionmesh/prepare_c03_label_bank.py",
    "actionmesh/research_math/correlated_calibration.py",
    "actionmesh/research_math/c03_calibration_artifacts.py",
    "actionmesh/prepare_c03_calibration.py",
    "actionmesh/research_math/correlated_calibration_candidate.py",
    "actionmesh/prepare_correlated_calibration_candidate.py")
_shared.LAUNCH_TICKET_KIND = "c03-staged-launch-ticket"
_shared.AUTHORIZATION_KIND = "c03-gpu-resume-authorization"
_shared.RESERVATION_KIND = "c03-gpu-resume-authorization-reservation"
_shared.CONSUMPTION_KIND = "c03-gpu-resume-authorization-consumption"
_shared.FAMILY_SPLIT_KIND = "c03-family-split"
_shared.FAMILY_ASSIGNMENTS_KIND = "c03-family-assignments"
_shared.FAMILY_DERIVATION_KIND = "c03-family-derivation"
_shared.FAMILY_REVIEW_KIND = "c03-family-derivation-review"
_shared.CRITERIA_KIND = "c03-outcome-criteria"
_shared.ANALYSIS_KIND = "c03-g01-analysis-plan"
_shared.ADMISSION_KIND = "c03-scientific-dispatch-admission"
_shared.CONTRACT_CONTRASTS = {"treatment": "full_smoothed_unsquared",
    "baseline": "b_star", "controls": ["b0", "intercept_squared",
        "intercept_unsquared", "unit_C01", "diagonal_squared",
        "diagonal_unsquared", "full_squared"]}
_shared.CONTRACT_ARM_NAMES = dict(_shared.ROLE_FOR_CONTRACT_ARM)
_shared.PRIMARY_METRIC = "cd_motion"
_shared.GUARDRAIL_METRICS = ("cd_3d", "cd_4d")
_shared.ALLOW_G01_STAGES = True


def validate_family_binding(root: Path, request: dict, family: dict) -> None:
    comparison = scoring.read_json(scoring.resolve_ref(root, request["comparison_ref"]))
    ref = comparison.get("g01_family_split_ref")
    split = scoring.read_json(scoring.resolve_ref(root, ref))
    if (comparison.get("g01_family_split_digest") != split.get("split_digest")
            or comparison.get("g01_application_stage") != request.get("application_stage")
            or split.get("d1_ids") != family.get("d1_ids")
            or split.get("d2_ids") != family.get("d2_ids")
            or split.get("confirmation_ids") != family.get("confirmation_ids")
            or split.get("family_by_uid") != family.get("family_by_uid")):
        raise ValueError("C03 admission uses a different split from its label/candidate/B* chain")


_shared.validate_family_binding = validate_family_binding


def bind_contract_arms(contract: dict, comparison: dict,
                       root: Path | None = None) -> list[dict]:
    rows = comparison.get("roles")
    by_role = {row.get("role"): row for row in rows or [] if isinstance(row, dict)}
    if set(by_role) != set(_shared.ROLE_FOR_CONTRACT_ARM.values()):
        raise ValueError("Frozen comparison lacks exact nine logical C03 roles")
    arms = contract["arm_requirements"]
    refs = []
    for arm_role, comparison_role in _shared.ROLE_FOR_CONTRACT_ARM.items():
        arm, row = arms[arm_role], by_role[comparison_role]
        identity = by_role[row["alias_of"]] if row.get("alias_of") else row
        implementation_refs = arm.get("implementation_refs")
        if (arm.get("name") != comparison_role
                or arm.get("revision") != row.get("method_id")
                or not isinstance(implementation_refs, list) or not implementation_refs
                or any(not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
                       for ref in implementation_refs)):
            raise ValueError("Contract arm differs from frozen C03 identity: " + arm_role)
        refs.extend(implementation_refs)
        implementation_sha = identity.get("implementation_sha256")
        if implementation_sha not in {ref["sha256"] for ref in implementation_refs}:
            raise ValueError("Contract implementation differs from frozen C03 arm")
        if root is not None and identity.get("report_ref") is not None:
            report = scoring.read_json(scoring.resolve_ref(root, identity["report_ref"]))
            reported = report.get("implementation_sha256")
            if identity.get("role") == "b0":
                source_ref = identity.get("implementation_ref")
                generation_ref = identity.get("generation_identity_ref")
                if root is None:
                    raise ValueError("C03 B0 contract binding requires project root")
                generation_path = scoring.resolve_ref(root, generation_ref)
                generation = scoring.read_json(generation_path)
                if (source_ref not in implementation_refs
                        or source_ref not in comparison.get("input_refs", [])
                        or generation_ref not in comparison.get("input_refs", [])
                        or generation_path !=
                           scoring.resolve_ref(root, identity["sequence_ref"]).parent.parent /
                           "generation-identity.json"
                        or generation.get("kind") != "native-context-generation-identity"
                        or generation.get("version") != 1
                        or generation.get("scope") !=
                           "paired engineering observer/replay; no candidate or scorer execution"
                        or generation.get("uid") != comparison.get("uid")
                        or generation.get("generation", {}).get("seed") !=
                           comparison.get("inference_seed")
                        or not isinstance(generation.get("verified_unit_manifest"), dict)
                        or not isinstance(generation.get("retained_input_refs"), dict)
                        or not generation["retained_input_refs"]
                        or not isinstance(generation.get("upstream_source_sha256"), dict)
                        or not generation["upstream_source_sha256"]
                        or not isinstance(generation.get("gpu_uuid"), str)
                        or not generation["gpu_uuid"].startswith("GPU-")
                        or generation.get("native_context_qualified") is not False
                        or generation.get("scientific_effect_qualification") is not False
                        or generation.get("instrument_code_sha256", {}).get(
                            "research_math/native_context_runner.py") != implementation_sha
                        or (reported is not None and reported != implementation_sha)):
                    raise ValueError("C03 B0 implementation is not pinned")
            elif reported != implementation_sha:
                raise ValueError("Frozen C03 arm/report implementation mismatch")
    alias = by_role["b_star"].get("alias_of")
    if alias is not None:
        target = next(key for key, role in _shared.ROLE_FOR_CONTRACT_ARM.items()
                      if role == alias)
        if arms["baseline"]["implementation_refs"] != arms[target]["implementation_refs"]:
            raise ValueError("Aliased C03 B* must bind target implementation")
    return list(_shared._unique_refs(refs, "arm implementation").values())


def require_c03_contract(contract: dict, *, benchmark_revision: str) -> None:
    arms = contract.get("arm_requirements")
    if (contract.get("benchmark_id") != "facebook/actionbench"
            or contract.get("benchmark_revision") != benchmark_revision
            or contract.get("primary_metric") != _shared.PRIMARY_METRIC
            or [row.get("name") for row in contract.get("metrics", [])] != list(scoring.METRICS)
            or contract.get("contrasts") != _shared.CONTRACT_CONTRASTS
            or not isinstance(arms, dict) or set(arms) != set(_shared.CONTRACT_ARM_NAMES)
            or any(arms[role].get("name") != name
                   for role, name in _shared.CONTRACT_ARM_NAMES.items())):
        raise ValueError("Exact C03 nine-role native contract required")
    if contract.get("scorer", {}).get("kind") != "official":
        raise ValueError("C03 requires official ActionBench scorer")


_shared.bind_contract_arms = bind_contract_arms
_shared.require_c14_contract = require_c03_contract


def __getattr__(name):
    return getattr(_shared, name)


def main() -> int:
    return _shared.main()


if __name__ == "__main__":
    raise SystemExit(main())
