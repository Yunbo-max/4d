"""C04 profile for the authorization-gated official-scoring plan builder."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from research_math import c04_native_scoring as scoring


_path = Path(__file__).with_name("prepare_c14_native_scoring.py")
_spec = importlib.util.spec_from_file_location("_c04_shared_plan_builder", _path)
if _spec is None or _spec.loader is None:
    raise ImportError("Unable to load shared native scoring plan builder")
_shared = importlib.util.module_from_spec(_spec)
_shared.scoring = scoring
_spec.loader.exec_module(_shared)

_shared.scoring = scoring
_shared.CANDIDATE_ID = "4d-math-20261006-c04"
_shared.ROLE_FOR_CONTRACT_ARM = {
    "treatment": "robust_conic_protection", "baseline": "b_star", "b0": "b0",
    "deterministic_protection": "deterministic_protection",
    "strength_matched_repair": "strength_matched_repair",
}
_shared.PROFILE = "c04"
_shared.TASK_ID = "c04-five-role-official-scoring"
_shared.AUTHORIZATION_SCOPE = "single_c04_scoring_attempt"
_shared.INPUT_NAMESPACE = "c04"
_shared.OUTPUT_DIRECTORY = "c04-scoring-output"
_shared.SCORING_MODULE = "research_math.c04_native_scoring"
_shared.COMPARISON_SOURCE = "actionmesh/research_math/c04_native_comparison.py"
_shared.SCORING_SOURCE = "actionmesh/research_math/c04_native_scoring.py"
_shared.LAUNCHER_SOURCE = "actionmesh/launch_c04_native_scoring.py"
_shared.EXTRA_CODE_SOURCES = (
    "actionmesh/prepare_c04_native_scoring.py",
    "actionmesh/prepare_c14_native_scoring.py",
    "actionmesh/launch_c14_native_scoring.py",
    "actionmesh/research_math/c14_native_scoring.py",
    "actionmesh/research_math/robust_motion_candidate.py",
    "actionmesh/research_math/protected_geometry_candidate.py",
    "actionmesh/research_ten/__init__.py",
    "actionmesh/research_ten/m01_elasticity.py",
    "actionmesh/prepare_robust_motion_candidate.py",
)
_shared.LAUNCH_TICKET_KIND = "c04-staged-launch-ticket"
_shared.AUTHORIZATION_KIND = "c04-gpu-resume-authorization"
_shared.RESERVATION_KIND = "c04-gpu-resume-authorization-reservation"
_shared.CONSUMPTION_KIND = "c04-gpu-resume-authorization-consumption"
_shared.FAMILY_SPLIT_KIND = "c04-family-split"
_shared.FAMILY_ASSIGNMENTS_KIND = "c04-family-assignments"
_shared.FAMILY_DERIVATION_KIND = "c04-family-derivation"
_shared.FAMILY_REVIEW_KIND = "c04-family-derivation-review"
_shared.CRITERIA_KIND = "c04-outcome-criteria"
_shared.ANALYSIS_KIND = "c04-g01-analysis-plan"
_shared.ADMISSION_KIND = "c04-scientific-dispatch-admission"
_shared.CONTRACT_CONTRASTS = {
    "treatment": "robust_conic_protection", "baseline": "b_star",
    "controls": ["b0", "deterministic_protection", "strength_matched_repair"],
}
_shared.CONTRACT_ARM_NAMES = dict(_shared.ROLE_FOR_CONTRACT_ARM)
_shared.PRIMARY_METRIC = "cd_3d"
_shared.GUARDRAIL_METRICS = ("cd_4d", "cd_motion")






# C04 has an additional scientific obligation: the chosen inference-available
# motion surrogate and uncertainty set need independent held-out semantic
# evidence.  The source constructor deliberately does not certify this claim.
_shared_admission = _shared.require_dispatch_admission
_shared_final_consumption = _shared.validate_final_consumption


def require_c04_semantic_review(root: Path, comparison: dict, *,
                                family_split_ref: dict,
                                authorized_at: str,
                                protocol_frozen_at: str) -> list[dict]:
    ref = comparison.get("semantic_review_ref")
    if not isinstance(ref, dict):
        raise ValueError("C04 requires a retained held-out semantic review")
    review = scoring.read_json(scoring.resolve_ref(root, ref))
    fields = {
        "kind", "version", "candidate_id", "status", "reviewer", "reviewed_at",
        "method_spec_ref", "family_split_ref",
        "surrogate", "uncertainty_set", "probability_coverage_claimed",
        "fit_ids", "held_out_ids", "evidence_refs", "review_digest",
    }
    if (set(review) != fields
            or review.get("kind") != "c04-motion-uncertainty-semantic-review"
            or review.get("version") != 1
            or review.get("candidate_id") != _shared.CANDIDATE_ID
            or review.get("status") != "accepted"
            or not isinstance(review.get("reviewer"), str)
            or not review["reviewer"].strip()
            or review.get("family_split_ref") != family_split_ref
            or review.get("probability_coverage_claimed") is not False
            or review.get("review_digest") != _shared.canonical_record_digest(
                review, "review_digest")):
        raise ValueError("Exact C04 semantic-review scope/source binding required")
    artifact = scoring.read_json(scoring.resolve_ref(
        root, comparison["candidate_artifact_ref"]))
    # The reviewed method spec predates confirmation outputs. Bind algorithm and
    # parameters, never require a review of a not-yet-generated confirmation mesh.
    specification = scoring.read_json(scoring.resolve_ref(root, review["method_spec_ref"]))
    spec_keys = {"kind", "version", "candidate_id", "implementation_ref",
                 "surrogate", "uncertainty_set", "parameters", "spec_digest"}
    candidate_role = next(row for row in comparison["roles"]
                          if row["role"] == "robust_conic_protection")
    if (set(specification) != spec_keys
            or specification.get("kind") != "c04-motion-surrogate-specification"
            or specification.get("version") != 1
            or specification.get("candidate_id") != _shared.CANDIDATE_ID
            or specification.get("implementation_ref") != candidate_role["implementation_ref"]
            or specification.get("parameters") != artifact.get("target_parameters")
            or specification.get("surrogate") != "normalized_predicted_kinetic_energy_v1"
            or specification.get("uncertainty_set") !=
            "positive_diagonal_acceleration_weighted_gradient_ellipsoid_v1"
            or review.get("surrogate") != specification["surrogate"]
            or review.get("uncertainty_set") != specification["uncertainty_set"]
            or specification.get("spec_digest") != _shared.canonical_record_digest(
                specification, "spec_digest")):
        raise ValueError("C04 reviewed algorithm/parameters differ from constructed method")
    family = scoring.read_json(scoring.resolve_ref(root, family_split_ref))
    fit, held_out = review.get("fit_ids"), review.get("held_out_ids")
    if (not isinstance(fit, list) or not isinstance(held_out, list)
            or not fit or not held_out
            or any(not isinstance(uid, str) or not uid for uid in fit + held_out)
            or len(set(fit)) != len(fit) or len(set(held_out)) != len(held_out)
            or set(fit) & set(held_out)
            or not set(fit + held_out).issubset(set(family.get("development_ids", [])))):
        raise ValueError("C04 semantic evidence needs disjoint held-out development IDs")
    family_by_uid = family.get("family_by_uid", {})
    if (any(uid not in family_by_uid for uid in fit + held_out)
            or {family_by_uid[uid] for uid in fit}
            & {family_by_uid[uid] for uid in held_out}):
        raise ValueError("C04 held-out semantic families overlap fit families")
    reviewed = _shared._timezone(review["reviewed_at"], "C04 semantic review")
    if (reviewed > _shared._timezone(authorized_at, "authorization")
            or reviewed > _shared._timezone(protocol_frozen_at, "protocol freeze")):
        raise ValueError("C04 semantic review must precede protocol and authorization")
    evidence_refs = review.get("evidence_refs")
    if not isinstance(evidence_refs, list) or not evidence_refs:
        raise ValueError("C04 review cannot self-certify without retained evidence")
    closure = [ref, family_split_ref, review["method_spec_ref"],
               specification["implementation_ref"]]
    observed = set()
    for evidence_ref in evidence_refs:
        record = scoring.read_json(scoring.resolve_ref(root, evidence_ref))
        required = {"kind", "version", "candidate_id", "uid", "asset_family",
                    "source_refs", "receipt_ref", "surrogate", "uncertainty_set",
                    "findings", "recorded_at", "evidence_digest", "native_plan_ref",
                    "harness_plan_ref", "harness_report_ref", "candidate_artifact_ref"}
        uid = record.get("uid")
        if (set(record) != required
                or record.get("kind") != "c04-held-out-semantic-evidence"
                or record.get("version") != 1
                or record.get("candidate_id") != _shared.CANDIDATE_ID
                or uid not in held_out or uid in observed
                or record.get("asset_family") != family_by_uid[uid]
                or record.get("surrogate") != review["surrogate"]
                or record.get("uncertainty_set") != review["uncertainty_set"]
                or record.get("evidence_digest") != _shared.canonical_record_digest(
                    record, "evidence_digest")
                or not isinstance(record.get("findings"), dict)
                or not record["findings"]
                or not isinstance(record.get("source_refs"), list)
                or not record["source_refs"]):
            raise ValueError("C04 semantic review lacks per-family source evidence")
        if _shared._timezone(record["recorded_at"], "semantic evidence") > reviewed:
            raise ValueError("C04 semantic evidence postdates independent review")
        # Reconcile an actual retained harness/native receipt pair, rather than
        # accepting a user-authored status/output list as execution evidence.
        receipt = scoring.read_json(scoring.resolve_ref(root, record["receipt_ref"]))
        native = scoring.read_json(scoring.resolve_ref(root, record["native_plan_ref"]))
        harness = scoring.read_json(scoring.resolve_ref(root, record["harness_plan_ref"]))
        harness_report = scoring.read_json(scoring.resolve_ref(root, record["harness_report_ref"]))
        jobs, attempts, tasks = native.get("jobs", []), receipt.get("attempts", []), harness.get("tasks", [])
        if (len(jobs) != 1 or len(attempts) != 1 or len(tasks) != 1
                or native.get("plan_digest") != _shared.canonical_record_digest(native, "plan_digest")
                or harness.get("plan_digest") != _shared.canonical_record_digest(harness, "plan_digest")
                or native.get("evidence_mode") != "developmental"
                or receipt.get("status") != "completed"
                or receipt.get("plan_digest") != native["plan_digest"]
                or receipt.get("provenance") != native.get("provenance")
                or tasks[0].get("plan_ref") != record["native_plan_ref"]
                or harness_report.get("status") != "completed"
                or harness_report.get("plan_digest") != harness["plan_digest"]):
            raise ValueError("C04 semantic evidence requires a reconciled harness/native receipt")
        job, attempt = jobs[0], attempts[0]
        if (attempt.get("status") != "completed" or attempt.get("exit_code") != 0
                or any(attempt.get(key) != job.get(key) for key in
                       ("trial_id", "input_refs", "code_refs", "seed", "group", "arm_role"))
                or specification["implementation_ref"] not in job.get("code_refs", [])):
            raise ValueError("C04 semantic receipt differs from exact method execution")
        command = job.get("command", [])
        if (not isinstance(command, list)
                or command[1:3] != ["-m", "research_math.robust_motion_candidate"]
                or command.count("--uid") != 1
                or command.index("--uid") + 1 >= len(command)
                or command[command.index("--uid") + 1] != uid):
            raise ValueError("C04 semantic plan command has a different native UID")
        for name, value in specification["parameters"].items():
            flag = "--" + name.replace("_", "-")
            try:
                matches = (command.count(flag) == 1
                           and float(command[command.index(flag) + 1]) == value)
            except (ValueError, IndexError, TypeError):
                matches = False
            if not matches:
                raise ValueError("C04 semantic plan differs from reviewed parameters: " + name)
        outputs = attempt.get("output_refs")
        if not isinstance(outputs, list) or record["candidate_artifact_ref"] not in outputs:
            raise ValueError("C04 held-out candidate must be an actual receipt output")
        held_artifact = scoring.read_json(scoring.resolve_ref(root, record["candidate_artifact_ref"]))
        if (held_artifact.get("candidate_id") != _shared.CANDIDATE_ID
                or held_artifact.get("uid") != uid
                or held_artifact.get("target_parameters") != specification["parameters"]
                or held_artifact.get("implementation_sha256") != specification["implementation_ref"]["sha256"]):
            raise ValueError("C04 held-out artifact differs from reviewed UID/method/parameters")
        artifact_time = _shared._timezone(held_artifact.get("recorded_at"), "held-out artifact")
        attempt_start = _shared._timezone(attempt.get("started_at"), "held-out attempt start")
        attempt_end = _shared._timezone(attempt.get("completed_at"), "held-out attempt completion")
        evidence_time = _shared._timezone(record["recorded_at"], "held-out evidence")
        if not attempt_start <= artifact_time <= attempt_end <= evidence_time <= reviewed:
            raise ValueError("C04 retained execution times contradict prospective review")
        for source_ref in record["source_refs"]:
            if source_ref not in outputs:
                raise ValueError("C04 semantic source is not receipt-bound")
            scoring.resolve_ref(root, source_ref)
        # A semantic conclusion remains a reviewed interpretation, not an
        # automatic gate derived from the constructor's conic certificate.
        closure.extend([evidence_ref, record["receipt_ref"], record["native_plan_ref"],
                        record["harness_plan_ref"], record["harness_report_ref"],
                        record["candidate_artifact_ref"], *record["source_refs"]])
        observed.add(uid)
    if observed != set(held_out):
        raise ValueError("C04 semantic review must cover every declared held-out ID")
    pinned = {item["path"]: item for item in comparison.get("input_refs", [])}
    for evidence_ref in closure:
        scoring.resolve_ref(root, evidence_ref)
        if pinned.get(evidence_ref["path"]) != evidence_ref:
            raise ValueError("C04 semantic evidence missing from frozen comparison closure")
    return list(_shared._unique_refs(closure, "C04 semantic evidence").values())


def require_dispatch_admission(root: Path, admission_path: Path, **kwargs):
    result = _shared_admission(root, admission_path, **kwargs)
    admission = scoring.read_json(admission_path)
    comparison = scoring.read_json(scoring.resolve_ref(
        root, kwargs["request"]["comparison_ref"]))
    refs = require_c04_semantic_review(
        root, comparison, family_split_ref=admission["family_split_ref"],
        authorized_at=result[3]["authorized_at"],
        protocol_frozen_at=kwargs["protocol"]["frozen_at"])
    return (result[0], list(_shared._unique_refs(
        result[1] + refs, "C04 admission").values()), result[2], result[3])


def require_c04_plan_semantics(root: Path, native_plan: dict) -> None:
    admission_ref = native_plan.get("provenance", {}).get("scientific_admission_ref")
    admission = scoring.read_json(scoring.resolve_ref(root, admission_ref))
    protocol = scoring.read_json(scoring.resolve_ref(root, admission["protocol_ref"]))
    authorization = scoring.read_json(scoring.resolve_ref(
        root, admission["gpu_resume_authorization_ref"]))
    request = scoring.read_json(scoring.resolve_ref(root, authorization["request_ref"]))
    comparison = scoring.read_json(scoring.resolve_ref(root, request["comparison_ref"]))
    refs = require_c04_semantic_review(
        root, comparison, family_split_ref=admission["family_split_ref"],
        authorized_at=authorization["authorized_at"],
        protocol_frozen_at=protocol["frozen_at"])
    declared = native_plan["jobs"][0]["input_refs"]
    if any(ref not in declared for ref in refs):
        raise ValueError("C04 installed plan omits frozen semantic evidence")


def validate_final_consumption(*args, **kwargs):
    result = _shared_final_consumption(*args, **kwargs)
    root = args[0] if args else kwargs["root"]
    require_c04_plan_semantics(root, result[1])
    return result


_shared.require_dispatch_admission = require_dispatch_admission
_shared.validate_final_consumption = validate_final_consumption


def __getattr__(name):
    return getattr(_shared, name)


def main() -> int:
    return _shared.main()


if __name__ == "__main__":
    raise SystemExit(main())
