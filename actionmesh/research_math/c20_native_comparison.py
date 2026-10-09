"""C20 six-role profile for prospective comparison freezing."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from research_math import phase_amplitude_artifacts as candidate_module


_path = Path(__file__).with_name("c11_native_comparison.py")
_spec = importlib.util.spec_from_file_location(
    "research_math._c20_shared_native_comparison", _path)
if _spec is None or _spec.loader is None:
    raise ImportError("Unable to load shared comparison implementation")
_shared = importlib.util.module_from_spec(_spec)
_shared.candidate_module = candidate_module
_spec.loader.exec_module(_shared)

_shared.candidate_module = candidate_module
_shared.CANDIDATE_ID = "4d-math-20261006-c20"
_shared.ROLES = ("b0", "b_star", "phase_only", "amplitude_only",
                 "simple_lag", "joint_monotone_phase_amplitude")
_shared.METHOD_ROLES = _shared.ROLES[2:]
_shared.CONTROL_ROLES = _shared.ROLES[2:5]
_shared.METHOD_IDS = candidate_module.METHOD_IDS
_shared.CANDIDATE_ROLE = "joint_monotone_phase_amplitude"
_shared.FREEZE_KIND = "c20-native-comparison-freeze"
_shared.DECISION_KIND = "c20-b-star-decision"
_shared.DECISION_NO_OUTCOMES_FIELD = "selected_without_c20_native_outcomes"
_shared.REQUEST_KIND = "c20-native-comparison-request"
_shared.PROFILE_LABEL = "C20"
_shared.ALLOWED_INFERENCE_SEEDS = (42, 314, 2718)
_shared.PRIMARY_METRIC = "cd_motion"
_shared.GUARDRAIL_METRICS = ("cd_3d", "cd_4d")

_standard_make_request = _shared.make_request
_standard_verify_request = _shared.verify_request


def _validate_split(split: dict) -> None:
    required = {
        "kind", "version", "candidate_id", "benchmark_revision",
        "d1_ids", "d2_ids", "confirmation_ids", "family_by_uid",
        "family_assignments_ref", "frozen_at", "split_digest",
    }
    groups = [split.get(name) for name in ("d1_ids", "d2_ids", "confirmation_ids")]
    if (set(split) != required or split.get("kind") != "c20-family-split"
            or split.get("version") != 1 or split.get("candidate_id") != CANDIDATE_ID
            or split.get("split_digest") != _shared.canonical_digest({
                key: value for key, value in split.items() if key != "split_digest"})
            or any(not isinstance(group, list) or not group for group in groups)
            or any(len(group) != len(set(group)) for group in groups)
            or set(groups[0]) & set(groups[1])
            or set(groups[0]) & set(groups[2])
            or set(groups[1]) & set(groups[2])
            or not isinstance(split.get("family_by_uid"), dict)
            or set(split["family_by_uid"]) != set().union(*map(set, groups))):
        raise ValueError("Exact digest-bound C20 G01 family split required")


def _candidate_receipt_refs(root: Path, freeze: dict,
                            artifact_path: Path) -> tuple[list[dict], list[dict]]:
    refs = [freeze.get("candidate_plan_ref"), freeze.get("candidate_receipt_ref")]
    if any(not isinstance(ref, dict) for ref in refs):
        raise ValueError("C20 freeze requires candidate plan/receipt references")
    plan_path, receipt_path = (_shared.resolve_ref(root, ref) for ref in refs)
    candidate_module.validate_candidate_execution(
        root, plan_path=plan_path, receipt_path=receipt_path,
        artifact_path=artifact_path)
    plan, receipt = map(_shared.read_json, (plan_path, receipt_path))
    run_plan = root / plan["output_root"] / plan["run_id"] / "plan.json"
    closure = [*refs, _shared.file_ref(root, run_plan)]
    job, attempt = plan["jobs"][0], receipt["attempts"][0]
    attempt_root = root / attempt["attempt_path"]
    workspace = attempt_root / "workspace"
    staged_refs = [_shared.file_ref(root, workspace / ref["path"])
                   for ref in job["input_refs"] + job["code_refs"]]
    closure += [*job["input_refs"], *job["code_refs"],
                _shared.file_ref(root, attempt_root / "attempt.json"),
                *staged_refs, *attempt["output_refs"]]
    unique = {}
    for ref in closure:
        if ref["path"] in unique and unique[ref["path"]] != ref:
            raise ValueError("Conflicting C20 candidate execution closure")
        unique[ref["path"]] = ref
    return refs, list(unique.values())


def make_request(root: Path, *, freeze_path: Path, _verify: bool = True) -> dict:
    root, freeze_path = Path(root).resolve(), Path(freeze_path).resolve()
    freeze = _shared.read_json(freeze_path)
    artifact_path = _shared.resolve_ref(root, freeze["candidate_artifact_ref"])
    receipt_refs, receipt_closure = _candidate_receipt_refs(root, freeze, artifact_path)
    split_ref = freeze.get("g01_family_split_ref")
    split = _shared.read_json(_shared.resolve_ref(root, split_ref))
    _validate_split(split)
    artifact = _shared.read_json(artifact_path)
    stage = artifact.get("application_stage")
    stage_ids = {"d1": split.get("d1_ids"), "d2": split.get("d2_ids"),
                 "confirmation": split.get("confirmation_ids")}
    if (split.get("candidate_id") != CANDIDATE_ID
            or freeze.get("g01_family_split_digest") != split.get("split_digest")
            or stage not in stage_ids or freeze.get("uid") not in (stage_ids[stage] or [])):
        raise ValueError("C20 artifact is not bound to the frozen G01 family split/stage")
    request = _standard_make_request(root, freeze_path=freeze_path, _verify=False)
    request["candidate_plan_ref"], request["candidate_receipt_ref"] = receipt_refs
    request["g01_family_split_ref"] = split_ref
    request["g01_family_split_digest"] = split["split_digest"]
    unique = {ref["path"]: ref for ref in [*request["input_refs"],
        *receipt_closure, split_ref]}
    request["input_refs"] = list(unique.values())
    request["request_digest"] = _shared.canonical_digest({
        key: value for key, value in request.items() if key != "request_digest"})
    if _verify:
        verify_request(root, request)
    return request


def verify_request(root: Path, request: dict) -> None:
    _standard_verify_request(root, request)
    pinned = {ref["path"]: ref for ref in request["input_refs"]}
    for key in ("candidate_plan_ref", "candidate_receipt_ref", "g01_family_split_ref"):
        if pinned.get(request[key]["path"]) != request[key]:
            raise ValueError("C20 comparison closure missing " + key)
    split = _shared.read_json(_shared.resolve_ref(root, request["g01_family_split_ref"]))
    _validate_split(split)
    if request.get("g01_family_split_digest") != split.get("split_digest"):
        raise ValueError("C20 G01 split digest mismatch")
    artifact_path = _shared.resolve_ref(root, request["candidate_artifact_ref"])
    _, closure = _candidate_receipt_refs(root, {
        "candidate_plan_ref": request["candidate_plan_ref"],
        "candidate_receipt_ref": request["candidate_receipt_ref"]}, artifact_path)
    if any(pinned.get(ref["path"]) != ref for ref in closure):
        raise ValueError("C20 comparison request omits candidate execution closure")


_shared.make_request = make_request
_shared.verify_request = verify_request

CANDIDATE_ID = _shared.CANDIDATE_ID
ROLES = _shared.ROLES
METHOD_ROLES = _shared.METHOD_ROLES
CONTROL_ROLES = _shared.CONTROL_ROLES
METHOD_IDS = _shared.METHOD_IDS
CANDIDATE_ROLE = _shared.CANDIDATE_ROLE
FREEZE_KIND = _shared.FREEZE_KIND
DECISION_KIND = _shared.DECISION_KIND
DECISION_NO_OUTCOMES_FIELD = _shared.DECISION_NO_OUTCOMES_FIELD
REQUEST_KIND = _shared.REQUEST_KIND
ALLOWED_INFERENCE_SEEDS = _shared.ALLOWED_INFERENCE_SEEDS
PRIMARY_METRIC = _shared.PRIMARY_METRIC
GUARDRAIL_METRICS = _shared.GUARDRAIL_METRICS


def __getattr__(name):
    return getattr(_shared, name)


def main() -> int:
    return _shared.main()


if __name__ == "__main__":
    raise SystemExit(main())
