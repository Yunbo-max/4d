"""Freeze C10's five logical native roles without scoring or choosing B*.

The assembler consumes a previously retained C10 artifact, revalidates the
common differential target and every produced arm through the candidate core,
and emits a content-addressed request for the separately admitted official
scorer.  It neither generates geometry nor advances any scientific gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from research_math import integrable_gradient_candidate as candidate_module


CANDIDATE_ID = "4d-math-20261006-c10"
ROLES = ("b0", "b_star", "direct_common_lift",
         "independent_local_repair", "pinned_integrable_solve")
METHOD_ROLES = ROLES[2:]
CONTROL_ROLES = ("direct_common_lift", "independent_local_repair")
METHOD_IDS = {
    "direct_common_lift": "c10-control-direct-common-lift-v1",
    "independent_local_repair": "c10-control-qualified-independent-local-arap-v1",
    "pinned_integrable_solve": CANDIDATE_ID,
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> dict:
    def no_duplicates(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = item
        return result
    value = json.loads(Path(path).read_text(), object_pairs_hook=no_duplicates)
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object: " + str(path))
    return value


def canonical_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def file_ref(root: Path, path: Path) -> dict:
    root, path = Path(root).resolve(), Path(path).resolve()
    return {"path": path.relative_to(root).as_posix(), "sha256": digest(path)}


def resolve_ref(root: Path, ref: dict) -> Path:
    if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
            or not isinstance(ref.get("path"), str)
            or not isinstance(ref.get("sha256"), str)
            or len(ref["sha256"]) != 64
            or any(character not in "0123456789abcdef"
                   for character in ref["sha256"])):
        raise ValueError("Exact path/sha256 reference required")
    relative = Path(ref["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Project-relative nonescaping reference required")
    root = Path(root).resolve()
    candidate = root / relative
    if candidate.is_symlink() or any(
            parent.is_symlink() for parent in candidate.parents
            if parent != root.parent):
        raise ValueError("Symlinked evidence is not accepted")
    path = candidate.resolve()
    path.relative_to(root)
    if not path.is_file() or digest(path) != ref["sha256"]:
        raise ValueError("Evidence reference missing or stale: " + ref["path"])
    return path


def _arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def _sequence_content_digest(arrays: dict) -> str:
    """Identify equal physical sequences independent of NPZ container bytes."""
    value = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        header = json.dumps([name, array.dtype.str, list(array.shape)],
                            separators=(",", ":")).encode()
        value.update(len(header).to_bytes(8, "little"))
        value.update(header)
        value.update(array.tobytes())
    return value.hexdigest()


def _same_native(source: dict, arm: dict) -> None:
    candidate_module.validate_native_arrays(source)
    candidate_module.validate_native_arrays(arm)
    if set(source) != set(arm):
        raise ValueError("Arm changed native sequence array inventory")
    for name in source:
        if name == "vertices":
            if (arm[name].shape != source[name].shape
                    or arm[name].dtype != source[name].dtype):
                raise ValueError("Arm changed vertex shape/dtype")
        elif not np.array_equal(arm[name], source[name]):
            raise ValueError("Arm changed native identity array: " + name)
    if not np.array_equal(arm["vertices"][0], source["vertices"][0]):
        raise ValueError("Arm changed exact native anchor frame")


def _freeze_core(freeze: dict) -> None:
    core = {key: value for key, value in freeze.items() if key != "freeze_digest"}
    if freeze.get("freeze_digest") != canonical_digest(core):
        raise ValueError("Freeze digest mismatch")
    if (freeze.get("kind") != "c10-native-comparison-freeze"
            or freeze.get("version") != 1
            or freeze.get("candidate_id") != CANDIDATE_ID
            or freeze.get("scoring_seed") != 44
            or freeze.get("primary_metric") != "cd_3d"
            or freeze.get("guardrail_metrics") != ["cd_4d", "cd_motion"]):
        raise ValueError("Current frozen C10 comparison record required")
    try:
        frozen = datetime.fromisoformat(freeze["frozen_at"].replace("Z", "+00:00"))
    except (KeyError, AttributeError, ValueError) as error:
        raise ValueError("Timezone-aware frozen_at required") from error
    if frozen.tzinfo is None:
        raise ValueError("Timezone-aware frozen_at required")
    uid = freeze.get("uid")
    if (not isinstance(uid, str) or not uid or Path(uid).name != uid
            or uid in (".", "..")):
        raise ValueError("Frozen native UID required")
    seed = freeze.get("inference_seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Frozen integer inference seed required")
    rows = freeze.get("roles")
    if (not isinstance(rows, list)
            or [row.get("role") if isinstance(row, dict) else None
                for row in rows] != list(ROLES)):
        raise ValueError("Exactly five ordered C10 roles must be frozen")


def _decision(root: Path, freeze: dict) -> tuple[dict, list[dict]]:
    decision = read_json(resolve_ref(root, freeze["b_star_decision_ref"]))
    fields = {
        "kind", "version", "candidate_id", "uid", "inference_seed",
        "decided_at", "selected_role", "selected_method_id",
        "selected_without_c10_native_outcomes", "selection_basis_refs",
        "decision_digest",
    }
    core = {key: value for key, value in decision.items()
            if key != "decision_digest"}
    if (set(decision) != fields
            or decision.get("kind") != "c10-b-star-decision"
            or decision.get("version") != 1
            or decision.get("candidate_id") != CANDIDATE_ID
            or decision.get("uid") != freeze["uid"]
            or decision.get("inference_seed") != freeze["inference_seed"]
            or decision.get("selected_without_c10_native_outcomes") is not True
            or decision.get("decision_digest") != canonical_digest(core)):
        raise ValueError("B* must be prospectively selected without C10 outcomes")
    try:
        decided = datetime.fromisoformat(decision["decided_at"].replace("Z", "+00:00"))
        frozen = datetime.fromisoformat(freeze["frozen_at"].replace("Z", "+00:00"))
    except (KeyError, AttributeError, ValueError) as error:
        raise ValueError("Timezone-aware B* decision required") from error
    if decided.tzinfo is None or decided > frozen:
        raise ValueError("B* decision must precede the comparison freeze")
    refs = decision.get("selection_basis_refs")
    if not isinstance(refs, list) or not refs:
        raise ValueError("B* requires prospective selection basis")
    for ref in refs:
        resolve_ref(root, ref)
    return decision, refs


def _artifact_closure(root: Path, artifact_path: Path,
                      freeze: dict) -> tuple[dict, list[dict], dict]:
    verified = candidate_module.validate_candidate_artifact(root, artifact_path)
    if (verified.get("candidate_id") != CANDIDATE_ID
            or verified.get("uid") != freeze["uid"]
            or verified.get("seed") != freeze["inference_seed"]):
        raise ValueError("Verified C10 artifact identity differs from freeze")
    artifact = read_json(artifact_path)
    if (artifact.get("candidate_id") != CANDIDATE_ID
            or artifact.get("uid") != freeze["uid"]
            or artifact.get("seed") != freeze["inference_seed"]
            or artifact.get("source_sequence_sha256")
            != freeze["source_sequence_ref"]["sha256"]
            or artifact.get("source_report_sha256")
            != freeze["source_report_ref"]["sha256"]
            or artifact.get("source_refs") != {
                "sequence": freeze["source_sequence_ref"],
                "report": freeze["source_report_ref"]}):
        raise ValueError("C10 artifact is not derived from the frozen B0 source")
    directory = artifact_path.parent
    terminal_reports = [read_json(directory / role / "report.json")
                        for role in METHOD_ROLES]
    expected_status = ("completed" if all(
        report.get("status") == "completed" for report in terminal_reports)
        else "incomplete")
    if (artifact.get("arms") != terminal_reports
            or artifact.get("status") != expected_status):
        raise ValueError("C10 candidate record differs from terminal role reports")
    manifest = read_json(directory / "manifest.json")
    completed_roles = [row["role"] for row in verified.get("arms", [])
                       if row.get("status") == "completed"]
    expected_cases = [{"case_id": freeze["uid"] + "-" + role,
                       "uid": freeze["uid"], "case_dir": role,
                       "arm_role": role} for role in completed_roles]
    if (manifest.get("expected_roles") != list(METHOD_ROLES)
            or manifest.get("cases") != expected_cases
            or manifest.get("common_target") != artifact.get("common_target")
            or manifest.get("scope") !=
            "C10 generated_unexecuted three-arm artifact; no B0/B*/scoring/admission"):
        raise ValueError("C10 artifact manifest differs from validated terminal arms")
    expected_paths = [artifact_path, directory / "manifest.json",
                      directory / "common-target.npz"]
    for role in METHOD_ROLES:
        report_path = directory / role / "report.json"
        expected_paths.append(report_path)
        report = read_json(report_path)
        for name in ("certificate.npz", "sequence.npz"):
            path = directory / role / name
            if report.get("status") == "completed" or path.exists():
                expected_paths.append(path)
    refs = []
    for path in expected_paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError("Incomplete physical C10 artifact closure: " + str(path))
        refs.append(file_ref(root, path))
    common_ref = file_ref(root, directory / "common-target.npz")
    if (verified.get("common_target") != common_ref
            or artifact.get("common_target", {}).get("sha256")
            != common_ref["sha256"]):
        raise ValueError("Verified C10 common target differs from artifact")
    return artifact, refs, common_ref


def make_request(root: Path, *, freeze_path: Path, _verify: bool = True) -> dict:
    root, freeze_path = Path(root).resolve(), Path(freeze_path).resolve()
    freeze_path.relative_to(root)
    freeze = read_json(freeze_path)
    _freeze_core(freeze)
    source_path = resolve_ref(root, freeze["source_sequence_ref"])
    source_report_path = resolve_ref(root, freeze["source_report_ref"])
    source_report = read_json(source_report_path)
    if (source_path.name != "sequence.npz"
            or source_report_path != source_path.with_name("report.json")
            or source_report.get("status") != "completed"
            or source_report.get("uid") != freeze["uid"]
            or source_report.get("seed") != freeze["inference_seed"]
            or source_report.get("sha256", {}).get("sequence.npz")
            != freeze["source_sequence_ref"]["sha256"]):
        raise ValueError("Exact completed B0 source pair required")
    source = _arrays(source_path)
    candidate_module.validate_native_arrays(source)
    decision, basis = _decision(root, freeze)
    artifact_path = resolve_ref(root, freeze["candidate_artifact_ref"])
    artifact, artifact_refs, common_target_ref = _artifact_closure(
        root, artifact_path, freeze)
    pinned = [file_ref(root, freeze_path), freeze["source_sequence_ref"],
              freeze["source_report_ref"], freeze["b_star_decision_ref"],
              freeze["candidate_artifact_ref"], *basis, *artifact_refs]
    normalized, physical, physical_content = [], {}, {}
    for row in freeze["roles"]:
        role = row["role"]
        if not isinstance(row.get("method_id"), str) or not row["method_id"]:
            raise ValueError("Every role needs a nonempty method_id")
        if "alias_of" in row:
            if (role != "b_star"
                    or row["alias_of"] not in ("b0", *CONTROL_ROLES)
                    or set(row) != {"role", "method_id", "alias_of"}
                    or decision["selected_role"] != row["alias_of"]
                    or decision["selected_method_id"] != row["method_id"]):
                raise ValueError("B* alias must match a prospective simple role")
            normalized.append(dict(row))
            continue
        required = {"role", "method_id", "report_ref", "sequence_ref",
                    "implementation_ref"}
        if role == "b0":
            required.add("generation_identity_ref")
        if role in METHOD_ROLES:
            required.add("certificate_ref")
        if set(row) != required:
            raise ValueError("Physical role requires exact refs: " + role)
        report_path = resolve_ref(root, row["report_ref"])
        implementation = resolve_ref(root, row["implementation_ref"])
        report = read_json(report_path)
        if (report_path.name != "report.json"
                or report.get("status") not in ("completed", "error")
                or report.get("uid") != freeze["uid"]
                or report.get("seed") != freeze["inference_seed"]):
            raise ValueError("Terminal role report required: " + role)
        completed = report["status"] == "completed"
        sequence_path = None
        if completed:
            sequence_path = resolve_ref(root, row["sequence_ref"])
            if (sequence_path != report_path.with_name("sequence.npz")
                    or report.get("sha256", {}).get("sequence.npz")
                    != row["sequence_ref"]["sha256"]):
                raise ValueError("Completed role lacks its retained sequence: " + role)
        else:
            error_type, error = report.get("exception_type"), report.get("error")
            if (row["sequence_ref"] is not None
                    or not isinstance(error_type, str) or not error_type.strip()
                    or len(error_type) > 256
                    or not isinstance(error, str) or not error.strip()
                    or len(error) > 4096):
                raise ValueError("Failed role requires bounded terminal evidence: " + role)
        if role == "b0":
            identity_path = resolve_ref(root, row["generation_identity_ref"])
            identity = read_json(identity_path)
            if (not completed
                    or row["report_ref"] != freeze["source_report_ref"]
                    or row["sequence_ref"] != freeze["source_sequence_ref"]
                    or identity_path != source_path.parent.parent / "generation-identity.json"
                    or identity.get("kind") != "native-context-generation-identity"
                    or identity.get("version") != 1
                    or identity.get("scope") !=
                    "paired engineering observer/replay; no candidate or scorer execution"
                    or identity.get("uid") != freeze["uid"]
                    or freeze["inference_seed"] != 42
                    or identity.get("generation", {}).get("seed") != 42
                    or not isinstance(identity.get("verified_unit_manifest"), dict)
                    or not isinstance(identity.get("retained_input_refs"), dict)
                    or not identity["retained_input_refs"]
                    or not isinstance(identity.get("upstream_source_sha256"), dict)
                    or not identity["upstream_source_sha256"]
                    or not isinstance(identity.get("gpu_uuid"), str)
                    or not identity["gpu_uuid"].startswith("GPU-")
                    or identity.get("native_context_qualified") is not False
                    or identity.get("scientific_effect_qualification") is not False
                    or identity.get("instrument_code_sha256", {}).get(
                        "research_math/native_context_runner.py") != digest(implementation)):
                raise ValueError("B0 must retain exact native generation identity")
        elif role == "b_star":
            if (decision.get("selected_role") != "b_star"
                    or decision.get("selected_method_id") != row["method_id"]
                    or report.get("source_sequence_sha256")
                    != freeze["source_sequence_ref"]["sha256"]
                    or report.get("source_report_sha256")
                    != freeze["source_report_ref"]["sha256"]
                    or report.get("method_id") != row["method_id"]
                    or report.get("implementation_sha256") != digest(implementation)):
                raise ValueError("Physical B* identity differs from prospective decision")
        else:
            expected_report = artifact_path.parent / role / "report.json"
            expected_certificate = artifact_path.parent / role / "certificate.npz"
            certificate = (resolve_ref(root, row["certificate_ref"])
                           if row["certificate_ref"] is not None else None)
            if (report_path != expected_report
                    or report.get("candidate_arm") != role
                    or report.get("source_sequence_sha256")
                    != freeze["source_sequence_ref"]["sha256"]
                    or report.get("source_report_sha256")
                    != freeze["source_report_ref"]["sha256"]
                    or report.get("implementation_sha256") != digest(implementation)):
                raise ValueError("C10 artifact/source identity mismatch: " + role)
            if completed:
                if (certificate != expected_certificate
                        or report.get("sha256", {}).get("certificate.npz")
                        != row["certificate_ref"]["sha256"]):
                    raise ValueError("Completed C10 arm lacks its certificate: " + role)
            elif certificate is not None:
                raise ValueError("Failed C10 arm must not invent a certificate: " + role)
            if row["method_id"] != METHOD_IDS[role]:
                raise ValueError("Frozen role relabels retained C10 method: " + role)
            if role == "pinned_integrable_solve":
                if row["method_id"] != CANDIDATE_ID:
                    raise ValueError("Pinned solve must be the actual C10 candidate")
            elif row["method_id"] == CANDIDATE_ID:
                raise ValueError("A C10 control cannot masquerade as the candidate")
        arrays = None
        if completed:
            arrays = _arrays(sequence_path)
            _same_native(source, arrays)
        current = {
            "role": role, "method_id": row["method_id"],
            "report_ref": row["report_ref"], "sequence_ref": row["sequence_ref"],
            "implementation_ref": row["implementation_ref"],
            "implementation_sha256": row["implementation_ref"]["sha256"],
            "case_id": freeze["uid"] + "-" + role if completed else None,
            "preparation_status": report["status"],
        }
        if not completed:
            current.update(exception_type=report["exception_type"], error=report["error"],
                           preparation_error=(report["exception_type"] + ": "
                                              + report["error"]))
        if role == "b0":
            current["generation_identity_ref"] = row["generation_identity_ref"]
            pinned.append(row["generation_identity_ref"])
        if role in METHOD_ROLES:
            current["certificate_ref"] = row["certificate_ref"]
            if row["certificate_ref"] is not None:
                pinned.append(row["certificate_ref"])
        normalized.append(current)
        pinned.extend([row["report_ref"], row["implementation_ref"]])
        if completed:
            physical[role] = current
            physical_content[role] = _sequence_content_digest(arrays)
            pinned.append(row["sequence_ref"])
    canonical_by_content, scoring_cases = {}, []
    for role in ROLES:
        item = physical.get(role)
        if item is None:
            continue
        content_sha = physical_content[role]
        canonical = canonical_by_content.get(content_sha)
        if canonical is None:
            canonical_by_content[content_sha] = item
            scoring_cases.append(item)
        else:
            item["case_id"] = canonical["case_id"]
            next(row for row in normalized if row["role"] == role)["case_id"] = canonical["case_id"]
    for row in normalized:
        if "alias_of" in row:
            target = next(item for item in normalized if item["role"] == row["alias_of"])
            if row["method_id"] != target["method_id"]:
                raise ValueError("Aliased B* method_id must match target")
            row.update(case_id=target["case_id"],
                       preparation_status=target["preparation_status"])
            if target["preparation_status"] == "error":
                row.update(exception_type=target["exception_type"], error=target["error"],
                           preparation_error=target["preparation_error"])
    role_to_case = {row["role"]: row["case_id"] for row in normalized}
    unique = {ref["path"]: ref for ref in pinned}
    request = {
        "kind": "c10-native-comparison-request", "version": 1,
        "candidate_id": CANDIDATE_ID, "uid": freeze["uid"],
        "inference_seed": freeze["inference_seed"], "scoring_seed": 44,
        "primary_metric": "cd_3d", "guardrail_metrics": ["cd_4d", "cd_motion"],
        "roles": normalized, "role_to_case": role_to_case,
        "logical_denominator": {
            "n_roles": 5,
            "roles": [{"role": row["role"],
                       "preparation_status": row["preparation_status"],
                       "case_id": row["case_id"]} for row in normalized],
            "failure_policy": (
                "Every frozen role remains; missing/error metrics are never zero-imputed."),
        },
        "scoring_cases": scoring_cases,
        "freeze_ref": file_ref(root, freeze_path),
        "b_star_decision_ref": freeze["b_star_decision_ref"],
        "candidate_artifact_ref": freeze["candidate_artifact_ref"],
        "source_sequence_ref": freeze["source_sequence_ref"],
        "source_report_ref": freeze["source_report_ref"],
        "common_target_ref": common_target_ref,
        "input_refs": list(unique.values()),
        "generated_unexecuted": True, "native_qualified": False,
        "scientific_verdict": "not_computed", "dispatch_ready": False,
    }
    request["request_digest"] = canonical_digest(request)
    if _verify:
        verify_request(root, request)
    return request


def verify_request(root: Path, request: dict) -> None:
    core = {key: value for key, value in request.items()
            if key != "request_digest"}
    if request.get("request_digest") != canonical_digest(core):
        raise ValueError("Request digest mismatch")
    if (request.get("kind") != "c10-native-comparison-request"
            or request.get("candidate_id") != CANDIDATE_ID
            or request.get("scoring_seed") != 44
            or request.get("generated_unexecuted") is not True
            or request.get("native_qualified") is not False
            or request.get("dispatch_ready") is not False):
        raise ValueError("Unexecuted C10 request scope required")
    if [row.get("role") for row in request.get("roles", [])] != list(ROLES):
        raise ValueError("Exactly five ordered C10 roles required")
    if set(request.get("role_to_case", {})) != set(ROLES):
        raise ValueError("Every logical role must map to a case or retained failure")
    cases = request.get("scoring_cases", [])
    if len({row.get("case_id") for row in cases}) != len(cases):
        raise ValueError("Physical scoring case IDs must be unique")
    refs = request.get("input_refs")
    if not isinstance(refs, list):
        raise ValueError("Pinned request closure required")
    pinned = {ref["path"]: ref for ref in refs}
    if len(pinned) != len(refs):
        raise ValueError("Input closure contains duplicate paths")
    for ref in refs:
        resolve_ref(root, ref)
    for key in ("freeze_ref", "b_star_decision_ref", "candidate_artifact_ref",
                "source_sequence_ref", "source_report_ref", "common_target_ref"):
        if pinned.get(request[key]["path"]) != request[key]:
            raise ValueError("Request closure missing: " + key)
    expected = make_request(
        root, freeze_path=resolve_ref(root, request["freeze_ref"]), _verify=False)
    if request != expected:
        raise ValueError("Request differs from current frozen C10 construction")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("request",))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = make_request(args.root, freeze_path=args.freeze)
    output = args.output.resolve()
    output.relative_to(args.root.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(request, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"request": str(output),
                      "request_digest": request["request_digest"],
                      "execution_started": False,
                      "native_qualified": False, "dispatch_ready": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
