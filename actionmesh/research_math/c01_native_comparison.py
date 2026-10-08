"""Freeze C01's five logical roles without scoring or selecting B*.

The request assembler revalidates native identity and recomputes the complete
C01 same-context decoder certificate.  It never chooses a comparator from
outcomes, calls ActionBench, or advances a scientific gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from research_math import self_map_candidate as candidate_module


CANDIDATE_ID = candidate_module.CANDIDATE_ID
ROLES = ("b0", "b_star", "raw_uncorrected", "mean_bias", "self_map_subtraction")
METHOD_ROLES = ROLES[2:]


def digest(path: Path) -> str:
    return candidate_module.digest(path)


def read_json(path: Path) -> dict:
    def no_duplicates(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate JSON key: " + key)
            value[key] = item
        return value
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
            or not isinstance(ref.get("sha256"), str) or len(ref["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in ref["sha256"])):
        raise ValueError("Exact path/sha256 reference required")
    relative = Path(ref["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Project-relative nonescaping reference required")
    root = Path(root).resolve(); path = root / relative
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != root.parent):
        raise ValueError("Symlinked evidence is not accepted")
    if not path.is_file() or digest(path) != ref["sha256"]:
        raise ValueError("Evidence reference missing or stale: " + ref["path"])
    path.resolve().relative_to(root)
    return path.resolve()


def _arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def _sequence_content_digest(arrays: dict) -> str:
    """De-duplicate physical meshes independently of ZIP container metadata."""
    value = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        header = json.dumps([name, array.dtype.str, list(array.shape)],
                            separators=(",", ":")).encode()
        value.update(len(header).to_bytes(8, "little")); value.update(header)
        value.update(array.tobytes())
    return value.hexdigest()


def _same_native(source: dict, arm: dict) -> None:
    candidate_module.validate_native_arrays(source)
    candidate_module.validate_native_arrays(arm)
    if set(source) != set(arm):
        raise ValueError("Arm changed native sequence array inventory")
    for name in source:
        if name == "vertices":
            if arm[name].shape != source[name].shape or arm[name].dtype != source[name].dtype:
                raise ValueError("Arm changed vertex shape/dtype")
        elif not np.array_equal(arm[name], source[name]):
            raise ValueError("Arm changed native identity array: " + name)
    if not np.array_equal(arm["vertices"][0], source["vertices"][0]):
        raise ValueError("Arm changed exact native anchor frame")


def _freeze_core(freeze: dict) -> None:
    core = {key: value for key, value in freeze.items() if key != "freeze_digest"}
    if freeze.get("freeze_digest") != canonical_digest(core):
        raise ValueError("Freeze digest mismatch")
    if (freeze.get("kind") != "c01-native-comparison-freeze"
            or freeze.get("version") != 1 or freeze.get("candidate_id") != CANDIDATE_ID
            or freeze.get("scoring_seed") != 44
            or freeze.get("primary_metric") != "cd_motion"
            or freeze.get("guardrail_metrics") != ["cd_3d", "cd_4d"]):
        raise ValueError("Current frozen C01 comparison record required")
    try:
        frozen = datetime.fromisoformat(freeze["frozen_at"].replace("Z", "+00:00"))
    except (KeyError, AttributeError, ValueError) as error:
        raise ValueError("Timezone-aware frozen_at required") from error
    if frozen.tzinfo is None:
        raise ValueError("Timezone-aware frozen_at required")
    if (not isinstance(freeze.get("uid"), str) or not freeze["uid"]
            or Path(freeze["uid"]).name != freeze["uid"] or freeze["uid"] in (".", "..")):
        raise ValueError("Frozen native UID required")
    if isinstance(freeze.get("inference_seed"), bool) or not isinstance(freeze.get("inference_seed"), int):
        raise ValueError("Frozen integer inference seed required")
    rows = freeze.get("roles")
    if (not isinstance(rows, list)
            or [row.get("role") if isinstance(row, dict) else None for row in rows] != list(ROLES)):
        raise ValueError("Exactly five ordered C01 roles must be frozen")


def _decision(root: Path, freeze: dict) -> tuple[dict, list[dict]]:
    decision = read_json(resolve_ref(root, freeze["b_star_decision_ref"]))
    fields = {"kind", "version", "candidate_id", "uid", "inference_seed",
              "decided_at", "selected_role", "selected_method_id",
              "selected_without_c01_native_outcomes", "selection_basis_refs",
              "decision_digest"}
    core = {key: value for key, value in decision.items() if key != "decision_digest"}
    if (set(decision) != fields or decision.get("kind") != "c01-b-star-decision"
            or decision.get("version") != 1 or decision.get("candidate_id") != CANDIDATE_ID
            or decision.get("uid") != freeze["uid"]
            or decision.get("inference_seed") != freeze["inference_seed"]
            or decision.get("selected_without_c01_native_outcomes") is not True
            or decision.get("decision_digest") != canonical_digest(core)):
        raise ValueError("B* must be prospectively selected without C01 outcomes")
    try:
        decided = datetime.fromisoformat(decision["decided_at"].replace("Z", "+00:00"))
        frozen = datetime.fromisoformat(freeze["frozen_at"].replace("Z", "+00:00"))
    except (KeyError, AttributeError, ValueError) as error:
        raise ValueError("Timezone-aware B* decision required") from error
    if decided.tzinfo is None or decided > frozen:
        raise ValueError("B* decision must be timezone-aware and precede freeze")
    refs = decision.get("selection_basis_refs")
    if not isinstance(refs, list) or not refs:
        raise ValueError("B* requires prospective selection basis")
    for ref in refs:
        resolve_ref(root, ref)
    return decision, refs


def _recompute_candidate(root: Path, source_path: Path, source_report_path: Path,
                         reports: dict[str, dict], arms: dict[str, dict],
                         certificate: Path) -> list[dict]:
    """Rebuild the method from the retained native same-context capture.

    No caller-provided displacement or reference map qualifies as native input.
    The core reopens the consumed producer closure, validates every receipt-bound
    tensor and correspondence, derives raw F_t/F_a and checks the exported arms.
    """
    verified = candidate_module.verify_candidate_artifacts(certificate.parent)
    if (verified["source_sequence"].resolve() != source_path.resolve()
            or verified["source_report"].resolve() != source_report_path.resolve()
            or verified["certificate"].resolve() != certificate.resolve()):
        raise ValueError("C01 source must be the exact receipt-bound native capture")
    refs = [file_ref(root, path) for path in verified["context_files"]]
    refs.append(file_ref(root, certificate.parent / "candidate.json"))
    for role in METHOD_ROLES:
        actual_report = read_json(certificate.parent / role / "report.json")
        if reports[role] != actual_report:
            raise ValueError("C01 role report must belong to verified artifact tree")
        if role in arms:
            expected = _arrays(verified["arms"][role])
            if (set(expected) != set(arms[role])
                    or any(not np.array_equal(expected[key], arms[role][key])
                           for key in expected)):
                raise ValueError("C01 role differs from recomputed native construction")
    return refs


def make_request(root: Path, *, freeze_path: Path, _verify: bool = True) -> dict:
    root, freeze_path = Path(root).resolve(), Path(freeze_path).resolve()
    freeze_path.relative_to(root)
    freeze = read_json(freeze_path); _freeze_core(freeze)
    source_path = resolve_ref(root, freeze["source_sequence_ref"])
    source_report_path = resolve_ref(root, freeze["source_report_ref"])
    source_report = read_json(source_report_path)
    if (source_path.name != "sequence.npz" or source_report_path != source_path.with_name("report.json")
            or source_report.get("status") != "completed"
            or source_report.get("uid") != freeze["uid"]
            or source_report.get("seed") != freeze["inference_seed"]
            or source_report.get("sha256", {}).get("sequence.npz") != freeze["source_sequence_ref"]["sha256"]):
        raise ValueError("Exact completed B0 source pair required")
    source = _arrays(source_path); candidate_module.validate_native_arrays(source)
    decision, basis = _decision(root, freeze)
    pinned = [file_ref(root, freeze_path), freeze["source_sequence_ref"],
              freeze["source_report_ref"], freeze["b_star_decision_ref"], *basis]
    normalized, physical, reports, arms = [], {}, {}, {}
    physical_content = {}
    certificate_ref = None
    for row in freeze["roles"]:
        role = row["role"]
        if not isinstance(row.get("method_id"), str) or not row["method_id"]:
            raise ValueError("Every role needs a nonempty method_id")
        if "alias_of" in row:
            if (role != "b_star" or row["alias_of"] not in ("b0", "raw_uncorrected", "mean_bias")
                    or set(row) != {"role", "method_id", "alias_of"}
                    or decision["selected_role"] != row["alias_of"]
                    or decision["selected_method_id"] != row["method_id"]):
                raise ValueError("B* alias must match one frozen simple role and decision")
            normalized.append(dict(row)); continue
        required = {"role", "method_id", "report_ref", "sequence_ref", "implementation_ref"}
        if role == "b0":
            required.add("generation_identity_ref")
        if role in METHOD_ROLES:
            required.add("certificate_ref")
        if set(row) != required:
            raise ValueError("Physical role requires exact report/sequence/implementation refs: " + role)
        report_path = resolve_ref(root, row["report_ref"])
        implementation = resolve_ref(root, row["implementation_ref"])
        report = read_json(report_path)
        if (report_path.name != "report.json" or report.get("status") not in ("completed", "error")
                or report.get("uid") != freeze["uid"]
                or report.get("seed") != freeze["inference_seed"]):
            raise ValueError("Terminal role report for current frozen identity required: " + role)
        completed = report["status"] == "completed"
        sequence_path = None
        if completed:
            sequence_path = resolve_ref(root, row["sequence_ref"])
            if (sequence_path != report_path.with_name("sequence.npz")
                    or report.get("sha256", {}).get("sequence.npz")
                    != row["sequence_ref"]["sha256"]):
                raise ValueError("Completed role report and current sequence required: " + role)
        else:
            exception_type, error = report.get("exception_type"), report.get("error")
            if (row["sequence_ref"] is not None
                    or not isinstance(exception_type, str) or not exception_type.strip()
                    or len(exception_type) > 256
                    or not isinstance(error, str) or not error.strip() or len(error) > 4096):
                raise ValueError(
                    "Failed role requires null sequence and bounded error evidence: " + role)
        if role == "b0":
            identity_path = resolve_ref(root, row["generation_identity_ref"])
            identity = read_json(identity_path)
            if (not completed or row["report_ref"] != freeze["source_report_ref"]
                    or row["sequence_ref"] != freeze["source_sequence_ref"]
                    or identity_path != source_path.parent.parent / "generation-identity.json"
                    or identity.get("kind") != "native-context-generation-identity"
                    or identity.get("uid") != freeze["uid"]
                    or freeze["inference_seed"] != 42
                    or identity.get("instrument_code_sha256", {}).get(
                        "research_math/native_context_runner.py") != digest(implementation)):
                raise ValueError("B0 must be exact source refs")
        elif role == "b_star":
            if (decision.get("selected_role") != "b_star"
                    or decision.get("selected_method_id") != row["method_id"]
                    or report.get("source_sequence_sha256") != freeze["source_sequence_ref"]["sha256"]
                    or report.get("source_report_sha256") != freeze["source_report_ref"]["sha256"]
                    or report.get("method_id") != row["method_id"]
                    or report.get("implementation_sha256") != digest(implementation)):
                raise ValueError("Physical B* source/implementation identity mismatch")
        else:
            if (report.get("source_sequence_sha256") != freeze["source_sequence_ref"]["sha256"]
                    or report.get("source_report_sha256") != freeze["source_report_ref"]["sha256"]
                    or report.get("candidate_role") != role
                    or report.get("implementation_sha256") != digest(implementation)):
                raise ValueError("C01 role source/implementation identity mismatch: " + role)
            if row["method_id"] != report.get("method_id"):
                raise ValueError("C01 frozen role relabels the retained method: " + role)
            if role == "self_map_subtraction" and row["method_id"] != CANDIDATE_ID:
                raise ValueError("Self-map role must be the actual C01 method")
            if role != "self_map_subtraction" and row["method_id"] == CANDIDATE_ID:
                raise ValueError("Operation control cannot masquerade as C01")
        arm = None
        if completed:
            arm = _arrays(sequence_path); _same_native(source, arm)
        current = {
            "role": role, "method_id": row["method_id"], "report_ref": row["report_ref"],
            "sequence_ref": row["sequence_ref"], "implementation_ref": row["implementation_ref"],
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
        if role in METHOD_ROLES:
            certificate = resolve_ref(root, row["certificate_ref"])
            if (report.get("sha256", {}).get("certificate.npz") != row["certificate_ref"]["sha256"]
                    or certificate.parent != report_path.parent.parent):
                raise ValueError("Shared C01 certificate missing/stale")
            if certificate_ref is None:
                certificate_ref = row["certificate_ref"]
            elif certificate_ref != row["certificate_ref"]:
                raise ValueError("All C01 operation roles require one certificate")
            current["certificate_ref"] = row["certificate_ref"]
            reports[role] = report; pinned.append(row["certificate_ref"])
            if completed:
                arms[role] = arm
        normalized.append(current)
        if completed:
            physical[role] = current
            physical_content[role] = _sequence_content_digest(arm)
        pinned.extend([row["report_ref"], row["implementation_ref"]])
        if role == "b0":
            pinned.append(row["generation_identity_ref"])
        if completed:
            pinned.append(row["sequence_ref"])
    if set(reports) != set(METHOD_ROLES) or certificate_ref is None:
        raise ValueError("Complete three-arm C01 artifact set required")
    pinned.extend(_recompute_candidate(
        root, source_path, source_report_path, reports, arms,
        resolve_ref(root, certificate_ref)))
    canonical_by_sha, scoring_cases = {}, []
    for role in ROLES:
        item = physical.get(role)
        if item is None:
            continue
        sequence_sha = physical_content[role]
        canonical = canonical_by_sha.get(sequence_sha)
        if canonical is None:
            canonical_by_sha[sequence_sha] = item; scoring_cases.append(item)
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
        "kind": "c01-native-comparison-request", "version": 1,
        "candidate_id": CANDIDATE_ID, "uid": freeze["uid"],
        "inference_seed": freeze["inference_seed"], "scoring_seed": 44,
        "primary_metric": "cd_motion", "guardrail_metrics": ["cd_3d", "cd_4d"],
        "roles": normalized, "role_to_case": role_to_case,
        "logical_denominator": {
            "n_roles": 5,
            "roles": [{"role": row["role"], "preparation_status": row["preparation_status"],
                       "case_id": row["case_id"]} for row in normalized],
            "failure_policy": "Every frozen role remains; missing/error metrics are never zero-imputed.",
        },
        "scoring_cases": scoring_cases, "freeze_ref": file_ref(root, freeze_path),
        "b_star_decision_ref": freeze["b_star_decision_ref"],
        "source_sequence_ref": freeze["source_sequence_ref"],
        "source_report_ref": freeze["source_report_ref"],
        "input_refs": list(unique.values()),
        "generated_unexecuted": True, "native_qualified": False,
        "scientific_verdict": "not_computed", "dispatch_ready": False,
    }
    request["request_digest"] = canonical_digest(request)
    if _verify:
        verify_request(root, request)
    return request


def verify_request(root: Path, request: dict) -> None:
    core = {key: value for key, value in request.items() if key != "request_digest"}
    if request.get("request_digest") != canonical_digest(core):
        raise ValueError("Request digest mismatch")
    if (request.get("kind") != "c01-native-comparison-request"
            or request.get("candidate_id") != CANDIDATE_ID
            or request.get("scoring_seed") != 44
            or request.get("generated_unexecuted") is not True
            or request.get("native_qualified") is not False
            or request.get("dispatch_ready") is not False):
        raise ValueError("Unexecuted C01 request scope required")
    if [row.get("role") for row in request.get("roles", [])] != list(ROLES):
        raise ValueError("Exactly five ordered C01 roles required")
    if set(request.get("role_to_case", {})) != set(ROLES):
        raise ValueError("Every logical role must map to a case")
    cases = request.get("scoring_cases", [])
    if len({row.get("case_id") for row in cases}) != len(cases):
        raise ValueError("Physical scoring case IDs must be unique")
    refs = request.get("input_refs")
    if not isinstance(refs, list):
        raise ValueError("Pinned request closure required")
    pinned = {ref["path"]: ref for ref in refs}
    for ref in refs:
        resolve_ref(root, ref)
    for key in ("freeze_ref", "b_star_decision_ref", "source_sequence_ref", "source_report_ref"):
        if pinned.get(request[key]["path"]) != request[key]:
            raise ValueError("Request closure missing: " + key)
    expected = make_request(root, freeze_path=resolve_ref(root, request["freeze_ref"]), _verify=False)
    if request != expected:
        raise ValueError("Request differs from current frozen C01 construction")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("request",))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = make_request(args.root, freeze_path=args.freeze)
    output = args.output.resolve(); output.relative_to(args.root.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(request, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"request": str(output), "request_digest": request["request_digest"],
                      "execution_started": False, "native_qualified": False,
                      "dispatch_ready": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
