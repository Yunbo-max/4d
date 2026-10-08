"""Freeze C02's five logical roles without scoring or selecting B*.

The request assembler revalidates native identity and recomputes the complete
C02 desired repair/projection certificate.  It never chooses a comparator from
outcomes, calls ActionBench, or advances a scientific gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from research_math import protected_geometry_candidate as candidate_module


CANDIDATE_ID = candidate_module.CANDIDATE_ID
ROLES = ("b0", "b_star", "geometry_only", "strength_matched_blend", "protected_step")
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
        raise ValueError("Arm changed exact anchor frame")


def _freeze_core(freeze: dict) -> None:
    core = {key: value for key, value in freeze.items() if key != "freeze_digest"}
    if freeze.get("freeze_digest") != canonical_digest(core):
        raise ValueError("Freeze digest mismatch")
    if (freeze.get("kind") != "c02-native-comparison-freeze"
            or freeze.get("version") != 1 or freeze.get("candidate_id") != CANDIDATE_ID
            or freeze.get("scoring_seed") != 44
            or freeze.get("primary_metric") != "cd_3d"
            or freeze.get("guardrail_metrics") != ["cd_4d", "cd_motion"]):
        raise ValueError("Current frozen C02 comparison record required")
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
        raise ValueError("Exactly five ordered C02 roles must be frozen")


def _decision(root: Path, freeze: dict) -> tuple[dict, list[dict]]:
    decision = read_json(resolve_ref(root, freeze["b_star_decision_ref"]))
    fields = {"kind", "version", "candidate_id", "uid", "inference_seed",
              "decided_at", "selected_role", "selected_method_id",
              "selected_without_c02_native_outcomes", "selection_basis_refs",
              "decision_digest"}
    core = {key: value for key, value in decision.items() if key != "decision_digest"}
    if (set(decision) != fields or decision.get("kind") != "c02-b-star-decision"
            or decision.get("version") != 1 or decision.get("candidate_id") != CANDIDATE_ID
            or decision.get("uid") != freeze["uid"]
            or decision.get("inference_seed") != freeze["inference_seed"]
            or decision.get("selected_without_c02_native_outcomes") is not True
            or decision.get("decision_digest") != canonical_digest(core)):
        raise ValueError("B* must be prospectively selected without C02 outcomes")
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


def _recompute_candidate(source: dict, reports: dict[str, dict], arms: dict[str, dict],
                         certificate: Path) -> None:
    protected = reports["protected_step"]
    parameters = protected.get("repair")
    if not isinstance(parameters, dict):
        raise ValueError("C02 report lacks frozen common-repair parameters")
    vertices, faces, times = candidate_module.validate_native_arrays(source)
    patch_count = protected.get("patch_count")
    labels, seeds = candidate_module.deterministic_patches(
        vertices[0].astype(np.float64), faces, patch_count)
    area = candidate_module.vertex_area_weights(vertices[0].astype(np.float64), faces)
    desired, repair = candidate_module.compute_common_geometry_repair(
        vertices, faces, times, arap_weight=parameters["arap_weight"],
        temporal_weight=parameters["temporal_weight"],
        iterations=parameters["outer_iterations_requested"],
        cg_tolerance=parameters["cg_tolerance"],
        cg_max_iterations=parameters["cg_max_iterations"])
    projected, projection = candidate_module.protected_projection(desired, area, labels)
    matched = projection["strength_matched_scale"] * desired
    def actual_update(role: str, update: np.ndarray) -> np.ndarray:
        if role in arms:
            return (arms[role]["vertices"].astype(np.float64)
                    - vertices.astype(np.float64))
        intended = (vertices.astype(np.float64) + update).astype(np.float32)
        intended[0] = vertices[0]
        return intended.astype(np.float64) - vertices.astype(np.float64)

    actual_protected = actual_update("protected_step", projected)
    actual_matched = actual_update("strength_matched_blend", matched)
    float32_centroid_residual = 0.0
    for patch in range(int(labels.max()) + 1):
        mask = labels == patch
        centroids32 = np.einsum(
            "v,tvc->tc", area[mask], actual_protected[:, mask]) / area[mask].sum()
        float32_centroid_residual = max(
            float32_centroid_residual,
            float(np.max(np.abs(np.diff(centroids32, axis=0)), initial=0.0)))
    protected_vertices = vertices.astype(np.float64) + actual_protected
    coordinate_scale = max(
        1.0, float(np.max(np.abs(vertices), initial=0.0)),
        float(np.max(np.abs(protected_vertices), initial=0.0)))
    float32_tolerance = 64.0 * np.finfo(np.float32).eps * coordinate_scale
    if float32_centroid_residual > float32_tolerance:
        raise ValueError("Float32 scored arm violates protected centroid velocity tolerance")
    projection["float32_protected_centroid_velocity_residual_linf"] = (
        float32_centroid_residual)
    projection["float32_protection_tolerance"] = float(float32_tolerance)
    projection["float32_protection_tolerance_units"] = (
        "native coordinate units per retained timestep index")
    projection["float32_projection_change_w_norm"] = float(np.sqrt(np.einsum(
        "v,tvc,tvc->", area, actual_protected - projected,
        actual_protected - projected)))
    protected32_norm2 = float(np.einsum(
        "v,tvc,tvc->", area, actual_protected, actual_protected))
    matched32_norm2 = float(np.einsum(
        "v,tvc,tvc->", area, actual_matched, actual_matched))
    protected_error2 = float(np.einsum(
        "v,tvc,tvc->", area, actual_protected - projected,
        actual_protected - projected))
    matched_error2 = float(np.einsum(
        "v,tvc,tvc->", area, actual_matched - matched, actual_matched - matched))
    strength_tolerance = (
        2.0 * np.sqrt(projection["projected_w_norm_squared"] * protected_error2)
        + protected_error2
        + 2.0 * np.sqrt(projection["strength_matched_w_norm_squared"] * matched_error2)
        + matched_error2 + projection["numerical_tolerance"])
    strength_delta = abs(protected32_norm2 - matched32_norm2)
    if strength_delta > strength_tolerance:
        raise ValueError("Float32 scored arms violate strength-matching tolerance")
    ideal_norm2 = projection["projected_w_norm_squared"]
    if ideal_norm2 <= projection["numerical_tolerance"]:
        if max(protected32_norm2, matched32_norm2) > 4.0 * projection["numerical_tolerance"]:
            raise ValueError("Near-zero ideal update gained material float32 scored strength")
        protected_relative_error = matched_relative_error = 0.0
        strength_relative_mismatch = 0.0
    else:
        protected_relative_error = float(np.sqrt(protected_error2 / ideal_norm2))
        matched_relative_error = float(np.sqrt(matched_error2 / ideal_norm2))
        strength_relative_mismatch = float(strength_delta / ideal_norm2)
        if (protected_relative_error
                > candidate_module.FLOAT32_RELATIVE_UPDATE_ERROR_CAP
                or matched_relative_error
                > candidate_module.FLOAT32_RELATIVE_UPDATE_ERROR_CAP
                or strength_relative_mismatch
                > candidate_module.FLOAT32_RELATIVE_STRENGTH_NORM2_MISMATCH_CAP):
            raise ValueError("Float32 quantization materially destroys scored strength matching")
    projection.update(
        float32_protected_w_norm_squared=protected32_norm2,
        float32_strength_matched_w_norm_squared=matched32_norm2,
        float32_strength_match_abs_delta=strength_delta,
        float32_strength_match_tolerance=float(strength_tolerance),
        float32_strength_match_tolerance_basis=(
            "weighted norm perturbation bound from both measured float32 quantization errors"),
        float32_protected_relative_update_error=protected_relative_error,
        float32_strength_matched_relative_update_error=matched_relative_error,
        float32_strength_norm2_relative_mismatch=strength_relative_mismatch,
        float32_relative_update_error_cap=(
            candidate_module.FLOAT32_RELATIVE_UPDATE_ERROR_CAP),
        float32_strength_norm2_relative_mismatch_cap=(
            candidate_module.FLOAT32_RELATIVE_STRENGTH_NORM2_MISMATCH_CAP),
        float32_near_zero_rule=(
            "if ideal W-norm squared <= numerical_tolerance, each actual W-norm squared "
            "must be <= 4*numerical_tolerance"),
    )
    with np.load(certificate, allow_pickle=False) as data:
        expected = {
            "vertex_area": area, "patch_labels": labels,
            "patch_seed_vertex_ids": seeds, "desired_step": desired,
            "protected_step": projected, "strength_matched_step": matched,
        }
        if set(data.files) != set(expected):
            raise ValueError("C02 certificate array inventory mismatch")
        for name, value in expected.items():
            if not np.array_equal(data[name], value):
                raise ValueError("C02 certificate differs from recomputation: " + name)
    for role, update in (("geometry_only", desired),
                         ("strength_matched_blend", matched),
                         ("protected_step", projected)):
        if reports[role].get("repair") != repair or reports[role].get("projection") != projection:
            raise ValueError("C02 role report differs from recomputed diagnostics: " + role)
        if role in arms:
            expected_vertices = (vertices.astype(np.float64) + update).astype(np.float32)
            expected_vertices[0] = vertices[0]
            if not np.array_equal(arms[role]["vertices"], expected_vertices):
                raise ValueError("C02 scored arm differs from recomputed method: " + role)


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
    certificate_ref = None
    for row in freeze["roles"]:
        role = row["role"]
        if not isinstance(row.get("method_id"), str) or not row["method_id"]:
            raise ValueError("Every role needs a nonempty method_id")
        if "alias_of" in row:
            if (role != "b_star" or row["alias_of"] not in ("b0", "geometry_only", "strength_matched_blend")
                    or set(row) != {"role", "method_id", "alias_of"}
                    or decision["selected_role"] != row["alias_of"]
                    or decision["selected_method_id"] != row["method_id"]):
                raise ValueError("B* alias must match one frozen simple role and decision")
            normalized.append(dict(row)); continue
        required = {"role", "method_id", "report_ref", "sequence_ref", "implementation_ref"}
        if role == "b0":
            required.add("command_ref")
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
            command_path = resolve_ref(root, row["command_ref"])
            command = read_json(command_path)
            if (not completed or row["report_ref"] != freeze["source_report_ref"]
                    or row["sequence_ref"] != freeze["source_sequence_ref"]
                    or command_path != report_path.with_name("command.json")
                    or command.get("uid") != freeze["uid"]
                    or command.get("seed") != freeze["inference_seed"]
                    or command.get("offline") is not True
                    or command.get("script_sha256") != digest(implementation)):
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
                raise ValueError("C02 role source/implementation identity mismatch: " + role)
            if row["method_id"] != report.get("method_id"):
                raise ValueError("C02 frozen role relabels the retained method: " + role)
            if role == "protected_step" and row["method_id"] != CANDIDATE_ID:
                raise ValueError("Protected role must be the actual C02 method")
            if role != "protected_step" and row["method_id"] == CANDIDATE_ID:
                raise ValueError("Operation control cannot masquerade as C02")
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
            current["command_ref"] = row["command_ref"]
        if role in METHOD_ROLES:
            certificate = resolve_ref(root, row["certificate_ref"])
            if (report.get("sha256", {}).get("certificate.npz") != row["certificate_ref"]["sha256"]
                    or certificate.parent != report_path.parent.parent):
                raise ValueError("Shared C02 certificate missing/stale")
            if certificate_ref is None:
                certificate_ref = row["certificate_ref"]
            elif certificate_ref != row["certificate_ref"]:
                raise ValueError("All C02 operation roles require one certificate")
            current["certificate_ref"] = row["certificate_ref"]
            reports[role] = report; pinned.append(row["certificate_ref"])
            if completed:
                arms[role] = arm
        normalized.append(current)
        if completed:
            physical[role] = current
        pinned.extend([row["report_ref"], row["implementation_ref"]])
        if role == "b0":
            pinned.append(row["command_ref"])
        if completed:
            pinned.append(row["sequence_ref"])
    if set(reports) != set(METHOD_ROLES) or certificate_ref is None:
        raise ValueError("Complete three-arm C02 artifact set required")
    _recompute_candidate(source, reports, arms, resolve_ref(root, certificate_ref))
    canonical_by_sha, scoring_cases = {}, []
    for role in ROLES:
        item = physical.get(role)
        if item is None:
            continue
        sequence_sha = item["sequence_ref"]["sha256"]
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
        "kind": "c02-native-comparison-request", "version": 1,
        "candidate_id": CANDIDATE_ID, "uid": freeze["uid"],
        "inference_seed": freeze["inference_seed"], "scoring_seed": 44,
        "primary_metric": "cd_3d", "guardrail_metrics": ["cd_4d", "cd_motion"],
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
    if (request.get("kind") != "c02-native-comparison-request"
            or request.get("candidate_id") != CANDIDATE_ID
            or request.get("scoring_seed") != 44
            or request.get("generated_unexecuted") is not True
            or request.get("native_qualified") is not False
            or request.get("dispatch_ready") is not False):
        raise ValueError("Unexecuted C02 request scope required")
    if [row.get("role") for row in request.get("roles", [])] != list(ROLES):
        raise ValueError("Exactly five ordered C02 roles required")
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
        raise ValueError("Request differs from current frozen C02 construction")


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
