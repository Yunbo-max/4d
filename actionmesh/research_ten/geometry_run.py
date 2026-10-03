"""Independent real-input pairs for H1/H5/H8, compatible with census evaluation.

No tracker, stable mask, camera, safe contact side or GT is invented. Inputs and
qualifications are explicit and hash locked. Each command changes ONE method;
every preset control receives the same source, anchor and solve settings.
"""
from __future__ import annotations
import json
from pathlib import Path
import time
import traceback

import numpy as np

from .case_io import BASE_COMMIT, digest, save_sequence, validate_mesh, write_json

COMMON = {"rest", "trajectories", "faces", "times"}
EXTRA = {1: {"observed_uv", "confidence", "projection"},
    5: {"stable_mask", "confidence"},
    8: {"contact_frame", "contact_indices", "contact_coefficients", "contact_normals", "contact_gap", "fixed_vertices"}}
REQUIRED = {1: EXTRA[1], 5: {"stable_mask"}, 8: EXTRA[8] - {"fixed_vertices"}}
QUALIFIER = {1: "observed_tracks_and_camera_verified", 5: "stable_region_verified", 8: "signed_contacts_verified"}
DEFAULTS = {1: dict(stiffness=10., min_stiffness=.1, evidence_floor=1., strain_scale=.05,
    data_weight=1., iterations=20, temporal_weight=.1, tol=1e-8, cg_maxiter=500),
    5: dict(camera_convention="fixed_metric_world", robust=True),
    8: dict(max_displacement=.01, max_iterations=1000, tolerance=1e-8)}


def _load(method, source, evidence):
    if method not in EXTRA:
        raise ValueError("geometry runner supports exactly one of methods 1, 5, 8")
    proof = json.loads(evidence.read_text())
    uid = proof.get("uid")
    qualification = proof.get("qualification", {})
    if (proof.get("schema_version") != 1 or proof.get("method") != method
            or not isinstance(uid, str) or not uid or Path(uid).name != uid or uid in (".", "..")
            or proof.get("input_evidence") not in ("natural_observations", "constructed_control")
            or not isinstance(proof.get("measurement_provenance"), str) or not proof["measurement_provenance"].strip()
            or not isinstance(qualification, dict) or qualification.get(QUALIFIER[method]) is not True):
        raise ValueError("matching method, UID, observation provenance and explicit qualification required")
    input_hash = digest(source)
    if proof.get("input_sha256") != input_hash:
        raise ValueError("input SHA256 differs from the frozen evidence document")
    supplied = proof.get("parameters", {})
    if not isinstance(supplied, dict) or set(supplied) - DEFAULTS[method].keys():
        raise ValueError("unknown method parameters; GT-based selection is not a runner option")
    parameters = dict(DEFAULTS[method], **supplied)
    for value in parameters.values():
        if isinstance(value, (float, int)) and not np.isfinite(value):
            raise ValueError("method parameters must be finite")
    with np.load(source, allow_pickle=False) as saved:
        if set(saved.files) - (COMMON | EXTRA[method]) or (COMMON | REQUIRED[method]) - set(saved.files):
            raise ValueError("missing input fields or unsupported arrays (including GT/evaluation labels)")
        data = {key: saved[key].copy() for key in saved.files}
    validate_mesh(data["trajectories"], data["faces"], data["times"])
    if (data["rest"].shape != data["trajectories"].shape[1:]
            or not np.array_equal(data["rest"], data["trajectories"][0])):
        raise ValueError("rest must be the exact shared frame-0 mesh anchor")
    if method == 8:
        frame = data["contact_frame"]
        k = len(frame) if frame.ndim == 1 else -1
        if (frame.ndim != 1 or frame.dtype.kind not in "iu"
                or np.any(frame < 0) or np.any(frame >= len(data["times"]))
                or data["contact_indices"].shape != (k, 4)
                or data["contact_coefficients"].shape != (k, 4)
                or data["contact_normals"].shape != (k, 3) or data["contact_gap"].shape != (k,)):
            raise ValueError("integer contact frames and equal-length signed vertex/triangle arrays required")
    for value in data.values():
        value.setflags(write=False)
    return data, proof, parameters, input_hash


def _variants(method, d, p):
    """Return independently callable arms; observations are computed once."""
    q, rest = d["trajectories"], d["rest"]
    arms = {"original": lambda: (q.copy(), {"converged": True, "solver_applied": False}, {})}
    if method == 5:
        from .m05_scale import stabilize_scale
        def candidate():
            r = stabilize_scale(rest, q, d["stable_mask"], confidence=d.get("confidence"), **p)
            return r.corrected, {"converged": True}, dict(scales=r.scales, rotations=r.rotations,
                translations=r.translations, centers=r.centers, fit_weights=r.fit_weights)
        arms["stable_region"] = candidate
        def bbox():
            size = np.linalg.norm(np.ptp(q, axis=1), axis=1)
            if np.any(size <= 1e-12):
                raise ValueError("bbox control cannot normalize zero-size geometry")
            centers = q.mean(axis=1, keepdims=True)
            result = centers + (q-centers)*(size[0]/size)[:, None, None]
            result[0] = q[0]
            return result, {"converged": True}, {}
        def global_similarity():
            r = stabilize_scale(rest, q, np.ones(len(rest), bool), robust=False,
                camera_convention=p["camera_convention"])
            return r.corrected, {"converged": True}, {}
        arms.update(bbox=bbox, global_sim3=global_similarity)
    elif method == 1:
        from .m01_elasticity import mesh_edges, observation_supported_weights, arap_fit, stiffness_controls
        edges = mesh_edges(d["faces"], len(rest))
        def project(vertices, frame):
            image = np.column_stack([vertices, np.ones(len(vertices))]) @ d["projection"][frame].T
            with np.errstate(divide="ignore", invalid="ignore"):
                return image[:, :2]/image[:, 2:3]
        observed = observation_supported_weights(rest, q, edges, d["observed_uv"], project, d["confidence"],
            **{k: p[k] for k in ("stiffness", "min_stiffness", "evidence_floor", "strain_scale")})
        settings = {k: p[k] for k in ("data_weight", "iterations", "temporal_weight", "tol", "cg_maxiter")}
        weights = {"observed_elasticity": observed.edge_weights,
            **stiffness_controls(rest, q, edges, observed.edge_weights)}
        for name, w in weights.items():
            def solve(w=w):
                r = arap_fit(rest, q, edges, w, times=d["times"], **settings)
                return r.trajectories, dict(converged=bool(r.converged), energies=r.energies,
                    cg_iterations=r.cg_iterations), dict(stiffness=w, support=observed.support)
            arms[name] = solve
    else:
        from .m08_contact import ContactConstraint, repair_contact_trajectory, project_contact_constraints, _constraints
        contacts = [[] for _ in range(len(q))]
        for frame, ids, coeff, normal, gap in zip(d["contact_frame"], d["contact_indices"],
                d["contact_coefficients"], d["contact_normals"], d["contact_gap"]):
            contacts[int(frame)].append(ContactConstraint(tuple(ids), tuple(coeff), tuple(normal), float(gap)))
        # Validate every constraint before starting any arm; zero contacts remain
        # valid software input, but do not qualify a natural contact failure.
        for row in contacts:
            _constraints(row, q.shape[1])
        def summarize(reports):
            return dict(converged=all(r.converged for r in reports),
                initial_violation=[r.initial_violation for r in reports],
                final_violation=[r.final_violation for r in reports],
                iterations=[r.iterations for r in reports], supplied_contacts=sum(map(len, contacts)))
        def candidate():
            repaired, reports = repair_contact_trajectory(q, contacts, fixed_vertices=d.get("fixed_vertices"), **p)
            return repaired, summarize(reports), {}
        def standard():
            reports = [project_contact_constraints(frame, contacts[t],
                fixed_vertices=np.arange(q.shape[1]) if t == 0 else d.get("fixed_vertices"),
                max_iterations=p["max_iterations"], tolerance=p["tolerance"]) for t, frame in enumerate(q)]
            return np.stack([r.vertices for r in reports]), summarize(reports), {}
        arms.update(bounded_normal=candidate, standard_projection=standard)
    return arms


def run_geometry(method, source, evidence, output):
    source, evidence, output = map(Path, (source, evidence, output))
    if output.exists():
        raise FileExistsError("output already exists: " + str(output))
    d, proof, parameters, input_hash = _load(method, source, evidence)
    evidence_hash = digest(evidence)
    prepared_at = time.perf_counter()
    arms = _variants(method, d, parameters)
    preparation_seconds = time.perf_counter() - prepared_at
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "method": method, "cases": [dict(
        case_id=f"h{method:02d}-{name}", uid=proof["uid"], case_dir="variants/"+name) for name in arms]}
    write_json(output/"manifest.json", manifest)
    report = dict(status="running", method=method, uid=proof["uid"], input_evidence=proof["input_evidence"],
        qualification_is_caller_assertion=True, measurement_provenance=proof["measurement_provenance"],
        input_sha256=input_hash, evidence_sha256=evidence_hash, parameters=parameters,
        preparation_seconds=preparation_seconds, model_calls=0, training=False, gt_used=False,
        natural_quality_claim=False, source_base_commit=BASE_COMMIT, previous_results_reinterpreted=False,
        runner_sha256=digest(__file__), arms={}, limitations=[
            "Requires natural failure qualification and separate official evaluation before interpreting efficacy.",
            "H5 global Sim(3) is a simple control; H8 has no IPC baseline; strong-baseline qualification remains open.",
            "Uniform H1 weights are preset; select the strongest on independent development data, never the test set."])
    write_json(output/"report.json", report)
    failed = False
    for name, execute in arms.items():
        folder = output/"variants"/name
        folder.mkdir(parents=True, exist_ok=False)
        row = dict(status="running", uid=proof["uid"], method=method, arm=name, frames=len(d["times"]),
            model_calls=0, training=False, gt_used=False, natural_quality_claim=False)
        report["arms"][name] = row
        write_json(folder/"report.json", row)
        started = time.perf_counter()
        try:
            tracks, diagnostics, arrays = execute()
            if not np.array_equal(tracks[0], d["trajectories"][0]):
                raise ValueError("arm changed the shared hard anchor")
            sequence_hash = save_sequence(folder, tracks, d["faces"], d["times"])
            if arrays:
                np.savez_compressed(folder/"diagnostics.npz", **arrays)
            row.update(diagnostics=diagnostics, sha256={"sequence.npz": sequence_hash}, anchor_exact=True)
            if not diagnostics.get("converged", False):
                raise RuntimeError("solver did not converge; preserve this attempt as a failed arm")
            row["status"] = "completed"
        except Exception as error:
            row.update(status="failed", error=str(error), traceback=traceback.format_exc())
            failed = True
        row["elapsed_seconds"] = time.perf_counter()-started
        write_json(folder/"report.json", row)
        write_json(output/"report.json", report)
        print(json.dumps({"method": method, "arm": name, "status": row["status"]}), flush=True)
    try:
        unchanged = digest(source) == input_hash and digest(evidence) == evidence_hash
    except Exception as error:
        unchanged = False
        report["input_verification_error"] = str(error)
    if not unchanged:
        for name, row in report["arms"].items():
            row.update(status="failed", error="input/evidence changed during run; all arms invalidated")
            write_json(output/"variants"/name/"report.json", row)
    report.update(status="failed" if failed or not unchanged else "completed", inputs_unchanged=unchanged)
    write_json(output/"report.json", report)
    return int(failed or not unchanged)
