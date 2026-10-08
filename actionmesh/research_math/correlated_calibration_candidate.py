"""C03: apply frozen development affine fits to a retained native context.

The fit is a plain normalized-coordinate surrogate learned only from development
tracked-GT decoder probes.  Application consumes no GT, scorer state, NN/ICP or
labels.  It transfers a global operator to the generated-query residual
distribution; that transfer is a falsifiable assumption, not vertex correspondence
or a proof about ActionBench's native metric.
"""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import time

import numpy as np

from research_math import correlated_calibration as calibration
from research_math import c03_calibration_artifacts as fitting
from research_math import self_map_candidate as c01


CANDIDATE_ID = "4d-math-20261006-c03"
ROLES = ("intercept_squared", "intercept_unsquared", "unit_C01",
         "diagonal_squared", "diagonal_unsquared", "full_squared",
         "full_smoothed_unsquared")
METHOD_IDS = {role: (CANDIDATE_ID if role == "full_smoothed_unsquared"
                     else "c03-control-" + role) for role in ROLES}
FRAME_COMPLETION = list(range(16))


def digest(path: Path) -> str:
    return c01.digest(path)


def read_json(path: Path) -> dict:
    value = json.loads(c01.physical(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    return value


def write_json(path: Path, value: dict) -> None:
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def validate_native_arrays(arrays: dict):
    c01.validate_native_arrays(arrays)
    return arrays["vertices"], arrays["faces"], arrays["timesteps"]


def file_ref(root: Path, path: Path) -> dict:
    root, path = Path(root).resolve(), Path(path).resolve()
    return {"path": path.relative_to(root).as_posix(), "sha256": digest(path)}


def resolve_ref(root: Path, ref: dict) -> Path:
    if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
            or not isinstance(ref.get("path"), str)
            or not isinstance(ref.get("sha256"), str) or len(ref["sha256"]) != 64):
        raise ValueError("Exact project-relative path/sha256 reference required")
    relative = Path(ref["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Nonescaping project-relative reference required")
    root = Path(root).resolve(); path = c01.physical(root / relative).resolve()
    path.relative_to(root)
    if digest(path) != ref["sha256"]:
        raise ValueError("Referenced C03 input changed: " + ref["path"])
    return path


def _development_refs(root: Path, bundle: dict) -> list[dict]:
    refs = bundle["input_refs"]
    ordered = [refs["data"], refs["policy"]]
    ordered += [refs["evidence"][name] for name in sorted(refs["evidence"])]
    ordered += refs["producer_inputs"]
    for ref in ordered:
        resolve_ref(root, ref)
    unique = {}
    for ref in ordered:
        if ref["path"] in unique and unique[ref["path"]] != ref:
            raise ValueError("Conflicting recursive C03 development input")
        unique[ref["path"]] = ref
    return list(unique.values())


def _g01_split_ref(root: Path, bundle: dict) -> dict:
    """Recover the one canonical G01 split frozen by the label producer."""
    matches = []
    for ref in bundle["input_refs"]["producer_inputs"]:
        path = resolve_ref(root, ref)
        try:
            value = read_json(path)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            continue
        if value.get("kind") == "c03-development-label-inventory":
            matches.append(value)
    if len(matches) != 1:
        raise ValueError("Exactly one C03 label inventory must bind the G01 split")
    split_ref = matches[0].get("family_split_ref")
    split = read_json(resolve_ref(root, split_ref))
    policy = bundle["policy"]
    if (split.get("kind") != "g01-family-split"
            or split.get("split_digest") != fitting.canonical_digest(
                {key: value for key, value in split.items()
                 if key != "split_digest"})
            or split.get("d1_ids") != policy["development_uids"]
            or split.get("d2_ids") != policy["d2_uids"]
            or split.get("confirmation_ids") != policy["confirmation_uids"]
            or split.get("family_by_uid") != policy["uid_to_family"]):
        raise ValueError("C03 fit policy differs from its canonical G01 split")
    return split_ref


def _fit_role(bundle: dict, role: str):
    if role == "unit_C01":
        return None
    row = bundle["role_results"][role]
    if row["status"] != "completed":
        raise ValueError("Frozen development fit incomplete: " + role)
    return row["fits"]


def _bounds(value: np.ndarray, lower: float, upper: float, policy: str) -> dict:
    below, above = value < lower, value > upper
    result = {"lower": lower, "upper": upper, "policy": policy,
              "below": int(np.count_nonzero(below)),
              "above": int(np.count_nonzero(above)), "clipped": False}
    if policy == "reject" and (result["below"] or result["above"]):
        raise ValueError("C03 output violates frozen coordinate bounds")
    return result


def _build_role(raw16, residual, anchor, bundle, role):
    if role == "unit_C01":
        return calibration.unit_c01(raw16, residual, anchor), None
    fits = _fit_role(bundle, role)
    return calibration.apply_frozen_fits(raw16, residual, anchor, fits), fits


def _members(output: Path, reports: list[dict]) -> list[Path]:
    paths = [output / "candidate.json", output / "manifest.json",
             output / "common-target.npz"]
    for report in reports:
        directory = output / report["candidate_arm"]
        paths.append(directory / "report.json")
        if report["status"] == "completed":
            paths += [directory / "sequence.npz", directory / "certificate.npz"]
    return sorted(paths, key=lambda path: path.relative_to(output).as_posix())


def _archive(output: Path, reports: list[dict], maximum: int) -> None:
    if type(maximum) is not int or maximum < 1:
        raise ValueError("Positive artifact byte ceiling required")
    rows = [{"path": path.relative_to(output).as_posix(), "sha256": digest(path),
             "size_bytes": path.stat().st_size} for path in _members(output, reports)]
    temporary = output / "artifact.tar.tmp"
    with tarfile.open(temporary, "w", format=tarfile.USTAR_FORMAT) as bundle:
        for row in rows:
            info = tarfile.TarInfo(row["path"]); info.size = row["size_bytes"]
            info.mode = 0o644; info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            with (output / row["path"]).open("rb") as stream:
                bundle.addfile(info, stream)
    if temporary.stat().st_size > maximum:
        temporary.unlink(); raise ValueError("C03 artifact exceeds frozen byte ceiling")
    temporary.replace(output / "artifact.tar")
    write_json(output / "artifact-archive.json", {
        "kind": "c03-terminal-artifact-archive", "version": 1,
        "archive": {"path": "artifact.tar", "sha256": digest(output / "artifact.tar"),
                    "size_bytes": (output / "artifact.tar").stat().st_size},
        "members": rows, "max_artifact_bytes": maximum})


def _validate_archive(root: Path, output: Path, reports: list[dict]) -> dict:
    record = read_json(output / "artifact-archive.json")
    rows = [{"path": path.relative_to(output).as_posix(), "sha256": digest(path),
             "size_bytes": path.stat().st_size} for path in _members(output, reports)]
    archive = c01.physical(output / "artifact.tar")
    if (record.get("kind") != "c03-terminal-artifact-archive"
            or record.get("version") != 1 or record.get("members") != rows
            or record.get("archive") != {"path": "artifact.tar",
                "sha256": digest(archive), "size_bytes": archive.stat().st_size}
            or archive.stat().st_size > record.get("max_artifact_bytes", -1)):
        raise ValueError("C03 archive record changed")
    with tarfile.open(archive, "r:") as bundle:
        members = bundle.getmembers()
        if [item.name for item in members] != [row["path"] for row in rows]:
            raise ValueError("C03 archive inventory changed")
        for item, row in zip(members, rows):
            stream = bundle.extractfile(item)
            if (not item.isfile() or item.size != row["size_bytes"]
                    or item.mode != 0o644 or item.uid or item.gid or item.mtime
                    or item.uname or item.gname or stream is None
                    or hashlib.sha256(stream.read()).hexdigest() != row["sha256"]):
                raise ValueError("C03 archive member changed")
    return {"archive": file_ref(root, archive),
            "record": file_ref(root, output / "artifact-archive.json")}


def export_candidate(root: Path, c01_candidate: Path, fit_bundle_path: Path,
                     output: Path, *, application_stage: str,
                     application_family: str,
                     coordinate_lower: float, coordinate_upper: float,
                     bounds_policy: str, max_artifact_bytes: int) -> dict:
    root = Path(root).resolve(); c01_candidate = Path(c01_candidate).resolve()
    fit_bundle_path = Path(fit_bundle_path).resolve(); output = Path(output)
    c01_candidate.relative_to(root); fit_bundle_path.relative_to(root)
    if (application_stage not in ("d1", "d2", "confirmation")
            or not application_family or not np.isfinite(coordinate_lower)
            or not np.isfinite(coordinate_upper) or coordinate_lower >= coordinate_upper
            or bounds_policy not in ("preserve_and_report", "reject")):
        raise ValueError("Exact confirmation family and finite no-clipping bounds required")
    source = c01.verify_candidate_artifacts(c01_candidate.parent)
    source_context_refs = [file_ref(root, path) for path in
                           (*source["context_files"], source["certificate"])]
    bundle = fitting.validate_bundle(read_json(fit_bundle_path), root=root)
    development_refs = _development_refs(root, bundle)
    g01_split_ref = _g01_split_ref(root, bundle)
    uid = read_json(source["source_report"])["uid"]
    policy = bundle["policy"]
    stage_uids = {"d1": policy["development_uids"], "d2": policy["d2_uids"],
                  "confirmation": policy["confirmation_uids"]}
    if (uid not in stage_uids[application_stage]
            or policy["uid_to_family"].get(uid) != application_family):
        raise ValueError("C03 application must be in its frozen G01 stage/family")
    with np.load(source["source_sequence"], allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    validate_native_arrays(arrays)
    with np.load(source["certificate"], allow_pickle=False) as saved:
        certificate = {name: saved[name].copy() for name in saved.files}
    raw16 = np.concatenate((certificate["anchor"][None], certificate["raw_targets"]), axis=0)
    residual, anchor = certificate["self_residual"], certificate["anchor"]
    if raw16.shape != arrays["vertices"].shape or residual.shape != anchor.shape:
        raise ValueError("Complete C01 raw/context certificate required")
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "common-target.npz", raw_vertices=raw16,
                        self_residual=residual, anchor=anchor,
                        faces=arrays["faces"], timesteps=arrays["timesteps"])
    implementation_sha = digest(Path(__file__))
    implementation_refs = [file_ref(root, path) for path in (
        Path(__file__), Path(fitting.__file__), Path(calibration.__file__))]
    reports = []
    started = time.monotonic()
    for role in ROLES:
        directory = output / role; directory.mkdir(); arm_started = time.monotonic()
        report = {"candidate_id": CANDIDATE_ID, "candidate_arm": role,
            "method_id": METHOD_IDS[role], "uid": uid,
            "seed": read_json(source["source_report"])["seed"],
            "implementation_sha256": implementation_sha,
            "implementation_refs": implementation_refs,
            "source_sequence_sha256": digest(source["source_sequence"]),
            "source_report_sha256": digest(source["source_report"]),
            "c01_candidate_ref": file_ref(root, c01_candidate),
            "fit_bundle_ref": file_ref(root, fit_bundle_path),
            "development_input_refs": development_refs,
            "g01_family_split_ref": g01_split_ref,
            "development_data_sha256": policy["data_sha256"],
            "application_stage": application_stage,
            "application_family": application_family,
            "coordinate_bounds": [coordinate_lower, coordinate_upper],
            "bounds_policy": bounds_policy,
            "input_scope": "frozen D1 affine operator applied label-free to generated-query residuals for the recorded G01 stage",
            "transfer_limitation": "plain-coordinate GT-query to generated-query distribution transfer; no native metric guarantee",
            "native_qualified": False, "scientific_admission": False,
            "local_method_verified": False, "frame_completion": FRAME_COMPLETION}
        try:
            values, fits = _build_role(raw16, residual, anchor, bundle, role)
            exported = values.astype(np.float32)
            exported[0] = arrays["vertices"][0]
            if (not np.isfinite(exported).all()
                    or not np.array_equal(exported[0], arrays["vertices"][0])):
                raise ValueError("C03 export changed exact anchor or produced nonfinite float32")
            bounds = _bounds(exported, coordinate_lower, coordinate_upper, bounds_policy)
            np.savez_compressed(directory / "sequence.npz", **{**arrays, "vertices": exported})
            fit_intercepts = (np.zeros((15, 3)) if fits is None else
                              np.asarray([fit["intercept"] for fit in fits]))
            fit_matrices = (np.repeat(np.eye(3)[None], 15, axis=0) if fits is None else
                            np.asarray([fit["matrix"] for fit in fits]))
            np.savez_compressed(directory / "certificate.npz",
                self_residual=residual, fit_intercepts=fit_intercepts,
                fit_matrices=fit_matrices,
                source_anchor=arrays["vertices"][0], output_vertices=exported)
            report.update(status="completed", bounds=bounds,
                sha256={"sequence.npz": digest(directory / "sequence.npz"),
                        "certificate.npz": digest(directory / "certificate.npz")})
        except Exception as error:
            for name in ("sequence.npz", "certificate.npz"):
                (directory / name).unlink(missing_ok=True)
            report.update(status="error", exception_type=type(error).__name__,
                          error=str(error)[:4096])
        report["elapsed_seconds"] = time.monotonic() - arm_started
        write_json(directory / "report.json", report); reports.append(report)
    completed = [report["candidate_arm"] for report in reports
                 if report["status"] == "completed"]
    common = {"path": "common-target.npz", "sha256": digest(output / "common-target.npz")}
    record = {"kind": "c03-correlated-calibration-candidate", "version": 1,
        "candidate_id": CANDIDATE_ID, "uid": uid, "seed": reports[0]["seed"],
        "roles": list(ROLES), "arms": reports,
        "status": "completed" if len(completed) == len(ROLES) else "incomplete",
        "source_sequence_sha256": digest(source["source_sequence"]),
        "source_report_sha256": digest(source["source_report"]),
        "source_refs": {"sequence": file_ref(root, source["source_sequence"]),
                        "report": file_ref(root, source["source_report"])},
        "source_context_refs": source_context_refs,
        "producer_provenance_ref": file_ref(root,
            source["source_sequence"].parent.parent / "generation-identity.json"),
        "c01_candidate_ref": file_ref(root, c01_candidate),
        "fit_bundle_ref": file_ref(root, fit_bundle_path),
        "development_input_refs": development_refs,
        "g01_family_split_ref": g01_split_ref,
        "implementation_refs": implementation_refs,
        "implementation_sha256": implementation_sha, "common_target": common,
        "application_stage": application_stage,
        "application_family": application_family,
        "coordinate_bounds": [coordinate_lower, coordinate_upper],
        "bounds_policy": bounds_policy,
        "source_delivery_status": "generated_unexecuted_at_authoring",
        "native_qualified": False, "scientific_admission": False,
        "local_method_verified": False,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started}
    write_json(output / "candidate.json", record)
    write_json(output / "manifest.json", {"expected_roles": list(ROLES),
        "cases": [{"case_id": uid + "-" + role, "uid": uid,
                   "case_dir": role, "arm_role": role} for role in completed],
        "common_target": common,
        "scope": "C03 stage-bound application artifacts; no GT/ICP/refit/scorer at application"})
    _archive(output, reports, max_artifact_bytes)
    return record


def validate_candidate_artifact(root: Path, candidate_path: Path) -> dict:
    root, candidate_path = Path(root).resolve(), Path(candidate_path).resolve()
    candidate_path.relative_to(root); output = candidate_path.parent
    record = read_json(candidate_path)
    if (record.get("kind") != "c03-correlated-calibration-candidate"
            or record.get("candidate_id") != CANDIDATE_ID
            or record.get("status") not in ("completed", "incomplete")
            or record.get("roles") != list(ROLES)
            or record.get("native_qualified") is not False
            or record.get("scientific_admission") is not False
            or record.get("application_stage") not in ("d1", "d2", "confirmation")
            or not isinstance(record.get("application_family"), str)
            or not record["application_family"]
            or record.get("bounds_policy") not in ("preserve_and_report", "reject")
            or not isinstance(record.get("coordinate_bounds"), list)
            or len(record["coordinate_bounds"]) != 2
            or any(type(value) not in (int, float) or not np.isfinite(value)
                   for value in record["coordinate_bounds"])
            or record["coordinate_bounds"][0] >= record["coordinate_bounds"][1]):
        raise ValueError("Current terminal unqualified C03 candidate required")
    source_candidate = resolve_ref(root, record["c01_candidate_ref"])
    fit_path = resolve_ref(root, record["fit_bundle_ref"])
    source = c01.verify_candidate_artifacts(source_candidate.parent)
    source_context_refs = [file_ref(root, path) for path in
                           (*source["context_files"], source["certificate"])]
    if record.get("source_context_refs") != source_context_refs:
        raise ValueError("C03 retained C01 context/certificate closure differs")
    bundle = fitting.validate_bundle(read_json(fit_path), root=root)
    g01_split_ref = _g01_split_ref(root, bundle)
    stage_uids = {"d1": bundle["policy"]["development_uids"],
                  "d2": bundle["policy"]["d2_uids"],
                  "confirmation": bundle["policy"]["confirmation_uids"]}
    if (record.get("uid") not in stage_uids[record["application_stage"]]
            or bundle["policy"]["uid_to_family"].get(record.get("uid")) !=
               record["application_family"]):
        raise ValueError("C03 candidate is outside its frozen G01 stage/family")
    source_sequence_ref = file_ref(root, source["source_sequence"])
    source_report_ref = file_ref(root, source["source_report"])
    source_report = read_json(source["source_report"])
    if (record.get("uid") != source_report.get("uid")
            or record.get("seed") != source_report.get("seed")
            or record.get("source_sequence_sha256") != source_sequence_ref["sha256"]
            or record.get("source_report_sha256") != source_report_ref["sha256"]
            or record.get("source_refs") != {
                "sequence": source_sequence_ref, "report": source_report_ref}):
        raise ValueError("C03 candidate is not bound to its retained C01 source")
    development_refs = _development_refs(root, bundle)
    if record.get("development_input_refs") != development_refs:
        raise ValueError("C03 recursive development provenance differs")
    if record.get("g01_family_split_ref") != g01_split_ref:
        raise ValueError("C03 candidate G01 split identity differs")
    expected_implementation_refs = [file_ref(root, path) for path in (
        Path(__file__), Path(fitting.__file__), Path(calibration.__file__))]
    if (record.get("implementation_refs") != expected_implementation_refs
            or record.get("implementation_sha256") != digest(Path(__file__))):
        raise ValueError("C03 implementation closure differs")
    producer_ref = record.get("producer_provenance_ref")
    producer_path = resolve_ref(root, producer_ref)
    if producer_path != source["source_sequence"].parent.parent / "generation-identity.json":
        raise ValueError("C03 producer provenance is not the retained C01 generation identity")
    with np.load(source["source_sequence"], allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    with np.load(source["certificate"], allow_pickle=False) as saved:
        context = {name: saved[name].copy() for name in saved.files}
    raw16 = np.concatenate((context["anchor"][None], context["raw_targets"]), axis=0)
    expected_common = {"raw_vertices": raw16, "self_residual": context["self_residual"],
                       "anchor": context["anchor"], "faces": arrays["faces"],
                       "timesteps": arrays["timesteps"]}
    common_path = output / "common-target.npz"
    if record.get("common_target") != {
            "path": "common-target.npz", "sha256": digest(common_path)}:
        raise ValueError("C03 common target reference differs")
    with np.load(common_path, allow_pickle=False) as saved:
        if set(saved.files) != set(expected_common) or any(
                saved[name].dtype != value.dtype or not np.array_equal(saved[name], value)
                for name, value in expected_common.items()):
            raise ValueError("C03 common target does not recompute")
    reports = []
    for role in ROLES:
        report = read_json(output / role / "report.json"); reports.append(report)
        if (report.get("candidate_arm") != role or report.get("method_id") != METHOD_IDS[role]
                or report.get("uid") != record["uid"] or report.get("seed") != record["seed"]
                or report.get("source_sequence_sha256") != record["source_sequence_sha256"]
                or report.get("source_report_sha256") != record["source_report_sha256"]
                or report.get("implementation_sha256") != digest(Path(__file__))
                or report.get("implementation_refs") != expected_implementation_refs
                or report.get("fit_bundle_ref") != record["fit_bundle_ref"]
                or report.get("c01_candidate_ref") != record["c01_candidate_ref"]
                or report.get("development_input_refs") != development_refs
                or report.get("g01_family_split_ref") != g01_split_ref
                or report.get("application_stage") != record.get("application_stage")
                or report.get("application_family") != record.get("application_family")):
            raise ValueError("C03 role identity differs: " + role)
        try:
            value, fits = _build_role(raw16, context["self_residual"], context["anchor"], bundle, role)
            exported = value.astype(np.float32); exported[0] = arrays["vertices"][0]
            if (not np.isfinite(exported).all()
                    or not np.array_equal(exported[0], arrays["vertices"][0])):
                raise ValueError("C03 export changed exact anchor or produced nonfinite float32")
            if (report.get("coordinate_bounds") != record.get("coordinate_bounds")
                    or report.get("bounds_policy") != record.get("bounds_policy")):
                raise ValueError("C03 role bounds contract differs")
            expected_bounds = _bounds(
                exported, *record["coordinate_bounds"], record["bounds_policy"])
            if report.get("bounds") != expected_bounds:
                raise ValueError("C03 retained bounds diagnostics differ: " + role)
            build_error = None
        except Exception as error:
            exported = fits = None
            build_error = error
        if build_error is not None:
            if (report.get("status") != "error"
                    or report.get("exception_type") != type(build_error).__name__
                    or report.get("error") != str(build_error)[:4096]):
                raise ValueError("C03 failed role does not match recomputation: " + role)
            continue
        if report.get("status") != "completed":
            raise ValueError("Recomputable role retained as failure: " + role)
        sequence = output / role / "sequence.npz"
        certificate = output / role / "certificate.npz"
        if (digest(sequence) != report["sha256"]["sequence.npz"]
                or digest(certificate) != report["sha256"]["certificate.npz"]):
            raise ValueError("C03 role artifact hash changed")
        with np.load(sequence, allow_pickle=False) as saved:
            expected = {**arrays, "vertices": exported}
            if set(saved.files) != set(expected) or any(
                    saved[name].dtype != expected[name].dtype
                    or not np.array_equal(saved[name], expected[name]) for name in expected):
                raise ValueError("C03 role sequence does not recompute")
        expected_certificate = {
            "self_residual": context["self_residual"],
            "fit_intercepts": (np.zeros((15, 3)) if fits is None else
                               np.asarray([fit["intercept"] for fit in fits])),
            "fit_matrices": (np.repeat(np.eye(3)[None], 15, axis=0) if fits is None else
                             np.asarray([fit["matrix"] for fit in fits])),
            "source_anchor": arrays["vertices"][0], "output_vertices": exported}
        with np.load(certificate, allow_pickle=False) as saved:
            if set(saved.files) != set(expected_certificate) or any(
                    saved[name].dtype != value.dtype or not np.array_equal(saved[name], value)
                    for name, value in expected_certificate.items()):
                raise ValueError("C03 role certificate does not recompute")
    expected = "completed" if all(row["status"] == "completed" for row in reports) else "incomplete"
    if record["status"] != expected or record["arms"] != reports:
        raise ValueError("C03 terminal inventory differs")
    archive = _validate_archive(root, output, reports)
    return {"candidate_id": CANDIDATE_ID, "uid": record["uid"],
            "seed": record["seed"], "status": record["status"],
            "artifact_archive": archive,
            "producer_provenance_ref": producer_ref,
            "application_stage": record["application_stage"],
            "g01_family_split_ref": g01_split_ref,
            "implementation_refs": record["implementation_refs"],
            "upstream_refs": [record["c01_candidate_ref"], record["fit_bundle_ref"],
                              *development_refs, *source_context_refs]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "c01-candidate", "fit-bundle", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--application-stage", choices=("d1", "d2", "confirmation"),
                        required=True)
    parser.add_argument("--application-family", required=True)
    parser.add_argument("--coordinate-bounds", type=float, nargs=2, required=True)
    parser.add_argument("--bounds-policy", choices=("preserve_and_report", "reject"), required=True)
    parser.add_argument("--max-artifact-bytes", type=int, required=True)
    args = parser.parse_args(argv)
    result = export_candidate(args.root, args.c01_candidate, args.fit_bundle,
        args.output, application_stage=args.application_stage,
        application_family=args.application_family,
        coordinate_lower=args.coordinate_bounds[0], coordinate_upper=args.coordinate_bounds[1],
        bounds_policy=args.bounds_policy, max_artifact_bytes=args.max_artifact_bytes)
    print(json.dumps({"status": result["status"], "native_qualified": False}))
    return 0 if result["status"] in ("completed", "incomplete") else 2


if __name__ == "__main__":
    raise SystemExit(main())
