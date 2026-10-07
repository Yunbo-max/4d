"""Freeze one deterministic current-release ActionBench calibration unit.

This CPU-only engineering admission binds already admitted source, model and
dataset bytes to the first released UID and to the exact generation/control/
scoring inventory needed by a later complete-unit timing run.  It performs no
model load, inference, mesh export or scoring and cannot qualify a scientific
result.  Execute only through the plan emitted by
``prepare_actionbench_unit_manifest.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import subprocess


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _physical_file(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if (relative_path.is_absolute() or ".." in relative_path.parts or
            not relative_path.parts):
        raise ValueError("Nonescaping relative unit input required: " + relative)
    path = root / relative_path
    parent = path.parent
    while parent != root:
        info = parent.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ValueError("Symlink or non-directory unit input parent: " + relative)
        parent = parent.parent
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError("Unit input must be a regular file: " + relative)
    return path


def _source_identity(source_root: Path) -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=source_root, check=True,
                              capture_output=True, text=True).stdout.strip()
    revision = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Official source tracked worktree is not clean")
    return {"revision": revision, "tree": tree, "tracked_worktree_clean": True}


def _validate(contract: dict, population: dict, snapshot: dict,
              semantics: dict) -> tuple[str, dict, dict]:
    kind = contract.get("kind")
    if (kind not in ("actionbench-current-release-unit-contract",
                    "actionbench-current-release-population-unit-contract") or
            contract.get("version") != "1.0.0"):
        raise ValueError("Unsupported current-release unit contract")
    uids = population.get("uids")
    expected = contract.get("population", {})
    if (population.get("dataset") != expected.get("dataset") or
            population.get("revision") != expected.get("revision") or
            not isinstance(uids, list) or len(uids) != expected.get("size") or
            len(uids) != len(set(uids)) or uids != sorted(uids)):
        raise ValueError("Released population identity/order mismatch")
    if hashlib.sha256(canonical(uids)).hexdigest() != expected.get("uid_set_sha256"):
        raise ValueError("Released population digest mismatch")
    uid = contract.get("calibration_unit", {}).get("uid")
    selection = contract.get("calibration_unit", {})
    if kind == "actionbench-current-release-unit-contract":
        if (selection.get("selection_rule") !=
                "first UID in the canonical lexicographically sorted released population" or
                not uids or uid != uids[0]):
            raise ValueError("Calibration UID was not selected prospectively")
    else:
        index = selection.get("population_index")
        if (selection.get("selection_rule") !=
                "canonical full-population index; no outcome-based exclusions" or
                type(index) is not int or not 0 <= index < len(uids) or
                uid != uids[index]):
            raise ValueError("Population UID does not match its frozen canonical index")
    if (snapshot.get("kind") != "actionbench-full128-snapshot-admission" or
            snapshot.get("version") != "1.0.0" or
            snapshot.get("status") != "admitted_engineering_snapshot" or
            snapshot.get("all_revisions_immutable") is not True or
            snapshot.get("all_content_files_hashed") is not True or
            snapshot.get("scientific_effect_qualification") is not False or
            snapshot.get("dispatch_ready") is not False):
        raise ValueError("Successful immutable snapshot admission required")
    if (semantics.get("kind") !=
            "actionbench-full128-dataset-semantics-admission" or
            semantics.get("version") != "1.0.0" or
            semantics.get("status") != "admitted_engineering_dataset_semantics" or
            semantics.get("dataset") != population.get("dataset") or
            semantics.get("revision") != population.get("revision") or
            semantics.get("population_size") != len(uids) or
            semantics.get("frames_per_sample") != 16 or
            semantics.get("all_consumed_bytes_revalidated") is not True or
            semantics.get("scientific_effect_qualification") is not False or
            semantics.get("dispatch_ready") is not False):
        raise ValueError("Successful full-population semantic admission required")
    dataset = snapshot.get("snapshots", {}).get("dataset", {})
    files = dataset.get("files")
    if (dataset.get("repository") != population.get("dataset") or
            dataset.get("revision") != population.get("revision") or
            not isinstance(files, list) or
            hashlib.sha256(canonical(files)).hexdigest() !=
            dataset.get("manifest_sha256") or
            semantics.get("snapshot_dataset_manifest_sha256") !=
            dataset.get("manifest_sha256")):
        raise ValueError("Dataset snapshot/semantic admission mismatch")
    admitted = {}
    for record in files:
        name = record.get("path") if isinstance(record, dict) else None
        if not isinstance(name, str) or not name or name in admitted:
            raise ValueError("Invalid admitted dataset file manifest")
        admitted[name] = record
    samples = semantics.get("samples")
    if not isinstance(samples, list) or len(samples) != len(uids):
        raise ValueError("Semantic sample inventory mismatch")
    semantic_by_uid = {row.get("uid"): row for row in samples
                       if isinstance(row, dict) and isinstance(row.get("uid"), str)}
    if set(semantic_by_uid) != set(uids):
        raise ValueError("Semantic UID inventory mismatch")
    return uid, admitted, semantic_by_uid[uid]


def freeze(contract: dict, population: dict, snapshot: dict, semantics: dict,
           source_root: Path, dataset_root: Path, output: Path,
           snapshot_admission_sha256: str,
           dataset_semantics_sha256: str) -> dict:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve existing unit manifest: " + str(output))
    uid, admitted, semantic_row = _validate(
        contract, population, snapshot, semantics)
    for label, value in (("snapshot admission", snapshot_admission_sha256),
                         ("dataset semantics", dataset_semantics_sha256)):
        if (not isinstance(value, str) or len(value) != 64 or
                any(character not in "0123456789abcdef" for character in value)):
            raise ValueError("Exact lowercase SHA-256 required for " + label)
    source_root = Path(source_root)
    dataset_root = Path(dataset_root)
    if (not source_root.is_dir() or source_root.is_symlink() or
            not dataset_root.is_dir() or dataset_root.is_symlink()):
        raise ValueError("Physical source and dataset roots are required")
    source_root, dataset_root = source_root.resolve(), dataset_root.resolve()
    source_identity = _source_identity(source_root)
    expected_source = contract["source"]
    if (source_identity["revision"] != expected_source["revision"] or
            source_identity["tree"] != expected_source["tree"] or
            snapshot.get("source", {}).get("revision") != expected_source["revision"] or
            snapshot.get("source", {}).get("tree") != expected_source["tree"] or
            snapshot.get("source", {}).get("tracked_worktree_clean") is not True):
        raise ValueError("Official source identity differs from admitted release")
    source_files = []
    for relative in contract["source"]["required_files"]:
        path = _physical_file(source_root, relative)
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=source_root, capture_output=True, text=True)
        if tracked.returncode != 0:
            raise ValueError("Required source file is not tracked: " + relative)
        source_files.append({"path": relative, "bytes": path.stat().st_size,
                             "sha256": digest(path)})
    prefix = "data/" + uid + "/"
    labels = [prefix + "camera.json", prefix + "surfaces.npy"] + [
        prefix + "imgs/" + f"{index:02d}.png" for index in range(16)]
    unit_inputs = []
    for label in labels:
        expected_record = admitted.get(label)
        if not isinstance(expected_record, dict):
            raise ValueError("Unit input absent from snapshot admission: " + label)
        path = _physical_file(dataset_root, label)
        before = path.stat()
        actual = {"path": label, "bytes": before.st_size, "sha256": digest(path),
                  "etag": expected_record.get("etag")}
        after = path.stat()
        if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) !=
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or
                actual != expected_record):
            raise ValueError("Unit input differs from admitted bytes: " + label)
        unit_inputs.append(actual)
    if hashlib.sha256(canonical(unit_inputs)).hexdigest() != \
            semantic_row.get("input_manifest_sha256"):
        raise ValueError("Unit bytes differ from semantic admission")
    snapshot_refs = {}
    for key in ("actionmesh", "triposg", "dinov2", "rmbg"):
        current = snapshot.get("snapshots", {}).get(key, {})
        required = contract["model_snapshots"][key]
        if (current.get("repository") != required["repository"] or
                current.get("revision") != required["revision"] or
                not isinstance(current.get("file_count"), int) or
                current["file_count"] < 1 or
                not isinstance(current.get("manifest_sha256"), str)):
            raise ValueError("Model snapshot admission mismatch: " + key)
        snapshot_refs[key] = {field: current[field] for field in
                              ("repository", "revision", "file_count",
                               "total_bytes", "manifest_sha256")}
    result = {
        "kind": "actionbench-current-release-unit-manifest",
        "version": "1.0.0",
        "status": "frozen_engineering_current_release_unit",
        "scope": ("Deterministic one-UID input/output manifest for later complete-unit "
                  "resource measurement; not execution, scorer qualification, a "
                  "published-run replay, candidate evidence or a queue"),
        "population": contract["population"],
        "calibration_unit": contract["calibration_unit"],
        "source": {**source_identity, "required_files": source_files},
        "model_snapshots": snapshot_refs,
        "dataset_inputs": unit_inputs,
        "dataset_input_manifest_sha256": hashlib.sha256(
            canonical(unit_inputs)).hexdigest(),
        "generation": contract["generation"],
        "multi_arm_unit": contract["multi_arm_unit"],
        "required_outputs": contract["required_outputs"],
        "natural_failure_policy": contract["natural_failure_policy"],
        "prerequisite_receipts": {
            "snapshot_admission_sha256": snapshot_admission_sha256,
            "dataset_semantics_sha256": dataset_semantics_sha256,
        },
        "inference_executed": False,
        "official_scorer_invocations": 0,
        "scientific_effect_qualification": False,
        "dispatch_ready": False,
        "queue_generated": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, ensure_ascii=False,
                                allow_nan=False) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("contract", "population", "snapshot-admission",
                 "dataset-semantics", "source-root", "dataset-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = freeze(
        json.loads(args.contract.read_text()),
        json.loads(args.population.read_text()),
        json.loads(args.snapshot_admission.read_text()),
        json.loads(args.dataset_semantics.read_text()),
        args.source_root, args.dataset_root, args.output,
        digest(args.snapshot_admission), digest(args.dataset_semantics))
    print(json.dumps({"output": str(args.output), "status": result["status"],
                      "inference_executed": False,
                      "scientific_effect_qualification": False,
                      "dispatch_ready": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
