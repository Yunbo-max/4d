"""Fail-closed engineering admission for pinned ActionBench release snapshots.

This module hashes already-downloaded bytes.  It does not download models, load
CUDA, run ActionMesh, invoke a scorer, or establish scientific qualification.
Execute it only as the inner command of the research-autopilot harness plan
emitted by ``prepare_actionbench_snapshots.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess


SNAPSHOT_KEYS = ("dataset", "actionmesh", "triposg", "dinov2", "rmbg")
SHA40 = re.compile(r"[0-9a-f]{40}")


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _regular_file(path: Path, label: str):
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode):
        raise ValueError("Symlink forbidden in admitted snapshot: " + label)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("Non-regular file forbidden in admitted snapshot: " + label)
    return info


def _stable_hash(path: Path, label: str):
    before = _regular_file(path, label)
    digest = hashlib.sha256()
    prefix = b""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            if len(prefix) < 512:
                prefix += chunk[:512 - len(prefix)]
            digest.update(chunk)
    after = _regular_file(path, label)
    identity_before = (before.st_dev, before.st_ino, before.st_size,
                       before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size,
                      after.st_mtime_ns)
    if identity_before != identity_after:
        raise ValueError("Snapshot file changed while hashing: " + label)
    stripped = prefix.lstrip().lower()
    if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise ValueError("Unresolved Git LFS pointer: " + label)
    if stripped.startswith(b"<!doctype html") or stripped.startswith(b"<html"):
        raise ValueError("Downloaded HTML response in snapshot: " + label)
    return {"path": label, "bytes": before.st_size,
            "sha256": digest.hexdigest()}


def _walk_content(root: Path):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError("Snapshot root is not a directory: " + str(root))
    records = []
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        relative_current = current_path.relative_to(root)
        kept = []
        for name in sorted(directories):
            child = current_path / name
            relative = (relative_current / name).as_posix()
            if relative == ".cache" or relative.startswith(".cache/huggingface"):
                continue
            if child.is_symlink():
                raise ValueError("Symlink forbidden in admitted snapshot: " + relative)
            kept.append(name)
        directories[:] = kept
        for name in sorted(filenames):
            path = current_path / name
            relative = (relative_current / name).as_posix()
            records.append(_stable_hash(path, relative))
    return sorted(records, key=lambda item: item["path"])


def _metadata(root: Path, relative: str, revision: str):
    metadata_root = root / ".cache" / "huggingface" / "download"
    for parent in (root / ".cache", root / ".cache" / "huggingface", metadata_root):
        if not parent.is_dir() or parent.is_symlink():
            raise ValueError("Invalid Hugging Face metadata directory")
    metadata = metadata_root / (relative + ".metadata")
    parent = metadata.parent
    while parent != metadata_root:
        if (metadata_root not in parent.parents or not parent.is_dir() or
                parent.is_symlink()):
            raise ValueError("Invalid Hugging Face metadata directory")
        parent = parent.parent
    if not metadata.is_file() or metadata.is_symlink():
        raise ValueError("Missing Hugging Face metadata for " + relative)
    _regular_file(metadata, ".cache/huggingface/download/" + relative + ".metadata")
    before = metadata.stat()
    lines = metadata.read_text(encoding="utf-8").splitlines()
    after = metadata.stat()
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) !=
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
        raise ValueError("Hugging Face metadata changed while reading: " + relative)
    if len(lines) < 2 or lines[0] != revision:
        raise ValueError("Hugging Face revision metadata mismatch for " + relative)
    if not lines[1] or any(character in lines[1] for character in "\r\n"):
        raise ValueError("Missing Hugging Face ETag for " + relative)
    return lines[1]


def snapshot_manifest(root: Path, descriptor: dict):
    root = Path(root).resolve()
    revision = descriptor.get("revision")
    repository = descriptor.get("repository")
    repository_type = descriptor.get("repository_type")
    if (not isinstance(repository, str) or repository.count("/") != 1 or
            repository_type not in {"dataset", "model"} or
            not isinstance(revision, str) or not SHA40.fullmatch(revision)):
        raise ValueError("Invalid immutable Hugging Face snapshot descriptor")
    files = _walk_content(root)
    if not files:
        raise ValueError(repository + " contains no regular files")
    for item in files:
        item["etag"] = _metadata(root, item["path"], revision)
    digest = hashlib.sha256(canonical(files)).hexdigest()
    return {
        "repository": repository,
        "repository_type": repository_type,
        "revision": revision,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "manifest_sha256": digest,
        "files": files,
    }


def verify_dataset_layout(root: Path, population: dict, layout: dict):
    root = Path(root).resolve()
    expected_revision = population.get("revision")
    uids = population.get("uids")
    if (population.get("dataset") != "facebook/actionbench" or
            not isinstance(expected_revision, str) or
            not isinstance(uids, list) or len(uids) != len(set(uids)) or
            any(not isinstance(uid, str) or not uid or uid in {".", ".."} or
                "/" in uid or "\\" in uid for uid in uids)):
        raise ValueError("Invalid released ActionBench population descriptor")
    count = layout.get("population_size")
    frames = layout.get("frames_per_sample")
    if count != len(uids) or not isinstance(frames, int) or frames <= 0:
        raise ValueError("Snapshot contract/population cardinality mismatch")
    data_root = root / layout.get("data_directory", "")
    if not data_root.is_dir() or data_root.is_symlink():
        raise ValueError("ActionBench data directory missing")
    actual_uids = sorted(path.name for path in data_root.iterdir()
                         if path.is_dir() and not path.is_symlink())
    if actual_uids != sorted(uids):
        raise ValueError("ActionBench population differs from released UID manifest")
    frame_dir = layout.get("frame_directory")
    pattern = layout.get("frame_pattern")
    base_files = layout.get("required_sample_files")
    if (not isinstance(frame_dir, str) or not frame_dir or
            not isinstance(pattern, str) or not isinstance(base_files, list) or
            any(not isinstance(item, str) or not item for item in base_files)):
        raise ValueError("Invalid ActionBench dataset layout contract")
    required = set(base_files)
    try:
        required.update(frame_dir + "/" + pattern.format(index=index)
                        for index in range(frames))
    except (KeyError, ValueError, IndexError) as error:
        raise ValueError("Invalid ActionBench frame pattern") from error
    for uid in uids:
        sample_root = data_root / uid
        actual = {item["path"] for item in _walk_content(sample_root)}
        if actual != required:
            raise ValueError("ActionBench dataset file closure mismatch for " + uid)
    return {"dataset": population["dataset"], "revision": expected_revision,
            "population_size": len(uids), "frames_per_sample": frames,
            "uid_manifest_sha256": hashlib.sha256(canonical(uids)).hexdigest(),
            "required_files_per_sample": len(required)}


def git_output(root: Path, arguments):
    completed = subprocess.run(
        ["git", "-C", str(Path(root).resolve()), *arguments], check=True,
        capture_output=True, text=True, timeout=30)
    return completed.stdout.rstrip("\r\n")


def verify_source(root: Path, descriptor: dict):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError("Official ActionMesh source root missing")
    revision = descriptor.get("revision")
    tree = descriptor.get("tree")
    submodules = descriptor.get("submodules")
    if (not isinstance(revision, str) or not SHA40.fullmatch(revision) or
            not isinstance(tree, str) or not SHA40.fullmatch(tree) or
            not isinstance(submodules, dict) or not submodules):
        raise ValueError("Invalid official source descriptor")
    if git_output(root, ["rev-parse", "HEAD"]) != revision:
        raise ValueError("Official ActionMesh source revision mismatch")
    if git_output(root, ["rev-parse", "HEAD^{tree}"]) != tree:
        raise ValueError("Official ActionMesh source tree mismatch")
    if git_output(root, ["status", "--porcelain", "--untracked-files=no"]):
        raise ValueError("Official ActionMesh source has tracked modifications")
    verified = {}
    for path, expected in sorted(submodules.items()):
        if (not isinstance(path, str) or Path(path).is_absolute() or ".." in Path(path).parts or
                not isinstance(expected, str) or not SHA40.fullmatch(expected)):
            raise ValueError("Invalid official source submodule descriptor")
        line = git_output(root, ["submodule", "status", "--", path])
        if not line.startswith(" " + expected + " " + path):
            raise ValueError("Official source submodule mismatch: " + path)
        verified[path] = expected
    return {"repository": descriptor.get("repository"), "revision": revision,
            "tree": tree, "tracked_worktree_clean": True,
            "submodules": verified}


def validate_contract(contract: dict, population: dict, roots: dict):
    if (contract.get("kind") != "actionbench-full128-snapshot-contract" or
            contract.get("version") != "1.0.0"):
        raise ValueError("Unsupported ActionBench snapshot contract")
    snapshots = contract.get("snapshots")
    if not isinstance(snapshots, dict) or set(snapshots) != set(SNAPSHOT_KEYS):
        raise ValueError("Snapshot contract must bind exactly five repositories")
    if set(roots) != set(SNAPSHOT_KEYS):
        raise ValueError("Exactly five staged snapshot roots are required")
    dataset = snapshots["dataset"]
    if (dataset.get("repository") != population.get("dataset") or
            dataset.get("revision") != population.get("revision")):
        raise ValueError("Snapshot contract differs from released population")
    uids = population.get("uids")
    expected_uid_digest = contract.get("dataset_layout", {}).get("uid_set_sha256")
    if (not isinstance(uids, list) or not isinstance(expected_uid_digest, str) or
            hashlib.sha256(canonical(uids)).hexdigest() != expected_uid_digest):
        raise ValueError("Snapshot contract released UID digest mismatch")


def admit(contract: dict, population: dict, source_root: Path, roots: dict,
          output: Path):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve existing snapshot admission: " + str(output))
    validate_contract(contract, population, roots)
    source = verify_source(source_root, contract["source"])
    snapshots = {key: snapshot_manifest(roots[key], contract["snapshots"][key])
                 for key in SNAPSHOT_KEYS}
    dataset = verify_dataset_layout(roots["dataset"], population,
                                    contract["dataset_layout"])
    result = {
        "kind": "actionbench-full128-snapshot-admission",
        "version": "1.0.0",
        "status": "admitted_engineering_snapshot",
        "scope": ("Local byte/revision/file-closure admission for the pinned current-public-"
                  "release path; not inference, native scoring, published-run identity, "
                  "runtime compatibility, or scientific qualification"),
        "source": source,
        "snapshots": snapshots,
        "dataset": dataset,
        "all_revisions_immutable": True,
        "all_content_files_hashed": True,
        "scientific_effect_qualification": False,
        "dispatch_ready": False,
        "queue_generated": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, ensure_ascii=False,
                                allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    for key in SNAPSHOT_KEYS:
        parser.add_argument("--" + key + "-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    population = json.loads(args.population.read_text())
    roots = {key: getattr(args, key + "_root") for key in SNAPSHOT_KEYS}
    result = admit(contract, population, args.source_root, roots, args.output)
    print(json.dumps({"output": str(args.output), "status": result["status"],
                      "scientific_effect_qualification": False,
                      "dispatch_ready": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
