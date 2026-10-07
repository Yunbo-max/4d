"""Freeze one UID from an admitted, priced Full128 ActionBench window.

This is a fail-closed engineering input verifier.  It performs no inference,
mesh export, official scoring, queue approval, or scientific qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from research_math.actionbench_queue_pricing import build_pricing_manifest
from research_math.actionbench_unit_manifest import (
    _physical_file,
    _source_identity,
    digest,
)
from research_math.complete_unit_contract import validate_generation_profile
from research_math.control_scoring import file_ref, resolve_ref


R7_SOURCE_REVISION = "84a94a60779b86e477c3488929097b76fdcebfec"
R7_ARCHIVE_MANIFEST_SHA256 = (
    "3e65c9347aab4b67329df72f6e14800a510b6c79d02ecabf337f0d0d105a1eb0")
HISTORICAL_PRICING_PATHS = (
    "inputs/actionbench-full128-queue/pricing.json",
    "docs/research-math-20261006/actionbench-full128-queue-pricing-contract.json",
    "actionmesh/research_overnight/assets/actionbench_population.json",
    "inputs/complete-unit-admissions/complete-lowram-r7.json",
    "inputs/fp16-lowram-v1/unit-manifest.json",
)


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _regular_nonsymlink(root: Path, relative: Path, label: str) -> Path:
    """Resolve a regular file beneath root while rejecting symlink components."""
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("Project-relative " + label + " required")
    root = Path(root).resolve()
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError(label + " may not be a symlink")
    resolved = candidate.resolve()
    resolved.relative_to(root)
    if not resolved.is_file():
        raise FileNotFoundError("Regular " + label + " required")
    return resolved


def validate_pricing_shape(pricing: dict) -> None:
    windows = pricing.get("windows")
    if (pricing.get("kind") != "actionbench-full128-queue-pricing" or
            pricing.get("version") != "1.0.0" or
            pricing.get("status") != "priced_engineering_only" or
            pricing.get("queue_priced") is not True or
            pricing.get("queue_approved") is not False or
            pricing.get("queue_generated") is not False or
            pricing.get("dispatch_ready") is not False or
            pricing.get("scientific_effect_qualification") is not False or
            pricing.get("native_scientific_qualification") is not False or
            pricing.get("candidate_methods_tested") is not False or
            not isinstance(pricing.get("unit_timeout_seconds"), int) or
            pricing["unit_timeout_seconds"] < 1 or
            not isinstance(windows, list) or not windows or
            pricing.get("window_count") != len(windows)):
        raise ValueError("Exact non-dispatchable priced queue required")
    seen = []
    for index, window in enumerate(windows, 1):
        uids = window.get("uids") if isinstance(window, dict) else None
        if (window.get("window_id") != f"full128-window-{index:02d}" or
                not isinstance(uids, list) or not uids or
                window.get("unit_count") != len(uids) or
                window.get("population_start_index") != len(seen) or
                window.get("population_stop_index_exclusive") != len(seen) + len(uids) or
                window.get("planned_workload_seconds") !=
                len(uids) * pricing["unit_timeout_seconds"] or
                any(not isinstance(uid, str) or not uid or Path(uid).name != uid
                    for uid in uids)):
            raise ValueError("Canonical priced window partition required")
        seen.extend(uids)
    if len(seen) != len(set(seen)) or len(seen) != pricing.get("population", {}).get("size"):
        raise ValueError("Complete unique priced population required")


def verify_pricing_receipt(root: Path, pricing: dict) -> tuple[dict, dict]:
    """Recompute pricing from its pinned contract, admission, and population."""
    root = Path(root).resolve()
    expected_paths = (
        root / "docs/research-math-20261006/actionbench-full128-queue-pricing-contract.json",
        root / "inputs/complete-unit-admissions/complete-lowram-r7.json",
        root / "actionmesh/research_overnight/assets/actionbench_population.json",
    )
    paths = []
    for key, expected in zip(
            ("contract_ref", "admission_ref", "population_ref"), expected_paths):
        ref = pricing.get(key)
        if not isinstance(ref, dict):
            raise ValueError("Pricing receipt source reference required: " + key)
        path = resolve_ref(root, ref)
        if path != expected:
            raise ValueError("Canonical pricing source path required: " + key)
        paths.append(path)
    contract, admission, population = (
        json.loads(path.read_text()) for path in paths)
    rebuilt = build_pricing_manifest(root, contract, admission, population)
    rebuilt.update(contract_ref=file_ref(root, paths[0]),
                   admission_ref=file_ref(root, paths[1]),
                   population_ref=file_ref(root, paths[2]))
    if rebuilt != pricing:
        raise ValueError("Pricing receipt differs from source recomputation")
    validate_pricing_shape(pricing)
    return pricing, admission


def verify_runtime_pricing_evidence(project_root: Path, historical_root: Path,
                                    historical_manifest: Path,
                                    pricing: dict) -> tuple[dict, dict]:
    """Recheck the exact historical pricing closure inside each unit attempt.

    The current attempt receives its own hash-pinned copy of the archive
    manifest and blobs. The old checkout is read only and is never used for
    executable source, dataset, weights, device, or output selection.
    """
    project_root = Path(project_root).resolve()
    raw_historical_root = Path(historical_root)
    if raw_historical_root.is_symlink():
        raise ValueError("Historical pricing root may not be a symlink")
    historical_root = raw_historical_root.resolve()
    if not historical_root.is_dir():
        raise FileNotFoundError("Exact historical pricing root required")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=historical_root, check=True,
        capture_output=True, text=True).stdout.strip()
    if revision != R7_SOURCE_REVISION:
        raise ValueError("Exact historical pricing source revision required")
    clean = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--"], cwd=historical_root,
        check=False, capture_output=True)
    if clean.returncode != 0:
        raise ValueError("Historical pricing tracked source must match its revision")

    raw_manifest = Path(historical_manifest)
    absolute_manifest = Path(os.path.abspath(raw_manifest))
    if not raw_manifest.is_absolute():
        absolute_manifest = Path(os.path.abspath(Path.cwd() / raw_manifest))
    # Canonicalize parent aliases such as macOS /var -> /private/var while
    # preserving the leaf so _regular_nonsymlink can still reject a symlinked
    # manifest file.
    absolute_manifest = absolute_manifest.parent.resolve() / absolute_manifest.name
    try:
        manifest_relative = absolute_manifest.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("Hash-verified historical evidence manifest required") from exc
    manifest_path = _regular_nonsymlink(
        project_root, manifest_relative, "historical evidence manifest")
    if (file_ref(project_root, manifest_path)["sha256"] !=
            R7_ARCHIVE_MANIFEST_SHA256):
        raise ValueError("Exact independently verified r7 closure manifest required")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("original_bytes") is not True or
            manifest.get("copied_to_canonical_project_paths") is not False or
            not isinstance(manifest.get("files"), list)):
        raise ValueError("Hash-verified historical archive manifest required")

    rows_by_source: dict[str, list[dict]] = {}
    seen_archive_paths: set[str] = set()
    for row in manifest["files"]:
        if (not isinstance(row, dict) or
                (row.get("matches") is not None and
                 not isinstance(row.get("matches"), bool)) or
                not isinstance(row.get("source"), str) or
                not isinstance(row.get("sha256"), str) or
                len(row["sha256"]) != 64 or
                not isinstance(row.get("bytes"), int) or row["bytes"] < 0 or
                not isinstance(row.get("archive_path"), str)):
            raise ValueError("Exact historical archive row required")
        if row.get("matches") is not True:
            continue
        relative = Path(row["archive_path"])
        if (relative.is_absolute() or not relative.parts or
                ".." in relative.parts or relative.as_posix() in seen_archive_paths):
            raise ValueError("Unique project-relative historical blob required")
        seen_archive_paths.add(relative.as_posix())
        blob = _regular_nonsymlink(
            project_root, relative, "archived historical evidence")
        if (not blob.is_file() or blob.stat().st_size != row["bytes"] or
                hashlib.sha256(blob.read_bytes()).hexdigest() != row["sha256"]):
            raise ValueError("Hash-mismatched archived historical evidence: " +
                             row["source"])
        rows_by_source.setdefault(Path(row["source"]).as_posix(), []).append(row)

    for relative in HISTORICAL_PRICING_PATHS:
        historical = _regular_nonsymlink(
            historical_root, Path(relative), "historical pricing source")
        current = _regular_nonsymlink(
            project_root, Path(relative), "current pricing source")
        if (
                file_ref(historical_root, historical)["sha256"] !=
                file_ref(project_root, current)["sha256"]):
            raise ValueError("Current pricing copy differs from historical evidence: " +
                             relative)
        # Match the canonical suffix while allowing the archive to preserve
        # the source checkout's original absolute root in its manifest.
        matches = [row for source, rows in rows_by_source.items()
                   if source == relative or source.endswith("/" + relative)
                   for row in rows]
        matching_hashes = {row["sha256"] for row in matches}
        if (len(matches) != 1 or matching_hashes !=
                {file_ref(historical_root, historical)["sha256"]}):
            raise ValueError("Historical archive lacks one exact pricing source: " +
                             relative)

    return verify_pricing_receipt(historical_root, pricing)


def _selected_window(pricing: dict, uid: str, window_id: str) -> tuple[dict, int]:
    matches = [window for window in pricing["windows"]
               if window.get("window_id") == window_id]
    if len(matches) != 1 or uid not in matches[0]["uids"]:
        raise ValueError("UID must belong to the exact priced window")
    return matches[0], matches[0]["uids"].index(uid)


def _validate_admissions(pricing: dict, snapshot: dict, semantics: dict,
                         template: dict) -> None:
    population = pricing["population"]
    dataset = snapshot.get("snapshots", {}).get("dataset", {})
    if (snapshot.get("kind") != "actionbench-full128-snapshot-admission" or
            snapshot.get("version") != "1.0.0" or
            snapshot.get("status") != "admitted_engineering_snapshot" or
            snapshot.get("all_revisions_immutable") is not True or
            snapshot.get("all_content_files_hashed") is not True or
            snapshot.get("scientific_effect_qualification") is not False or
            snapshot.get("dispatch_ready") is not False or
            semantics.get("kind") !=
            "actionbench-full128-dataset-semantics-admission" or
            semantics.get("version") != "1.0.0" or
            semantics.get("status") != "admitted_engineering_dataset_semantics" or
            semantics.get("dataset") != population.get("dataset") or
            semantics.get("revision") != population.get("revision") or
            semantics.get("population_size") != population.get("size") or
            semantics.get("frames_per_sample") != 16 or
            semantics.get("all_consumed_bytes_revalidated") is not True or
            semantics.get("scientific_effect_qualification") is not False or
            semantics.get("dispatch_ready") is not False or
            dataset.get("repository") != population.get("dataset") or
            dataset.get("revision") != population.get("revision") or
            semantics.get("snapshot_dataset_manifest_sha256") !=
            dataset.get("manifest_sha256")):
        raise ValueError("Matching admitted Full128 snapshot and semantics required")
    template_population = template.get("population", {})
    if (template.get("kind") != "actionbench-current-release-unit-manifest" or
            template.get("version") != "1.0.0" or
            template.get("status") != "frozen_engineering_current_release_unit" or
            any(template_population.get(key) != value
                for key, value in population.items()) or
            template.get("inference_executed") is not False or
            template.get("scientific_effect_qualification") is not False or
            template.get("dispatch_ready") is not False or
            template.get("queue_generated") is not False):
        raise ValueError("Exact non-executed complete-unit template required")
    if validate_generation_profile(template.get("generation", {})) != "fp16-lowram-v1":
        raise ValueError("Priced Full128 units require the admitted FP16 low-RAM profile")
    for key in ("actionmesh", "triposg", "dinov2", "rmbg"):
        current = snapshot.get("snapshots", {}).get(key)
        frozen = template.get("model_snapshots", {}).get(key)
        fields = ("repository", "revision", "file_count", "total_bytes",
                  "manifest_sha256")
        if not isinstance(current, dict) or frozen != {field: current.get(field)
                                                       for field in fields}:
            raise ValueError("Model snapshot differs from priced template: " + key)


def _verify_template_source(source_root: Path, descriptor: dict) -> None:
    source_root = Path(source_root).resolve()
    identity = _source_identity(source_root)
    if (identity.get("revision") != descriptor.get("revision") or
            identity.get("tree") != descriptor.get("tree")):
        raise ValueError("Official source differs from priced template")
    files = descriptor.get("required_files")
    if not isinstance(files, list):
        raise ValueError("Priced template source inventory required")
    for record in files:
        relative = record.get("path") if isinstance(record, dict) else None
        if not isinstance(relative, str):
            raise ValueError("Priced template source record required")
        path = _physical_file(source_root, relative)
        if (record.get("bytes") != path.stat().st_size or
                record.get("sha256") != digest(path)):
            raise ValueError("Official source file differs from priced template: " + relative)


def _verify_sample_directory_closure(dataset_root: Path, uid: str) -> None:
    sample = dataset_root / "data" / uid
    images = sample / "imgs"
    if (not sample.is_dir() or sample.is_symlink() or
            not images.is_dir() or images.is_symlink()):
        raise ValueError("Physical selected UID directory closure required")
    sample_entries = {path.name for path in sample.iterdir()}
    image_entries = {path.name for path in images.iterdir()}
    expected_images = {f"{index:02d}.png" for index in range(16)}
    if (sample_entries != {"camera.json", "surfaces.npy", "imgs"} or
            image_entries != expected_images):
        raise ValueError("Exact selected UID directory closure required")
    for path in (sample / "camera.json", sample / "surfaces.npy",
                 *(images / name for name in sorted(expected_images))):
        if not path.is_file() or path.is_symlink():
            raise ValueError("Regular selected UID directory closure required")


def freeze_unit(root: Path, pricing: dict, snapshot: dict, semantics: dict,
                template: dict, source_root: Path, dataset_root: Path, uid: str,
                window_id: str, output: Path, snapshot_admission_sha256: str,
                dataset_semantics_sha256: str, *,
                historical_root: Path | None = None,
                historical_manifest: Path | None = None) -> dict:
    """Revalidate and freeze one selected UID without running model or scorer."""
    root = Path(root).resolve()
    output = Path(output).resolve()
    output.relative_to(root)
    if output.exists():
        raise FileExistsError("Preserve existing Full128 unit manifest: " + str(output))
    if historical_root is None or historical_manifest is None:
        raise ValueError("Runtime historical pricing evidence is required")
    verified, admission = verify_runtime_pricing_evidence(
        root, historical_root, historical_manifest, pricing)
    _validate_admissions(verified, snapshot, semantics, template)
    expected_template = admission.get("unit_manifest_ref")
    template_path = resolve_ref(root, expected_template) if isinstance(expected_template, dict) else None
    if (not isinstance(expected_template, dict) or
            expected_template != file_ref(root, template_path) or
            json.loads(template_path.read_text()) != template):
        raise ValueError("Admission-bound template manifest reference required")
    window, offset = _selected_window(verified, uid, window_id)
    receipts = template.get("prerequisite_receipts", {})
    if receipts != {"snapshot_admission_sha256": snapshot_admission_sha256,
                    "dataset_semantics_sha256": dataset_semantics_sha256}:
        raise ValueError("Snapshot or semantics receipt differs from priced template")
    source_root = Path(source_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    _verify_template_source(source_root, template["source"])
    _verify_sample_directory_closure(dataset_root, uid)
    records = snapshot["snapshots"]["dataset"].get("files")
    if (not isinstance(records, list) or
            hashlib.sha256(canonical(records)).hexdigest() !=
            snapshot["snapshots"]["dataset"].get("manifest_sha256")):
        raise ValueError("Canonical admitted dataset inventory required")
    admitted = {record.get("path"): record for record in records
                if isinstance(record, dict) and isinstance(record.get("path"), str)}
    if len(admitted) != len(records):
        raise ValueError("Unique admitted dataset paths required")
    prefix = f"data/{uid}/"
    labels = [prefix + "camera.json", prefix + "surfaces.npy"] + [
        prefix + f"imgs/{index:02d}.png" for index in range(16)]
    unit_inputs = []
    for label in labels:
        expected = admitted.get(label)
        if not isinstance(expected, dict):
            raise ValueError("Selected UID input absent from admission: " + label)
        path = _physical_file(dataset_root, label)
        before = path.stat()
        actual = {"path": label, "bytes": before.st_size, "sha256": digest(path),
                  "etag": expected.get("etag")}
        after = path.stat()
        if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) !=
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or
                actual != expected):
            raise ValueError("Selected UID input differs from admitted bytes: " + label)
        unit_inputs.append(actual)
    samples = semantics.get("samples")
    semantic_rows = [row for row in samples if isinstance(row, dict) and
                     row.get("uid") == uid] if isinstance(samples, list) else []
    if (len(semantic_rows) != 1 or
            semantic_rows[0].get("input_manifest_sha256") !=
            hashlib.sha256(canonical(unit_inputs)).hexdigest()):
        raise ValueError("Selected UID bytes differ from semantic admission")
    result = {
        "kind": "actionbench-full128-unit-manifest", "version": "1.0.0",
        "status": "frozen_engineering_full128_unit",
        "scope": ("One priced Full128 three-arm unit; input verification only, "
                  "not execution, queue approval, baseline qualification, or a "
                  "candidate effect."),
        "population": verified["population"],
        "selected_unit": {"uid": uid, "window_id": window_id,
                          "window_offset": offset,
                          "population_index":
                          window["population_start_index"] + offset,
                          "unit_timeout_seconds": verified["unit_timeout_seconds"]},
        "source": template["source"],
        "model_snapshots": template["model_snapshots"],
        "dataset_inputs": unit_inputs,
        "dataset_input_manifest_sha256":
            hashlib.sha256(canonical(unit_inputs)).hexdigest(),
        "generation": template["generation"],
        "multi_arm_unit": template["multi_arm_unit"],
        "required_outputs": template["required_outputs"],
        "natural_failure_policy": template["natural_failure_policy"],
        "prerequisite_receipts": receipts,
        "pricing_canonical_sha256": hashlib.sha256(canonical(verified)).hexdigest(),
        "template_manifest_ref": expected_template,
        "inference_executed": False, "official_scorer_invocations": 0,
        "queue_priced": True, "queue_approved": False, "queue_generated": False,
        "scientific_effect_qualification": False,
        "native_scientific_qualification": False,
        "candidate_methods_tested": False, "dispatch_ready": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, ensure_ascii=False,
                                allow_nan=False) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "pricing", "snapshot-admission", "dataset-semantics",
                 "unit-manifest", "source-root", "dataset-root", "output",
                 "historical-root", "historical-manifest"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--uid", required=True)
    parser.add_argument("--window-id", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    expected_pricing = root / "inputs/actionbench-full128-queue/pricing.json"
    if args.pricing.resolve() != expected_pricing:
        raise ValueError("Canonical full128 pricing path required")
    result = freeze_unit(
        root, json.loads(args.pricing.read_text()),
        json.loads(args.snapshot_admission.read_text()),
        json.loads(args.dataset_semantics.read_text()),
        json.loads(args.unit_manifest.read_text()), args.source_root,
        args.dataset_root, args.uid, args.window_id, args.output,
        digest(args.snapshot_admission), digest(args.dataset_semantics),
        historical_root=args.historical_root,
        historical_manifest=args.historical_manifest)
    print(json.dumps({"output": str(args.output), "status": result["status"],
                      "uid": args.uid, "inference_executed": False,
                      "scientific_effect_qualification": False,
                      "dispatch_ready": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
