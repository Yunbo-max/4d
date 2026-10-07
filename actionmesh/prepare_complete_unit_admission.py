"""Build, but never execute, a CPU-only complete-unit admission plan."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from research_math.control_scoring import file_ref, resolve_ref


SOURCE_NAMES = (
    "research_math/__init__.py",
    "research_math/control_scoring.py",
    "research_math/complete_unit_contract.py",
    "research_math/complete_unit_admission.py",
    "official_actionbench_adapter.py",
    "prepare_complete_unit_admission.py",
)


def canonical_origin_paths(root: Path, harness_plan: dict, native_plan: dict,
                           native_receipt: dict) -> list[Path]:
    root = Path(root).resolve()
    tasks = harness_plan.get("tasks", [])
    attempts = native_receipt.get("attempts", [])
    if len(tasks) != 1 or len(attempts) != 1:
        raise ValueError("Exact one-task, one-attempt complete unit required")
    batch = root / harness_plan.get("output_root", "") / harness_plan.get("batch_id", "")
    task = batch / "tasks" / tasks[0].get("task_id", "")
    run = root / native_plan.get("output_root", "") / native_plan.get("run_id", "")
    attempt = run / attempts[0].get("attempt_id", "")
    return [
        batch / "plan.json", batch / "state.json", batch / "report.json",
        task / "task.json", task / "result.json", run / "plan.json",
        task / "execution-context.json", run / "receipt.json",
        attempt / "attempt.json",
    ]


def runner_output_paths(root: Path, native_receipt: dict) -> list[Path]:
    """Return every file in the runner's pre-result inventory, including caches."""
    root = Path(root).resolve()
    attempts = native_receipt.get("attempts", [])
    if len(attempts) != 1:
        raise ValueError("Exact one-attempt complete unit required")
    attempt_path = Path(attempts[0].get("attempt_path", ""))
    if attempt_path.is_absolute() or ".." in attempt_path.parts:
        raise ValueError("Project-relative source attempt path required")
    output = root / attempt_path / "workspace/actionmesh/unit-output"
    result = json.loads((output / "result.json").read_text())
    rows = result.get("outputs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Runner pre-result output inventory required")
    paths, seen = [], set()
    for row in rows:
        relative = Path(row.get("path", "")) if isinstance(row, dict) else Path()
        if (relative.is_absolute() or not relative.parts or ".." in relative.parts or
                relative.as_posix() in seen):
            raise ValueError("Unique relative runner output path required")
        path = (output / relative).resolve()
        path.relative_to(output.resolve())
        ref = file_ref(root, path)
        if (set(row) != {"path", "bytes", "sha256"} or
                row["path"] != relative.as_posix() or
                row["bytes"] != path.stat().st_size or
                row["sha256"] != ref["sha256"]):
            raise ValueError("Runner output inventory hash or size mismatch")
        seen.add(relative.as_posix())
        paths.append(path)
    return sorted(paths)


def _nested_refs(value, trail=()):
    if isinstance(value, dict):
        if set(value) == {"path", "sha256"}:
            yield trail, value
            return
        for key, nested in value.items():
            yield from _nested_refs(nested, trail + (key,))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            yield from _nested_refs(nested, trail + (index,))


def reference_closure(root: Path, seed_paths: list[Path], *,
                      historical_manifest: Path | None = None) -> list[dict]:
    """Resolve a JSON evidence graph, optionally using byte-pinned old paths.

    Historical evidence is kept under its archive path in the current project.
    A manifest row maps the original absolute/relative source name to that
    archived byte blob.  Current project files are always preferred when their
    bytes match the reference; a same-name file with different bytes is never
    silently rebound.
    """
    root = Path(root).resolve()
    archive_index = {}
    manifest_path = None
    if historical_manifest is not None:
        raw_manifest_path = Path(historical_manifest)
        if raw_manifest_path.is_symlink():
            raise ValueError("Regular historical evidence manifest required")
        manifest_path = raw_manifest_path.resolve()
        manifest_path.relative_to(root)
        if not manifest_path.is_file():
            raise ValueError("Regular historical evidence manifest required")
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("original_bytes") is not True or
                manifest.get("copied_to_canonical_project_paths") is not False or
                not isinstance(manifest.get("files"), list)):
            raise ValueError("Hash-verified historical archive manifest required")
        for row in manifest["files"]:
            matches = row.get("matches") if isinstance(row, dict) else None
            if (not isinstance(row, dict) or
                    (matches is not None and not isinstance(matches, bool)) or
                    not isinstance(row.get("source"), str) or
                    not isinstance(row.get("sha256"), str) or
                    len(row["sha256"]) != 64 or
                    not isinstance(row.get("bytes"), int) or row["bytes"] < 0 or
                    not isinstance(row.get("archive_path"), str)):
                raise ValueError("Exact historical archive row required")
            # A retained manifest may also inventory files from the newer live
            # checkout. They are metadata for that run, not original historical
            # bytes. Bind them through the manifest itself, but never expose
            # their blobs as historical aliases.
            if row.get("matches") is not True:
                continue
            relative = Path(row["archive_path"])
            if (relative.is_absolute() or not relative.parts or
                    ".." in relative.parts):
                raise ValueError("Project-relative historical blob path required")
            raw_blob = root / relative
            if raw_blob.is_symlink():
                raise ValueError("Archived historical evidence may not be a symlink")
            blob = raw_blob.resolve()
            blob.relative_to(root)
            if (not blob.is_file() or
                    blob.stat().st_size != row["bytes"] or
                    hashlib.sha256(blob.read_bytes()).hexdigest() != row["sha256"]):
                raise ValueError("Hash-mismatched archived historical evidence: " +
                                 row["source"])
            source = Path(row["source"])
            aliases = {source.as_posix()}
            parts = source.parts
            aliases.update(Path(*parts[index:]).as_posix()
                           for index in range(1, len(parts)))
            item = {"path": blob, "source": row["source"],
                    "sha256": row["sha256"], "logical_suffix": source.suffix}
            for alias in aliases:
                archive_index.setdefault(alias, []).append(item)

    queue = [(Path(path).resolve(), Path(path).suffix) for path in seed_paths]
    refs = {}
    if manifest_path is not None:
        manifest_ref = file_ref(root, manifest_path)
        refs[manifest_ref["path"]] = manifest_ref

    def archived_reference(reference: dict) -> dict | None:
        raw_path = reference["path"]
        aliases = [Path(raw_path).as_posix()]
        if Path(raw_path).is_absolute():
            aliases.extend(Path(*Path(raw_path).parts[index:]).as_posix()
                           for index in range(1, len(Path(raw_path).parts)))
        candidates = []
        for alias in aliases:
            candidates = [item for item in archive_index.get(alias, [])
                          if item["sha256"] == reference["sha256"]]
            if candidates:
                break
        if not candidates:
            return None
        paths = {candidate["path"] for candidate in candidates}
        if len(paths) != 1:
            raise ValueError("Ambiguous archived historical evidence: " + raw_path)
        return candidates[0]

    while queue:
        path, logical_suffix = queue.pop()
        path = Path(path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("Evidence seed escapes project root") from exc
        ref = file_ref(root, path)
        if ref["path"] in refs:
            if refs[ref["path"]] != ref:
                raise ValueError("Conflicting complete-unit reference")
            continue
        refs[ref["path"]] = ref
        if logical_suffix == ".json":
            document = json.loads(path.read_text())
            for trail, nested in _nested_refs(document):
                candidate = None
                if Path(nested["path"]).is_absolute():
                    absolute = Path(nested["path"]).resolve()
                    try:
                        relative = absolute.relative_to(root).as_posix()
                    except ValueError:
                        archived = archived_reference(nested)
                        if archived is not None:
                            queue.append((archived["path"],
                                          archived["logical_suffix"]))
                            continue
                        # The official adapter retains this diagnostic input
                        # path outside the project-root harness. Its bytes were
                        # admitted by the frozen snapshot/unit manifest and the
                        # complete-unit finalizer does not consume this path.
                        if (len(trail) >= 4 and trail[-4] == "cases" and
                                isinstance(trail[-3], int) and
                                tuple(trail[-2:]) == ("inputs", "ground_truth")):
                            continue
                        raise ValueError("Unexpected absolute evidence ref escapes "
                                         "project root: " + "/".join(map(str, trail)))
                    nested = {"path": relative, "sha256": nested["sha256"]}
                    candidate = root / relative
                else:
                    candidate = root / nested["path"]
                try:
                    resolved_candidate = candidate.resolve()
                    resolved_candidate.relative_to(root)
                except ValueError as exc:
                    raise ValueError("Evidence reference escapes project root: " +
                                     nested["path"]) from exc
                if (candidate.is_file() and not candidate.is_symlink() and
                        resolved_candidate.is_file()):
                    actual = hashlib.sha256(resolved_candidate.read_bytes()).hexdigest()
                    if actual == nested["sha256"]:
                        queue.append((resolved_candidate, candidate.suffix))
                        continue
                archived = archived_reference(nested)
                if archived is None:
                    # Retain the original precise error for ordinary closures.
                    resolve_ref(root, nested)
                    raise ValueError("Unreachable reference-closure branch")
                queue.append((archived["path"], archived["logical_suffix"]))
    return [refs[path] for path in sorted(refs)]


def build_plans(root: Path, *, contract_path: Path, harness_plan_path: Path, harness_report_path: Path,
                native_plan_path: Path, native_receipt_path: Path,
                approved_unit_plan_digest: str, output: Path, skill_dir: Path,
                plan_dir: Path, run_id: str, wall_seconds: int,
                ram_mib: int, cpu_cores: int):
    root = Path(root).resolve()
    (contract_path, harness_plan_path, harness_report_path, native_plan_path,
     native_receipt_path, output, plan_dir) = [Path(path).resolve() for path in (
         contract_path, harness_plan_path, harness_report_path, native_plan_path,
         native_receipt_path, output, plan_dir)]
    output.relative_to(root)
    plan_dir.relative_to(root)
    if output != root / "inputs/complete-unit-admissions/complete-lowram-r7.json":
        raise ValueError("Exact first complete-unit admission target required")
    if output.exists():
        raise FileExistsError("Complete-unit admission output is single-use")
    if not 1 <= wall_seconds <= 900 or ram_mib < 1 or cpu_cores < 1:
        raise ValueError("Bounded positive CPU finalization resources required")
    scripts = Path(skill_dir).resolve() / "scripts"
    if not (scripts / "run_harness.py").is_file():
        raise ValueError("Complete installed research-autopilot skill required")
    sys.path.insert(0, str(scripts))
    import run_experiments as native
    import run_harness as harness

    from research_math.complete_unit_admission import validate_admission_contract
    contract = json.loads(contract_path.read_text())
    validate_admission_contract(contract)
    harness_plan = json.loads(harness_plan_path.read_text())
    harness_report = json.loads(harness_report_path.read_text())
    native_plan = json.loads(native_plan_path.read_text())
    native_receipt = json.loads(native_receipt_path.read_text())
    if (harness_plan.get("plan_digest") != approved_unit_plan_digest or
            harness_report.get("plan_digest") != approved_unit_plan_digest or
            harness_report.get("status") != "completed" or
            native_receipt.get("plan_digest") != native_plan.get("plan_digest") or
            native_receipt.get("status") != "completed"):
        raise ValueError("Completed source unit is required before admission")
    seeds = [contract_path, harness_plan_path, harness_report_path, native_plan_path,
             native_receipt_path]
    seeds += canonical_origin_paths(root, harness_plan, native_plan, native_receipt)
    seeds += runner_output_paths(root, native_receipt)
    input_refs = reference_closure(root, seeds)

    sources = [root / "actionmesh" / name for name in SOURCE_NAMES]
    missing = [path for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing admission source: " + str(missing[0]))
    code_refs = [file_ref(root, path) for path in sources]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                              capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "diff", "HEAD", "--binary"], cwd=root,
                           check=True, capture_output=True).stdout
    untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"],
                               cwd=root, check=True, capture_output=True,
                               text=True).stdout.splitlines()
    if any(path.endswith(".py") for path in untracked):
        raise ValueError("Commit all executable Python sources before Local acceptance")
    plan_dir.mkdir(parents=True, exist_ok=False)
    dirty_path = plan_dir / "dirty.patch"
    dirty_path.write_bytes(dirty)
    refs = {ref["path"]: ref for ref in input_refs}
    dirty_ref = file_ref(root, dirty_path)
    refs[dirty_ref["path"]] = dirty_ref
    command = [
        sys.executable, "-m", "research_math.complete_unit_admission",
        "--root", "..",
        "--contract", str(contract_path),
        "--harness-plan", str(harness_plan_path),
        "--harness-report", str(harness_report_path),
        "--native-plan", str(native_plan_path),
        "--native-receipt", str(native_receipt_path),
        "--approved-plan-digest", approved_unit_plan_digest,
        "--output", "../" + output.relative_to(root).as_posix(),
    ]
    plan = native.make_plan(
        root, run_id=run_id, purpose="engineering", evidence_mode="developmental",
        jobs=[{
            "trial_id": "admit-complete-lowram-r7",
            "command": command,
            "cwd": "actionmesh",
            "input_refs": list(refs.values()),
            "code_refs": code_refs,
            "output_paths": [output.relative_to(root).as_posix()],
            "seed": 42,
            "group": "engineering",
            "arm_role": "complete-unit-admission",
        }],
        provenance={
            "git_revision": revision,
            "git_refs": [dirty_ref],
            "model_revision": "bound by source complete-unit receipt",
            "data_revision": native_plan.get("provenance", {}).get("data_revision", "unknown"),
            "environment_digest": hashlib.sha256(sys.version.encode()).hexdigest(),
        },
        limits={
            "max_attempts": 1, "max_development_trials": 1,
            "max_confirmation_trials": 0, "max_retries_per_trial": 0,
            "wall_time_seconds": wall_seconds,
            "attempt_timeout_seconds": wall_seconds,
        },
    )
    native_path = plan_dir / "native.json"
    native_path.write_text(json.dumps(plan, indent=2) + "\n")
    outer = harness.make_plan(
        root, batch_id=run_id,
        tasks=[{
            "task_id": "complete-unit-admission",
            "idea_id": "baseline-qualification",
            "depends_on": [], "priority": 1,
            "plan_ref": file_ref(root, native_path),
            "resources": {
                "cpu_cores": cpu_cores, "ram_mib": ram_mib, "gpu_count": 0,
                "gpu_peak_mib": None, "allow_gpu_share": False,
                "memory_profile_ref": None,
                "exclusive_keys": ["complete-unit-admission"],
            },
        }],
        limits={
            "total_wall_seconds": wall_seconds, "window_seconds": wall_seconds,
            "max_parallel_tasks": 1, "cpu_cores": cpu_cores,
            "ram_mib": ram_mib, "max_gpu_task_seconds": 0,
        },
    )
    (plan_dir / "harness.json").write_text(json.dumps(outer, indent=2) + "\n")
    return outer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "contract", "harness-plan", "harness-report", "native-plan",
                 "native-receipt", "output", "skill-dir", "plan-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--approved-unit-plan-digest", required=True)
    parser.add_argument("--run-id", required=True)
    for name in ("wall-seconds", "ram-mib", "cpu-cores"):
        parser.add_argument("--" + name, type=int, required=True)
    args = parser.parse_args()
    plan = build_plans(
        args.root, contract_path=args.contract, harness_plan_path=args.harness_plan,
        harness_report_path=args.harness_report, native_plan_path=args.native_plan,
        native_receipt_path=args.native_receipt,
        approved_unit_plan_digest=args.approved_unit_plan_digest,
        output=args.output, skill_dir=args.skill_dir, plan_dir=args.plan_dir,
        run_id=args.run_id, wall_seconds=args.wall_seconds,
        ram_mib=args.ram_mib, cpu_cores=args.cpu_cores)
    print(json.dumps({
        "plan": str(args.plan_dir / "harness.json"),
        "approved_plan_digest": plan["plan_digest"],
        "execution_started": False,
        "gpu_count": 0,
        "scope": "complete-unit evidence admission only; no queue or scientific claim",
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
