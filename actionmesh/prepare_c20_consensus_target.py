"""Emit C20's single-attempt decoder-consensus producer plan; never run it."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

from research_math import c20_consensus_target as producer
from prepare_actionbench_full128_window import validate_environment_closure


def _copy_exact(source: Path, destination: Path) -> None:
    source, destination = Path(source).resolve(), Path(destination)
    if source.is_symlink() or not source.is_file():
        raise ValueError("Physical pinned C20 cache source required")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_symlink() or producer.digest(destination) != producer.digest(source):
            raise ValueError("Existing C20 cache differs from pinned source")
        return
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        if temporary.is_symlink() or producer.digest(temporary) != producer.digest(source):
            raise ValueError("Stale partial C20 cache differs from pinned source")
    else:
        with source.open("rb") as incoming, temporary.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=8 * 1024 * 1024)
    temporary.replace(destination)


def _stage_runtime_inputs(root: Path, *, source_root: Path, weights_root: Path,
                          freeze: dict, identity: dict) -> tuple[Path, Path, list[dict]]:
    """Create/reuse an immutable project-relative source+weight snapshot."""
    revision = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True).stdout.strip()
    if revision != producer.ACTIONMESH_REVISION:
        raise ValueError("Exact pinned ActionMesh checkout revision required")
    clean = subprocess.run(
        ["git", "-C", str(source_root), "diff", "--quiet", "HEAD", "--"],
        check=False)
    if clean.returncode != 0:
        raise ValueError("Pinned ActionMesh checkout has tracked modifications")
    listing = subprocess.run(
        ["git", "-C", str(source_root), "ls-files", "-z"], check=True,
        capture_output=True).stdout.split(b"\0")
    tracked = [value.decode() for value in listing if value]
    tree_listing = subprocess.run(
        ["git", "-C", str(source_root), "ls-tree", "-r", "--name-only", "-z",
         "HEAD"], check=True, capture_output=True).stdout.split(b"\0")
    head_tracked = [value.decode() for value in tree_listing if value]
    if not tracked or any(Path(value).is_absolute() or ".." in Path(value).parts
                          for value in tracked) or tracked != head_tracked:
        raise ValueError("Canonical tracked ActionMesh source inventory required")
    for relative, expected in identity["upstream_source_sha256"].items():
        if relative not in tracked or producer.digest(source_root / relative) != expected:
            raise ValueError("Pinned ActionMesh identity differs before staging")
    manifest_path = producer.resolve_ref(root, freeze["weights_manifest_ref"])
    rows = json.loads(manifest_path.read_text())
    selected = [row for row in rows if isinstance(row, dict)
                and row.get("repo") == "facebook/ActionMesh"
                and row.get("revision") == producer.ACTIONMESH_REVISION
                and isinstance(row.get("path"), str)
                and row["path"].startswith("weights/ActionMesh/autoencoder/")]
    if {row.get("file") for row in selected} != {
            "autoencoder/config.json", "autoencoder/model.safetensors"}:
        raise ValueError("Exact pinned C20 weight inventory required")
    cache = (root / "inputs/c20-target/staged" /
             (freeze["uid"] + "-" + str(freeze["generation_seed"])) /
             producer.ACTIONMESH_REVISION).resolve()
    cache.relative_to(root)
    staged_source, staged_weights = cache / "source", cache / "weights"
    expected_paths = set()
    for relative in tracked:
        source, destination = source_root / relative, staged_source / relative
        _copy_exact(source, destination); expected_paths.add(destination.resolve())
    for row in selected:
        relative = Path(row["path"]).relative_to("weights")
        source, destination = weights_root / relative, staged_weights / relative
        if (source.stat().st_size != row.get("size")
                or producer.digest(source) != row.get("sha256")):
            raise ValueError("Pinned C20 weight differs before staging")
        _copy_exact(source, destination); expected_paths.add(destination.resolve())
    actual = {path.resolve() for parent in (staged_source, staged_weights)
              for path in parent.rglob("*") if path.is_file()}
    if actual != expected_paths or any(path.is_symlink() for path in expected_paths):
        raise ValueError("C20 immutable staged runtime inventory differs")
    refs = [producer.file_ref(root, path) for path in sorted(
        expected_paths, key=lambda path: path.as_posix())]
    return staged_source, staged_weights, refs


def _authorization(root: Path, path: Path, *, freeze: Path, environment: Path,
                   run_id: str, gpu_uuid: str) -> tuple[dict, list[dict]]:
    path = Path(path).resolve(); path.relative_to(root)
    value = producer.read_json(path)
    core = {key: item for key, item in value.items() if key != "authorization_digest"}
    required = {"kind", "version", "candidate_id", "scope", "run_id", "uid",
                "generation_seed", "gpu_uuid", "input_freeze_ref", "environment_ref",
                "issued_at", "expires_at", "wall_seconds", "explicit_resume_for_exact_attempt",
                "no_scientific_retry", "authorization_digest"}
    if (set(value) != required
            or value.get("authorization_digest") != producer.canonical_digest(core)
            or value.get("kind") != "c20-target-gpu-resume-authorization"
            or value.get("version") != 1 or value.get("candidate_id") != producer.CANDIDATE_ID
            or value.get("scope") != "single_c20_target_attempt"
            or value.get("run_id") != run_id or value.get("gpu_uuid") != gpu_uuid
            or value.get("input_freeze_ref") != producer.file_ref(root, freeze)
            or value.get("environment_ref") != producer.file_ref(root, environment)
            or value.get("explicit_resume_for_exact_attempt") is not True
            or value.get("no_scientific_retry") is not True):
        raise ValueError("Exact single-use C20 target GPU-resume authorization required")
    try:
        issued = datetime.fromisoformat(value["issued_at"].replace("Z", "+00:00"))
        expires = datetime.fromisoformat(value["expires_at"].replace("Z", "+00:00"))
    except (KeyError, ValueError, AttributeError) as error:
        raise ValueError("Timezone-aware C20 authorization window required") from error
    now = datetime.now(timezone.utc)
    if issued.tzinfo is None or expires.tzinfo is None or not issued <= now < expires:
        raise ValueError("C20 target GPU authorization is not currently valid")
    if (type(value.get("wall_seconds")) is not int
            or not 60 <= value["wall_seconds"] <= 26940):
        raise ValueError("Bounded C20 target wall budget required")
    freeze_value = producer.read_json(freeze)
    if (value.get("uid") != freeze_value.get("uid")
            or value.get("generation_seed") != freeze_value.get("generation_seed")):
        raise ValueError("C20 authorization and input freeze identities differ")
    return value, [producer.file_ref(root, path)]


def build_plans(root: Path, *, skill_dir: Path, freeze: Path,
                source_root: Path, weights_root: Path, environment: Path,
                authorization: Path, plan_dir: Path, run_id: str,
                gpu_uuid: str, cpu_cores: int, ram_mib: int):
    root, skill_dir, freeze, environment, plan_dir = map(
        lambda item: Path(item).resolve(),
        (root, skill_dir, freeze, environment, plan_dir))
    plan_dir.relative_to(root)
    if plan_dir.exists():
        raise FileExistsError("Preserve prior C20 target plan")
    if (not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-")
            or any(character.isspace() or character == "," for character in gpu_uuid)):
        raise ValueError("One exact physical GPU UUID required")
    if any(type(value) is not int or value < 1 for value in (cpu_cores, ram_mib)):
        raise ValueError("Positive C20 CPU/RAM resources required")
    for name, path in (("source_root", source_root), ("weights_root", weights_root)):
        if not Path(path).is_absolute() or not Path(path).is_dir():
            raise ValueError("Existing absolute pinned " + name + " required")
    scripts = skill_dir / "scripts"
    if not (scripts / "run_harness.py").is_file():
        raise ValueError("Complete installed research-autopilot required")
    freeze_value = producer.read_json(freeze)
    identity, refs = producer.validate_freeze(root, freeze_value)
    environment_value = producer.read_json(environment)
    environment_paths = validate_environment_closure(
        root, environment, environment_value, gpu_uuid)
    auth, auth_refs = _authorization(
        root, authorization, freeze=freeze, environment=environment,
        run_id=run_id, gpu_uuid=gpu_uuid)
    staged_source, staged_weights, runtime_refs = _stage_runtime_inputs(
        root, source_root=Path(source_root).resolve(),
        weights_root=Path(weights_root).resolve(), freeze=freeze_value,
        identity=identity)
    inputs = [producer.file_ref(root, path) for path in environment_paths]
    inputs += [producer.file_ref(root, freeze), *refs, *auth_refs, *runtime_refs]
    unique = {}
    for ref in inputs:
        if ref["path"] in unique and unique[ref["path"]] != ref:
            raise ValueError("Conflicting C20 producer input identity")
        unique[ref["path"]] = ref
    code_paths = [root / "actionmesh/research_math" / name for name in (
        "__init__.py", "c20_consensus_target.py", "actionbench_parity.py",
        "control_scoring.py", "decoder_observer.py",
        "pipeline_decoder_observer.py", "native_context_runner.py")]
    code_paths += [root / "actionmesh" / name for name in (
        "official_actionbench_adapter.py", "research_census_eval.py")]
    code_paths.append(root / "actionmesh/prepare_c20_consensus_target.py")
    output_relative = "actionmesh/c20-consensus-target-output"
    command = [sys.executable, "-m", "research_math.c20_consensus_target",
        "--root", "..", "--freeze", str(freeze),
        "--source-root", "../" + staged_source.relative_to(root).as_posix(),
        "--weights-root", "../" + staged_weights.relative_to(root).as_posix(),
        "--environment", str(environment), "--authorization", str(authorization),
        "--gpu-uuid", gpu_uuid,
        "--output", "c20-consensus-target-output"]
    sys.path.insert(0, str(scripts))
    import run_experiments as native
    import run_harness as harness
    plan = native.make_plan(root, run_id=run_id, purpose="engineering",
        evidence_mode="developmental", jobs=[{
            "trial_id": "c20-decoder-consensus-target", "command": command,
            "cwd": "actionmesh", "input_refs": list(unique.values()),
            "code_refs": [producer.file_ref(root, path) for path in code_paths],
            "output_paths": [output_relative + "/producer-report.json",
                             output_relative + "/producer-manifest.json",
                             output_relative + "/consensus-target.npz"],
            "seed": freeze_value["generation_seed"], "group": "c20-target-producer",
            "arm_role": "decoder-consensus-no-camera-no-gt"}],
        provenance={"git_revision": "exact code refs; no clean-tree claim",
            "model_revision": "environment/freeze-bound ActionMesh autoencoder",
            "data_revision": producer.digest(freeze),
            "environment_digest": producer.digest(environment)},
        limits={"max_attempts": 1, "max_development_trials": 1,
            "max_confirmation_trials": 0, "max_retries_per_trial": 0,
            "wall_time_seconds": auth["wall_seconds"],
            "attempt_timeout_seconds": auth["wall_seconds"]})
    plan_dir.mkdir(parents=True)
    native_path = plan_dir / "native.json"
    native_path.write_text(json.dumps(plan, indent=2) + "\n")
    outer = harness.make_plan(root, batch_id=run_id, tasks=[{
        "task_id": "c20-decoder-consensus-target",
        "idea_id": producer.CANDIDATE_ID, "depends_on": [], "priority": 1,
        "plan_ref": producer.file_ref(root, native_path),
        "resources": {"cpu_cores": cpu_cores, "ram_mib": ram_mib,
            "gpu_count": 1, "gpu_peak_mib": None, "allow_gpu_share": False,
            "memory_profile_ref": None,
            "exclusive_keys": ["c20-target-producer", "actionmesh-stage2"]}}],
        limits={"total_wall_seconds": auth["wall_seconds"],
            "window_seconds": auth["wall_seconds"] + 1800,
            "max_parallel_tasks": 1, "cpu_cores": cpu_cores, "ram_mib": ram_mib,
            "max_gpu_task_seconds": auth["wall_seconds"]},
        gpus={"uuids": [gpu_uuid], "safety_margin_mib": 1024,
               "max_tasks_per_gpu": 1})
    (plan_dir / "harness.json").write_text(json.dumps(outer, indent=2) + "\n")
    return plan, outer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "skill-dir", "freeze", "source-root", "weights-root",
                 "environment", "authorization", "plan-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--run-id", required=True); parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--cpu-cores", type=int, required=True)
    parser.add_argument("--ram-mib", type=int, required=True)
    args = parser.parse_args(argv)
    _, outer = build_plans(args.root, skill_dir=args.skill_dir,
        freeze=args.freeze, source_root=args.source_root, weights_root=args.weights_root,
        environment=args.environment, authorization=args.authorization,
        plan_dir=args.plan_dir, run_id=args.run_id, gpu_uuid=args.gpu_uuid,
        cpu_cores=args.cpu_cores, ram_mib=args.ram_mib)
    print(json.dumps({"plan": str(args.plan_dir.resolve() / "harness.json"),
        "approved_plan_digest": outer["plan_digest"], "execution_started": False,
        "dispatch_ready": False, "scientific_admission": False,
        "source_delivery_status": "generated_unexecuted",
        "gpu_stop_remains_effective_without_exact_authorization": True}))


if __name__ == "__main__":
    main()
