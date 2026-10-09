"""Consume one C20 target authorization and delegate to the installed harness.

The claim is stable across controller restarts and exclusive across owners.  It
does not retry the scientific task; it only lets the existing harness resume
the same exact run/plan identity.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys

from research_math import c20_consensus_target as target


def launch(root: Path, *, skill_dir: Path, authorization: Path,
           plan_dir: Path, approved_plan_digest: str,
           stop_after_report: bool = False) -> dict:
    root, authorization, plan_dir = map(lambda path: Path(path).resolve(),
                                        (root, authorization, plan_dir))
    scripts = Path(skill_dir).resolve() / "scripts"
    sys.path.insert(0, str(scripts))
    import run_experiments as native
    import run_harness as harness
    auth = target.read_json(authorization)
    auth_core = {key: value for key, value in auth.items()
                 if key != "authorization_digest"}
    required = {
        "kind", "version", "candidate_id", "scope", "run_id", "uid",
        "generation_seed", "gpu_uuid", "input_freeze_ref", "environment_ref",
        "issued_at", "expires_at", "wall_seconds",
        "explicit_resume_for_exact_attempt", "no_scientific_retry",
        "authorization_digest",
    }
    try:
        issued = datetime.fromisoformat(auth["issued_at"].replace("Z", "+00:00"))
        expires = datetime.fromisoformat(auth["expires_at"].replace("Z", "+00:00"))
    except (KeyError, AttributeError, ValueError) as error:
        raise ValueError("Timezone-aware C20 target authorization window required") from error
    if (set(auth) != required
            or auth.get("kind") != "c20-target-gpu-resume-authorization"
            or auth.get("version") != 1
            or auth.get("candidate_id") != target.CANDIDATE_ID
            or auth.get("scope") != "single_c20_target_attempt"
            or auth.get("explicit_resume_for_exact_attempt") is not True
            or auth.get("no_scientific_retry") is not True
            or auth.get("authorization_digest") != target.canonical_digest(auth_core)
            or issued.tzinfo is None or expires.tzinfo is None or not issued < expires
            or type(auth.get("wall_seconds")) is not int
            or not 60 <= auth["wall_seconds"] <= 26940
            or not isinstance(auth.get("gpu_uuid"), str)
            or not auth["gpu_uuid"].startswith("GPU-")):
        raise ValueError("Exact C20 target authorization required")
    freeze_path = target.resolve_ref(root, auth["input_freeze_ref"])
    environment_path = target.resolve_ref(root, auth["environment_ref"])
    freeze = target.read_json(freeze_path)
    target.validate_freeze(root, freeze)
    environment = target.read_json(environment_path)
    dependency_refs = environment.get("dependency_lock_refs")
    if (freeze.get("uid") != auth["uid"]
            or freeze.get("generation_seed") != auth["generation_seed"]
            or environment.get("gpu_uuid") != auth["gpu_uuid"]
            or not isinstance(dependency_refs, list) or len(dependency_refs) != 1):
        raise ValueError("C20 authorization and frozen source identities differ")
    target.resolve_ref(root, dependency_refs[0])
    native_path, harness_path = plan_dir / "native.json", plan_dir / "harness.json"
    native_plan, harness_plan = map(target.read_json, (native_path, harness_path))
    native.validate_plan(root, native_plan)
    harness.validate_plan(root, harness_plan)
    target.validate_target_plan(root, native_plan, auth, authorization)
    if (harness_plan.get("plan_digest") != approved_plan_digest
            or target.canonical_digest({key: value for key, value in harness_plan.items()
                if key != "plan_digest"}) != approved_plan_digest
            or auth.get("run_id") != native_plan.get("run_id")):
        raise ValueError("Exact authorized C20 target plans required")
    jobs = native_plan.get("jobs")
    tasks = harness_plan.get("tasks")
    if not isinstance(jobs, list) or len(jobs) != 1:
        raise ValueError("Exact one-job C20 target plan required")
    job = jobs[0]
    staged_refs = job.get("input_refs", [])
    required_refs = [target.file_ref(root, authorization),
                     auth["input_freeze_ref"], auth["environment_ref"],
                     dependency_refs[0]]
    command = job.get("command", [])
    def argument(name: str) -> str:
        try:
            return command[command.index(name) + 1]
        except (ValueError, IndexError) as error:
            raise ValueError("C20 target command lacks " + name) from error
    if (any(ref not in staged_refs for ref in required_refs)
            or Path(argument("--freeze")).resolve() != freeze_path
            or Path(argument("--environment")).resolve() != environment_path
            or argument("--gpu-uuid") != auth["gpu_uuid"]
            or job.get("seed") != auth["generation_seed"]
            or native_plan.get("limits", {}).get("max_attempts") != 1
            or native_plan.get("limits", {}).get("max_retries_per_trial") != 0
            or native_plan.get("limits", {}).get("wall_time_seconds") != auth["wall_seconds"]
            or not isinstance(tasks, list) or len(tasks) != 1
            or tasks[0].get("plan_ref") != target.file_ref(root, native_path)
            or harness_plan.get("gpus", {}).get("uuids") != [auth["gpu_uuid"]]
            or harness_plan.get("limits", {}).get("max_gpu_task_seconds") !=
               auth["wall_seconds"]):
        raise ValueError("C20 target plan differs from exact authorization")
    claim_dir = root / "inputs/c20-target/authorization-consumption"
    claim_dir.mkdir(parents=True, exist_ok=True)
    stem = target.digest(authorization)
    lock = (claim_dir / (stem + ".lock")).open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("C20 target authorization already has a live owner")
    claim_path = claim_dir / (stem + ".json")
    identity = {
        "kind": "c20-target-authorization-consumption", "version": 1,
        "state": "consumed_for_exact_plan",
        "authorization_ref": target.file_ref(root, authorization),
        "run_id": auth["run_id"], "gpu_uuid": auth["gpu_uuid"],
        "native_plan_ref": target.file_ref(root, native_path),
        "native_plan_digest": native_plan["plan_digest"],
        "harness_plan_ref": target.file_ref(root, harness_path),
        "harness_plan_digest": approved_plan_digest,
        "no_scientific_retry": True,
    }
    now = datetime.now(timezone.utc)
    if claim_path.exists():
        current = target.read_json(claim_path)
        try:
            consumed = datetime.fromisoformat(
                current["consumed_at"].replace("Z", "+00:00"))
        except (KeyError, AttributeError, ValueError) as error:
            lock.close()
            raise ValueError("Existing C20 consumption timestamp is invalid") from error
        if (any(current.get(key) != value for key, value in identity.items())
                or current.get("consumption_digest") != target.canonical_digest({
                    key: value for key, value in current.items()
                    if key != "consumption_digest"})
                or consumed.tzinfo is None or not issued <= consumed < expires):
            lock.close()
            raise ValueError("Existing C20 target consumption has different identity")
    else:
        if not issued <= now < expires:
            lock.close()
            raise ValueError("First C20 target consumption is outside authorization window")
        record = {**identity, "consumed_at": now.isoformat()}
        record["consumption_digest"] = target.canonical_digest(record)
        try:
            with claim_path.open("x") as stream:
                stream.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
        except FileExistsError:
            lock.close()
            raise RuntimeError("C20 target authorization was concurrently consumed")
    previous = {name: os.environ.get(name) for name in (
        target.CONTROLLER_ENV_PREFIX + "_ROOT",
        target.CONTROLLER_ENV_PREFIX + "_CONSUMPTION_PATH",
        target.CONTROLLER_ENV_PREFIX + "_CONSUMPTION_SHA256")}
    os.environ.update({
        target.CONTROLLER_ENV_PREFIX + "_ROOT": str(root),
        target.CONTROLLER_ENV_PREFIX + "_CONSUMPTION_PATH": str(claim_path),
        target.CONTROLLER_ENV_PREFIX + "_CONSUMPTION_SHA256": target.digest(claim_path),
    })
    try:
        result = harness.run_harness(
            root, harness_plan,
            authorizer=lambda scope: (scope.get("plan") == harness_plan
                and scope.get("plan_digest") == approved_plan_digest),
            stop_after_report=stop_after_report)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        lock.close()
    return {"candidate_id": target.CANDIDATE_ID,
            "consumption_ref": target.file_ref(root, claim_path),
            "harness_plan_digest": approved_plan_digest,
            "harness_result": result, "scientific_admission": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--skill-dir", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--approved-plan-digest", required=True)
    parser.add_argument("--stop-after-report", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(launch(args.root, skill_dir=args.skill_dir,
        authorization=args.authorization, plan_dir=args.plan_dir,
        approved_plan_digest=args.approved_plan_digest,
        stop_after_report=args.stop_after_report), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
