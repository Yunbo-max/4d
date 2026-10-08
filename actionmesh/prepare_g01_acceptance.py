"""Prepare a CPU-only G01 family/split acceptance task in the existing harness.

The generated task validates the immutable stage design plus Local-supplied,
independently reviewed family evidence and writes a deterministic split.  It
does not load scientific arrays, run a candidate, invoke the scorer or resume a
GPU.  Web authored this source without executing it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from research_math import g01_design
from prepare_control_scoring_checks import G01_SOURCES


def _command_path(root: Path, cwd: Path, path: Path) -> str:
    root = Path(root).resolve()
    cwd = Path(cwd).resolve()
    path = Path(path).resolve()
    path.relative_to(root)
    return Path("..", path.relative_to(root)).as_posix() if cwd == root / "actionmesh" \
        else path.relative_to(cwd).as_posix()


def g01_job_contract(*, python: str, design: str, family_evidence: str,
                     output_split: str,
                     b_star_selections: str | None) -> tuple[list[str], dict, dict]:
    command = [
        python, "-m", "research_math.g01_design", "--root", "..",
        "--design", design,
        "--family-evidence", family_evidence,
        "--output-split", output_split,
    ]
    if b_star_selections is not None:
        command += ["--b-star-selections", b_star_selections]
    limits = {
        "max_attempts": 1,
        "max_development_trials": 1,
        "max_confirmation_trials": 0,
        "max_retries_per_trial": 0,
        "wall_time_seconds": 120,
        "attempt_timeout_seconds": 120,
    }
    resources = {
        "cpu_cores": 1,
        "ram_mib": 2048,
        "gpu_count": 0,
        "gpu_peak_mib": None,
        "allow_gpu_share": False,
        "memory_profile_ref": None,
        "exclusive_keys": ["g01-family-split"],
    }
    return command, limits, resources


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "skill-dir", "design", "family-evidence",
                 "output-split", "plan-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--b-star-selections", type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    actionmesh = root / "actionmesh"
    skill_scripts = args.skill_dir.resolve() / "scripts"
    if not (skill_scripts / "run_harness.py").is_file():
        parser.error("complete installed research-autopilot skill required")
    for path in (args.design, args.family_evidence):
        path.resolve().relative_to(root)
    if args.b_star_selections:
        args.b_star_selections.resolve().relative_to(root)
    output_split = args.output_split.resolve()
    output_split.relative_to(root)
    if output_split.exists() or output_split.is_symlink():
        parser.error("output split already exists; preserve the old evidence")
    plan_dir = args.plan_dir.resolve()
    plan_dir.relative_to(root)
    if plan_dir.exists() or plan_dir.is_symlink():
        parser.error("plan directory already exists")

    design = g01_design.validate_design(root, args.design.resolve())
    evidence = g01_design.read_json(args.family_evidence.resolve())
    dynamic_paths = g01_design.family_evidence_closure(root, design, evidence)
    expected_split = g01_design.derive_family_split(root, design, evidence)
    if args.b_star_selections:
        selections = g01_design.read_json(args.b_star_selections.resolve())
        dynamic_paths.extend(g01_design.validate_b_star_freeze(
            root, design, selections, expected_split=expected_split))
        dynamic_paths.append(args.b_star_selections.resolve())
    bound_paths = [root / path for path in G01_SOURCES]
    for row in design["candidates"]:
        for ref in g01_design.approved_native_code_refs(design, row) or []:
            bound_paths.append(g01_design.resolve_ref(root, ref))
    bound_paths.extend([args.design.resolve(), args.family_evidence.resolve()])
    bound_paths.extend(dynamic_paths)
    unique_inputs = {path.resolve(): path.resolve() for path in bound_paths}
    input_refs = [g01_design.file_ref(root, unique_inputs[path])
                  for path in sorted(unique_inputs, key=str)]
    code_paths = [
        root / "actionmesh/research_math/__init__.py",
        root / "actionmesh/research_math/g01_design.py",
        root / "actionmesh/prepare_g01_acceptance.py",
    ]
    code_refs = [g01_design.file_ref(root, path) for path in code_paths]
    command, limits, resources = g01_job_contract(
        python=sys.executable,
        design=_command_path(root, actionmesh, args.design),
        family_evidence=_command_path(root, actionmesh, args.family_evidence),
        output_split=_command_path(root, actionmesh, output_split),
        b_star_selections=(
            _command_path(root, actionmesh, args.b_star_selections)
            if args.b_star_selections else None),
    )

    sys.path.insert(0, str(skill_scripts))
    import run_experiments as native
    import run_harness as harness

    native_plan = native.make_plan(
        root, run_id=args.run_id,
        jobs=[{
            "trial_id": "g01-family-split-acceptance",
            "command": command,
            "cwd": "actionmesh",
            "input_refs": input_refs,
            "code_refs": code_refs,
            "output_paths": [output_split.relative_to(root).as_posix()],
            "seed": 0,
            "group": "engineering",
            "arm_role": "software-only",
        }],
        provenance={
            "git_revision": "exact current G01 source bound by code_refs",
            "model_revision": "none; no model or scientific scorer execution",
            "data_revision": BENCHMARK_REVISION,
            "environment_digest": hashlib.sha256(
                sys.version.encode("utf-8")).hexdigest(),
        },
        limits=limits,
    )
    plan_dir.mkdir(parents=True, exist_ok=False)
    native_path = plan_dir / "native.json"
    native_path.write_text(json.dumps(native_plan, indent=2) + "\n")
    harness_plan = harness.make_plan(
        root, batch_id=args.run_id,
        tasks=[{
            "task_id": "g01-family-split-acceptance",
            "idea_id": "stage-wide-g01",
            "depends_on": [],
            "priority": 1,
            "plan_ref": g01_design.file_ref(root, native_path),
            "resources": resources,
        }],
        limits={
            "total_wall_seconds": 180,
            "window_seconds": 180,
            "max_parallel_tasks": 1,
            "cpu_cores": 1,
            "ram_mib": 2048,
            "max_gpu_task_seconds": 0,
        },
    )
    harness_path = plan_dir / "harness.json"
    harness_path.write_text(json.dumps(harness_plan, indent=2) + "\n")
    print(json.dumps({
        "plan": str(harness_path),
        "approved_plan_digest": harness_plan["plan_digest"],
        "execution_started": False,
        "scope": "G01 source/family evidence acceptance only; GPU STOP unchanged",
    }, sort_keys=True))
    return 0


BENCHMARK_REVISION = g01_design.BENCHMARK_REVISION


if __name__ == "__main__":
    raise SystemExit(main())
