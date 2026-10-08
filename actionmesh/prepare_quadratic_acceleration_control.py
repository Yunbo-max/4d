"""Build the CPU-only harness plan for C13's quadratic strong control."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys


def file_ref(root, path):
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return {'path': relative, 'sha256': digest.hexdigest()}


def build_plans(root: Path, *, source_sequence: Path, run_id: str, weight: float,
                plan_dir: Path, wall_seconds: int):
    import run_experiments as native
    import run_harness as harness

    root, source_sequence = Path(root).resolve(), Path(source_sequence).resolve()
    plan_dir = Path(plan_dir).resolve()
    if source_sequence.name != 'sequence.npz':
        raise ValueError('Expected sequence.npz')
    if isinstance(weight, bool) or not math.isfinite(weight) or weight <= 0.:
        raise ValueError('Finite positive acceleration weight required')
    if wall_seconds < 1 or wall_seconds > 600:
        raise ValueError('Adapter-only wall budget must be 1..600 seconds')
    plan_dir.relative_to(root)
    source_report = source_sequence.with_name('report.json')
    inputs = [file_ref(root, source_sequence), file_ref(root, source_report)]
    report = json.loads(source_report.read_text())
    if (report.get('status') != 'completed'
            or report.get('sha256', {}).get('sequence.npz') != inputs[0]['sha256']):
        raise ValueError('Completed source report and current sequence hash required')
    uid, seed = report.get('uid'), report.get('seed')
    if not isinstance(uid, str) or not uid or Path(uid).name != uid or uid in ('.', '..'):
        raise ValueError('Native UID required')
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('Integer inference seed required')
    code = [file_ref(root, root / 'actionmesh/research_math' / name)
            for name in ('__init__.py', 'quadratic_acceleration_control.py')]
    environment = {
        'python_executable': sys.executable, 'python': platform.python_version(),
        'numpy': importlib.metadata.version('numpy'),
        'scope': 'CPU C13 quadratic strong-control preparation only',
    }
    environment_digest = hashlib.sha256(json.dumps(
        environment, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    outputs = [
        'actionmesh/quadratic-control-output/controls.json',
        'actionmesh/quadratic-control-output/manifest.json',
        'actionmesh/quadratic-control-output/quadratic_acceleration/sequence.npz',
        'actionmesh/quadratic-control-output/quadratic_acceleration/report.json',
    ]
    plan = native.make_plan(root, run_id=run_id, jobs=[{
        'trial_id': 'prepare-quadratic-acceleration-control',
        'command': [sys.executable, '-m', 'research_math.quadratic_acceleration_control',
                    '--source-sequence', str(source_sequence), '--output', 'quadratic-control-output',
                    '--uid', uid, '--expected-sequence-sha256', inputs[0]['sha256'],
                    '--weight', str(weight)],
        'cwd': 'actionmesh', 'input_refs': inputs, 'code_refs': code,
        'output_paths': outputs, 'seed': seed, 'group': 'baseline-artifacts-only',
        'arm_role': 'strong-simple-control-no-scorer',
    }], provenance={
        'git_revision': 'source files pinned in code_refs; no implicit clean-tree claim',
        'model_revision': 'no model loaded; cached generated sequence identity only',
        'data_revision': inputs[0]['sha256'], 'environment_digest': environment_digest,
    }, limits={
        'max_attempts': 1, 'max_development_trials': 1, 'max_confirmation_trials': 0,
        'max_retries_per_trial': 0, 'wall_time_seconds': wall_seconds,
        'attempt_timeout_seconds': wall_seconds,
    })
    plan_dir.mkdir(parents=True, exist_ok=False)
    native_path = plan_dir / 'native.json'
    native_path.write_text(json.dumps(plan, indent=2) + '\n')
    (plan_dir / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    outer = harness.make_plan(root, batch_id=run_id, tasks=[{
        'task_id': 'prepare-quadratic-acceleration-control',
        'idea_id': 'c13-required-strong-control', 'depends_on': [], 'priority': 1,
        'plan_ref': file_ref(root, native_path),
        'resources': {'cpu_cores': 1, 'ram_mib': 2048, 'gpu_count': 0,
                      'gpu_peak_mib': None, 'allow_gpu_share': False,
                      'memory_profile_ref': None, 'exclusive_keys': []},
    }], limits={'total_wall_seconds': wall_seconds + 60, 'window_seconds': wall_seconds + 60,
                'max_parallel_tasks': 1, 'cpu_cores': 1, 'ram_mib': 2048,
                'max_gpu_task_seconds': 0})
    (plan_dir / 'harness.json').write_text(json.dumps(outer, indent=2) + '\n')
    return plan, outer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'skill-dir', 'source-sequence', 'plan-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--weight', type=float, required=True)
    parser.add_argument('--wall-seconds', type=int, default=600)
    args = parser.parse_args()
    scripts = args.skill_dir.resolve() / 'scripts'
    if not (scripts / 'run_harness.py').is_file():
        parser.error('Full installed research-autopilot skill required')
    sys.path.insert(0, str(scripts))
    _, outer = build_plans(args.root, source_sequence=args.source_sequence,
        run_id=args.run_id, weight=args.weight, plan_dir=args.plan_dir,
        wall_seconds=args.wall_seconds)
    print(json.dumps({'plan': str(args.plan_dir.resolve() / 'harness.json'),
                      'approved_plan_digest': outer['plan_digest'],
                      'execution_started': False, 'native_qualified': False,
                      'candidate_methods_tested': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
