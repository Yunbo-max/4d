"""Build the CPU-only artifact plan for C13's group-acceleration source.

This builder does not run the method, score it, admit it scientifically, or
resume GPU work.  Every solver/time-scale parameter is explicit so Local can
freeze a development choice before plan emission.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np


def file_ref(root: Path, path: Path) -> dict:
    root, path = Path(root).resolve(), Path(path).resolve()
    relative = path.relative_to(root).as_posix()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return {'path': relative, 'sha256': digest.hexdigest()}


def _positive_scalar(name: str, value) -> float:
    if isinstance(value, (bool, np.bool_)) or not math.isfinite(value) or value <= 0.:
        raise ValueError(f'{name} must be finite and positive')
    return float(value)


def build_plans(root: Path, *, source_sequence: Path,
                observation_metric: Path | None, run_id: str,
                group_weight: float, rho: float, absolute_tolerance: float,
                relative_tolerance: float, gap_tolerance: float,
                max_iterations: int, plan_dir: Path, wall_seconds: int):
    import run_experiments as native
    import run_harness as harness

    root = Path(root).resolve()
    source_sequence = Path(source_sequence).resolve()
    plan_dir = Path(plan_dir).resolve()
    plan_dir.relative_to(root)
    if source_sequence.name != 'sequence.npz':
        raise ValueError('Expected sequence.npz')
    for name, value in (
            ('group_weight', group_weight), ('rho', rho),
            ('absolute_tolerance', absolute_tolerance),
            ('relative_tolerance', relative_tolerance),
            ('gap_tolerance', gap_tolerance)):
        _positive_scalar(name, value)
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError('max_iterations must be a positive integer')
    if isinstance(wall_seconds, bool) or not 1 <= wall_seconds <= 26940:
        raise ValueError('CPU job budget must be 1..26940 seconds so outer total stays <=27000')

    source_report = source_sequence.with_name('report.json')
    inputs = [file_ref(root, source_sequence), file_ref(root, source_report)]
    source = json.loads(source_report.read_text())
    if (source.get('status') != 'completed'
            or source.get('sha256', {}).get('sequence.npz') != inputs[0]['sha256']):
        raise ValueError('Completed source report and current sequence hash required')
    uid, seed = source.get('uid'), source.get('seed')
    if not isinstance(uid, str) or not uid or Path(uid).name != uid or uid in ('.', '..'):
        raise ValueError('Native UID required')
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('Integer inference seed required')

    command = [
        sys.executable, '-m', 'research_math.group_acceleration_candidate',
        '--source-sequence', str(source_sequence), '--output', 'c13-group-output',
        '--uid', uid, '--expected-sequence-sha256', inputs[0]['sha256'],
    ]
    metric_scope = 'identity_temporal_spd'
    if observation_metric is None:
        command.append('--identity-observation-metric')
    else:
        observation_metric = Path(observation_metric).resolve()
        metric_ref = file_ref(root, observation_metric)
        inputs.append(metric_ref)
        command += ['--observation-metric', str(observation_metric),
                    '--expected-metric-sha256', metric_ref['sha256']]
        metric_scope = 'explicit_hash_pinned_temporal_spd_npz'
    command += [
        '--group-weight', str(group_weight), '--rho', str(rho),
        '--absolute-tolerance', str(absolute_tolerance),
        '--relative-tolerance', str(relative_tolerance),
        '--gap-tolerance', str(gap_tolerance),
        '--max-iterations', str(max_iterations),
    ]
    code = [file_ref(root, root / 'actionmesh/research_math' / name)
            for name in ('__init__.py', 'quadratic_acceleration_control.py',
                         'group_acceleration_candidate.py')]
    environment = {
        'python_executable': sys.executable,
        'python': platform.python_version(),
        'numpy': importlib.metadata.version('numpy'),
        'scope': 'CPU C13 candidate artifact preparation only; no scorer or model',
        'metric_scope': metric_scope,
    }
    environment_digest = hashlib.sha256(json.dumps(
        environment, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    outputs = [
        'actionmesh/c13-group-output/candidate.json',
        'actionmesh/c13-group-output/manifest.json',
        'actionmesh/c13-group-output/group_acceleration/sequence.npz',
        'actionmesh/c13-group-output/group_acceleration/certificate.npz',
        'actionmesh/c13-group-output/group_acceleration/report.json',
    ]
    plan = native.make_plan(
        root, run_id=run_id,
        jobs=[{
            'trial_id': 'prepare-c13-group-acceleration-candidate',
            'command': command, 'cwd': 'actionmesh',
            'input_refs': inputs, 'code_refs': code, 'output_paths': outputs,
            'seed': seed, 'group': 'candidate-artifacts-only',
            'arm_role': 'candidate-artifact-no-scorer-no-admission',
        }],
        provenance={
            'git_revision': 'source files pinned in code_refs; no implicit clean-tree claim',
            'model_revision': 'no model loaded; cached generated sequence identity only',
            'data_revision': inputs[0]['sha256'],
            'environment_digest': environment_digest,
        },
        limits={
            'max_attempts': 1, 'max_development_trials': 1,
            'max_confirmation_trials': 0, 'max_retries_per_trial': 0,
            'wall_time_seconds': wall_seconds,
            'attempt_timeout_seconds': wall_seconds,
        })
    plan_dir.mkdir(parents=True, exist_ok=False)
    native_path = plan_dir / 'native.json'
    native_path.write_text(json.dumps(plan, indent=2) + '\n')
    (plan_dir / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    outer = harness.make_plan(
        root, batch_id=run_id,
        tasks=[{
            'task_id': 'prepare-c13-group-acceleration-candidate',
            'idea_id': '4d-math-20261006-c13', 'depends_on': [], 'priority': 1,
            'plan_ref': file_ref(root, native_path),
            'resources': {
                'cpu_cores': 1, 'ram_mib': 4096, 'gpu_count': 0,
                'gpu_peak_mib': None, 'allow_gpu_share': False,
                'memory_profile_ref': None, 'exclusive_keys': [],
            },
        }],
        limits={
            'total_wall_seconds': wall_seconds + 60,
            'window_seconds': wall_seconds + 60,
            'max_parallel_tasks': 1, 'cpu_cores': 1, 'ram_mib': 4096,
            'max_gpu_task_seconds': 0,
        })
    (plan_dir / 'harness.json').write_text(json.dumps(outer, indent=2) + '\n')
    return plan, outer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'skill-dir', 'source-sequence', 'plan-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--observation-metric', type=Path)
    parser.add_argument('--identity-observation-metric', action='store_true')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--group-weight', type=float, required=True)
    parser.add_argument('--rho', type=float, required=True)
    parser.add_argument('--absolute-tolerance', type=float, required=True)
    parser.add_argument('--relative-tolerance', type=float, required=True)
    parser.add_argument('--gap-tolerance', type=float, required=True)
    parser.add_argument('--max-iterations', type=int, required=True)
    parser.add_argument('--wall-seconds', type=int, required=True)
    args = parser.parse_args()
    if args.identity_observation_metric == (args.observation_metric is not None):
        parser.error('Choose exactly one of --identity-observation-metric or --observation-metric')
    scripts = args.skill_dir.resolve() / 'scripts'
    if not (scripts / 'run_harness.py').is_file():
        parser.error('Full installed research-autopilot skill required')
    sys.path.insert(0, str(scripts))
    _, outer = build_plans(
        args.root, source_sequence=args.source_sequence,
        observation_metric=args.observation_metric, run_id=args.run_id,
        group_weight=args.group_weight, rho=args.rho,
        absolute_tolerance=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
        gap_tolerance=args.gap_tolerance,
        max_iterations=args.max_iterations, plan_dir=args.plan_dir,
        wall_seconds=args.wall_seconds)
    print(json.dumps({
        'plan': str(args.plan_dir.resolve() / 'harness.json'),
        'approved_plan_digest': outer['plan_digest'],
        'execution_started': False, 'candidate_id': '4d-math-20261006-c13',
        'native_qualified': False, 'scientific_admission': False,
        'local_method_verified': False,
    }))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
