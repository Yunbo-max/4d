"""Emit a CPU-only C03 calibration artifact plan; do not execute or admit it."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

from research_math.self_map_candidate import digest, physical, load
from research_math.c03_calibration_artifacts import validate_policy


def file_ref(root, path):
    path = physical(path)
    return {'path': path.resolve().relative_to(root).as_posix(), 'sha256': digest(path)}


def build_plans(root, *, data, policy, evidence_files, plan_dir, run_id,
                wall_seconds, ram_mib):
    import run_experiments as native
    import run_harness as harness
    root = Path(root).resolve(); plan_dir = Path(plan_dir).resolve()
    plan_dir.relative_to(root)
    if type(wall_seconds) is not int or not 1 <= wall_seconds <= 26940:
        raise ValueError('Explicit bounded CPU wall time required')
    if type(ram_mib) is not int or ram_mib < 1:
        raise ValueError('Explicit positive RAM admission required')
    record = validate_policy(load(policy))
    if set(evidence_files) != set(record['correspondence_evidence']):
        raise ValueError('Every actual correspondence evidence file must be staged explicitly')
    if digest(physical(data)) != record['data_sha256']:
        raise ValueError('Development data bytes differ from frozen policy')
    inputs = [file_ref(root, p) for p in (data, policy)]
    command = [sys.executable, '-m', 'research_math.c03_calibration_artifacts',
        '--data', str(Path(data).resolve()), '--policy', str(Path(policy).resolve()),
        '--output', 'c03-calibration-output']
    for name, path in sorted(evidence_files.items()):
        ref = file_ref(root, path)
        if ref['sha256'] != record['correspondence_evidence'][name]:
            raise ValueError('Correspondence source changed: ' + name)
        inputs.append(ref)
        command += ['--evidence-file', name, str(Path(path).resolve())]
    paths = [root / 'actionmesh/research_math' / name for name in
        ('__init__.py', 'correlated_calibration.py', 'c03_calibration_artifacts.py',
         'self_map_candidate.py')]
    paths.append(root / 'actionmesh/prepare_c03_calibration.py')
    environment = {'python': sys.version, 'executable': sys.executable,
        'numpy': importlib.metadata.version('numpy'), 'ram_mib': ram_mib,
        'scope': 'CPU development-surrogate fit only; no native data producer/model/scorer'}
    environment_digest = hashlib.sha256(json.dumps(environment, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    plan = native.make_plan(root, run_id=run_id, jobs=[{
        'trial_id': 'c03-calibration-artifact', 'command': command, 'cwd': 'actionmesh',
        'input_refs': inputs, 'code_refs': [file_ref(root, p) for p in paths],
        'output_paths': ['actionmesh/c03-calibration-output/fit-bundle.json'],
        'seed': 42, 'group': 'candidate-artifacts-only',
        'arm_role': 'development-calibration-no-native-admission'}],
        provenance={'git_revision': 'exact code_refs; no clean-tree claim',
            'model_revision': 'none loaded; precomputed pointwise label bank',
            'data_revision': record['data_sha256'], 'environment_digest': environment_digest},
        limits={'max_attempts': 1, 'max_development_trials': 1, 'max_confirmation_trials': 0,
            'max_retries_per_trial': 0, 'wall_time_seconds': wall_seconds,
            'attempt_timeout_seconds': wall_seconds})
    plan_dir.mkdir(parents=True, exist_ok=False)
    native_path = plan_dir / 'native.json'
    native_path.write_text(json.dumps(plan, indent=2) + '\n')
    (plan_dir / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    outer = harness.make_plan(root, batch_id=run_id, tasks=[{
        'task_id': 'c03-calibration-artifact', 'idea_id': '4d-math-20261006-c03',
        'depends_on': [], 'priority': 1, 'plan_ref': file_ref(root, native_path),
        'resources': {'cpu_cores': 1, 'ram_mib': ram_mib, 'gpu_count': 0,
            'gpu_peak_mib': None, 'allow_gpu_share': False,
            'memory_profile_ref': None, 'exclusive_keys': []}}],
        limits={'total_wall_seconds': wall_seconds + 60, 'window_seconds': wall_seconds + 60,
            'max_parallel_tasks': 1, 'cpu_cores': 1, 'ram_mib': ram_mib,
            'max_gpu_task_seconds': 0})
    (plan_dir / 'harness.json').write_text(json.dumps(outer, indent=2) + '\n')
    return plan, outer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'skill-dir', 'data', 'policy', 'plan-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--evidence-file', nargs=2, action='append', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--wall-seconds', type=int, required=True)
    parser.add_argument('--ram-mib', type=int, required=True)
    args = parser.parse_args(argv)
    scripts = args.skill_dir.resolve() / 'scripts'
    if not (scripts / 'run_harness.py').is_file():
        parser.error('Complete installed research-autopilot skill required')
    sys.path.insert(0, str(scripts))
    evidence = dict(args.evidence_file)
    if len(evidence) != len(args.evidence_file):
        parser.error('Duplicate evidence names forbidden')
    _, outer = build_plans(args.root, data=args.data, policy=args.policy,
        evidence_files=evidence, plan_dir=args.plan_dir, run_id=args.run_id,
        wall_seconds=args.wall_seconds, ram_mib=args.ram_mib)
    print(json.dumps({'plan': str(args.plan_dir.resolve() / 'harness.json'),
        'approved_plan_digest': outer['plan_digest'], 'execution_started': False,
        'source_delivery_status': 'generated_unexecuted', 'scientific_admission': False}))


if __name__ == '__main__':
    main()
