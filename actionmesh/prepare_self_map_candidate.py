"""Emit a one-attempt, zero-GPU C01 artifact harness plan; never execute it."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys


def file_ref(root, path):
    from research_math.self_map_candidate import digest, physical
    path = physical(path)
    return {'path': path.resolve().relative_to(Path(root).resolve()).as_posix(), 'sha256': digest(path)}


def build_plans(root, *, context_root, run_id, plan_dir, wall_seconds,
                coordinate_bounds, bounds_policy, ram_mib):
    import run_experiments as native
    import run_harness as harness
    from research_math.self_map_candidate import digest, context_inventory, METADATA, ROLES
    root, context_root, plan_dir = (Path(p).resolve() for p in (root, context_root, plan_dir))
    context_root.relative_to(root); plan_dir.relative_to(root)
    if type(wall_seconds) is not int or not 1 <= wall_seconds <= 26940:
        raise ValueError('Explicit CPU wall budget must be 1..26940 seconds')
    if type(ram_mib) is not int or ram_mib < 1:
        raise ValueError('Explicit positive CPU RAM admission required')
    import math
    if (len(coordinate_bounds) != 2 or any(type(x) not in (int, float) or not math.isfinite(x)
            for x in coordinate_bounds) or coordinate_bounds[0] >= coordinate_bounds[1]
            or bounds_policy not in ('preserve_and_report', 'reject')):
        raise ValueError('Explicit finite ordered bounds and no-clipping policy required')
    consumption_hash = digest(context_root / METADATA[0])
    consumption, manifest, _, files = context_inventory(context_root, consumption_hash)
    inputs = [file_ref(root, path) for path in files]
    command = [sys.executable, '-m', 'research_math.self_map_candidate',
        '--consumption', str(context_root / METADATA[0]),
        '--manifest', str(context_root / METADATA[1]), '--result', str(context_root / METADATA[2]),
        '--output', 'c01-self-map-output', '--expected-consumption-sha256', consumption_hash,
        '--coordinate-bounds', *(str(v) for v in coordinate_bounds), '--bounds-policy', bounds_policy]
    for row in manifest['files']:
        # A directory argument is not rewritten by _stage. Every actual
        # source is a separate absolute argument and a corresponding input_ref.
        command += ['--raw-file', row['path'], str(context_root / 'raw' / row['path'])]
    names = ('__init__.py', 'self_map_candidate.py', 'native_context_delivery.py',
             'native_context_runner.py', 'pipeline_decoder_observer.py',
             'decoder_observer.py', 'complete_unit_export.py')
    code = [file_ref(root, root / 'actionmesh/research_math' / name) for name in names]
    code.append(file_ref(root, root / 'actionmesh/prepare_self_map_candidate.py'))
    environment = {'python_executable': sys.executable, 'python': platform.python_version(),
        'packages': {name: importlib.metadata.version(name) for name in ('numpy', 'torch', 'safetensors')},
        'scope': 'CPU cached native tensor correction only; no model, scorer or GPU',
        'ram_mib': ram_mib}
    environment_digest = hashlib.sha256(json.dumps(environment, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    output_root = 'actionmesh/c01-self-map-output/'
    outputs = [output_root + name for name in ('candidate.json', 'manifest.json', 'certificate.npz')]
    outputs += [output_root + 'context/' + name for name in METADATA]
    outputs += [output_root + 'context/raw/' + row['path'] for row in manifest['files']]
    outputs += [output_root + role + '/' + name for role in ROLES for name in ('sequence.npz', 'report.json')]
    plan = native.make_plan(root, run_id=run_id, jobs=[{
        'trial_id': 'prepare-c01-self-map-candidate', 'command': command, 'cwd': 'actionmesh',
        'input_refs': inputs, 'code_refs': code, 'output_paths': outputs, 'seed': 42,
        'group': 'candidate-artifacts-only', 'arm_role': 'candidate-artifact-no-scorer-no-admission'}],
        provenance={'git_revision': 'exact code_refs; no clean-tree claim',
            'model_revision': 'cached source-time context only; no model loaded',
            'data_revision': consumption['generation_identity_sha256'],
            'environment_digest': environment_digest},
        limits={'max_attempts': 1, 'max_development_trials': 1, 'max_confirmation_trials': 0,
            'max_retries_per_trial': 0, 'wall_time_seconds': wall_seconds,
            'attempt_timeout_seconds': wall_seconds})
    plan_dir.mkdir(parents=True, exist_ok=False)
    native_path = plan_dir / 'native.json'
    native_path.write_text(json.dumps(plan, indent=2) + '\n')
    (plan_dir / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    outer = harness.make_plan(root, batch_id=run_id, tasks=[{
        'task_id': 'prepare-c01-self-map-candidate', 'idea_id': '4d-math-20261006-c01',
        'depends_on': [], 'priority': 1, 'plan_ref': file_ref(root, native_path),
        'resources': {'cpu_cores': 1, 'ram_mib': ram_mib, 'gpu_count': 0,
            'gpu_peak_mib': None, 'allow_gpu_share': False,
            'memory_profile_ref': None, 'exclusive_keys': []}}],
        limits={'total_wall_seconds': wall_seconds + 60, 'window_seconds': wall_seconds + 60,
            'max_parallel_tasks': 1, 'cpu_cores': 1, 'ram_mib': ram_mib, 'max_gpu_task_seconds': 0})
    (plan_dir / 'harness.json').write_text(json.dumps(outer, indent=2) + '\n')
    return plan, outer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'skill-dir', 'context-root', 'plan-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--wall-seconds', type=int, required=True)
    parser.add_argument('--ram-mib', type=int, required=True)
    parser.add_argument('--coordinate-bounds', type=float, nargs=2, required=True)
    parser.add_argument('--bounds-policy', choices=('preserve_and_report', 'reject'), required=True)
    args = parser.parse_args(argv)
    scripts = args.skill_dir.resolve() / 'scripts'
    if not (scripts / 'run_harness.py').is_file():
        parser.error('Complete installed research-autopilot skill required')
    sys.path.insert(0, str(scripts))
    _, outer = build_plans(args.root, context_root=args.context_root, run_id=args.run_id,
        plan_dir=args.plan_dir, wall_seconds=args.wall_seconds, coordinate_bounds=args.coordinate_bounds,
        bounds_policy=args.bounds_policy, ram_mib=args.ram_mib)
    print(json.dumps({'plan': str(args.plan_dir.resolve() / 'harness.json'),
        'approved_plan_digest': outer['plan_digest'], 'execution_started': False,
        'source_delivery_status': 'generated_unexecuted', 'native_qualified': False,
        'local_method_verified': False, 'scientific_admission': False}))


if __name__ == '__main__':
    main()
