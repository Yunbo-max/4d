"""Emit an unapproved paired-context engineering plan; never execute it.

GPU STOP is not lifted by this builder. It uses the existing single harness and
calibration UID only, with a separate explicit budget for two generations,
replay and byte collection. No Full128 pricing or candidate admission is reused.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from prepare_actionbench_full128_window import validate_environment_closure
from research_math.complete_unit_plan import validate_inventory
from research_math.control_scoring import file_ref
from research_math.native_context_delivery import INPUT_NAMES, receipt_output_paths
from research_math.native_context_runner import generation_settings, tolerance


def build_plan(args):
    root, plan_dir = args.root.resolve(), args.plan_dir.resolve()
    plan_dir.relative_to(root)
    if plan_dir.exists():
        raise FileExistsError('Preserve existing native-context plan')
    if type(args.instrument_wall_seconds) is not int or not 60 <= args.instrument_wall_seconds <= 27000:
        raise ValueError('Separate instrument budget must be 60..27000 seconds')
    for name in ('cpu_cores', 'ram_mib', 'max_capture_bytes'):
        if type(getattr(args, name)) is not int or getattr(args, name) < 1:
            raise ValueError('Positive integer required: ' + name)
    if not args.gpu_uuid.startswith('GPU-') or any(c.isspace() or c == ',' for c in args.gpu_uuid):
        raise ValueError('One physical GPU UUID required')
    atol, rtol = tolerance(args.atol), tolerance(args.rtol)
    records = validate_inventory(args)
    manifest = records['unit_manifest']
    # Pure configuration inspection; no model imports, freeze workload, or device probe.
    generation_settings(manifest['generation'], args.source_root, Path('unused'), Path('unused'))
    if args.environment.resolve() != root / 'inputs/native-runtime/environment.json':
        raise ValueError('Canonical environment path required for attempt-relative dependency verification')
    environment = json.loads(args.environment.read_text())
    env_paths = validate_environment_closure(root, args.environment, environment, args.gpu_uuid)
    for name in ('source_root', 'dataset_root', 'weights_root'):
        path = getattr(args, name)
        if not path.is_absolute() or not path.is_dir():
            raise ValueError('Existing absolute external asset directory required: ' + name)
    scripts = args.skill_dir.resolve() / 'scripts'
    if not (scripts / 'run_harness.py').is_file():
        raise ValueError('Complete installed research-autopilot required')
    sys.path.insert(0, str(scripts))
    import run_experiments as native
    import run_harness as harness
    inputs = [file_ref(root, getattr(args, name)) for name in INPUT_NAMES]
    inputs += [file_ref(root, p) for p in env_paths if p != args.environment.resolve()]
    sources = sorted((root / 'actionmesh/research_math').glob('*.py'))
    sources += [root / 'actionmesh' / name for name in
                ('official_actionbench_adapter.py', 'research_census_eval.py',
                 'deterministic_actionbench_entry.py', 'prepare_native_context.py')]
    code = [file_ref(root, path) for path in sources]
    command = [sys.executable, '-m', 'research_math.native_context_runner']
    for name in INPUT_NAMES:
        command += ['--' + name.replace('_', '-'), str(getattr(args, name).resolve())]
    for name in ('source_root', 'dataset_root', 'weights_root'):
        command += ['--' + name.replace('_', '-'), str(getattr(args, name))]
    command += ['--gpu-uuid', args.gpu_uuid, '--output', 'context-output',
                '--wall-seconds', str(args.instrument_wall_seconds),
                '--instrument-wall-seconds', str(args.instrument_wall_seconds),
                '--atol', str(atol), '--rtol', str(rtol),
                '--max-capture-bytes', str(args.max_capture_bytes)]
    if args.source_time_query:
        command.append('--source-time-query')
    task_id = 'paired-native-context'
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    plan = native.make_plan(root, run_id=args.run_id, purpose='engineering', evidence_mode='developmental',
        jobs=[{'trial_id': task_id, 'command': command, 'cwd': 'actionmesh',
               'input_refs': inputs, 'code_refs': code, 'output_paths': receipt_output_paths(),
               'seed': 42, 'group': 'engineering', 'arm_role': 'paired-native-context-instrument'}],
        provenance={'git_revision': revision, 'model_revision': 'four snapshot-bound current-release models',
                    'data_revision': manifest['population']['revision'],
                    'environment_digest': file_ref(root, args.environment)['sha256']},
        limits={'max_attempts': 1, 'max_development_trials': 1, 'max_confirmation_trials': 0,
                'max_retries_per_trial': 0, 'wall_time_seconds': args.instrument_wall_seconds,
                'attempt_timeout_seconds': args.instrument_wall_seconds})
    plan_dir.mkdir(parents=True, exist_ok=False)
    inner = plan_dir / 'native.json'
    inner.write_text(json.dumps(plan, indent=2) + '\n')
    outer = harness.make_plan(root, batch_id=args.run_id,
        tasks=[{'task_id': task_id, 'idea_id': 'baseline-qualification', 'depends_on': [],
                'priority': 1, 'plan_ref': file_ref(root, inner),
                'resources': {'cpu_cores': args.cpu_cores, 'ram_mib': args.ram_mib,
                    'gpu_count': 1, 'gpu_peak_mib': None, 'allow_gpu_share': False,
                    'memory_profile_ref': None,
                    'exclusive_keys': ['actionbench-complete-unit', 'paired-native-context']}}],
        limits={'total_wall_seconds': args.instrument_wall_seconds,
                'window_seconds': args.instrument_wall_seconds + 1800,
                'max_parallel_tasks': 1, 'cpu_cores': args.cpu_cores, 'ram_mib': args.ram_mib,
                'max_gpu_task_seconds': args.instrument_wall_seconds},
        gpus={'uuids': [args.gpu_uuid], 'safety_margin_mib': 1024, 'max_tasks_per_gpu': 1})
    target = plan_dir / 'harness.json'
    target.write_text(json.dumps(outer, indent=2) + '\n')
    return {'plan': str(target), 'approved_plan_digest': outer['plan_digest'],
            'execution_started': False, 'dispatch_ready': False, 'queue_approved': False,
            'native_context_qualified': False, 'budget_measured': False,
            'scope': 'calibration-UID paired engineering instrument; GPU STOP remains effective'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (*INPUT_NAMES, 'root', 'plan_dir', 'skill_dir', 'source_root', 'dataset_root', 'weights_root'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--instrument-wall-seconds', type=int, required=True)
    parser.add_argument('--atol', type=float, required=True)
    parser.add_argument('--rtol', type=float, required=True)
    parser.add_argument('--max-capture-bytes', type=int, required=True)
    parser.add_argument('--cpu-cores', type=int, required=True)
    parser.add_argument('--ram-mib', type=int, required=True)
    parser.add_argument('--source-time-query', action='store_true')
    print(json.dumps(build_plan(parser.parse_args())))


if __name__ == '__main__':
    main()
