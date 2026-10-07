"""Emit a CPU-only plan to capture installed native Conda metadata; never execute."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'skill-dir', 'plan-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    plan_dir = args.plan_dir.resolve()
    plan_dir.relative_to(root)
    if not args.gpu_uuid.startswith('GPU-') or any(c in args.gpu_uuid for c in '\n\r, '):
        parser.error('One controller-observed physical GPU UUID required')
    scripts = args.skill_dir.resolve()/'scripts'
    if not (scripts/'run_harness.py').is_file(): parser.error('Complete installed skill required')
    sys.path.insert(0, str(scripts))
    import run_experiments as native
    import run_harness as harness
    paths = [root/'actionmesh/research_math/__init__.py',
             root/'actionmesh/research_math/native_runtime.py']
    refs = [{'path': path.relative_to(root).as_posix(),
             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]
    # These paths refer to the isolated attempt root, never the controller checkout.
    command = [sys.executable, '-m', 'research_math.native_runtime', '--root', '..',
               '--output', '../inputs/native-runtime', '--gpu-uuid', args.gpu_uuid]
    plan = native.make_plan(root, run_id=args.run_id,
        jobs=[{'trial_id': 'native-runtime-metadata', 'command': command, 'cwd': 'actionmesh',
               'input_refs': [], 'code_refs': refs,
               'output_paths': ['inputs/native-runtime/environment.json',
                                'inputs/native-runtime/dependencies.json'],
               'seed': 0, 'group': 'engineering', 'arm_role': 'software-only'}],
        provenance={'git_revision': 'Actual collector source identities pinned by code_refs',
                    'model_revision': 'none; metadata only',
                    'data_revision': 'none; no native benchmark execution',
                    'environment_digest': hashlib.sha256(sys.version.encode()).hexdigest()},
        limits={'max_attempts': 1, 'max_development_trials': 1, 'max_confirmation_trials': 0,
                'max_retries_per_trial': 0, 'wall_time_seconds': 120, 'attempt_timeout_seconds': 120})
    plan_dir.mkdir(parents=True, exist_ok=False)
    native_path = plan_dir/'native.json'
    native_path.write_text(json.dumps(plan, indent=2)+'\n')
    outer = harness.make_plan(root, batch_id=args.run_id,
        tasks=[{'task_id': 'native-runtime-metadata', 'idea_id': 'baseline-qualification',
                'depends_on': [], 'priority': 1,
                'plan_ref': {'path': native_path.relative_to(root).as_posix(),
                             'sha256': hashlib.sha256(native_path.read_bytes()).hexdigest()},
                'resources': {'cpu_cores': 1, 'ram_mib': 1024, 'gpu_count': 0,
                              'gpu_peak_mib': None, 'allow_gpu_share': False,
                              'memory_profile_ref': None, 'exclusive_keys': []}}],
        limits={'total_wall_seconds': 180, 'window_seconds': 180, 'max_parallel_tasks': 1,
                'cpu_cores': 1, 'ram_mib': 1024, 'max_gpu_task_seconds': 0})
    (plan_dir/'harness.json').write_text(json.dumps(outer, indent=2)+'\n')
    print(json.dumps({'plan': str(plan_dir/'harness.json'),
                      'approved_plan_digest': outer['plan_digest'],
                      'execution_started': False, 'scope': 'Installed environment metadata only'}))
    return 0


if __name__ == '__main__': raise SystemExit(main())
