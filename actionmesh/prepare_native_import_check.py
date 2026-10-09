"""Emit a bounded zero-GPU import diagnostic; do not import native libraries."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'source-root', 'skill-dir', 'plan-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    root, source, plans = args.root.resolve(), args.source_root.resolve(), args.plan_dir.resolve()
    plans.relative_to(root)
    if plans.exists():
        raise FileExistsError('Preserve existing diagnostic plan')
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=source):
        raise ValueError('Clean official source required')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    if revision != 'd5c01f5045df55819e337369c9617f603c667e00':
        raise ValueError('Pinned official ActionMesh source required')
    closure = []
    for prefix in ('actionmesh', 'inference', 'third_party/TripoSG/triposg'):
        for path in sorted((source / prefix).rglob('*.py')):
            if path.is_symlink():
                raise ValueError('Physical source files required')
            closure.append({'path': path.relative_to(source).as_posix(), 'sha256': sha(path)})
    if not closure:
        raise ValueError('Native source closure required')
    plans.mkdir(parents=True)
    staged_source = plans / 'official-source'
    for row in closure:
        target = staged_source / row['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        data = (source / row['path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('Source changed while preparing plan')
        target.write_bytes(data)
    inventory = plans / 'source-inventory.json'
    inventory.write_text(json.dumps({'original_source_root': str(source), 'revision': revision,
                                    'python_executable': sys.executable,
                                    'packages': {name: importlib.metadata.version(name) for name in
                                        ('torch', 'torchvision', 'pytorch3d', 'diffusers', 'transformers', 'diso')},
                                    'files': closure}, indent=2) + '\n')
    def ref(path):
        return {'path': path.relative_to(root).as_posix(), 'sha256': sha(path)}
    scripts = args.skill_dir.resolve() / 'scripts'
    sys.path.insert(0, str(scripts))
    import run_experiments as native
    import run_harness as harness
    task = 'native-import-check'
    command = [sys.executable, '-m', 'research_math.native_import_check',
               '--inventory', str(inventory), '--output', 'native-import-output.json']
    inner = native.make_plan(root, run_id=args.run_id, purpose='engineering', evidence_mode='developmental',
        jobs=[{'trial_id': task, 'command': command, 'cwd': 'actionmesh',
               'input_refs': [ref(inventory)],
               'code_refs': [ref(root / 'actionmesh' / name) for name in (
                   'research_math/__init__.py', 'research_math/native_import_check.py',
                   'prepare_native_import_check.py')] + [ref(staged_source / row['path']) for row in closure],
               'output_paths': ['actionmesh/native-import-output.json'],
               'seed': 0, 'group': 'engineering', 'arm_role': 'software-only'}],
        provenance={'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                    'model_revision': 'none; no model loaded', 'data_revision': revision,
                    'environment_digest': hashlib.sha256(sys.version.encode()).hexdigest()},
        limits={'max_attempts': 1, 'max_development_trials': 1, 'max_confirmation_trials': 0,
                'max_retries_per_trial': 0, 'wall_time_seconds': 180, 'attempt_timeout_seconds': 180})
    native_path = plans / 'native.json'
    native_path.write_text(json.dumps(inner, indent=2) + '\n')
    outer = harness.make_plan(root, batch_id=args.run_id,
        tasks=[{'task_id': task, 'idea_id': 'native-runtime', 'depends_on': [], 'priority': 1,
                'plan_ref': ref(native_path), 'resources': {'cpu_cores': 1, 'ram_mib': 4096,
                'gpu_count': 0, 'gpu_peak_mib': None, 'allow_gpu_share': False,
                'memory_profile_ref': None, 'exclusive_keys': ['native-import-check']}}],
        limits={'total_wall_seconds': 240, 'window_seconds': 240, 'max_parallel_tasks': 1,
                'cpu_cores': 1, 'ram_mib': 4096, 'max_gpu_task_seconds': 0})
    (plans / 'harness.json').write_text(json.dumps(outer, indent=2) + '\n')
    print(json.dumps({'plan': str(plans / 'harness.json'), 'approved_plan_digest': outer['plan_digest'],
                      'execution_started': False, 'gpu_count': 0}))


if __name__ == '__main__':
    main()
