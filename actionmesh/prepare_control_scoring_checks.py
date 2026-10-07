"""Prepare CPU-only Local engineering acceptance through the existing harness."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT_SOURCES = (
    'research_census_eval.py',
    'prepare_mesh_controls.py',
    'prepare_control_scoring.py',
    'prepare_control_scoring_checks.py',
    'prepare_native_runtime.py',
    'official_actionbench_adapter.py',
    'prepare_actionbench_parity.py',
    'prepare_actionbench_snapshots.py',
    'prepare_actionbench_dataset_semantics.py',
    'finalize_actionbench_parity.py',
    'prepare_actionbench_parity_finalization.py',
    'prepare_complete_unit_admission.py',
    'prepare_actionbench_queue_pricing.py',
    'prepare_actionbench_full128_window.py',
    'prepare_actionbench_active_batch_snapshot.py',
    'prepare_actionbench_active_batch_reconciliation.py',
    'deterministic_actionbench_entry.py',
)


def acceptance_sources(root: Path) -> list[Path]:
    """Return the complete import closure exercised by the acceptance suite."""
    root = Path(root).resolve()
    files = sorted((root/'actionmesh/research_math').rglob('*.py'))
    files += [root/'actionmesh'/name for name in ROOT_SOURCES]
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing acceptance source: '+str(missing[0]))
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'skill-dir', 'plan-dir'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    root = args.root.resolve(); plan_dir = args.plan_dir.resolve(); plan_dir.relative_to(root)
    scripts = args.skill_dir.resolve()/'scripts'
    if not (scripts/'run_harness.py').is_file(): parser.error('Complete installed skill required')
    sys.path.insert(0, str(scripts))
    import run_experiments as native
    import run_harness as harness
    from research_math.control_scoring import file_ref
    files = acceptance_sources(root)
    refs = [file_ref(root, path) for path in files]
    # Test_control_plan imports the installed harness modules. Carry their actual
    # directory to the child rather than relying on an inherited PYTHONPATH.
    program = ('import sys,unittest;sys.path.insert(0,'+repr(str(scripts))+');'
        'suite=unittest.defaultTestLoader.discover("research_math/tests",pattern="test_*.py");'
        'result=unittest.TextTestRunner(verbosity=2).run(suite);raise SystemExit(not result.wasSuccessful())')
    command = [sys.executable, '-c', program]
    plan = native.make_plan(root, run_id=args.run_id,
        jobs=[{'trial_id': 'all-research-math-checks', 'command': command, 'cwd': 'actionmesh',
               'input_refs': [], 'code_refs': refs, 'output_paths': [],
               'seed': 0, 'group': 'engineering', 'arm_role': 'software-only'}],
        provenance={'git_revision': 'All current software sources pinned by code_refs',
                    'model_revision': 'none; no model or scorer execution',
                    'data_revision': 'engineering fixtures only; no benchmark evidence',
                    'environment_digest': hashlib.sha256(sys.version.encode()).hexdigest()},
        limits={'max_attempts': 1, 'max_development_trials': 1, 'max_confirmation_trials': 0,
                'max_retries_per_trial': 0, 'wall_time_seconds': 120, 'attempt_timeout_seconds': 120})
    plan_dir.mkdir(parents=True, exist_ok=False)
    native_path = plan_dir/'native.json'; native_path.write_text(json.dumps(plan, indent=2)+'\n')
    outer = harness.make_plan(root, batch_id=args.run_id,
        tasks=[{'task_id': 'all-research-math-checks', 'idea_id': 'baseline-qualification',
                'depends_on': [], 'priority': 1, 'plan_ref': file_ref(root, native_path),
                'resources': {'cpu_cores': 1, 'ram_mib': 2048, 'gpu_count': 0, 'gpu_peak_mib': None,
                              'allow_gpu_share': False, 'memory_profile_ref': None, 'exclusive_keys': []}}],
        limits={'total_wall_seconds': 180, 'window_seconds': 180, 'max_parallel_tasks': 1,
                'cpu_cores': 1, 'ram_mib': 2048, 'max_gpu_task_seconds': 0})
    (plan_dir/'harness.json').write_text(json.dumps(outer, indent=2)+'\n')
    print(json.dumps({'plan': str(plan_dir/'harness.json'), 'approved_plan_digest': outer['plan_digest'],
                      'execution_started': False, 'scope': 'software checks only'}))
    return 0


if __name__ == '__main__': raise SystemExit(main())
