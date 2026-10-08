"""Prepare CPU-only Local engineering acceptance through the existing harness."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT_SOURCES = (
    'research_census_eval.py',
    'prepare_mesh_controls.py',
    'prepare_quadratic_acceleration_control.py',
    'prepare_group_acceleration_candidate.py',
    'prepare_corotational_residual_candidate.py',
    'prepare_protected_geometry_candidate.py',
    'prepare_self_map_candidate.py',
    'prepare_c03_calibration.py',
    'prepare_c03_label_bank.py',
    'prepare_correlated_calibration_candidate.py',
    'prepare_c03_native_acceptance.py',
    'prepare_c03_native_scoring.py',
    'launch_c03_native_scoring.py',
    'prepare_protected_lowrank_candidate.py',
    'prepare_trajectory_bridge_candidate.py',
    'prepare_c08_native_acceptance.py',
    'prepare_c08_native_scoring.py',
    'launch_c08_native_scoring.py',
    'prepare_c15_native_scoring.py',
    'launch_c15_native_scoring.py',
    'prepare_c01_native_scoring.py',
    'prepare_c01_native_acceptance.py',
    'launch_c01_native_scoring.py',
    'prepare_c13_native_scoring.py',
    'prepare_c14_native_scoring.py',
    'prepare_c02_native_scoring.py',
    'prepare_integrable_gradient_candidate.py',
    'prepare_c10_native_scoring.py',
    'prepare_strain_projection_candidate.py',
    'prepare_c11_native_scoring.py',
    'prepare_area_admission_candidate.py',
    'prepare_c12_native_scoring.py',
    'prepare_area_transport_candidate.py',
    'prepare_c06_native_scoring.py',
    'prepare_c06_native_acceptance.py',
    'prepare_partial_transport_candidate.py',
    'prepare_c07_native_scoring.py',
    'prepare_c07_native_acceptance.py',
    'prepare_robust_motion_candidate.py',
    'prepare_c04_native_acceptance.py',
    'prepare_c04_native_scoring.py',
    'launch_c14_native_scoring.py',
    'launch_c02_native_scoring.py',
    'launch_c10_native_scoring.py',
    'launch_c11_native_scoring.py',
    'launch_c12_native_scoring.py',
    'launch_c06_native_scoring.py',
    'launch_c07_native_scoring.py',
    'launch_c04_native_scoring.py',
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
    'observe_actionmesh_generation.py',
    'prepare_native_context.py',
    'prepare_native_context_consumption.py',
    'prepare_g01_acceptance.py',
)

G01_SOURCES = (
    'docs/research-math-20261006/longgoal-20261007/G01_DESIGN.json',
    'docs/research-math-20261006/longgoal-20261007/g01-source-review.json',
    'docs/research-math-20261006/longgoal-20261007/g01-c06-incremental-source-review.json',
    'docs/research-math-20261006/longgoal-20261007/g01-c07-incremental-source-review.json',
    'docs/research-math-20261006/longgoal-20261007/g01-c08-incremental-source-review.json',
    'docs/research-math-20261006/longgoal-20261007/c20-g01-incremental-source-review.json',
    'docs/research-math-20261006/revisions/20261006-mechanism-boundaries/selection.json',
    'docs/research-math-20261006/evidence/exposure-and-contract.json',
    'actionmesh/research_overnight/assets/actionbench_population.json',
    'actionmesh/repo/actionbench/README.md',
    'actionmesh/repo/actionbench/benchmark.py',
    'actionmesh/repo/actionbench/chamfer.py',
    'actionmesh/repo/actionbench/evaluate_dataset.py',
    'actionmesh/repo/actionbench/icp.py',
    'actionmesh/repo/actionbench/sample_mesh.py',
    'actionmesh/repo/actionbench/sample_point_cloud.py',
    'docs/research-math-20261006/actionbench-full128-dataset-semantics-contract.json',
    'docs/research-math-20261006/actionbench-full128-reproduction-contract.json',
    *(
        'docs/research-math-20261006/revisions/20261006-mechanism-boundaries/specs/'
        + candidate + '.json'
        for candidate in (
            '4d-math-20261006-c01', '4d-math-20261006-c02',
            '4d-math-20261006-c03', '4d-math-20261006-c04',
            '4d-math-20261006-c05', '4d-math-20261006-c06',
            '4d-math-20261006-c07', '4d-math-20261006-c08',
            '4d-math-20261006-c10', '4d-math-20261006-c11',
            '4d-math-20261006-c12', '4d-math-20261006-c13',
            '4d-math-20261006-c14', '4d-math-20261006-c15',
            '4d-math-20261006-c20')),
)


def acceptance_sources(root: Path) -> list[Path]:
    """Return the complete import closure exercised by the acceptance suite."""
    root = Path(root).resolve()
    files = sorted((root/'actionmesh/research_math').rglob('*.py'))
    files += [root/'actionmesh'/name for name in ROOT_SOURCES]
    # C02/C04 share the actual ARAP implementation; staged tests must not
    # accidentally import this dependency from a live source checkout.
    files += [root/'actionmesh/research_ten'/name
              for name in ('__init__.py', 'm01_elasticity.py')]
    files.append(root/'scripts/research_supervisor.py')
    files.append(root/'scripts/research_repair_bridge.py')
    files.append(root/'docs/research-math-20261006/longgoal-20261007/CANDIDATE_INPUT_AUDIT.json')
    files += [root/path for path in G01_SOURCES]
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
    # Bind both harness imports and supervisor acceptance to the selected full
    # installed skill; do not inherit an unrelated skill path from the shell.
    program = ('import os,sys,unittest;sys.path.insert(0,'+repr(str(scripts))+');'
        'os.environ["RESEARCH_AUTOPILOT_SKILL_DIR"]='+repr(str(args.skill_dir.resolve()))+';'
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
