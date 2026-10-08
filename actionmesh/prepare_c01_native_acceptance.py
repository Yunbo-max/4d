"""Prepare CPU-only acceptance of retained C01 native artifacts; no execution.

Only metadata, explicit files and hashes are inspected here. The actual context
reconstruction and cross-root staging test run once inside the existing harness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from research_math import self_map_candidate as candidate


SOURCE_NAMES = (
    'research_math/__init__.py', 'research_math/self_map_candidate.py',
    'research_math/c01_native_comparison.py', 'research_math/native_context_delivery.py',
    'research_math/native_context_runner.py', 'research_math/pipeline_decoder_observer.py',
    'research_math/decoder_observer.py', 'research_math/complete_unit_export.py',
    'research_math/tests/__init__.py', 'research_math/tests/test_c01_native_comparison.py',
    'prepare_c01_native_acceptance.py',
)


def file_ref(root, path):
    path = candidate.physical(path)
    return {'path': path.resolve().relative_to(Path(root).resolve()).as_posix(),
            'sha256': candidate.digest(path)}


def artifact_refs(root, candidate_path):
    """Read/hash exact closure without loading tensors or recomputing methods."""
    root = Path(root).resolve()
    candidate_path = candidate.physical(candidate_path)
    candidate_path.resolve().relative_to(root)
    if candidate_path.name != 'candidate.json':
        raise ValueError('Explicit retained candidate.json required')
    output = candidate_path.parent
    result = candidate.load(candidate_path)
    if (result.get('candidate_id') != candidate.CANDIDATE_ID
            or result.get('status') not in ('completed', 'incomplete')
            or result.get('native_qualified') is not False
            or result.get('scientific_admission') is not False):
        raise ValueError('Completed or explicitly rejected unqualified C01 artifact required')
    _, _, _, files = candidate.context_inventory(output / 'context', result['consumption_sha256'])
    certificate = candidate.physical(output / 'certificate.npz')
    files += [candidate_path, candidate.physical(output / 'manifest.json'), certificate]
    for role in candidate.ROLES:
        report_path = candidate.physical(output / role / 'report.json')
        report = candidate.load(report_path)
        if (report.get('candidate_role') != role
                or report.get('candidate_id') != candidate.CANDIDATE_ID
                or report.get('status') not in ('completed', 'error')
                or report.get('sha256', {}).get('certificate.npz') != candidate.digest(certificate)):
            raise ValueError('Current retained C01 role/certificate metadata required')
        files.append(report_path)
        sequence = output / role / 'sequence.npz'
        if report['status'] == 'completed':
            sequence = candidate.physical(sequence)
            if candidate.digest(sequence) != report.get('sha256', {}).get('sequence.npz'):
                raise ValueError('Retained C01 output hash changed')
            files.append(sequence)
        elif sequence.exists() or sequence.is_symlink():
            raise ValueError('Failed role cannot retain a scoreable sequence')
    actual = {path.resolve() for path in output.rglob('*') if path.is_file() or path.is_symlink()}
    if actual != {path.resolve() for path in files}:
        raise ValueError('Artifact contains undeclared files; preserve and inspect before acceptance')
    return [file_ref(root, path) for path in sorted(set(files))]


def prospective_freeze_refs(root, candidate_path, freeze_path):
    """Pin only project-level freeze refs; context-internal refs are not paths here."""
    from research_math import c01_native_comparison as comparison
    root, candidate_path, freeze_path = map(Path, (root, candidate_path, freeze_path))
    output = candidate_path.parent
    freeze = comparison.read_json(candidate.physical(freeze_path))
    comparison._freeze_core(freeze)
    _, basis = comparison._decision(root, freeze)
    source = output / 'context/raw/observed/sequence.npz'
    source_report = source.with_name('report.json')
    actual_report = comparison.read_json(source_report)
    if (freeze['source_sequence_ref'] != file_ref(root, source)
            or freeze['source_report_ref'] != file_ref(root, source_report)
            or freeze['uid'] != actual_report['uid']
            or freeze['inference_seed'] != actual_report['seed']):
        raise ValueError('Prospective freeze must use the supplied C01 artifact source')
    refs = [file_ref(root, freeze_path), freeze['source_sequence_ref'],
            freeze['source_report_ref'], freeze['b_star_decision_ref'], *basis]
    rows = {row['role']: row for row in freeze['roles']}
    b0 = rows['b0']
    if (b0.get('report_ref') != freeze['source_report_ref']
            or b0.get('sequence_ref') != freeze['source_sequence_ref']
            or b0.get('generation_identity_ref') != file_ref(
                root, source.parent.parent / 'generation-identity.json')):
        raise ValueError('Prospective B0 must bind the supplied native producer')
    for role in candidate.ROLES:
        row = rows[role]
        report_path = output / role / 'report.json'
        report = comparison.read_json(report_path)
        sequence_ref = (file_ref(root, output / role / 'sequence.npz')
                        if report['status'] == 'completed' else None)
        if (row.get('report_ref') != file_ref(root, report_path)
                or row.get('sequence_ref') != sequence_ref
                or row.get('certificate_ref') != file_ref(root, output / 'certificate.npz')
                or row.get('method_id') != report.get('method_id')
                or row.get('implementation_ref', {}).get('sha256') != report.get('implementation_sha256')):
            raise ValueError('Prospective method role must bind supplied C01 artifact: ' + role)
    # These are the entire direct root-relative role reference vocabulary.
    # Bundle metadata/manifest JSON has its own relative namespace and is pinned
    # by artifact_refs, never traversed as arbitrary root-relative references.
    for row in freeze['roles']:
        for name in ('report_ref', 'sequence_ref', 'implementation_ref',
                     'certificate_ref', 'generation_identity_ref'):
            if row.get(name) is not None:
                refs.append(row[name])
    unique = {}
    for ref in refs:
        comparison.resolve_ref(root, ref)
        if ref['path'] in unique and unique[ref['path']] != ref:
            raise ValueError('Conflicting prospective freeze input identity')
        unique[ref['path']] = ref
    return list(unique.values())


def build_plans(root, *, skill_dir, artifact_candidate, plan_dir, run_id,
                wall_seconds, ram_mib, freeze=None):
    root, skill_dir, artifact_candidate, plan_dir = (
        Path(path).resolve() for path in (root, skill_dir, artifact_candidate, plan_dir))
    plan_dir.relative_to(root)
    if type(wall_seconds) is not int or not 1 <= wall_seconds <= 26940:
        raise ValueError('Explicit CPU acceptance budget must be 1..26940 seconds')
    if type(ram_mib) is not int or ram_mib < 1:
        raise ValueError('Explicit positive CPU RAM budget required')
    scripts = skill_dir / 'scripts'
    if not (scripts / 'run_harness.py').is_file():
        raise ValueError('Complete installed research-autopilot skill required')
    inputs = artifact_refs(root, artifact_candidate)
    if freeze is not None:
        freeze = Path(freeze).resolve()
        freeze.relative_to(root)
        inputs += prospective_freeze_refs(root, artifact_candidate, freeze)
    unique_inputs = {}
    for ref in inputs:
        if ref['path'] in unique_inputs and unique_inputs[ref['path']] != ref:
            raise ValueError('Conflicting C01 acceptance/freeze input identity')
        unique_inputs[ref['path']] = ref
    inputs = list(unique_inputs.values())
    sources = [root / 'actionmesh' / name for name in SOURCE_NAMES]
    code = [file_ref(root, path) for path in sources]
    # The installed runtime is a library dependency, not scientific input.
    # candidate.json is a separate argv path so the harness actually rewrites it.
    program = (
        'import os,sys,unittest,json\nfrom pathlib import Path\n'
        'sys.path.insert(0,' + repr(str(scripts)) + ')\n'
        'os.environ["RESEARCH_AUTOPILOT_SKILL_DIR"]=' + repr(str(skill_dir)) + '\n'
        'os.environ["C01_NATIVE_ARTIFACT"]=str(Path(sys.argv[1]).resolve().parent)\n'
        'suite=unittest.defaultTestLoader.loadTestsFromName('
        '"research_math.tests.test_c01_native_comparison.C01RetainedNativeComparison")\n'
        'result=unittest.TextTestRunner(verbosity=2).run(suite)\n'
        'if not result.wasSuccessful() or result.skipped: raise SystemExit(1)\n')
    command = [sys.executable, '-c', program, str(artifact_candidate)]
    output_paths = []
    if freeze is not None:
        program += (
            'from research_math import c01_native_comparison as comparison\n'
            'root=Path(comparison.__file__).resolve().parents[2]\n'
            'request=comparison.make_request(root,freeze_path=Path(sys.argv[2]))\n'
            'with Path("c01-comparison-request.json").open("x") as stream:\n'
            '    stream.write(json.dumps(request,indent=2,allow_nan=False)+"\\n")\n')
        command = [sys.executable, '-c', program, str(artifact_candidate), str(freeze)]
        output_paths = ['actionmesh/c01-comparison-request.json']
    sys.path.insert(0, str(scripts))
    import run_experiments as native
    import run_harness as harness
    plan = native.make_plan(root, run_id=run_id, jobs=[{
        'trial_id': 'c01-retained-native-acceptance', 'command': command,
        'cwd': 'actionmesh', 'input_refs': inputs, 'code_refs': code,
        'output_paths': output_paths, 'seed': 42, 'group': 'engineering',
        'arm_role': 'retained-native-context-software-acceptance'}],
        provenance={'git_revision': 'Exact C01 acceptance code_refs; no clean-tree claim',
            'model_revision': 'none; no model/scorer execution',
            'data_revision': candidate.digest(artifact_candidate),
            'environment_digest': hashlib.sha256(json.dumps({
                'python': sys.version, 'installed_skill': str(skill_dir),
                'run_experiments_sha256': candidate.digest(scripts / 'run_experiments.py'),
                'run_harness_sha256': candidate.digest(scripts / 'run_harness.py'),
            }, sort_keys=True).encode()).hexdigest()},
        limits={'max_attempts': 1, 'max_development_trials': 1,
            'max_confirmation_trials': 0, 'max_retries_per_trial': 0,
            'wall_time_seconds': wall_seconds, 'attempt_timeout_seconds': wall_seconds})
    plan_dir.mkdir(parents=True, exist_ok=False)
    native_path = plan_dir / 'native.json'
    native_path.write_text(json.dumps(plan, indent=2) + '\n')
    outer = harness.make_plan(root, batch_id=run_id, tasks=[{
        'task_id': 'c01-retained-native-acceptance', 'idea_id': candidate.CANDIDATE_ID,
        'depends_on': [], 'priority': 1, 'plan_ref': file_ref(root, native_path),
        'resources': {'cpu_cores': 1, 'ram_mib': ram_mib, 'gpu_count': 0,
            'gpu_peak_mib': None, 'allow_gpu_share': False,
            'memory_profile_ref': None, 'exclusive_keys': []}}],
        limits={'total_wall_seconds': wall_seconds + 60,
            'window_seconds': wall_seconds + 60, 'max_parallel_tasks': 1,
            'cpu_cores': 1, 'ram_mib': ram_mib, 'max_gpu_task_seconds': 0})
    (plan_dir / 'harness.json').write_text(json.dumps(outer, indent=2) + '\n')
    return plan, outer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'skill-dir', 'artifact-candidate', 'plan-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--freeze', type=Path)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--wall-seconds', type=int, required=True)
    parser.add_argument('--ram-mib', type=int, required=True)
    args = parser.parse_args()
    _, outer = build_plans(args.root, skill_dir=args.skill_dir,
        artifact_candidate=args.artifact_candidate, plan_dir=args.plan_dir,
        run_id=args.run_id, wall_seconds=args.wall_seconds, ram_mib=args.ram_mib,
        freeze=args.freeze)
    print(json.dumps({'plan': str(args.plan_dir.resolve() / 'harness.json'),
        'approved_plan_digest': outer['plan_digest'], 'execution_started': False,
        'source_delivery_status': 'generated_unexecuted', 'native_qualified': False,
        'local_method_verified': False, 'scientific_admission': False}))


if __name__ == '__main__':
    main()
