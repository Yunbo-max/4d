"""Plan-only, zero-GPU acceptance of retained real C07 artifacts.

This builder reads metadata and hashes. Reconstruction, numerical certificates
and real cross-root staging are checked only by the emitted Local harness job.
No result, model, scorer or scientific authorization is created here.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

from prepare_control_scoring_checks import acceptance_sources


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _physical(root, path):
    root = Path(root).resolve()
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError('Acceptance inputs must be regular physical files')
    path = path.resolve()
    path.relative_to(root)
    if not path.is_file():
        raise ValueError('Missing acceptance input: ' + str(path))
    return path


def _ref(root, path):
    path = _physical(root, path)
    return {'path': path.relative_to(Path(root).resolve()).as_posix(),
            'sha256': _digest(path)}


def _load(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    result = json.loads(Path(path).read_text(), object_pairs_hook=unique)
    if not isinstance(result, dict):
        raise ValueError('JSON object required')
    return result


def artifact_refs(root, artifact_candidate):
    """Pin exact terminal archive and its materialized members without NumPy."""
    root = Path(root).resolve()
    path = _physical(root, artifact_candidate)
    if path.name != 'candidate.json':
        raise ValueError('Retained candidate.json path required')
    candidate = _load(path)
    if (candidate.get('candidate_id') != '4d-math-20261006-c07'
            or candidate.get('status') not in ('completed', 'incomplete')
            or candidate.get('native_qualified') is not False
            or candidate.get('scientific_admission') is not False):
        raise ValueError('Unqualified terminal C07 artifact required')
    artifact = path.parent
    record_path = _physical(root, artifact / 'artifact-archive.json')
    record = _load(record_path)
    archive = _physical(root, artifact / 'artifact.tar')
    if (record.get('kind') != 'c07-terminal-artifact-archive'
            or record.get('version') != 1
            or record.get('archive') != {'path': 'artifact.tar',
                'sha256': _digest(archive), 'size_bytes': archive.stat().st_size}):
        raise ValueError('Retained archive metadata/hash mismatch')
    members = record.get('members')
    if not isinstance(members, list) or not members or len(members) > 12:
        raise ValueError('Bounded complete C07 artifact inventory required')
    files = [record_path, archive]
    names = set()
    for row in members:
        if not isinstance(row, dict) or set(row) != {'path', 'sha256', 'size_bytes'}:
            raise ValueError('Exact archive member record required')
        name = Path(row['path'])
        if (name.is_absolute() or '..' in name.parts or name.as_posix() != row['path']
                or row['path'] in names):
            raise ValueError('Noncanonical or duplicate artifact member')
        names.add(row['path'])
        member = _physical(root, artifact / name)
        if (type(row['size_bytes']) is not int
                or member.stat().st_size != row['size_bytes']
                or _digest(member) != row['sha256']):
            raise ValueError('Retained artifact member size/hash mismatch')
        files.append(member)
    expected = {'candidate.json', 'manifest.json', 'common-target.npz'}
    roles = ('confidence_threshold_fallback', 'full_mass_transport', 'partial_mass_native_fallback')
    for role in roles:
        report = _load(_physical(root, artifact / role / 'report.json'))
        expected.add(role + '/report.json')
        if report.get('status') == 'completed':
            expected.update({role + '/sequence.npz', role + '/certificate.npz'})
        elif report.get('status') != 'error':
            raise ValueError('Terminal role report required')
    if names != expected:
        raise ValueError('Artifact does not retain the complete terminal role inventory')
    inventory = list(artifact.rglob('*'))
    if any(p.is_symlink() for p in inventory):
        raise ValueError('Symlinks are not accepted in retained artifacts')
    actual = {p.resolve() for p in inventory if p.is_file()}
    if actual != set(files):
        raise ValueError('Undeclared files in retained artifact directory')
    source_refs = candidate.get('source_refs')
    if not isinstance(source_refs, dict) or set(source_refs) != {'sequence', 'report'}:
        raise ValueError('Exact original native sequence/report refs required')
    for ref in source_refs.values():
        if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
            raise ValueError('Exact source reference required')
        source = _physical(root, ref['path'])
        if _ref(root, source) != ref:
            raise ValueError('Native source identity changed')
        files.append(source)
    # The producer provenance is an input, separate from the terminal archive.
    provenance_ref = candidate.get('producer_provenance_ref')
    provenance_path = _checked_ref(root, provenance_ref)
    files.append(provenance_path)
    provenance = _load(provenance_path)
    for ref in provenance.get('code_refs', []):
        files.append(_checked_ref(root, ref))
    return [_ref(root, item) for item in sorted(set(files))]



def _checked_ref(root, ref):
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
        raise ValueError('Exact project path/hash reference required')
    if (not isinstance(ref['path'], str) or Path(ref['path']).is_absolute()
            or '..' in Path(ref['path']).parts):
        raise ValueError('Safe project-relative evidence path required')
    physical = _physical(root, ref['path'])
    if _ref(root, physical) != ref:
        raise ValueError('Prospective evidence identity mismatch: ' + ref['path'])
    return physical


def _nested_refs(value):
    """Traverse containers, never automatically recurse into referenced files."""
    if isinstance(value, dict):
        if set(value) == {'path', 'sha256'}:
            yield value
        else:
            for child in value.values():
                yield from _nested_refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _nested_refs(child)


def prospective_freeze_refs(root, artifact_candidate, freeze_path):
    """Metadata-only closure; artifact-internal paths have a different namespace."""
    freeze_path = _physical(root, freeze_path)
    freeze = _load(freeze_path)
    if freeze.get('candidate_artifact_ref') != _ref(root, artifact_candidate):
        raise ValueError('Freeze must bind the supplied retained C07 artifact')
    candidate = _load(artifact_candidate)
    if (freeze.get('source_sequence_ref') != candidate['source_refs']['sequence']
            or freeze.get('source_report_ref') != candidate['source_refs']['report']):
        raise ValueError('Freeze must bind the supplied original native input pair')
    refs = [_ref(root, freeze_path), *_nested_refs(freeze)]
    decision = _load(_checked_ref(root, freeze['b_star_decision_ref']))
    basis = decision.get('selection_basis_refs')
    if not isinstance(basis, list) or not basis:
        raise ValueError('Prospective B* selection basis required')
    refs.extend(basis)
    if freeze.get('semantic_review_ref') is not None:
        review = _load(_checked_ref(root, freeze['semantic_review_ref']))
        refs.extend(_nested_refs(review))
        for ref in [review.get('method_spec_ref'), *review.get('evidence_refs', [])]:
            if ref is not None:
                # Exactly the same two-level JSON closure as comparison.make_request.
                # A held-out candidate ref is pinned but its internal common_target
                # must not be interpreted as a project-relative filesystem path.
                document = _load(_checked_ref(root, ref))
                refs.extend(_nested_refs(document))
    for ref in refs:
        _checked_ref(root, ref)
    return refs


def official_source_names(root):
    """Read official.OFFICIAL_FILES from its literal source without importing it."""
    adapter = _physical(root, Path(root) / 'actionmesh/official_actionbench_adapter.py')
    tree = ast.parse(adapter.read_text())
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'OFFICIAL_FILES'
                           for target in node.targets)]
    if len(assignments) != 1:
        raise ValueError('One literal official source closure required')
    names = ast.literal_eval(assignments[0].value)
    if (not isinstance(names, (tuple, list)) or not names
            or len(names) != len(set(names))
            or any(not isinstance(name, str) or Path(name).name != name
                   or name in ('.', '..') for name in names)):
        raise ValueError('Canonical official source basenames required')
    return tuple(names)

def build_plans(root, *, skill_dir, artifact_candidate, plan_dir, run_id,
                wall_seconds, ram_mib, freeze=None, ground_truth=None, population=None,
                dataset_admission=None, dataset_semantics=None, repo_root=None,
                timeout_seconds=None):
    root, skill_dir, plan_dir = (Path(p).resolve() for p in (root, skill_dir, plan_dir))
    artifact_candidate = _physical(root, artifact_candidate)
    plan_dir.relative_to(root)
    if type(wall_seconds) is not int or not 1 <= wall_seconds <= 26940:
        raise ValueError('Explicit remaining CPU budget must be 1..26940 seconds')
    if type(ram_mib) is not int or ram_mib <= 0:
        raise ValueError('Explicit positive RAM budget required')
    scripts = skill_dir / 'scripts'
    if not (scripts / 'run_harness.py').is_file():
        raise ValueError('Complete installed research-autopilot required')
    inputs = artifact_refs(root, artifact_candidate)
    request_paths = (ground_truth, population, dataset_admission, dataset_semantics, repo_root)
    wants_scoring_request = any(value is not None for value in (*request_paths, timeout_seconds))
    if wants_scoring_request and (freeze is None or any(value is None for value in request_paths)
                                 or type(timeout_seconds) is not int or timeout_seconds < 1):
        raise ValueError('Scoring request preparation requires freeze and all six scoring arguments')
    if freeze is not None:
        freeze = _physical(root, freeze)
        inputs.extend(prospective_freeze_refs(root, artifact_candidate, freeze))
    scoring_paths = []
    official_paths = []
    if wants_scoring_request:
        scoring_paths = [_physical(root, path) for path in request_paths[:4]]
        repo_root = Path(repo_root)
        if not repo_root.is_absolute():
            repo_root = root / repo_root
        for name in official_source_names(root):
            official_paths.append(_physical(root, repo_root / 'actionbench' / name))
        inputs.extend(_ref(root, path) for path in [*scoring_paths, *official_paths])
    unique = {}
    for ref in inputs:
        if ref['path'] in unique and unique[ref['path']] != ref:
            raise ValueError('Conflicting C07 acceptance/preparation input identity')
        unique[ref['path']] = ref
    inputs = list(unique.values())
    code = [_ref(root, p) for p in acceptance_sources(root)]
    mode = 'request_preparation' if freeze is not None else 'native_method_acceptance'
    program = (
        'import os,sys,unittest,json\nfrom pathlib import Path\n'
        'from research_math import partial_transport_candidate as candidate\n'
        'sys.path.insert(0,' + repr(str(scripts)) + ')\n'
        'os.environ["RESEARCH_AUTOPILOT_SKILL_DIR"]=' + repr(str(skill_dir)) + '\n'
        'root=Path(candidate.__file__).resolve().parents[2]\n')
    if freeze is None:
        program += (
            'os.environ["C07_NATIVE_ROOT"]=str(root)\n'
            'os.environ["C07_NATIVE_ARTIFACT"]=str(Path(sys.argv[1]).resolve())\n'
            'suite=unittest.defaultTestLoader.loadTestsFromName('
            '"research_math.tests.test_partial_transport_candidate.C07RetainedNativeAcceptance")\n'
            'result=unittest.TextTestRunner(verbosity=2).run(suite)\n'
            'if not result.wasSuccessful() or result.skipped or result.testsRun == 0: raise SystemExit(1)\n')
    else:
        # Terminal-integrity verification accepts accurate retained failures.
        # It is deliberately distinct from all-completed method acceptance.
        program += (
            'terminal=candidate.validate_candidate_artifact(root,Path(sys.argv[1]))\n'
            'print(json.dumps({"mode":"request_preparation",'
            '"terminal_status":terminal["status"],"local_method_verified":False}))\n')
    command = [sys.executable, '-c', program, str(artifact_candidate)]
    output_paths = []
    if freeze is not None:
        program += (
            'from research_math import c07_native_comparison as comparison\n'
            'root=Path(candidate.__file__).resolve().parents[2]\n'
            'request=comparison.make_request(root,freeze_path=Path(sys.argv[2]))\n'
            'comparison_path=Path("c07-comparison-request.json")\n'
            'with comparison_path.open("x") as stream: json.dump(request,stream,indent=2,allow_nan=False)\n')
        command.append(str(freeze))
        output_paths.append('actionmesh/c07-comparison-request.json')
    if wants_scoring_request:
        program += (
            'from research_math import c07_native_scoring as scoring\n'
            'request=scoring.make_scoring_request(root,comparison_path=comparison_path,\n'
            '    ground_truth=Path(sys.argv[3]),population=Path(sys.argv[4]),\n'
            '    dataset_admission=Path(sys.argv[5]),dataset_semantics=Path(sys.argv[6]),\n'
            '    repo_root=Path(sys.argv[7]).parent.parent,timeout_seconds=int(sys.argv[8]))\n'
            'with Path("c07-scoring-request.json").open("x") as stream: json.dump(request,stream,indent=2,allow_nan=False)\n')
        # Every physical input is a separate argv token. The directory comes
        # only from the actual staged official file; never from a live root.
        command.extend(str(path) for path in scoring_paths)
        command.extend((str(official_paths[0]), str(timeout_seconds)))
        output_paths.append('actionmesh/c07-scoring-request.json')
    command[2] = program
    sys.path.insert(0, str(scripts))
    import run_experiments as native
    import run_harness as harness
    plan = native.make_plan(root, run_id=run_id, jobs=[{
        'trial_id': 'c07-retained-native-acceptance',
        'command': command,
        'cwd': 'actionmesh', 'input_refs': inputs, 'code_refs': code,
        'output_paths': output_paths, 'seed': 42, 'group': 'engineering',
        'arm_role': mode}],
        provenance={'git_revision': 'Exact current C07 acceptance code_refs',
            'model_revision': 'none; no generation or scoring',
            'data_revision': _digest(artifact_candidate),
            'environment_digest': hashlib.sha256(json.dumps({
                'python': sys.version, 'skill': str(skill_dir),
                'native': _digest(scripts/'run_experiments.py'),
                'harness': _digest(scripts/'run_harness.py'),
            }, sort_keys=True).encode()).hexdigest()},
        limits={'max_attempts': 1, 'max_development_trials': 1,
            'max_confirmation_trials': 0, 'max_retries_per_trial': 0,
            'wall_time_seconds': wall_seconds, 'attempt_timeout_seconds': wall_seconds})
    plan_dir.mkdir(parents=True, exist_ok=False)
    native_path = plan_dir/'native.json'
    native_path.write_text(json.dumps(plan, indent=2)+'\n')
    outer = harness.make_plan(root, batch_id=run_id, tasks=[{
        'task_id': 'c07-retained-native-acceptance', 'idea_id': '4d-math-20261006-c07',
        'depends_on': [], 'priority': 1, 'plan_ref': _ref(root, native_path),
        'resources': {'cpu_cores': 1, 'ram_mib': ram_mib, 'gpu_count': 0,
            'gpu_peak_mib': None, 'allow_gpu_share': False,
            'memory_profile_ref': None, 'exclusive_keys': []}}],
        limits={'total_wall_seconds': wall_seconds+60, 'window_seconds': wall_seconds+60,
            'max_parallel_tasks': 1, 'cpu_cores': 1, 'ram_mib': ram_mib,
            'max_gpu_task_seconds': 0})
    (plan_dir/'harness.json').write_text(json.dumps(outer, indent=2)+'\n')
    return plan, outer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'skill-dir', 'artifact-candidate', 'plan-dir'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--wall-seconds', type=int, required=True)
    parser.add_argument('--ram-mib', type=int, required=True)
    parser.add_argument('--freeze', type=Path)
    for key in ('ground-truth', 'population', 'dataset-admission', 'dataset-semantics', 'repo-root'):
        parser.add_argument('--'+key, type=Path)
    parser.add_argument('--timeout-seconds', type=int)
    args = parser.parse_args()
    _, outer = build_plans(args.root, skill_dir=args.skill_dir,
        artifact_candidate=args.artifact_candidate, plan_dir=args.plan_dir,
        run_id=args.run_id, wall_seconds=args.wall_seconds, ram_mib=args.ram_mib,
        freeze=args.freeze, ground_truth=args.ground_truth, population=args.population,
        dataset_admission=args.dataset_admission, dataset_semantics=args.dataset_semantics,
        repo_root=args.repo_root, timeout_seconds=args.timeout_seconds)
    print(json.dumps({'plan': str(args.plan_dir.resolve()/'harness.json'),
        'approved_plan_digest': outer['plan_digest'], 'execution_started': False,
        'source_delivery_status': 'generated_unexecuted',
        'mode': 'request_preparation' if args.freeze is not None else 'native_method_acceptance',
        'native_qualified': False, 'local_method_verified': False}))


if __name__ == '__main__':
    main()
