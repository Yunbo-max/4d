"""Run C14's frozen physical cases once through the official ActionBench adapter.

This module is an execution/collection component for a separately admitted
research-autopilot native plan.  It does not select B*, generate a method arm,
approve dispatch, qualify the native scorer, compute confidence intervals, or
issue a scientific verdict.  Every logical role remains in the denominator;
aliases share one physical measurement and failures are never zero-imputed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback

import official_actionbench_adapter as official
from research_math import c14_native_comparison as comparison_module


METRICS = ('cd_3d', 'cd_4d', 'cd_motion')
RUNTIME_POLICY = 'strict-cuda-forward-upstream-cpu-knn-backward-v1'
MAX_TOTAL_SECONDS = 27_000
RAW_LIMITS = {
    'max_files': 10_000,
    'max_member_bytes': 1024 * 1024 * 1024,
    'max_unpacked_bytes': 4 * 1024 * 1024 * 1024,
    'max_archive_bytes': 4 * 1024 * 1024 * 1024,
    'max_metadata_bytes': 16 * 1024 * 1024,
}


digest = comparison_module.digest
file_ref = comparison_module.file_ref
resolve_ref = comparison_module.resolve_ref
read_json = comparison_module.read_json
canonical_digest = comparison_module.canonical_digest


def read_json_bytes(value: bytes, label: str) -> dict:
    """Parse retained JSON while rejecting ambiguous duplicate keys."""
    def object_without_duplicates(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key in ' + label + ': ' + key)
            result[key] = item
        return result
    try:
        result = json.loads(value, object_pairs_hook=object_without_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError('Invalid retained JSON: ' + label) from error
    if not isinstance(result, dict):
        raise ValueError('Retained JSON must be an object: ' + label)
    return result


def write_json(path: Path, value: dict, *, exclusive: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open('x') as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        return
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _copy_exact(source: Path, target: Path, expected_sha256: str) -> None:
    source, target = Path(source), Path(target)
    if (source.is_symlink() or not source.is_file()
            or any(parent.is_symlink() for parent in source.parents)):
        raise ValueError('Physical source file required: ' + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as src, target.open('xb') as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    if digest(source) != expected_sha256 or digest(target) != expected_sha256:
        raise ValueError('Evidence changed while staging: ' + str(source))


def stage_cases(root: Path, comparison: dict, raw_root: Path) -> dict:
    """Copy each unique completed physical case into an immutable scorer tree."""
    root, raw_root = Path(root).resolve(), Path(raw_root).resolve()
    raw_root.relative_to(root)
    if raw_root.is_symlink() or any(parent.is_symlink()
                                   for parent in raw_root.parents):
        raise ValueError('Physical raw staging root required')
    cases_root = raw_root / 'cases'
    cases_root.mkdir(exist_ok=False)
    rows = []
    seen = set()
    for case in comparison.get('scoring_cases', []):
        case_id = case.get('case_id')
        if (not isinstance(case_id, str) or not case_id
                or Path(case_id).name != case_id or case_id in ('.', '..')
                or case_id in seen):
            raise ValueError('Unique filesystem-safe physical case ID required')
        if case.get('preparation_status') != 'completed':
            raise ValueError('Only completed physical cases may be staged')
        report = resolve_ref(root, case.get('report_ref'))
        sequence = resolve_ref(root, case.get('sequence_ref'))
        certificate_ref = case.get('certificate_ref')
        certificate = (resolve_ref(root, certificate_ref)
                       if certificate_ref is not None else None)
        poses_ref = case.get('poses_ref')
        poses = (resolve_ref(root, poses_ref) if poses_ref is not None else None)
        directory = cases_root / case_id
        directory.mkdir(exist_ok=False)
        _copy_exact(report, directory / 'report.json', case['report_ref']['sha256'])
        _copy_exact(sequence, directory / 'sequence.npz', case['sequence_ref']['sha256'])
        if certificate is not None:
            _copy_exact(certificate, directory / 'certificate.npz',
                        certificate_ref['sha256'])
        if poses is not None:
            _copy_exact(poses, directory / 'poses.npz', poses_ref['sha256'])
        rows.append({'case_id': case_id, 'uid': comparison['uid'],
                     'case_dir': case_id})
        seen.add(case_id)
    if not rows:
        raise ValueError('At least one completed physical scoring case required')
    return {'cases': rows}


def _validate_dataset_closure(population: dict, admission: dict,
                              semantics: dict, *, uid: str,
                              ground_truth_sha256: str) -> str:
    uids = population.get('uids')
    revision = population.get('revision')
    dataset = admission.get('snapshots', {}).get('dataset', {})
    files = dataset.get('files')
    samples = semantics.get('samples')
    if (population.get('dataset') != 'facebook/actionbench'
            or not isinstance(revision, str) or len(revision) != 40
            or not isinstance(uids, list) or len(uids) != 128
            or len(set(uids)) != 128 or uid not in uids
            or admission.get('kind') != 'actionbench-full128-snapshot-admission'
            or admission.get('version') != '1.0.0'
            or admission.get('status') != 'admitted_engineering_snapshot'
            or admission.get('all_revisions_immutable') is not True
            or admission.get('all_content_files_hashed') is not True
            or admission.get('scientific_effect_qualification') is not False
            or admission.get('dispatch_ready') is not False
            or dataset.get('repository') != 'facebook/actionbench'
            or dataset.get('revision') != revision
            or not isinstance(files, list)
            or hashlib.sha256(json.dumps(
                files, sort_keys=True, separators=(',', ':'),
                ensure_ascii=False, allow_nan=False).encode()).hexdigest()
            != dataset.get('manifest_sha256')
            or semantics.get('kind') !=
            'actionbench-full128-dataset-semantics-admission'
            or semantics.get('version') != '1.0.0'
            or semantics.get('status') != 'admitted_engineering_dataset_semantics'
            or semantics.get('dataset') != 'facebook/actionbench'
            or semantics.get('revision') != revision
            or semantics.get('population_size') != 128
            or semantics.get('frames_per_sample') != 16
            or semantics.get('snapshot_dataset_manifest_sha256') !=
            dataset.get('manifest_sha256')
            or semantics.get('all_consumed_bytes_revalidated') is not True
            or semantics.get('scientific_effect_qualification') is not False
            or semantics.get('dispatch_ready') is not False
            or not isinstance(samples, list) or len(samples) != 128):
        raise ValueError('Canonical admitted ActionBench closure required')
    semantic_by_uid = {row.get('uid'): row for row in samples
                       if isinstance(row, dict)}
    if set(semantic_by_uid) != set(uids):
        raise ValueError('ActionBench semantic UID inventory mismatch')
    expected_gt_path = 'data/' + uid + '/surfaces.npy'
    admitted_by_path = {row.get('path'): row for row in files
                        if isinstance(row, dict)}
    if len(admitted_by_path) != len(files):
        raise ValueError('Unique admitted ActionBench file inventory required')
    gt = admitted_by_path.get(expected_gt_path, {})
    sample = semantic_by_uid[uid]
    if (gt.get('sha256') != ground_truth_sha256
            or sample.get('surfaces', {}).get('shape') != [16, 100000, 6]
            or sample.get('surfaces', {}).get('dtype') != '<f4'
            or sample.get('frames', {}).get('count') != 16):
        raise ValueError('Admitted ActionBench UID/GT semantics mismatch')
    return revision


def make_scoring_request(root: Path, *, comparison_path: Path,
                         ground_truth: Path, population: Path,
                         dataset_admission: Path, dataset_semantics: Path,
                         repo_root: Path, timeout_seconds: int) -> dict:
    """Pin the official scorer closure without granting scientific dispatch."""
    root = Path(root).resolve()
    comparison_path, ground_truth, population, dataset_admission, \
        dataset_semantics, repo_root = map(
        lambda item: Path(item).resolve(),
        (comparison_path, ground_truth, population, dataset_admission,
         dataset_semantics, repo_root))
    frozen = read_json(comparison_path)
    comparison_module.verify_request(root, frozen)
    released = read_json(population)
    if (released.get('dataset') != 'facebook/actionbench'
            or not isinstance(released.get('revision'), str)
            or not released['revision']
            or frozen['uid'] not in released.get('uids', [])):
        raise ValueError('Retained released ActionBench population/UID required')
    if (ground_truth.name != 'surfaces.npy'
            or ground_truth.parent.name != frozen['uid']
            or not ground_truth.is_file() or ground_truth.is_symlink()):
        raise ValueError('Exact released UID/surfaces.npy ground truth required')
    admission = read_json(dataset_admission)
    semantics = read_json(dataset_semantics)
    admitted_revision = _validate_dataset_closure(
        released, admission, semantics, uid=frozen['uid'],
        ground_truth_sha256=digest(ground_truth))
    if admitted_revision != released['revision']:
        raise ValueError('Population/admission revision mismatch')
    if type(timeout_seconds) is not int or timeout_seconds < 1:
        raise ValueError('Positive integer per-case timeout required')
    physical_cases = len(frozen['scoring_cases'])
    if timeout_seconds * physical_cases > MAX_TOTAL_SECONDS:
        raise ValueError('Per-case timeouts exceed the frozen total scoring budget')
    adapter = Path(official.__file__).resolve()
    deterministic = adapter.with_name('deterministic_actionbench_entry.py')
    official_paths = [repo_root / 'actionbench' / name
                      for name in official.OFFICIAL_FILES]
    refs = [file_ref(root, path) for path in official_paths]
    request = {
        'kind': 'c14-native-scoring-request', 'version': 1,
        'candidate_id': comparison_module.CANDIDATE_ID,
        'uid': frozen['uid'], 'inference_seed': frozen['inference_seed'],
        'scoring_seed': 44, 'device': 'cuda:0',
        'metrics': list(METRICS),
        'runtime_policy': RUNTIME_POLICY,
        'timeout_seconds_per_case': timeout_seconds,
        'max_total_seconds': MAX_TOTAL_SECONDS,
        'comparison_ref': file_ref(root, comparison_path),
        'comparison_request_digest': frozen['request_digest'],
        'ground_truth_ref': file_ref(root, ground_truth),
        'population_ref': file_ref(root, population),
        'dataset_admission_ref': file_ref(root, dataset_admission),
        'dataset_semantics_ref': file_ref(root, dataset_semantics),
        'benchmark_revision': released['revision'],
        'repo_root': repo_root.relative_to(root).as_posix(),
        'adapter_ref': file_ref(root, adapter),
        'deterministic_entry_ref': file_ref(root, deterministic),
        'official_source_refs': refs,
        'generated_unexecuted': True,
        'native_qualified': False,
        'scientific_verdict': 'not_computed',
        'dispatch_ready': False,
    }
    request['request_digest'] = canonical_digest(request)
    verify_scoring_request(root, request)
    return request


def verify_scoring_request(root: Path, request: dict) -> dict:
    root = Path(root).resolve()
    expected_keys = {
        'kind', 'version', 'candidate_id', 'uid', 'inference_seed',
        'scoring_seed', 'device', 'metrics', 'runtime_policy',
        'timeout_seconds_per_case', 'max_total_seconds', 'comparison_ref',
        'comparison_request_digest', 'ground_truth_ref', 'population_ref',
        'dataset_admission_ref', 'dataset_semantics_ref',
        'benchmark_revision', 'repo_root', 'adapter_ref',
        'deterministic_entry_ref', 'official_source_refs',
        'generated_unexecuted', 'native_qualified', 'scientific_verdict',
        'dispatch_ready', 'request_digest',
    }
    core = {key: value for key, value in request.items()
            if key != 'request_digest'}
    if request.get('request_digest') != canonical_digest(core):
        raise ValueError('C14 scoring request digest mismatch')
    if (set(request) != expected_keys
            or request.get('kind') != 'c14-native-scoring-request'
            or request.get('version') != 1
            or request.get('candidate_id') != comparison_module.CANDIDATE_ID
            or request.get('scoring_seed') != 44
            or request.get('device') != 'cuda:0'
            or request.get('metrics') != list(METRICS)
            or request.get('runtime_policy') != RUNTIME_POLICY
            or request.get('max_total_seconds') != MAX_TOTAL_SECONDS
            or request.get('generated_unexecuted') is not True
            or request.get('native_qualified') is not False
            or request.get('scientific_verdict') != 'not_computed'
            or request.get('dispatch_ready') is not False):
        raise ValueError('Frozen unexecuted C14 official-scoring closure required')
    frozen = read_json(resolve_ref(root, request.get('comparison_ref')))
    comparison_module.verify_request(root, frozen)
    if (request.get('comparison_request_digest') != frozen['request_digest']
            or request.get('uid') != frozen['uid']
            or request.get('inference_seed') != frozen['inference_seed']):
        raise ValueError('C14 scoring/comparison identity mismatch')
    population = read_json(resolve_ref(root, request.get('population_ref')))
    if (population.get('dataset') != 'facebook/actionbench'
            or population.get('revision') != request.get('benchmark_revision')
            or request['uid'] not in population.get('uids', [])):
        raise ValueError('C14 scoring population identity mismatch')
    ground_truth = resolve_ref(root, request.get('ground_truth_ref'))
    if ground_truth.name != 'surfaces.npy' or ground_truth.parent.name != request['uid']:
        raise ValueError('C14 scoring ground-truth UID mismatch')
    admission = read_json(resolve_ref(root, request.get('dataset_admission_ref')))
    semantics = read_json(resolve_ref(root, request.get('dataset_semantics_ref')))
    if (_validate_dataset_closure(
            population, admission, semantics, uid=request['uid'],
            ground_truth_sha256=digest(ground_truth))
            != request['benchmark_revision']):
        raise ValueError('C14 admitted snapshot/semantics closure mismatch')
    timeout = request.get('timeout_seconds_per_case')
    if (type(timeout) is not int or timeout < 1
            or timeout * len(frozen['scoring_cases']) > MAX_TOTAL_SECONDS):
        raise ValueError('C14 scoring timeout exceeds frozen total budget')
    repo = (root / request.get('repo_root', '')).resolve()
    repo.relative_to(root)
    expected_adapter = Path(official.__file__).resolve()
    expected_deterministic = expected_adapter.with_name(
        'deterministic_actionbench_entry.py')
    if resolve_ref(root, request.get('adapter_ref')) != expected_adapter:
        raise ValueError('Current official adapter source must be pinned')
    if resolve_ref(root, request.get('deterministic_entry_ref')) != expected_deterministic:
        raise ValueError('Current deterministic ActionBench entry must be pinned')
    refs = request.get('official_source_refs')
    if not isinstance(refs, list) or len(refs) != len(official.OFFICIAL_FILES):
        raise ValueError('Complete official ActionBench source closure required')
    expected = [file_ref(root, repo / 'actionbench' / name)
                for name in official.OFFICIAL_FILES]
    if refs != expected:
        raise ValueError('Official ActionBench source closure changed')
    return frozen


def _metric_values(row: dict) -> dict | None:
    if row.get('status') != 'success':
        return None
    values = {}
    for metric in METRICS:
        value = row.get(metric)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            return None
        values[metric] = float(value)
    return values


def build_logical_readout(comparison: dict, official_report: dict) -> dict:
    """Expand unique physical scores to five logical roles without imputation."""
    physical_rows = {row.get('case_id'): row
                     for row in official_report.get('cases', [])
                     if isinstance(row, dict) and isinstance(row.get('case_id'), str)}
    role_lookup = {row['role']: row for row in comparison['roles']}
    logical = []
    for role in comparison_module.ROLES:
        frozen = role_lookup[role]
        row = {'role': role, 'method_id': frozen['method_id'],
               'case_id': frozen.get('case_id')}
        alias = frozen.get('alias_of')
        if alias:
            row.update(alias_of=alias, shared_measurement_with=alias)
        if frozen.get('preparation_status') != 'completed':
            row.update(status='preparation_error',
                       error=frozen.get('preparation_error'))
        else:
            physical = physical_rows.get(frozen.get('case_id'))
            metrics = _metric_values(physical or {})
            if metrics is None:
                row.update(status='scoring_error',
                           error=(physical or {}).get('error',
                                                          'missing physical score'))
            else:
                row.update(status='success', metrics=metrics, n_frames=16)
        logical.append(row)
    group = next(row for row in logical if row['role'] == 'corotational_residual')
    contrasts = []
    if group['status'] == 'success':
        for control_role in ('b0', 'b_star', 'world_gaussian',
                             'body_gaussian'):
            control = next(row for row in logical if row['role'] == control_role)
            if control['status'] == 'success':
                contrasts.append({
                    'candidate_role': 'corotational_residual',
                    'control_role': control_role,
                    'shared_measurement': bool(control.get('alias_of')),
                    'deltas': {metric: group['metrics'][metric]
                               - control['metrics'][metric]
                               for metric in METRICS},
                    'direction': 'corotational residual minus control; lower is better',
                })
    successful = sum(row['status'] == 'success' for row in logical)
    return {
        'logical_denominator': {
            'n_roles': 5, 'roles': list(comparison_module.ROLES),
            'failure_policy': ('All frozen roles remain in the denominator; '
                               'missing/error metrics are never zero-imputed.'),
        },
        'roles': logical,
        'unique_physical_measurements': len({
            row['case_id'] for row in logical
            if row['status'] == 'success' and row['case_id'] is not None}),
        'n_successful_roles': successful,
        'n_failed_or_missing_roles': 5 - successful,
        'contrasts': contrasts,
        'primary_contrast': next((row for row in contrasts
                                  if row['control_role'] == 'b_star'), None),
        'confidence_intervals': None,
        'scientific_verdict': 'not_computed',
        'native_qualified': False,
    }


def _expected_original_hashes(root: Path, request: dict) -> dict:
    return {Path(ref['path']).name: ref['sha256']
            for ref in request['official_source_refs']}


def _expected_official_provenance(root: Path, request: dict) -> dict:
    originals = _expected_original_hashes(root, request)
    paths = {Path(ref['path']).name: resolve_ref(root, ref)
             for ref in request['official_source_refs']}
    old = 'devices=[verts.device], enabled=True'
    new = 'devices=([verts.device] if verts.is_cuda else []), enabled=True'
    sample = paths['sample_mesh.py'].read_text()
    if sample.count(old) != 1:
        raise ValueError('Pinned official CPU RNG source shape changed')
    patched = dict(originals)
    patched['sample_mesh.py'] = hashlib.sha256(
        sample.replace(old, new).encode()).hexdigest()
    return {
        'original_sha256': originals,
        'patched_source_sha256': patched,
        'compatibility_patch': {
            'file': 'sample_mesh.py',
            'function': 'get_baryc_sampling_mesh',
            'old': old, 'new': new,
            'reason': ('CPU surface sampler must not query CUDA RNG state '
                       'for a CPU device'),
            'metric_or_draw_change': False,
        },
    }


def _validate_output_ref(raw_root: Path, ref: dict, label: str) -> Path:
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
        raise ValueError('Exact official output ref required: ' + label)
    raw_root = Path(raw_root).resolve()
    unresolved = Path(ref['path'])
    path = unresolved.resolve()
    path.relative_to(raw_root)
    if (unresolved.is_symlink()
            or any(parent.is_symlink() for parent in unresolved.parents
                   if parent != raw_root.parent)
            or not path.is_file() or digest(path) != ref['sha256']):
        raise ValueError('Changed official output: ' + label)
    return path


def observe_gpu_identity(gpu_uuid: str, raw_root: Path) -> dict:
    command = [
        'nvidia-smi', '--id=' + gpu_uuid,
        '--query-gpu=uuid,name,memory.total', '--format=csv,noheader,nounits',
    ]
    completed = subprocess.run(command, capture_output=True, text=True,
                               check=True, timeout=10)
    values = [value.strip() for value in completed.stdout.strip().split(',')]
    if len(values) != 3 or values[0] != gpu_uuid:
        raise ValueError('Allocated GPU UUID differs from nvidia-smi')
    try:
        memory_mib = float(values[2])
    except ValueError as error:
        raise ValueError('Invalid allocated GPU memory observation') from error
    if not math.isfinite(memory_mib) or memory_mib <= 0:
        raise ValueError('Invalid allocated GPU memory observation')
    record = {
        'kind': 'c14-gpu-identity-observation', 'version': 1,
        'gpu_uuid': gpu_uuid, 'name': values[1],
        'memory_total_mib': memory_mib,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'command': command, 'exit_code': completed.returncode,
        'stdout': completed.stdout, 'stderr': completed.stderr,
        'stdout_sha256': hashlib.sha256(completed.stdout.encode()).hexdigest(),
        'stderr_sha256': hashlib.sha256(completed.stderr.encode()).hexdigest(),
    }
    if record['cuda_visible_devices'] != gpu_uuid:
        raise ValueError('CUDA_VISIBLE_DEVICES differs from observed physical GPU')
    write_json(Path(raw_root)/'gpu-identity.json', record, exclusive=True)
    return record


def _validate_gpu_identity_record(gpu: dict, gpu_uuid: str) -> None:
    keys = {
        'kind', 'version', 'gpu_uuid', 'name', 'memory_total_mib',
        'cuda_visible_devices', 'command', 'exit_code', 'stdout', 'stderr',
        'stdout_sha256', 'stderr_sha256',
    }
    expected_command = [
        'nvidia-smi', '--id=' + gpu_uuid,
        '--query-gpu=uuid,name,memory.total', '--format=csv,noheader,nounits',
    ]
    if (set(gpu) != keys
            or gpu.get('kind') != 'c14-gpu-identity-observation'
            or gpu.get('version') != 1 or gpu.get('gpu_uuid') != gpu_uuid
            or gpu.get('cuda_visible_devices') != gpu_uuid
            or gpu.get('command') != expected_command
            or gpu.get('exit_code') != 0
            or not isinstance(gpu.get('name'), str) or not gpu['name']
            or isinstance(gpu.get('memory_total_mib'), bool)
            or not isinstance(gpu.get('memory_total_mib'), (int, float))
            or not math.isfinite(gpu['memory_total_mib'])
            or gpu['memory_total_mib'] <= 0
            or not isinstance(gpu.get('stdout'), str)
            or not isinstance(gpu.get('stderr'), str)
            or hashlib.sha256(gpu['stdout'].encode()).hexdigest()
            != gpu.get('stdout_sha256')
            or hashlib.sha256(gpu['stderr'].encode()).hexdigest()
            != gpu.get('stderr_sha256')):
        raise ValueError('Physical GPU identity observation mismatch')


def validate_official_report(root: Path, request: dict, comparison: dict,
                             raw_root: Path, report: dict,
                             adapter_exit_code: int | None,
                             gpu_uuid: str) -> dict:
    """Validate the adapter's complete physical denominator and retained bytes."""
    root, raw_root = Path(root).resolve(), Path(raw_root).resolve()
    manifest_path = raw_root / 'manifest.json'
    expected_cases = comparison['scoring_cases']
    if (report.get('schema_version') != 1
            or report.get('device') != request['device']
            or report.get('seed') != request['scoring_seed']
            or report.get('adapter_sha256') != request['adapter_ref']['sha256']):
        raise ValueError('Official adapter schema/device/seed/source mismatch')
    gpu = read_json(raw_root/'gpu-identity.json')
    _validate_gpu_identity_record(gpu, gpu_uuid)
    denominator = report.get('denominator')
    expected_denominator = {
        'frozen': True, 'manifest': str(manifest_path),
        'manifest_sha256': digest(manifest_path),
        'n_declared': len(expected_cases),
    }
    if denominator != expected_denominator:
        raise ValueError('Official physical denominator mismatch')
    source = report.get('official_source')
    expected_provenance = _expected_official_provenance(root, request)
    if source != expected_provenance:
        raise ValueError('Official source/compatibility provenance mismatch')
    executed_source = (raw_root/'official-scores.json.official'/'official-source')
    for name, expected_hash in expected_provenance['patched_source_sha256'].items():
        path = executed_source/name
        if path.is_symlink() or not path.is_file() or digest(path) != expected_hash:
            raise ValueError('Executed official source bytes changed: ' + name)
    rows = report.get('cases')
    if not isinstance(rows, list) or len(rows) != len(expected_cases):
        raise ValueError('Official report changed the physical denominator')
    expected_by_id = {case['case_id']: case for case in expected_cases}
    if [row.get('case_id') for row in rows] != [case['case_id']
                                                for case in expected_cases]:
        raise ValueError('Official cases are missing, extra, or reordered')
    successes = 0
    for row in rows:
        case = expected_by_id[row['case_id']]
        case_dir = raw_root / 'cases' / row['case_id']
        if (row.get('uid') != request['uid']
                or row.get('case_dir') != str(case_dir.resolve())):
            raise ValueError('Official case UID/directory mismatch')
        if row.get('status') == 'error':
            if not isinstance(row.get('error'), str) or not row['error']:
                raise ValueError('Official failure must retain an error')
            continue
        metrics = _metric_values(row)
        if metrics is None or row.get('n_frames') != 16:
            raise ValueError('Official success requires 16 finite metric frames')
        inputs = row.get('inputs')
        expected_sequence = {
            'path': str((case_dir / 'sequence.npz').resolve()),
            'sha256': case['sequence_ref']['sha256'],
        }
        expected_gt = {
            'path': str(resolve_ref(root, request['ground_truth_ref'])),
            'sha256': request['ground_truth_ref']['sha256'],
        }
        if not isinstance(inputs, dict) or inputs.get('sequence') != expected_sequence:
            raise ValueError('Official sequence provenance mismatch')
        if inputs.get('ground_truth') != expected_gt:
            raise ValueError('Official ground-truth provenance mismatch')
        for label in ('csv', 'summary', 'export_manifest', 'execution'):
            expected_name = {
                'csv': 'official.csv', 'summary': 'official.summary.json',
                'export_manifest': 'export-manifest.json',
                'execution': 'execution.json',
            }[label]
            path = _validate_output_ref(
                raw_root, row.get('official_outputs', {}).get(label),
                row['case_id'] + ':' + label)
            expected_path = raw_root/'official-scores.json.official'/row['case_id']/expected_name
            if path != expected_path.resolve():
                raise ValueError('Official output path mismatch: ' + label)
        csv_path = Path(row['official_outputs']['csv']['path'])
        parsed = official.parse_official_csv(csv_path, request['uid'])
        if (parsed['n_frames'] != row['n_frames']
                or any(float(parsed[metric]) != float(row[metric])
                       for metric in METRICS)):
            raise ValueError('Official CSV metrics differ from adapter report')
        summary = read_json(Path(row['official_outputs']['summary']['path']))
        expected_case_summary = {
            'n_total': 1, 'n_success': 1, 'n_failed': 0,
            'success_rate': 1.0,
            **{metric + '_mean': float(row[metric]) for metric in METRICS},
        }
        if summary != expected_case_summary:
            raise ValueError('Official summary metrics differ from adapter report')
        export = read_json(Path(row['official_outputs']['export_manifest']['path']))
        if (export.get('source_sequence') != expected_sequence['path']
                or export.get('source_sequence_sha256') != expected_sequence['sha256']
                or export.get('n_frames') != 16
                or export.get('vertices_dtype') != 'float32'):
            raise ValueError('Official export manifest differs from staged sequence')
        exported = export.get('files')
        if not isinstance(exported, list) or len(exported) != 16:
            raise ValueError('Official export must retain all 16 GLB frames')
        case_work = raw_root/'official-scores.json.official'/row['case_id']
        prediction_root = case_work/'predictions'/request['uid']
        for index, item in enumerate(exported):
            expected_glb = prediction_root/f'mesh_{index:05d}.glb'
            if (not isinstance(item, dict)
                    or set(item) != {'frame_index', 'path', 'sha256', 'bytes'}
                    or item.get('frame_index') != index
                    or item.get('path') != str(expected_glb)
                    or type(item.get('bytes')) is not int
                    or item['bytes'] <= 0
                    or expected_glb.is_symlink() or not expected_glb.is_file()
                    or expected_glb.stat().st_size != item['bytes']
                    or digest(expected_glb) != item.get('sha256')):
                raise ValueError('Official GLB frame inventory mismatch')
        execution_path = Path(row['official_outputs']['execution']['path'])
        execution = read_json(execution_path)
        expected_command = official.official_command(
            executed_source/'evaluate_dataset.py',
            resolve_ref(root, request['ground_truth_ref']).parent.parent,
            case_work/'predictions', case_work/'official.csv',
            request['device'], request['scoring_seed'], True)
        if (execution.get('exit_code') != 0
                or execution.get('command') != expected_command
                or execution.get('cwd') != str(executed_source)):
            raise ValueError('Official execution did not use the pinned deterministic entry')
        for name, key in (('stdout.log', 'stdout_sha256'),
                          ('stderr.log', 'stderr_sha256')):
            log = case_work/name
            if log.is_symlink() or not log.is_file() or digest(log) != execution.get(key):
                raise ValueError('Official execution log differs: ' + name)
        backend = Path(row['official_outputs']['csv']['path']).with_suffix(
            '.backend.json')
        if (not backend.is_file() or backend.is_symlink()
                or row.get('additional_runtime_compatibility') != read_json(backend)):
            raise ValueError('Pinned CPU-kNN-backward receipt missing')
        successes += 1
    expected_summary = {
        'n_total': len(rows), 'n_success': successes,
        'n_failed': len(rows) - successes,
        'success_rate': successes / len(rows),
    }
    if report.get('summary') != expected_summary:
        raise ValueError('Official success/failure summary mismatch')
    expected_exit = 0 if successes == len(rows) else 1
    if adapter_exit_code != expected_exit:
        raise ValueError('Official adapter exit code disagrees with retained cases')
    return report


def _member_name(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or '\\' in value:
        raise ValueError('Canonical POSIX raw member path required')
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ('', '.', '..') for part in path.parts):
        raise ValueError('Nonescaping raw member path required')
    return path


def _raw_inventory(raw_root: Path) -> list[dict]:
    rows = []
    unpacked_bytes = 0
    for path in sorted(Path(raw_root).rglob('*')):
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError('Physical regular raw evidence required: ' + str(path))
        if len(rows) >= RAW_LIMITS['max_files']:
            raise ValueError('C14 raw file count exceeds frozen limit')
        size = path.stat().st_size
        if size > RAW_LIMITS['max_member_bytes']:
            raise ValueError('C14 raw member exceeds frozen limit: ' + str(path))
        unpacked_bytes += size
        if unpacked_bytes > RAW_LIMITS['max_unpacked_bytes']:
            raise ValueError('C14 raw expansion exceeds frozen limit')
        relative = path.relative_to(raw_root).as_posix()
        _member_name(relative)
        rows.append({'path': relative, 'bytes': size,
                     'sha256': digest(path)})
    if not rows:
        raise ValueError('Nonempty raw evidence required')
    return rows


def finalize_raw_bundle(raw_root: Path, output_root: Path, *,
                        request_digest: str,
                        comparison_request_digest: str,
                        excluded_ground_truth_ref: dict) -> dict:
    """Create a deterministic tar plus an external hash inventory."""
    raw_root, output_root = Path(raw_root).resolve(), Path(output_root).resolve()
    raw_root.relative_to(output_root)
    if (raw_root.is_symlink() or output_root.is_symlink()
            or any(parent.is_symlink() for parent in output_root.parents)):
        raise ValueError('Physical C14 bundle root required')
    archive = output_root / 'raw-evidence.tar'
    manifest_path = output_root / 'raw-manifest.json'
    if any(path.exists() or path.is_symlink() for path in (archive, manifest_path)):
        raise FileExistsError('Preserve existing C14 raw bundle')
    rows = _raw_inventory(raw_root)
    estimated_archive_bytes = sum(
        512 + ((row['bytes'] + 511) // 512) * 512
        + (512 + ((len(row['path'].encode()) + 1 + 511) // 512) * 512
           if len(row['path'].encode()) > 100 else 0)
        for row in rows) + 20 * 512
    if estimated_archive_bytes > RAW_LIMITS['max_archive_bytes']:
        raise ValueError('Estimated C14 archive exceeds frozen limit')
    if shutil.disk_usage(output_root).free < estimated_archive_bytes + 1024 * 1024:
        raise ValueError('Insufficient free disk for bounded C14 archive')
    with tarfile.open(archive, mode='x:', format=tarfile.GNU_FORMAT) as handle:
        for row in rows:
            path = raw_root / row['path']
            info = handle.gettarinfo(str(path), arcname=row['path'])
            if not info.isfile() or info.size != row['bytes']:
                raise ValueError('Raw evidence changed during collection')
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            info.mtime = 0
            info.mode = 0o600
            with path.open('rb') as stream:
                handle.addfile(info, stream)
            if handle.fileobj.tell() > RAW_LIMITS['max_archive_bytes']:
                raise ValueError('C14 archive exceeded frozen limit while streaming')
    if archive.stat().st_size > RAW_LIMITS['max_archive_bytes']:
        raise ValueError('C14 archive exceeds frozen limit')
    archive_ref = {'path': 'raw-evidence.tar', 'bytes': archive.stat().st_size,
                   'sha256': digest(archive)}
    bundle_ref = {'archive': archive_ref, 'raw_file_count': len(rows)}
    manifest = {
        'kind': 'c14-native-scoring-raw-bundle', 'version': 1,
        'request_digest': request_digest,
        'comparison_request_digest': comparison_request_digest,
        'files': rows, 'bundle_ref': bundle_ref,
        'excluded_ground_truth_ref': excluded_ground_truth_ref,
        'exclusion_reason': 'Released ActionBench ground truth is hash-bound but not redistributed',
        'validation_limits': dict(RAW_LIMITS),
        'native_qualified': False, 'scientific_verdict': 'not_computed',
    }
    write_json(manifest_path, manifest, exclusive=True)
    validate_delivery_manifest(manifest_path, archive)
    return {
        'archive': archive_ref,
        'manifest': {'path': 'raw-manifest.json',
                     'sha256': digest(manifest_path)},
        'raw_file_count': len(rows),
    }


def validate_delivery_manifest(manifest_path: Path, archive_path: Path) -> dict:
    manifest_path, archive_path = Path(manifest_path), Path(archive_path)
    if (not manifest_path.is_file() or manifest_path.is_symlink()
            or manifest_path.stat().st_size > RAW_LIMITS['max_metadata_bytes']
            or not archive_path.is_file() or archive_path.is_symlink()
            or archive_path.stat().st_size > RAW_LIMITS['max_archive_bytes']):
        raise ValueError('Physical bounded C14 manifest/archive required')
    manifest = read_json(manifest_path)
    manifest_keys = {
        'kind', 'version', 'request_digest', 'comparison_request_digest',
        'files', 'bundle_ref', 'excluded_ground_truth_ref',
        'exclusion_reason', 'validation_limits', 'native_qualified',
        'scientific_verdict',
    }
    if (set(manifest) != manifest_keys
            or manifest.get('kind') != 'c14-native-scoring-raw-bundle'
            or manifest.get('version') != 1
            or not isinstance(manifest.get('request_digest'), str)
            or len(manifest['request_digest']) != 64
            or any(character not in '0123456789abcdef'
                   for character in manifest['request_digest'])
            or not isinstance(manifest.get('comparison_request_digest'), str)
            or len(manifest['comparison_request_digest']) != 64
            or any(character not in '0123456789abcdef'
                   for character in manifest['comparison_request_digest'])
            or manifest.get('exclusion_reason') !=
            'Released ActionBench ground truth is hash-bound but not redistributed'
            or manifest.get('native_qualified') is not False
            or manifest.get('scientific_verdict') != 'not_computed'
            or manifest.get('validation_limits') != RAW_LIMITS):
        raise ValueError('Unqualified C14 raw manifest required')
    rows = manifest.get('files')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Nonempty C14 raw inventory required')
    if len(rows) > RAW_LIMITS['max_files']:
        raise ValueError('C14 raw file count exceeds frozen limit')
    names = [row.get('path') for row in rows]
    if names != sorted(names) or len(names) != len(set(names)):
        raise ValueError('Sorted unique raw inventory required')
    expected_archive = manifest.get('bundle_ref', {}).get('archive')
    if set(manifest.get('bundle_ref', {})) != {'archive', 'raw_file_count'}:
        raise ValueError('Exact C14 raw bundle reference required')
    actual_archive = {'path': 'raw-evidence.tar',
                      'bytes': archive_path.stat().st_size,
                      'sha256': digest(archive_path)}
    if expected_archive != actual_archive:
        raise ValueError('C14 raw archive receipt mismatch')
    if manifest['bundle_ref'].get('raw_file_count') != len(rows):
        raise ValueError('C14 raw member count mismatch')
    gt_ref = manifest.get('excluded_ground_truth_ref')
    if (not isinstance(gt_ref, dict) or set(gt_ref) != {'path', 'sha256'}
            or not isinstance(gt_ref['path'], str) or not gt_ref['path']
            or not isinstance(gt_ref['sha256'], str)
            or len(gt_ref['sha256']) != 64
            or any(character not in '0123456789abcdef'
                   for character in gt_ref['sha256'])):
        raise ValueError('Exact excluded ground-truth ref required')
    unpacked_bytes = 0
    with tarfile.open(archive_path, mode='r:') as handle:
        members = handle.getmembers()
        if [member.name for member in members] != names:
            raise ValueError('C14 archive member inventory mismatch')
        for member, row in zip(members, rows):
            _member_name(member.name)
            if (not isinstance(row, dict)
                    or set(row) != {'path', 'bytes', 'sha256'}
                    or type(row['bytes']) is not int or row['bytes'] < 0
                    or row['bytes'] > RAW_LIMITS['max_member_bytes']
                    or not member.isfile() or member.issym() or member.islnk()
                    or getattr(member, 'sparse', None)
                    or member.size != row['bytes']):
                raise ValueError('C14 archive contains an invalid member')
            unpacked_bytes += row['bytes']
            if unpacked_bytes > RAW_LIMITS['max_unpacked_bytes']:
                raise ValueError('C14 raw expansion exceeds frozen limit')
            stream = handle.extractfile(member)
            if stream is None:
                raise ValueError('C14 archive member unavailable')
            value = hashlib.sha256()
            with stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    value.update(block)
            if value.hexdigest() != row['sha256']:
                raise ValueError('C14 archive member hash mismatch: ' + member.name)
    return manifest


def _extract_validated_archive(archive_path: Path, manifest: dict,
                               destination: Path) -> None:
    destination = Path(destination).resolve()
    inventory = {row['path']: row for row in manifest['files']}
    with tarfile.open(archive_path, mode='r:') as handle:
        for member in handle.getmembers():
            row = inventory[member.name]
            target = destination/_member_name(member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = handle.extractfile(member)
            if stream is None:
                raise ValueError('C14 archive member unavailable')
            with stream, target.open('xb') as output:
                shutil.copyfileobj(stream, output, length=1024 * 1024)
            if (target.stat().st_size != row['bytes']
                    or digest(target) != row['sha256']):
                raise ValueError('C14 extracted member changed')


def _validate_archived_frozen_inputs(manifest: dict, request_path: Path,
                                     request: dict, comparison: dict) -> None:
    inventory = {row['path']: row for row in manifest['files']}
    request_member = inventory.get('scoring-request.json')
    comparison_member = inventory.get('comparison-request.json')
    if (not isinstance(request_member, dict)
            or request_member.get('sha256') != digest(request_path)
            or not isinstance(comparison_member, dict)
            or comparison_member.get('sha256') != request[
                'comparison_ref']['sha256']):
        raise ValueError('Archived request bytes differ from the frozen inputs')
    for case in comparison['scoring_cases']:
        prefix = 'cases/' + case['case_id'] + '/'
        expected = {
            'report.json': case['report_ref'],
            'sequence.npz': case['sequence_ref'],
        }
        if case.get('certificate_ref') is not None:
            expected['certificate.npz'] = case['certificate_ref']
        if case.get('poses_ref') is not None:
            expected['poses.npz'] = case['poses_ref']
        for name, ref in expected.items():
            member = inventory.get(prefix + name)
            if (not isinstance(member, dict)
                    or member.get('sha256') != ref['sha256']):
                raise ValueError('Archived case evidence differs from frozen input: '
                                 + case['case_id'] + '/' + name)


def _relocate_value(value, old_root: Path, new_root: Path):
    if isinstance(value, str):
        old = str(old_root)
        if value == old or value.startswith(old + os.sep):
            return str(new_root) + value[len(old):]
        return value
    if isinstance(value, list):
        return [_relocate_value(item, old_root, new_root) for item in value]
    if isinstance(value, dict):
        return {key: _relocate_value(item, old_root, new_root)
                for key, item in value.items()}
    return value


def _revalidate_archived_official(root: Path, request: dict, comparison: dict,
                                  archive_path: Path, manifest: dict,
                                  report: dict, adapter_exit_code: int | None,
                                  gpu_uuid: str) -> None:
    old_manifest = report.get('denominator', {}).get('manifest')
    if not isinstance(old_manifest, str) or not Path(old_manifest).is_absolute():
        raise ValueError('Archived official denominator root required')
    old_root = Path(old_manifest).parent
    with tempfile.TemporaryDirectory(prefix='c14-delivery-') as directory:
        extracted = Path(directory).resolve()
        _extract_validated_archive(archive_path, manifest, extracted)
        for case in comparison['scoring_cases']:
            case_directory = extracted/'cases'/case['case_id']
            expected = {
                'report.json': case['report_ref'],
                'sequence.npz': case['sequence_ref'],
            }
            if case.get('certificate_ref') is not None:
                expected['certificate.npz'] = case['certificate_ref']
            if case.get('poses_ref') is not None:
                expected['poses.npz'] = case['poses_ref']
            for name, ref in expected.items():
                path = case_directory/name
                if not path.is_file() or path.is_symlink() or digest(path) != ref['sha256']:
                    raise ValueError('Archived physical case input differs: '
                                     + case['case_id'] + '/' + name)
        relocated = _relocate_value(report, old_root, extracted)
        for row in relocated.get('cases', []):
            if row.get('status') != 'success':
                continue
            outputs = row.get('official_outputs', {})
            for label in ('export_manifest', 'execution'):
                path = Path(outputs[label]['path'])
                value = read_json(path)
                value = _relocate_value(value, old_root, extracted)
                write_json(path, value)
                outputs[label]['sha256'] = digest(path)
        validate_official_report(
            root, request, comparison, extracted, relocated,
            adapter_exit_code, gpu_uuid)


def validate_delivery(root: Path, request_path: Path, result_path: Path,
                      manifest_path: Path, archive_path: Path,
                      *, expected_gpu_uuid: str, expected_result_sha256: str,
                      expected_manifest_sha256: str,
                      expected_archive_sha256: str,
                      expected_request_digest: str) -> dict:
    root, request_path = Path(root).resolve(), Path(request_path).resolve()
    request_path.relative_to(root)
    actual_request = read_json(request_path)
    actual_comparison = verify_scoring_request(root, actual_request)
    if (actual_request.get('request_digest') != expected_request_digest
            or not isinstance(expected_gpu_uuid, str)
            or not expected_gpu_uuid.startswith('GPU-')):
        raise ValueError('Current request/GPU identity differs from delivery receipt')
    result_path, manifest_path, archive_path = map(
        Path, (result_path, manifest_path, archive_path))
    if (not result_path.is_file() or result_path.is_symlink()
            or result_path.stat().st_size > RAW_LIMITS['max_metadata_bytes']):
        raise ValueError('Physical bounded C14 result required')
    actual = {
        'result': digest(result_path), 'manifest': digest(manifest_path),
        'archive': digest(archive_path),
    }
    expected = {
        'result': expected_result_sha256,
        'manifest': expected_manifest_sha256,
        'archive': expected_archive_sha256,
    }
    if actual != expected:
        raise ValueError('C14 delivery differs from the pinned receipt')
    result = read_json(result_path)
    manifest = validate_delivery_manifest(manifest_path, archive_path)
    inventory = {row['path']: row for row in manifest['files']}
    _validate_archived_frozen_inputs(
        manifest, request_path, actual_request, actual_comparison)
    expected_bundle = {
        'archive': manifest['bundle_ref']['archive'],
        'manifest': {'path': 'raw-manifest.json',
                     'sha256': actual['manifest']},
        'raw_file_count': len(manifest['files']),
    }
    archived_official_report = None
    with tarfile.open(archive_path, mode='r:') as handle:
        def archived_json(name: str) -> dict:
            try:
                member = handle.getmember(name)
            except KeyError as error:
                raise ValueError('Required C14 raw JSON missing: ' + name) from error
            stream = handle.extractfile(member)
            if stream is None:
                raise ValueError('Required C14 raw JSON unavailable: ' + name)
            with stream:
                return read_json_bytes(stream.read(), name)
        comparison = archived_json('comparison-request.json')
        request = archived_json('scoring-request.json')
        gpu = archived_json('gpu-identity.json')
        execution = archived_json('adapter-execution.json')
        if result.get('status') == 'failed_validation':
            expected_readout = build_logical_readout(comparison, {'cases': []})
            if not isinstance(result.get('validation_error'), str):
                raise ValueError('Failed validation result must retain its error')
            expected_readout['evidence_validation_error'] = result['validation_error']
        else:
            official_report = archived_json('official-scores.json')
            archived_official_report = official_report
            expected_readout = build_logical_readout(comparison, official_report)
            for row in official_report.get('cases', []):
                if row.get('status') != 'success':
                    continue
                case_id = row.get('case_id')
                if case_id not in {case['case_id']
                                   for case in actual_comparison['scoring_cases']}:
                    raise ValueError('Archived official case is outside current request')
                prefix = 'official-scores.json.official/' + case_id + '/'
                for label, filename in (
                        ('csv', 'official.csv'),
                        ('summary', 'official.summary.json'),
                        ('export_manifest', 'export-manifest.json'),
                        ('execution', 'execution.json')):
                    ref = row.get('official_outputs', {}).get(label)
                    member = inventory.get(prefix + filename)
                    if (not isinstance(ref, dict) or not isinstance(member, dict)
                            or ref.get('sha256') != member.get('sha256')):
                        raise ValueError('Archived official output is not byte-bound')
                export = archived_json(prefix + 'export-manifest.json')
                files = export.get('files')
                if not isinstance(files, list) or len(files) != 16:
                    raise ValueError('Archived official export lacks 16 frames')
                for index, exported in enumerate(files):
                    member = inventory.get(
                        prefix + 'predictions/' + request['uid'] +
                        f'/mesh_{index:05d}.glb')
                    if (not isinstance(exported, dict) or not isinstance(member, dict)
                            or exported.get('frame_index') != index
                            or exported.get('sha256') != member.get('sha256')
                            or exported.get('bytes') != member.get('bytes')):
                        raise ValueError('Archived official GLB inventory mismatch')
    if result.get('status') != 'failed_validation':
        if archived_official_report is None:
            raise ValueError('Archived official report required')
        _revalidate_archived_official(
            root, actual_request, actual_comparison, archive_path, manifest,
            archived_official_report, result.get('adapter_exit_code'),
            expected_gpu_uuid)
    expected_status = ('execution_completed_with_failures'
                       if expected_readout['n_failed_or_missing_roles']
                       else 'execution_completed')
    if result.get('status') == 'failed_validation':
        expected_status = 'failed_validation'
    result_keys = {
        'kind', 'version', 'candidate_id', 'uid', 'status', 'gpu_uuid',
        'request_digest', 'comparison_request_digest', 'ground_truth_ref',
        'adapter_exit_code', 'adapter_error', 'validation_error', 'readout',
        'raw_bundle', 'native_qualified', 'scientific_effect_qualification',
        'scientific_verdict', 'dispatch_ready', 'scope',
    }
    if (set(result) != result_keys
            or result.get('kind') != 'c14-native-scoring-result'
            or result.get('version') != 1
            or result.get('candidate_id') != comparison_module.CANDIDATE_ID
            or result.get('uid') != comparison.get('uid')
            or result.get('status') != expected_status
            or (result.get('status') == 'failed_validation')
            != isinstance(result.get('validation_error'), str)
            or result.get('request_digest') != expected_request_digest
            or request.get('request_digest') != expected_request_digest
            or request != actual_request
            or comparison != actual_comparison
            or manifest.get('request_digest') != expected_request_digest
            or request.get('comparison_request_digest')
            != comparison.get('request_digest')
            or result.get('comparison_request_digest')
            != manifest.get('comparison_request_digest')
            or result.get('comparison_request_digest')
            != comparison.get('request_digest')
            or result.get('ground_truth_ref')
            != manifest.get('excluded_ground_truth_ref')
            or request.get('ground_truth_ref') != result.get('ground_truth_ref')
            or result.get('gpu_uuid') != expected_gpu_uuid
            or result.get('gpu_uuid') != gpu.get('gpu_uuid')
            or execution.get('gpu_uuid') != result.get('gpu_uuid')
            or execution.get('exit_code') != result.get('adapter_exit_code')
            or execution.get('error') != result.get('adapter_error')
            or result.get('raw_bundle') != expected_bundle
            or result.get('readout') != expected_readout
            or result.get('native_qualified') is not False
            or result.get('scientific_effect_qualification') is not False
            or result.get('scientific_verdict') != 'not_computed'
            or result.get('dispatch_ready') is not False
            or result.get('scope') !=
            ('One frozen C14 physical scoring pass and receipt-bound raw '
             'collection; no confidence interval, gate, qualification or verdict')):
        raise ValueError('C14 result/raw receipt cross-link mismatch')
    _validate_gpu_identity_record(gpu, result['gpu_uuid'])
    return {'status': 'validated_unqualified',
            'request_digest': expected_request_digest,
            'raw_file_count': len(manifest['files']),
            'archive_sha256': manifest['bundle_ref']['archive']['sha256'],
            'native_qualified': False, 'scientific_verdict': 'not_computed'}


def adapter_command(root: Path, request: dict, raw_root: Path) -> list[str]:
    ground_truth = resolve_ref(root, request['ground_truth_ref'])
    return [
        sys.executable, str(resolve_ref(root, request['adapter_ref'])),
        '--case-dir', str(raw_root/'cases'),
        '--gt-dir', str(ground_truth.parent.parent),
        '--output', str(raw_root/'official-scores.json'),
        '--manifest', str(raw_root/'manifest.json'),
        '--repo-root', str((Path(root)/request['repo_root']).resolve()),
        '--device', request['device'], '--seed', str(request['scoring_seed']),
        '--timeout-seconds', str(request['timeout_seconds_per_case']),
        '--cpu-knn-backward',
    ]


def validate_execution_authorization(root: Path, request_path: Path,
                                     output: Path, gpu_uuid: str,
                                     launch_ticket_path: Path) -> dict:
    """Validate the hash-staged non-circular ticket before scorer/GPU action."""
    root = Path(root).resolve()
    ticket_path = Path(launch_ticket_path).resolve()
    ticket_path.relative_to(root)
    if ticket_path.is_symlink() or not ticket_path.is_file():
        raise ValueError('Physical staged C14 launch ticket required')
    ticket = read_json(ticket_path)
    expected_keys = {
        'kind', 'version', 'authorization_ref', 'authorization_nonce',
        'request_ref', 'method_batch_ref', 'environment_ref', 'run_id',
        'task_id', 'group', 'contract_digest', 'reservation_ref',
        'gpu_uuid', 'expected_output', 'execution_contract', 'issued_at',
        'expires_at', 'ticket_digest',
    }
    request_ref = file_ref(root, request_path)
    expected_contract = {
        'purpose': 'scientific', 'evidence_mode': 'prospective_confirmatory',
        'max_attempts': 1, 'max_confirmation_trials': 1,
        'max_retries_per_trial': 0,
    }
    if (set(ticket) != expected_keys
            or ticket.get('kind') != 'c14-staged-launch-ticket'
            or ticket.get('version') != 1
            or ticket.get('request_ref') != request_ref
            or ticket.get('gpu_uuid') != gpu_uuid
            or ticket.get('expected_output') != output.relative_to(root).as_posix()
            or ticket.get('execution_contract') != expected_contract
            or ticket.get('ticket_digest') != canonical_digest(
                {key: value for key, value in ticket.items()
                 if key != 'ticket_digest'})):
        raise ValueError('Staged C14 launch ticket binding mismatch')
    authorization = read_json(resolve_ref(root, ticket['authorization_ref']))
    reservation = read_json(resolve_ref(root, ticket['reservation_ref']))
    if (authorization.get('authorization_nonce') != ticket.get(
            'authorization_nonce')
            or authorization.get('gpu_uuid') != gpu_uuid
            or authorization.get('status') != 'authorized'
            or authorization.get('request_ref') != request_ref
            or authorization.get('run_id') != ticket.get('run_id')
            or authorization.get('task_id') != ticket.get('task_id')
            or authorization.get('group') != ticket.get('group')
            or authorization.get('contract_digest') != ticket.get(
                'contract_digest')
            or authorization.get('method_batch_ref') != ticket.get(
                'method_batch_ref')
            or authorization.get('environment_ref') != ticket.get(
                'environment_ref')
            or authorization.get('expires_at') != ticket.get('expires_at')
            or reservation.get('authorization_ref') != ticket.get(
                'authorization_ref')
            or reservation.get('request_ref') != request_ref
            or reservation.get('run_id') != ticket.get('run_id')
            or reservation.get('state') != 'reserved'):
        raise ValueError('Staged C14 authorization source/reservation mismatch')
    try:
        expires = datetime.fromisoformat(
            authorization['expires_at'].replace('Z', '+00:00'))
        issued = datetime.fromisoformat(ticket['issued_at'].replace('Z', '+00:00'))
    except (KeyError, AttributeError, ValueError) as error:
        raise ValueError('Timezone-aware C14 authorization times required') from error
    if (expires.tzinfo is None or issued.tzinfo is None
            or issued > expires or datetime.now(timezone.utc) > expires):
        raise ValueError('C14 execution authorization expired')
    controller_root_value = os.environ.get('C14_CONTROLLER_ROOT')
    claim_value = os.environ.get('C14_LAUNCH_CLAIM_PATH')
    claim_sha = os.environ.get('C14_LAUNCH_CLAIM_SHA256')
    consumption_value = os.environ.get('C14_CONSUMPTION_PATH')
    if not all((controller_root_value, claim_value, claim_sha, consumption_value)):
        raise ValueError('C14 finalized controller claim injection required')
    controller_root = Path(controller_root_value).resolve()
    claim_path = Path(claim_value).resolve()
    consumption_path = Path(consumption_value).resolve()
    claim_path.relative_to(controller_root)
    consumption_path.relative_to(controller_root)
    if (claim_path.is_symlink() or consumption_path.is_symlink()
            or not claim_path.is_file() or not consumption_path.is_file()
            or digest(claim_path) != claim_sha):
        raise ValueError('Physical finalized C14 controller claim required')
    claim = read_json(claim_path)
    consumption = read_json(consumption_path)
    if (claim.get('kind') != 'c14-launch-claim'
            or claim.get('claim_digest') != canonical_digest(
                {key: value for key, value in claim.items()
                 if key != 'claim_digest'})
            or claim.get('consumption_ref') != file_ref(
                controller_root, consumption_path)
            or consumption.get('kind') !=
            'c14-gpu-resume-authorization-consumption'
            or consumption.get('consumption_digest') != canonical_digest(
                {key: value for key, value in consumption.items()
                 if key != 'consumption_digest'})
            or consumption.get('authorization_ref') != ticket[
                'authorization_ref']
            or consumption.get('reservation_ref') != ticket['reservation_ref']
            or consumption.get('launch_ticket_ref', {}).get('sha256') != digest(
                ticket_path)
            or consumption.get('request_ref', {}).get('sha256') != digest(
                request_path)
            or claim.get('authorization_ref') != consumption.get(
                'authorization_ref')
            or claim.get('native_plan_ref') != consumption.get('native_plan_ref')
            or claim.get('native_plan_digest') != consumption.get(
                'native_plan_digest')
            or claim.get('harness_plan_ref') != consumption.get(
                'harness_plan_ref')
            or claim.get('harness_plan_digest') != consumption.get(
                'harness_plan_digest')):
        raise ValueError('C14 finalized claim/consumption closure mismatch')
    for ref_key, digest_key in (
            ('native_plan_ref', 'native_plan_digest'),
            ('harness_plan_ref', 'harness_plan_digest')):
        plan = read_json(resolve_ref(controller_root, consumption[ref_key]))
        if (plan.get('plan_digest') != consumption[digest_key]
                or canonical_digest({key: value for key, value in plan.items()
                                     if key != 'plan_digest'})
                != consumption[digest_key]):
            raise ValueError('C14 controller plan digest changed')
    return ticket


def score_request(root: Path, request_path: Path, output: Path,
                  gpu_uuid: str, *, launch_ticket: Path) -> dict:
    """Execute once, retain failures/interruptions, and package every raw byte."""
    root, request_path, output = map(
        lambda item: Path(item).resolve(), (root, request_path, output))
    request_path.relative_to(root)
    output.relative_to(root)
    if (request_path.is_symlink() or not request_path.is_file()
            or any(parent.is_symlink() for parent in output.parents)):
        raise ValueError('Physical in-project request/output paths required')
    validate_execution_authorization(
        root, request_path, output, gpu_uuid, launch_ticket)
    request = read_json(request_path)
    comparison = verify_scoring_request(root, request)
    if (not isinstance(gpu_uuid, str) or not gpu_uuid.startswith('GPU-')
            or any(character in gpu_uuid for character in '\n\r, ')
            or os.environ.get('CUDA_VISIBLE_DEVICES') != gpu_uuid):
        raise ValueError('Outer harness must allocate the exact physical GPU UUID')
    if output.exists() or output.is_symlink():
        raise FileExistsError('Preserve existing C14 scoring output')
    output.mkdir(parents=True)
    raw = output/'raw'; raw.mkdir()
    _copy_exact(request_path, raw/'scoring-request.json', digest(request_path))
    comparison_path = resolve_ref(root, request['comparison_ref'])
    _copy_exact(comparison_path, raw/'comparison-request.json',
                request['comparison_ref']['sha256'])
    manifest = stage_cases(root, comparison, raw)
    write_json(raw/'manifest.json', manifest, exclusive=True)
    observe_gpu_identity(gpu_uuid, raw)
    command = adapter_command(root, request, raw)
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    adapter_exit_code = None
    adapter_error = None
    try:
        with (raw/'adapter.stdout.log').open('x') as stdout, \
                (raw/'adapter.stderr.log').open('x') as stderr:
            completed = subprocess.run(
                command, cwd=root, stdout=stdout, stderr=stderr, check=False,
                timeout=request['max_total_seconds'] + 60)
        adapter_exit_code = completed.returncode
    except (Exception, KeyboardInterrupt) as error:
        adapter_error = type(error).__name__ + ': ' + str(error)
        (raw/'adapter.exception.log').write_text(traceback.format_exc())
    execution = {
        'command': command, 'cwd': str(root), 'gpu_uuid': gpu_uuid,
        'started_at': started_at,
        'completed_at': datetime.now(timezone.utc).isoformat(),
        'elapsed_seconds': time.monotonic() - started,
        'exit_code': adapter_exit_code, 'error': adapter_error,
        'stdout_sha256': digest(raw/'adapter.stdout.log')
        if (raw/'adapter.stdout.log').is_file() else None,
        'stderr_sha256': digest(raw/'adapter.stderr.log')
        if (raw/'adapter.stderr.log').is_file() else None,
    }
    write_json(raw/'adapter-execution.json', execution, exclusive=True)
    official_path = raw/'official-scores.json'
    official_report = {'cases': []}
    validation_error = None
    if official_path.is_file():
        try:
            official_report = read_json(official_path)
            validate_official_report(root, request, comparison, raw,
                                     official_report, adapter_exit_code, gpu_uuid)
            comparison_module.verify_request(root, comparison)
            verify_scoring_request(root, request)
        except Exception as error:
            validation_error = type(error).__name__ + ': ' + str(error)
            (raw/'validation.exception.log').write_text(traceback.format_exc())
    else:
        validation_error = adapter_error or 'Official adapter produced no report'
    if validation_error is not None:
        readout = build_logical_readout(comparison, {'cases': []})
        readout['evidence_validation_error'] = validation_error
        status = 'failed_validation'
    else:
        readout = build_logical_readout(comparison, official_report)
        status = ('execution_completed_with_failures'
                  if readout['n_failed_or_missing_roles']
                  else 'execution_completed')
    bundle = finalize_raw_bundle(
        raw, output, request_digest=request['request_digest'],
        comparison_request_digest=request['comparison_request_digest'],
        excluded_ground_truth_ref=request['ground_truth_ref'])
    result = {
        'kind': 'c14-native-scoring-result', 'version': 1,
        'candidate_id': comparison_module.CANDIDATE_ID,
        'uid': request['uid'], 'status': status,
        'gpu_uuid': gpu_uuid,
        'request_digest': request['request_digest'],
        'comparison_request_digest': request['comparison_request_digest'],
        'ground_truth_ref': request['ground_truth_ref'],
        'adapter_exit_code': adapter_exit_code,
        'adapter_error': adapter_error,
        'validation_error': validation_error,
        'readout': readout, 'raw_bundle': bundle,
        'native_qualified': False,
        'scientific_effect_qualification': False,
        'scientific_verdict': 'not_computed',
        'dispatch_ready': False,
        'scope': ('One frozen C14 physical scoring pass and receipt-bound raw '
                  'collection; no confidence interval, gate, qualification or verdict'),
    }
    write_json(output/'result.json', result, exclusive=True)
    validate_delivery(root, request_path, output/'result.json', output/'raw-manifest.json',
                      output/'raw-evidence.tar',
                      expected_gpu_uuid=gpu_uuid,
                      expected_result_sha256=digest(output/'result.json'),
                      expected_manifest_sha256=digest(output/'raw-manifest.json'),
                      expected_archive_sha256=digest(output/'raw-evidence.tar'),
                      expected_request_digest=request['request_digest'])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    prepare = sub.add_parser('request')
    prepare.add_argument('--root', type=Path, required=True)
    prepare.add_argument('--comparison', type=Path, required=True)
    prepare.add_argument('--ground-truth', type=Path, required=True)
    prepare.add_argument('--population', type=Path, required=True)
    prepare.add_argument('--dataset-admission', type=Path, required=True)
    prepare.add_argument('--dataset-semantics', type=Path, required=True)
    prepare.add_argument('--repo-root', type=Path, required=True)
    prepare.add_argument('--timeout-seconds', type=int, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    execute = sub.add_parser('score')
    execute.add_argument('--root', type=Path, required=True)
    execute.add_argument('--request', type=Path, required=True)
    execute.add_argument('--output', type=Path, required=True)
    execute.add_argument('--gpu-uuid', required=True)
    execute.add_argument('--launch-ticket', type=Path, required=True)
    validate = sub.add_parser('validate-delivery')
    validate.add_argument('--root', type=Path, required=True)
    validate.add_argument('--request', type=Path, required=True)
    validate.add_argument('--gpu-uuid', required=True)
    validate.add_argument('--result', type=Path, required=True)
    validate.add_argument('--manifest', type=Path, required=True)
    validate.add_argument('--archive', type=Path, required=True)
    validate.add_argument('--expected-request-digest', required=True)
    validate.add_argument('--expected-result-sha256', required=True)
    validate.add_argument('--expected-manifest-sha256', required=True)
    validate.add_argument('--expected-archive-sha256', required=True)
    args = parser.parse_args()
    if args.operation == 'request':
        request = make_scoring_request(
            args.root, comparison_path=args.comparison,
            ground_truth=args.ground_truth, population=args.population,
            dataset_admission=args.dataset_admission,
            dataset_semantics=args.dataset_semantics,
            repo_root=args.repo_root, timeout_seconds=args.timeout_seconds)
        output = args.output.resolve(); output.relative_to(args.root.resolve())
        write_json(output, request, exclusive=True)
        print(json.dumps({'request': str(output),
                          'request_digest': request['request_digest'],
                          'execution_started': False,
                          'native_qualified': False,
                          'dispatch_ready': False}))
        return 0
    if args.operation == 'validate-delivery':
        print(json.dumps(validate_delivery(
            args.root, args.request, args.result, args.manifest, args.archive,
            expected_gpu_uuid=args.gpu_uuid,
            expected_result_sha256=args.expected_result_sha256,
            expected_manifest_sha256=args.expected_manifest_sha256,
            expected_archive_sha256=args.expected_archive_sha256,
            expected_request_digest=args.expected_request_digest)))
        return 0
    result = score_request(
        args.root, args.request, args.output, args.gpu_uuid,
        launch_ticket=args.launch_ticket)
    print(json.dumps({'status': result['status'],
                      'native_qualified': False,
                      'scientific_verdict': 'not_computed'}))
    return 0 if result['status'] == 'execution_completed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
