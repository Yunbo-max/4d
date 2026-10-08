"""C01 same-context direct-coordinate correction; authored, not executed on Web.

The frozen native producer supplies one complete 16-frame Stage-II window.
This CPU consumer never calls a model, clamps coordinates or qualifies a method.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import time

import numpy as np

CANDIDATE_ID = '4d-math-20261006-c01'
ROLES = ('raw_uncorrected', 'mean_bias', 'self_map_subtraction')
METADATA = ('bundle-consumption.json', 'bundle-manifest.json', 'bundle-result.json')
G01_GENERATION_SEEDS = (42, 314, 2718)


def validate_native_arrays(arrays):
    from research_math.complete_unit_export import validate_sequence
    expected = validate_sequence(arrays['vertices'], arrays['faces'])
    if (set(arrays) != set(expected) or arrays['vertices'].dtype != np.float32
            or any(not np.array_equal(arrays[k], v) for k, v in expected.items())):
        raise ValueError('Complete float32 native sequence and original identity/time metadata required')
    return arrays


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def physical(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or any(p.is_symlink() for p in path.parents):
        raise ValueError('Physical regular input required: ' + str(path))
    return path


def load(path):
    return json.loads(physical(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def relative_name(value):
    path = PurePosixPath(value)
    if (not isinstance(value, str) or not value or path.is_absolute()
            or '..' in path.parts or path.as_posix() != value or '\\' in value):
        raise ValueError('Canonical relative context path required')
    return value


def correction_arrays(anchor, raw_targets, source_time, *, coordinate_bounds, bounds_policy):
    """s1/s2: full pointwise subtraction, spatial mean control, no clipping.

    Here F_t(X) denotes decoder(context, cast(X), a, t), including the
    captured native query cast and normals. Both calls use that identical
    representation; X is the original native float32 mesh anchor.
    Compute in float64 from the captured decoder values, then export float32.
    The exact native frame-zero coordinates are carried unchanged in every arm.
    No GT error estimate or claim about the sign of the s3 cross term is made.
    """
    anchor = np.asarray(anchor)
    targets = np.asarray(raw_targets)
    source = np.asarray(source_time)
    if (anchor.dtype != np.float32 or anchor.ndim != 2 or anchor.shape[1] != 3
            or len(anchor) < 3 or targets.shape != (15,) + anchor.shape
            or source.shape != anchor.shape
            or any(not np.issubdtype(x.dtype, np.floating) or not np.isfinite(x).all()
                   for x in (anchor, targets, source))):
        raise ValueError('Complete finite native anchor, 15 raw targets and full source query required')
    bounds = np.asarray(coordinate_bounds, dtype=np.float64)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
        raise ValueError('Explicit ordered finite coordinate bounds required')
    if bounds_policy not in ('preserve_and_report', 'reject'):
        raise ValueError('Explicit preserve_and_report or reject policy required; clipping is prohibited')
    x, ft, fa = (a.astype(np.float64) for a in (anchor, targets, source))
    residual = fa - x
    mean = residual.mean(axis=0, dtype=np.float64)
    values = {'raw_uncorrected': ft, 'mean_bias': ft - mean[None, None, :],
              'self_map_subtraction': ft - residual[None, :, :]}
    arms, details = {}, {}
    for role, value in values.items():
        full = np.concatenate([x[None], value], axis=0)
        exported = full.astype(np.float32)
        exported[0] = anchor
        if not np.isfinite(exported).all():
            raise ValueError('Nonfinite float32 export: ' + role)
        outside = (exported < bounds[0]) | (exported > bounds[1])
        details[role] = {
            'min_coordinate': float(exported.min()), 'max_coordinate': float(exported.max()),
            'outside_coordinate_count': int(outside.sum()),
            'outside_vertex_count': int(outside.any(axis=-1).sum()),
            'outside_frame_ids': np.flatnonzero(outside.any(axis=(1, 2))).tolist(),
            'float32_roundoff_linf': float(np.abs(exported.astype(np.float64) - full).max()),
            'anchor_exact': bool(np.array_equal(exported[0], anchor)), 'clipped': False,
            'rejected': bool(bounds_policy == 'reject' and outside.any()),
        }
        arms[role] = exported
    return arms, {'coordinate_bounds': bounds.tolist(), 'bounds_policy': bounds_policy,
                  'self_residual_l2': float(np.linalg.norm(residual)),
                  'mean_residual_xyz': mean.tolist(), 'arms': details,
                  'error_cross_term': 'not estimated; requires native qualified correspondence/error evidence',
                  'source_frame_policy': 'retain exact native frame zero; self-map identity by construction'}


def context_inventory(context, expected_consumption_sha256):
    """Verify the transported raw closure against its immutable consumed manifest."""
    context = Path(context)
    consumption_path = physical(context / METADATA[0])
    if digest(consumption_path) != expected_consumption_sha256:
        raise ValueError('Consumed context receipt hash mismatch')
    c = load(consumption_path)
    if (c.get('status') != 'consumed_unqualified' or c.get('source_time_query') is not True
            or any(c.get(k) is not False for k in ('native_context_qualified',
                'scientific_effect_qualification', 'replay_qualified', 'candidate_methods_tested', 'dispatch_ready'))):
        raise ValueError('Source-time queried, unqualified consumption required')
    m, result = load(context / METADATA[1]), load(context / METADATA[2])
    for filename, key in ((METADATA[1], 'raw-manifest.json'), (METADATA[2], 'result.json')):
        if digest(context / filename) != c.get('input_sha256', {}).get(key):
            raise ValueError('Consumed metadata hash mismatch: ' + filename)
    if (m.get('kind') != 'paired-native-context-raw-bundle' or m.get('version') != 1
            or m.get('source_time_query') is not True
            or m.get('archive', {}).get('sha256') != c.get('input_sha256', {}).get('raw-evidence.tar')):
        raise ValueError('Source-time raw manifest binding required')
    if (result.get('kind') != 'paired-native-context-instrument'
            or result.get('status') != 'completed_unqualified'
            or result.get('all_comparisons_match') is not True
            or result.get('final_integrity') != 'matched'
            or result.get('gpu_uuid') != c.get('producer_gpu_uuid')
            or result.get('source_time_query_requested') is not True):
        raise ValueError('Completed matching producer result required')
    from research_math.native_context_delivery import required_raw_paths, _false_scope, _reject_positive_scope
    _false_scope(result, 'C01 producer')
    _reject_positive_scope(m, 'C01 raw manifest')
    rows = m.get('files')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Complete raw inventory required')
    names = [relative_name(row['path']) for row in rows]
    if (names != sorted(set(names)) or not set(required_raw_paths(source_time_query=True)).issubset(names)
            or c.get('extracted_files') != len(names)):
        raise ValueError('Canonical complete raw closure required')
    files = [context / name for name in METADATA]
    for row in rows:
        path = physical(context / 'raw' / row['path'])
        if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError('Raw context input changed: ' + row['path'])
        files.append(path)
    actual = {p.relative_to(context / 'raw').as_posix() for p in (context / 'raw').rglob('*')
              if p.is_file() or p.is_symlink()}
    if actual != set(names):
        raise ValueError('Unlisted raw context member')
    expected_bundle = {'archive': m['archive'],
        'manifest': {'path': 'raw-manifest.json', 'sha256': digest(context / METADATA[1])},
        'raw_file_count': len(rows)}
    if result.get('raw_bundle') != expected_bundle:
        raise ValueError('Producer raw closure differs')
    return c, m, result, files


def load_context(context, expected_consumption_sha256):
    """Recompute paired/replay semantics from retained real tensors and meshes."""
    import torch
    from safetensors.torch import load_file
    from research_math.native_context_runner import (verify_capture_export, read_sequence,
                                                       compare_sequences, capture_windows)
    from research_math.pipeline_decoder_observer import load_window
    from research_math.decoder_observer import load_capture
    context = Path(context)
    c, manifest, result, files = context_inventory(context, expected_consumption_sha256)
    raw = context / 'raw'
    identity = load(raw / 'generation-identity.json')
    from research_math.native_context_delivery import _reject_positive_scope, _false_scope
    _reject_positive_scope(identity, 'C01 generation identity')
    if (digest(raw / 'generation-identity.json') != c['generation_identity_sha256']
            or digest(raw / 'capture/identity.json') != c['generation_identity_sha256']
            or identity.get('kind') != 'native-context-generation-identity'
            or identity.get('uid') != c['producer_uid']
            or identity.get('gpu_uuid') != c['producer_gpu_uuid']):
        raise ValueError('Same producer/generation/capture identity required')
    pair = load(raw / 'paired-comparison.json')
    actual_pair = compare_sequences(raw / 'unobserved/sequence.npz', raw / 'observed/sequence.npz',
                                    atol=pair['atol'], rtol=pair['rtol'])
    if pair != actual_pair or pair.get('matches') is not True or result.get('paired_comparison') != pair:
        raise ValueError('Paired full native sequence mismatch')
    verify_capture_export(raw / 'capture', raw / 'observed/sequence.npz')
    window = capture_windows(raw / 'capture')[0]
    meshes, window_record = load_window(window)
    captured, decoder_record = load_capture(window / 'decoder/call-0000')
    replay = raw / 'replay/window-0000'
    report = load(replay / 'report.json')
    _reject_positive_scope(report, 'C01 replay report')
    if (report.get('kind') != 'native-window-replay' or report.get('version') != 1
            or report.get('status') != 'replayed_unqualified'
            or report.get('raw_matches') is not True or report.get('mesh_matches') is not True
            or report.get('source_time_requested') is not True
            or report.get('window_record_sha256') != digest(window / 'record.json')
            or report.get('identity_sha256') != c['generation_identity_sha256']
            or report.get('source_time_sha256') != digest(replay / 'source-time.safetensors')
            or report.get('replay_sha256') != digest(replay / 'replay.safetensors')
            or decoder_record.get('step_callback_present') or window_record.get('callback_present')):
        raise ValueError('Matched same-context source-time replay required')
    from research_math.pipeline_decoder_observer import _comparison, _tolerance
    atol, rtol = _tolerance(report['atol']), _tolerance(report['rtol'])
    tensors = load_file(str(replay / 'replay.safetensors'), device='cpu')
    if set(tensors) != {'raw', 'bounded'}:
        raise ValueError('Exact replay fields required')
    raw_check = _comparison(tensors['raw'], captured['output'], atol=atol, rtol=rtol)
    mesh_check = _comparison(tensors['bounded'][0], meshes['vertices'], atol=atol, rtol=rtol)
    if (raw_check != report.get('raw') or mesh_check != report.get('mesh')
            or raw_check['matches'] is not True or raw_check.get('dtype_matches') is not True
            or mesh_check['matches'] is not True
            or not torch.equal(tensors['bounded'], tensors['raw'].clamp(-1, 1))):
        raise ValueError('Replay coordinate comparisons do not recompute')
    stage = load(raw / 'replay/stage-result.json')
    _false_scope(stage, 'C01 replay stage')
    if (stage.get('status') != 'completed_unqualified' or stage.get('all_comparisons_match') is not True
            or stage.get('windows') != [report] or result.get('replay_reports') != [report]
            or stage.get('paired_comparison') != pair):
        raise ValueError('Replay stage/report/result binding differs')
    source = load_file(str(replay / 'source-time.safetensors'), device='cpu')
    if (set(source) != {'output'} or source['output'].shape != (1, 1, len(meshes['anchor_vertices']), 3)
            or source['output'].dtype != captured['output'].dtype
            or not torch.isfinite(source['output']).all().item()):
        raise ValueError('Finite same-dtype complete raw source-time field required')
    sequence_path = raw / 'observed/sequence.npz'
    sequence = read_sequence(sequence_path)
    source_report = load(sequence_path.with_name('report.json'))
    report_seed = source_report.get('seed')
    identity_seed = identity.get('generation', {}).get('seed')
    if (source_report.get('status') != 'completed' or source_report.get('uid') != c['producer_uid']
            or source_report.get('sha256', {}).get('sequence.npz') != digest(sequence_path)
            or type(report_seed) is not int or report_seed not in G01_GENERATION_SEEDS
            or identity_seed != report_seed):
        raise ValueError('Completed original source sequence report required')
    values = {'anchor': sequence['vertices'][0],
              'raw_targets': captured['output'][0].double().numpy(),
              'source_time': source['output'][0, 0].double().numpy(),
              'query_xyz': captured['query'][0, :, :3].double().numpy()}
    return sequence, values, files, source_report


def _copy_file(source, target, expected, expected_bytes):
    source = physical(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.stat().st_size != expected_bytes:
        raise ValueError('Snapshot source size differs')
    with source.open('rb') as reader, target.open('xb') as writer:
        remaining = expected_bytes
        while remaining:
            block = reader.read(min(1024 * 1024, remaining))
            if not block:
                raise ValueError('Snapshot input truncated')
            writer.write(block)
            remaining -= len(block)
        if reader.read(1):
            raise ValueError('Snapshot input grew')
    if digest(target) != expected:
        raise ValueError('Input changed during snapshot: ' + str(source))


def export_self_map_candidate(consumption_path, manifest_path, result_path, raw_files, output, *,
                              expected_consumption_sha256, coordinate_bounds, bounds_policy):
    """Harness inner operation; every input is an explicit staged regular file."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {'candidate_id': CANDIDATE_ID, 'status': 'started',
              'source_delivery_status': 'generated_unexecuted_at_authoring',
              'native_qualified': False, 'scientific_admission': False,
              'candidate_methods_tested': False, 'local_method_verified': False,
              'coordinate_bounds': list(coordinate_bounds), 'bounds_policy': bounds_policy,
              'consumption_sha256': expected_consumption_sha256, 'roles': list(ROLES)}
    try:
        c = load(consumption_path)
        if digest(consumption_path) != expected_consumption_sha256:
            raise ValueError('Consumption hash differs')
        manifest = load(manifest_path)
        limits = c['limits']
        rows = manifest['files']
        if (len(rows) > limits['max_files']
                or any(type(row.get('bytes')) is not int or not 0 <= row['bytes'] <= limits['max_member_bytes'] for row in rows)
                or sum(row['bytes'] for row in rows) > limits['max_unpacked_bytes']):
            raise ValueError('Context snapshot exceeds frozen consumption limits')
        metadata_bytes = [physical(p).stat().st_size for p in (consumption_path, manifest_path, result_path)]
        if any(n > limits['max_metadata_bytes'] for n in metadata_bytes):
            raise ValueError('Context metadata exceeds frozen limit')
        # Snapshot + three exported sequences/certificate. This conservative
        # bound reserves eight times raw tensor/archive size, not a guessed mesh.
        required_free = 8 * sum(row['bytes'] for row in rows) + sum(metadata_bytes) + 1024 * 1024
        if shutil.disk_usage(output).free < required_free:
            raise ValueError('Insufficient free disk for snapshot and complete C01 artifacts')
        context = output / 'context'
        for source, name, expected in (
                (consumption_path, METADATA[0], expected_consumption_sha256),
                (manifest_path, METADATA[1], c['input_sha256']['raw-manifest.json']),
                (result_path, METADATA[2], c['input_sha256']['result.json'])):
            _copy_file(source, context / name, expected, physical(source).stat().st_size)
        names = [relative_name(row['path']) for row in rows]
        if names != sorted(set(names)) or set(raw_files) != set(names):
            raise ValueError('Exact explicit raw-file staging closure required')
        for row in rows:
            _copy_file(raw_files[row['path']], context / 'raw' / row['path'], row['sha256'], row['bytes'])
        sequence, values, _, source_report = load_context(context, expected_consumption_sha256)
        arms, diagnostics = correction_arrays(values['anchor'], values['raw_targets'],
            values['source_time'], coordinate_bounds=coordinate_bounds, bounds_policy=bounds_policy)
        certificate = dict(values, self_residual=values['source_time'] - values['anchor'].astype(np.float64),
                           mean_residual=(values['source_time'] - values['anchor']).mean(axis=0))
        np.savez_compressed(output / 'certificate.npz', **certificate)
        result['diagnostics'] = diagnostics
        result['query_cast_linf'] = float(np.max(np.abs(values['query_xyz'] - values['anchor'])))
        result['uid'], result['seed'] = source_report['uid'], source_report['seed']
        source_path = context / 'raw/observed/sequence.npz'
        cases = []
        for role in ROLES:
            arm = output / role; arm.mkdir()
            rejected = diagnostics['arms'][role]['rejected']
            report = {'status': 'error' if rejected else 'completed',
                'candidate_id': CANDIDATE_ID, 'candidate_role': role,
                'method_id': CANDIDATE_ID if role == 'self_map_subtraction' else 'c01-control-' + role,
                'uid': result['uid'], 'seed': result['seed'],
                'source_sequence_sha256': digest(source_path),
                'source_report_sha256': digest(source_path.with_name('report.json')),
                'implementation_sha256': digest(__file__),
                'certificate_sha256': digest(output / 'certificate.npz'),
                'sha256': {'certificate.npz': digest(output / 'certificate.npz')},
                'coordinate_bounds': list(coordinate_bounds), 'bounds_policy': bounds_policy,
                'diagnostics': diagnostics['arms'][role], 'query_cast_linf': result['query_cast_linf'],
                'native_qualified': False, 'scientific_admission': False,
                'source_delivery_status': 'generated_unexecuted_at_authoring',
                'input_scope': 'one complete native direct decoder context; no GT/labels/scorer fit',
                'information_cost': 'all three arms share/payout the identical source-time query'}
            if rejected:
                report.update(exception_type='CoordinateBoundsRejection',
                              error='Frozen reject policy: output outside coordinate bounds; no fallback')
            else:
                np.savez_compressed(arm / 'sequence.npz', **{**sequence, 'vertices': arms[role]})
                report['sha256']['sequence.npz'] = digest(arm / 'sequence.npz')
            write(arm / 'report.json', report)
            cases.append({'case_id': result['uid'] + '-' + role, 'uid': result['uid'],
                          'case_dir': role, 'preparation_status': report['status']})
        write(output / 'manifest.json', {'cases': cases, 'scope': 'C01 artifacts only; retain failed roles'})
        result['status'] = 'completed' if all(not d['rejected'] for d in diagnostics['arms'].values()) else 'incomplete'
    except Exception as error:
        result.update(status='error', exception_type=type(error).__name__, error=str(error))
    finally:
        result['elapsed_seconds'] = time.monotonic() - started
        result['timing_scope'] = 'CPU snapshot, full-context validation and correction; excludes model/scorer'
        write(output / 'candidate.json', result)
    return result


def verify_candidate_artifacts(output):
    """Recompute the method and its full closure; used before native comparison."""
    output = Path(output)
    result = load(output / 'candidate.json')
    if (result.get('candidate_id') != CANDIDATE_ID or result.get('status') not in ('completed', 'incomplete')
            or result.get('native_qualified') is not False or result.get('scientific_admission') is not False):
        raise ValueError('Completed or explicitly rejected C01 artifact set required')
    sequence, values, files, source_report = load_context(output / 'context', result['consumption_sha256'])
    arms, diagnostics = correction_arrays(values['anchor'], values['raw_targets'], values['source_time'],
        coordinate_bounds=result['coordinate_bounds'], bounds_policy=result['bounds_policy'])
    cast_error = float(np.max(np.abs(values['query_xyz'] - values['anchor'])))
    expected_status = 'incomplete' if any(d['rejected'] for d in diagnostics['arms'].values()) else 'completed'
    if (diagnostics != result.get('diagnostics') or result.get('query_cast_linf') != cast_error
            or result.get('uid') != source_report['uid'] or result.get('seed') != source_report['seed']
            or result.get('status') != expected_status or result.get('roles') != list(ROLES)):
        raise ValueError('C01 diagnostics do not recompute')
    expected_certificate = dict(values,
        self_residual=values['source_time'] - values['anchor'].astype(np.float64),
        mean_residual=(values['source_time'] - values['anchor']).mean(axis=0))
    certificate_path = physical(output / 'certificate.npz')
    with np.load(certificate_path, allow_pickle=False) as certificate:
        if (set(certificate.files) != set(expected_certificate)
                or any(certificate[k].dtype != v.dtype or not np.array_equal(certificate[k], v)
                       for k, v in expected_certificate.items())):
            raise ValueError('C01 complete-context certificate differs')
    paths = {}
    source_path = output / 'context/raw/observed/sequence.npz'
    for role in ROLES:
        report = load(output / role / 'report.json')
        rejected = diagnostics['arms'][role]['rejected']
        if (report.get('status') != ('error' if rejected else 'completed')
                or report.get('candidate_role') != role or report.get('candidate_id') != CANDIDATE_ID
                or report.get('method_id') != (CANDIDATE_ID if role == 'self_map_subtraction' else 'c01-control-' + role)
                or report.get('uid') != source_report['uid'] or report.get('seed') != source_report['seed']
                or report.get('query_cast_linf') != cast_error
                or report.get('native_qualified') is not False or report.get('scientific_admission') is not False
                or report.get('source_sequence_sha256') != digest(source_path)
                or report.get('source_report_sha256') != digest(source_path.with_name('report.json'))
                or report.get('implementation_sha256') != digest(__file__)
                or report.get('certificate_sha256') != digest(certificate_path)
                or report.get('sha256', {}).get('certificate.npz') != digest(certificate_path)
                or report.get('bounds_policy') != result['bounds_policy']
                or report.get('coordinate_bounds') != result['coordinate_bounds']
                or report.get('diagnostics') != diagnostics['arms'][role]):
            raise ValueError('C01 report/identity/implementation mismatch: ' + role)
        path = output / role / 'sequence.npz'
        if rejected:
            if path.exists() or path.is_symlink():
                raise ValueError('Rejected role must not retain a scoreable sequence')
        else:
            if digest(physical(path)) != report.get('sha256', {}).get('sequence.npz'):
                raise ValueError('C01 sequence hash differs: ' + role)
            with np.load(path, allow_pickle=False) as saved:
                expected = {**sequence, 'vertices': arms[role]}
                if set(saved.files) != set(expected) or any(saved[k].dtype != v.dtype or not np.array_equal(saved[k], v) for k, v in expected.items()):
                    raise ValueError('C01 output does not recompute: ' + role)
            paths[role] = path
    files.extend([output / 'candidate.json', output / 'manifest.json'])
    return {'source_sequence': source_path, 'source_report': source_path.with_name('report.json'),
            'context_files': files, 'arms': paths, 'certificate': certificate_path,
            'diagnostics': diagnostics}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('consumption', 'manifest', 'result', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--raw-file', nargs=2, action='append', required=True, metavar=('RELATIVE', 'PATH'))
    parser.add_argument('--expected-consumption-sha256', required=True)
    parser.add_argument('--coordinate-bounds', type=float, nargs=2, required=True)
    parser.add_argument('--bounds-policy', choices=('preserve_and_report', 'reject'), required=True)
    args = parser.parse_args(argv)
    raw = dict(args.raw_file)
    if len(raw) != len(args.raw_file):
        parser.error('Duplicate raw-file names are forbidden')
    result = export_self_map_candidate(args.consumption, args.manifest, args.result, raw, args.output,
        expected_consumption_sha256=args.expected_consumption_sha256,
        coordinate_bounds=args.coordinate_bounds, bounds_policy=args.bounds_policy)
    print(json.dumps({'status': result['status'], 'native_qualified': False}))
    return 0 if result['status'] == 'completed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
