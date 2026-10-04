"""Run/collect a frozen natural-failure diagnostic on cached official cases.

No training or model inference. Native outputs, static anchor and one fixed
temporal filter are scored with unchanged official ActionBench functions. This
round is developmental diagnosis on inspected assets, not a novel-method test.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback

from research_census_eval import PROTOCOL, digest, load_arrays, load_official, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / 'docs/failure-diagnosis/manifest-20261004.json'
ARMS = ('native', 'static_anchor', 'temporal_smooth')
KEYS = ('cd_3d', 'cd_4d', 'cd_motion')


def now():
    return datetime.now(timezone.utc).isoformat()


def manifest_id(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def git_blob_hash(path):
    value = path.read_bytes()
    return hashlib.sha1(b'blob ' + str(len(value)).encode() + b'\0' + value).hexdigest()


def load_manifest(path):
    value = read_json(path)
    if value.get('schema_version') != 1 or tuple(value.get('arms', [])) != ARMS or value.get('frames') != 16:
        raise ValueError('Unsupported frozen diagnostic contract')
    if value.get('dataset_revision') != '2796071cbe6248422fcbeab3101fa9f9886cb7b9':
        raise ValueError('Official data revision changed; freeze a new round')
    if value.get('protocol') != dict(PROTOCOL, sampling_seed=44):
        raise ValueError('Native evaluation budget/settings changed')
    if value.get('budgets') != {'case_seconds': 600, 'round_seconds': 5400,
            'allocator_mib': 2048, 'free_reserve_mib': 2048, 'serial': True}:
        raise ValueError('Frozen resource budget changed; freeze a new round')
    if value.get('replay_tolerance', {}).get('atol') != 1e-6 or value.get('replay_tolerance', {}).get('rtol') != 1e-4:
        raise ValueError('Frozen numerical replay tolerance changed')
    cases = value.get('cases', [])
    if len(cases) != 10 or len({c['case_id'] for c in cases}) != 10:
        raise ValueError('Require all ten frozen runs without duplicate IDs')
    primary = [c for c in cases if c['seed'] == 42]
    repeats = [c for c in cases if c['seed'] == 43]
    if (len(primary) != 8 or len({c['uid'] for c in primary}) != 8 or len(repeats) != 2
            or [c['uid'] for c in repeats] != [c['uid'] for c in primary[:2]]):
        raise ValueError('Independent-asset and repeated-seed denominator changed')
    for c in cases:
        for name in ('uid', 'case_id', 'case_dir'):
            if not c[name] or Path(c[name]).name != c[name] or c[name] in ('.', '..'):
                raise ValueError('Unsafe frozen case path')
        for name in ('sequence_sha256', 'generation_report_sha256', 'gt_sha256'):
            if len(c[name]) != 64 or any(x not in '0123456789abcdef' for x in c[name]):
                raise ValueError('Missing source content identity')
        if set(c['reference_metrics']) != set(KEYS) or not all(
                math.isfinite(c['reference_metrics'][k]) and c['reference_metrics'][k] >= 0 for k in KEYS):
            raise ValueError('Invalid historical native metric')
    return value


def replay_check(reference, observed, tolerance):
    components = {k: {'reference': reference[k], 'observed': observed[k],
        'abs_difference': abs(observed[k] - reference[k]),
        'pass': math.isfinite(observed[k]) and math.isclose(observed[k], reference[k],
            rel_tol=tolerance['rtol'], abs_tol=tolerance['atol'])} for k in KEYS}
    return {'pass': all(v['pass'] for v in components.values()), 'components': components,
            'scope': tolerance['scope']}


def historical_stage_summary(manifest, evidence_root):
    rows, errors = [], []
    for case in manifest['cases']:
        if case['seed'] != 42:
            continue
        try:
            path = evidence_root / case['stage_report']
            report = read_json(path)
            if report['uid'] != case['uid'] or report['seed'] != 42 or report['status'] != 'completed':
                raise ValueError('Historical Stage-GT identity/status mismatch')
            if [f['frame'] for f in report['frames']] != [0, 8, 15] or report['future_frame_alignment']:
                raise ValueError('Historical Stage-GT scoring contract changed')
            first = [f['stageI_aligned_to_gt']['sum_unsquared'] for f in report['frames']]
            second = [f['stageII_aligned_to_gt']['sum_unsquared'] for f in report['frames']]
            rows.append({'uid': case['uid'], 'stageI_cd': first, 'stageII_cd': second,
                'stageI_future_mean_minus_anchor': sum(first[1:]) / 2 - first[0],
                'stageII_future_mean_minus_anchor': sum(second[1:]) / 2 - second[0],
                'stageII_future_mean_minus_stageI': sum(second[1:]) / 2 - sum(first[1:]) / 2,
                'sampled_monotonic_increase': second[0] < second[1] < second[2],
                'source': case['stage_report'], 'source_sha256': digest(path)})
        except Exception as error:
            errors.append({'uid': case['uid'], 'error': str(error)})
    return {'n_planned_assets': 8, 'n_assets_with_stage_report': len(rows), 'measured_frames': [0, 8, 15],
        'stageI_future_mean_above_anchor': sum(r['stageI_future_mean_minus_anchor'] > 0 for r in rows),
        'stageII_future_mean_above_anchor': sum(r['stageII_future_mean_minus_anchor'] > 0 for r in rows),
        'stageII_sampled_monotonic_increase': sum(r['sampled_monotonic_increase'] for r in rows),
        'stageII_future_mean_above_stageI': sum(r['stageII_future_mean_minus_stageI'] > 0 for r in rows),
        'meaningful_failure_threshold_established': False, 'rows': rows, 'errors': errors,
        'scope': 'Three-frame shared-anchor geometry observation; not full16 native score or causal decomposition'}


def frame_conditions(gt):
    """Evaluation-only covariates; never inference input or outcome labels."""
    import numpy as np
    diagonal = float(np.linalg.norm(np.ptp(gt[0].astype(np.float64), axis=0)))
    if diagonal <= 0 or not math.isfinite(diagonal):
        raise ValueError('Invalid native GT anchor extent')
    delta = gt.astype(np.float64) - gt[0].astype(np.float64)
    return {'frame_indices': list(range(16)), 'gt_anchor_diagonal': diagonal,
        'gt_material_rms_departure_from_anchor': np.sqrt(np.mean(np.sum(delta ** 2, axis=-1), axis=1)).tolist(),
        'gt_centroid_departure_from_anchor': np.linalg.norm(delta.mean(axis=1), axis=-1).tolist(),
        'scope': 'Native GT material-index assumption; descriptive conditions, not new quality metrics'}


def error_profile(per_frame, conditions):
    import numpy as np
    from scipy.stats import spearmanr
    result = {}
    for key, values in per_frame.items():
        values = np.asarray(values, dtype=np.float64)
        def correlation(predictor):
            if np.ptp(values[1:]) == 0 or np.ptp(predictor[1:]) == 0:
                return None
            return float(spearmanr(values[1:], predictor[1:]).statistic)
        result[key] = {'future_mean_minus_anchor': float(values[1:].mean() - values[0]),
            'peak_frame': int(values.argmax()), 'last_minus_peak': float(values[-1] - values.max()),
            'n_decreasing_adjacent_intervals': int((np.diff(values) < 0).sum()),
            'spearman_vs_time_future_only': correlation(np.arange(16)),
            'spearman_vs_gt_departure_future_only': correlation(np.asarray(
                conditions['gt_material_rms_departure_from_anchor'])),
            'scope': 'Descriptive autocorrelated frame observations; no p-value, causal verdict or failure cutoff'}
    return result


def verify_sources(args, manifest, case):
    source = args.root / 'outputs/census-20261002/cases' / case['case_dir']
    gt = args.root / 'data/actionbench-census-20261002/data' / case['uid'] / 'surfaces.npy'
    paths = {'sequence': source / 'sequence.npz', 'generation_report': source / 'report.json', 'ground_truth': gt}
    expected = {'sequence': case['sequence_sha256'], 'generation_report': case['generation_report_sha256'],
                'ground_truth': case['gt_sha256']}
    for name, path in paths.items():
        if digest(path) != expected[name]:
            raise ValueError('Pinned natural source differs: ' + name)
    generation = read_json(paths['generation_report'])
    if generation.get('status') != 'completed' or generation.get('uid') != case['uid'] or generation.get('seed') != case['seed']:
        raise ValueError('Completed source generation identity mismatch')
    reference = ROOT / case['reference_evaluation']
    if git_blob_hash(reference) != case['reference_evaluation_git_blob']:
        raise ValueError('Pinned historical evaluator receipt changed')
    benchmark_root = args.root / 'repo/actionbench'
    for name, expected_hash in manifest['official_sha256'].items():
        if digest(benchmark_root / name) != expected_hash:
            raise ValueError('Official source changed: ' + name)
    paths.update({'official/' + name: benchmark_root / name for name in manifest['official_sha256']})
    paths.update({'runner': Path(__file__), 'trace': Path(__file__).with_name('research_failure_trace.py'),
                  'evaluator_loader': Path(__file__).with_name('research_census_eval.py'),
                  'resource_helper': Path(__file__).with_name('research_three_ideas.py'),
                  'manifest': args.manifest, 'reference_evaluation': reference})
    return paths, {name: digest(path) for name, path in paths.items()}


def variants(vertices):
    import numpy as np
    # One noniterated symmetric [1,2,1]/4 filter; both endpoints stay native.
    # Native fixed topology/query identities are required; no GT fitting/tuning.
    smooth = vertices.copy()
    smooth[1:-1] = (vertices[:-2] + 2 * vertices[1:-1] + vertices[2:]) / 4
    return {'native': vertices, 'static_anchor': np.repeat(vertices[:1], 16, axis=0),
            'temporal_smooth': smooth}


def run_case(args):
    folder = args.output
    folder.mkdir(parents=True, exist_ok=False)
    manifest = load_manifest(args.manifest)
    case = next(c for c in manifest['cases'] if c['case_id'] == args.case_id)
    report = dict(round_id=manifest['round_id'], manifest_id=manifest_id(manifest),
        case_id=case['case_id'], uid=case['uid'], seed=case['seed'], status='running', started_utc=now(),
        arms={name: {'status': 'pending'} for name in ARMS}, resources={}, scientific_status='INCONCLUSIVE')
    started, resources = time.monotonic(), None
    write_json(folder / 'report.json', report)
    try:
        paths, hashes = verify_sources(args, manifest, case)
        report['source_sha256'] = hashes
        report['source_paths'] = {name: str(path.resolve()) for name, path in paths.items()}
        vertices, faces, gt = load_arrays(paths['sequence'], paths['ground_truth'])
        if len(vertices) != 16:
            raise ValueError('Full native16 timeline required')
        report['conditions'] = frame_conditions(gt)
        report['packages'] = {name: importlib.metadata.version(name) for name in ('numpy', 'torch', 'trimesh', 'scipy', 'pytorch3d')}
        import numpy as np
        import torch
        import trimesh
        from research_three_ideas import Resources
        from research_failure_trace import trace_official, finish_trace
        device = torch.device(args.device)
        if device.type != 'cuda' or device.index is None or not torch.cuda.is_available():
            raise ValueError('Explicit available CUDA device required')
        torch.cuda.set_device(device)
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(min(1., manifest['budgets']['allocator_mib'] * 1024 ** 2 /
            torch.cuda.get_device_properties(device).total_memory), device)
        resources = Resources(folder, torch, str(device.index))
        resources.start()
        backend, report['official_source'] = load_official(args.root / 'repo')
        import benchmark
        unified_reference = None
        for arm, values in variants(vertices).items():
            row = report['arms'][arm]
            arm_folder = folder / arm
            arm_folder.mkdir()
            row.update(status='running', started_utc=now(), anchor_bitwise_unchanged=values[0].tobytes() == vertices[0].tobytes())
            if not row['anchor_bitwise_unchanged']:
                raise ValueError('Control changed source anchor')
            for frame in values:
                tri = frame[faces]
                if not np.isfinite(frame).all() or np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=-1).sum() <= 0:
                    raise ValueError('Control has nonfinite/zero-area surface')
            sequence = arm_folder / 'sequence.npz'
            np.savez_compressed(sequence, vertices=values, faces=faces, frame_indices=np.arange(16))
            row['sequence_sha256'] = digest(sequence)
            write_json(folder / 'report.json', report)
            meshes = [trimesh.Trimesh(vertices=v, faces=faces, process=False) for v in values]
            arm_started = time.monotonic()
            with trace_official(benchmark, arm_folder) as trace:
                returned = backend(torch.from_numpy(gt), meshes, device=args.device, is_4D=True,
                    n_pts_icp=10000, n_pts_chamfer=100000, seed=44)
            metrics = dict(zip(KEYS, map(float, returned)))
            if not all(math.isfinite(v) and v >= 0 for v in metrics.values()):
                raise ValueError('Nonfinite/negative native metric')
            trace, unified = finish_trace(trace, metrics, arm_folder)
            row.update(metrics=metrics, per_frame=trace['per_frame'], trace=trace,
                elapsed_seconds=time.monotonic() - arm_started,
                artifact_sha256={p.name: digest(p) for p in arm_folder.iterdir() if p.is_file()})
            if arm == 'native':
                report['native_replay'] = replay_check(case['reference_metrics'], metrics, manifest['replay_tolerance'])
                if not report['native_replay']['pass']:
                    row['status'] = 'qualification_failed'
                    raise ValueError('Native scalar replay differs; controls blocked, no efficacy interpretation')
                unified_reference = unified.copy()
            elif not np.allclose(unified, unified_reference, rtol=1e-6, atol=1e-6):
                row['status'] = 'qualification_failed'
                raise ValueError('Identical anchor does not reproduce unified ICP; paired control invalid')
            row['unified_anchor_alignment_qualified'] = True
            row['profile'] = error_profile(row['per_frame'], report['conditions'])
            row['status'] = 'completed'
            write_json(folder / 'report.json', report)
        for name, path in paths.items():
            if digest(path) != hashes[name]:
                raise ValueError('Source/scorer/runner changed during scoring: ' + name)
        report.update(status='completed', source_end_check='passed', scientific_status='DEVELOPMENT_DIAGNOSTIC_ONLY')
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        for row in report['arms'].values():
            if row['status'] == 'running':
                row['status'] = 'failed'
            elif row['status'] == 'pending':
                row['status'] = 'blocked'
        # No final source check means retained partial outputs are not qualified.
        report['source_end_check'] = 'not_passed'
    finally:
        if resources is not None:
            try:
                report['resources'] = resources.finish()
                if report['resources']['gpu_samples'] == 0 or report['resources']['gpu_monitor_error']:
                    report.update(status='failed', resource_qualification='failed', scientific_status='INCONCLUSIVE')
                    report.setdefault('error', 'Resource monitor returned no valid samples')
                else:
                    report['resource_qualification'] = 'passed'
                    report['case_artifact_sha256'] = {'gpu-memory.csv': digest(folder / 'gpu-memory.csv')}
            except Exception:
                report.update(status='failed', resource_collection_error=traceback.format_exc(), scientific_status='INCONCLUSIVE')
        report.update(ended_utc=now(), elapsed_seconds=time.monotonic() - started)
        write_json(folder / 'report.json', report)
    return int(report['status'] != 'completed')


def verify_resource_receipt(report, folder):
    resources = report.get('resources', {})
    if report.get('resource_qualification') != 'passed' or resources.get('gpu_samples', 0) < 1 or resources.get('gpu_monitor_error'):
        raise ValueError('Missing qualified GPU resource evidence')
    expected = report.get('case_artifact_sha256', {}).get('gpu-memory.csv')
    if expected is None or digest(folder / 'gpu-memory.csv') != expected:
        raise ValueError('GPU resource log changed or has no recorded digest')


def collect(output, manifest, evidence_root):
    runs, paired = [], []
    mid = manifest_id(manifest)
    queue_path = output / 'queue.json'
    jobs = {}
    if queue_path.is_file():
        queue = read_json(queue_path)
        if queue.get('manifest_id') != mid:
            raise ValueError('Queue identity differs from diagnostic manifest')
        jobs = {job['case_id']: job for job in queue.get('jobs', [])}
    for case in manifest['cases']:
        row = dict(case_id=case['case_id'], uid=case['uid'], seed=case['seed'], status='pending',
            arms={arm: {'status': 'pending'} for arm in ARMS})
        path = output / case['case_id'] / 'report.json'
        if path.is_file():
            try:
                report = read_json(path)
                if any(report.get(k) != row[k] for k in ('case_id', 'uid', 'seed')) or report.get('manifest_id') != mid:
                    raise ValueError('Report identity/round differs from frozen manifest')
                row.update(report)
                row['report_sha256'] = digest(path)
                if row['status'] == 'completed':
                    if row.get('source_end_check') != 'passed' or not row.get('native_replay', {}).get('pass'):
                        raise ValueError('Missing native/source qualification')
                    verify_resource_receipt(row, output / case['case_id'])
                    for name, expected in (('sequence', case['sequence_sha256']),
                            ('ground_truth', case['gt_sha256']), ('generation_report', case['generation_report_sha256'])):
                        if row.get('source_sha256', {}).get(name) != expected:
                            raise ValueError('Collected source identity differs: ' + name)
                    if row.get('official_source', {}).get('sha256') != manifest['official_sha256']:
                        raise ValueError('Collected scorer differs from native source')
                    for name, source_path in row['source_paths'].items():
                        if digest(Path(source_path)) != row['source_sha256'][name]:
                            raise ValueError('Source changed before collection: ' + name)
                    for arm in ARMS:
                        a = row['arms'][arm]
                        if a['status'] != 'completed' or not a['unified_anchor_alignment_qualified']:
                            raise ValueError('Unqualified paired control')
                        for key in KEYS:
                            values = a['per_frame'][key]
                            if len(values) != 16 or not all(math.isfinite(v) and v >= 0 for v in values):
                                raise ValueError('Incomplete native frame score')
                            if not math.isclose(sum(values) / 16, a['metrics'][key], rel_tol=1e-6, abs_tol=1e-6):
                                raise ValueError('Native frame contributions do not reduce to scalar')
                        for name, expected in a['artifact_sha256'].items():
                            if digest(output / case['case_id'] / arm / name) != expected:
                                raise ValueError('Collected score artifact changed: ' + name)
                    if case['seed'] == 42:
                        paired.append({'uid': case['uid'], 'native': row['arms']['native']['metrics'],
                            'delta_control_minus_native': {arm: {key: row['arms'][arm]['metrics'][key]
                                - row['arms']['native']['metrics'][key] for key in KEYS} for arm in ARMS[1:]}})
            except Exception as error:
                row.update(status='invalid_report', collection_error=str(error))
                row['arms'] = {arm: {'status': 'invalid_report'} for arm in ARMS}
        if case['case_id'] in jobs:
            job = jobs[case['case_id']]
            row['queue_job'] = job
            if job['status'] in ('failed', 'timeout') and row['status'] in ('pending', 'running', 'completed'):
                row.update(status=job['status'], queue_failure_reason=job.get('error'))
                row['arms'] = {arm: {'status': 'blocked'} for arm in ARMS}
        runs.append(row)
    complete_primary_uids = {r['uid'] for r in runs if r['status'] == 'completed' and r['seed'] == 42}
    paired = [p for p in paired if p['uid'] in complete_primary_uids]
    historical = read_json(evidence_root / 'results/census-20261002/source-and-receipts/aggregate-final10.json')
    summary = {'round_id': manifest['round_id'], 'manifest_id': mid, 'collected_utc': now(),
        'n_planned_runs': 10, 'n_unique_assets': 8, 'n_planned_arm_scores': 30,
        'run_status_counts': dict(Counter(r['status'] for r in runs)),
        'arm_status_counts': dict(Counter(a['status'] for r in runs for a in r['arms'].values())),
        'arm_status_counts_scope': 'Execution states; arms under an unqualified run are excluded from paired evidence',
        'n_qualified_arm_scores': 3 * sum(r['status'] == 'completed' for r in runs),
        'n_complete_primary_pairs': len(paired), 'primary_seed42_paired_results': paired,
        'historical_failed_attempts': historical['failed_attempts'],
        'historical_stage_observation': historical_stage_summary(manifest, evidence_root), 'cases': runs,
        'scientific_status': 'INCONCLUSIVE' if any(r['status'] != 'completed' for r in runs) else 'DEVELOPMENT_DIAGNOSTIC_ONLY',
        'natural_gate_0': 'UNRESOLVED', 'gate_a': 'NOT_FROZEN',
        'limitations': ['Inspected eight-asset development set; repeats are not independent assets.',
            'No established task-relevant success/failure cutoff or population prevalence.',
            'Frame correlations do not establish causal mechanism; ICP allows anisotropic scale.',
            'Static and one smoothing baseline do not qualify strongest guidance/output-fitting alternatives.',
            'New hypotheses, closest-work collision and fresh confirmation remain outstanding.']}
    summary['mean_paired_delta_control_minus_native_complete_primary_only'] = {
        arm: {key: sum(p['delta_control_minus_native'][arm][key] for p in paired) / len(paired)
            if paired else None for key in KEYS} for arm in ARMS[1:]}
    return summary


def gpu_free(device):
    index = device.split(':')[1] if device.startswith('cuda:') else None
    if index is None or not index.isdigit() or os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Use physical cuda:N without CUDA_VISIBLE_DEVICES remapping')
    result = subprocess.run(['nvidia-smi', '-i', index, '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10, check=True)
    return int(result.stdout.strip())


def fail_owned_case(folder, status, reason):
    path = folder / 'report.json'
    if not path.is_file():
        return
    report = read_json(path)
    report.update(status=status, queue_failure_reason=reason, scientific_status='INCONCLUSIVE', source_end_check='not_passed')
    for row in report['arms'].values():
        if row['status'] == 'running':
            row['status'] = status
        elif row['status'] == 'pending':
            row['status'] = 'blocked'
    write_json(path, report)


def stop_owned_child(process):
    """Terminate only the newly-created session; tolerate an already exited group."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def should_halt_after_case(index, job, detail):
    return ((index == 0 and job['status'] != 'completed')
        or 'out of memory' in detail.get('error', '').lower()
        or detail.get('resource_qualification') == 'failed'
        or 'resource_collection_error' in detail)


def run_round(args):
    manifest = load_manifest(args.manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    queue = {'round_id': manifest['round_id'], 'manifest_id': manifest_id(manifest), 'started_utc': now(),
        'python': platform.python_version(), 'runner_sha256': digest(Path(__file__)),
        'manifest_file_sha256': digest(args.manifest), 'budget': manifest['budgets'],
        'jobs': [dict(case_id=c['case_id'], status='pending') for c in manifest['cases']]}
    write_json(args.output / 'queue.json', queue)
    write_json(args.output / 'manifest.json', manifest)
    started = time.monotonic()
    active = active_folder = active_job = None
    stop_signal = None
    previous_handlers = {}
    def request_stop(signum, _frame):
        nonlocal stop_signal
        stop_signal = signum
    for name in ('SIGTERM', 'SIGINT', 'SIGHUP'):
        signum = getattr(signal, name, None)
        if signum is not None:
            previous_handlers[signum] = signal.signal(signum, request_stop)
    try:
        for index, (case, job) in enumerate(zip(manifest['cases'], queue['jobs'])):
            if stop_signal is not None:
                queue['halt_reason'] = f'Queue cancelled by signal {stop_signal}'
                break
            remaining = manifest['budgets']['round_seconds'] - (time.monotonic() - started)
            if remaining <= 0:
                queue['halt_reason'] = 'Frozen round wall-time budget exhausted'
                break
            free = gpu_free(args.device)
            if free < manifest['budgets']['allocator_mib'] + manifest['budgets']['free_reserve_mib']:
                queue['halt_reason'] = 'Insufficient physical GPU reserve; admissions stopped'
                break
            folder = args.output / case['case_id']
            command = [sys.executable, str(Path(__file__).resolve()), '_case', '--root', str(args.root),
                '--manifest', str(args.manifest.resolve()), '--case-id', case['case_id'],
                '--output', str(folder.resolve()), '--device', args.device]
            job.update(status='running', command=command, started_utc=now())
            write_json(args.output / 'queue.json', queue)
            active_folder, active_job = folder, job
            with (args.output / (case['case_id'] + '.log')).open('w') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active, active_folder, active_job = process, folder, job
                timeout = min(remaining, manifest['budgets']['case_seconds'])
                child_started, last_gpu = time.monotonic(), time.monotonic()
                try:
                    while process.poll() is None:
                        if stop_signal is not None:
                            raise InterruptedError(f'Queue cancelled by signal {stop_signal}')
                        if time.monotonic() - child_started > timeout:
                            raise TimeoutError('Owned case exceeded frozen wall-time budget')
                        if time.monotonic() - last_gpu >= 2:
                            if gpu_free(args.device) < manifest['budgets']['free_reserve_mib']:
                                raise RuntimeError('Physical GPU free reserve breached')
                            last_gpu = time.monotonic()
                        time.sleep(.5)
                    job.update(returncode=process.returncode, status='completed' if process.returncode == 0 else 'failed')
                except BaseException as error:
                    # Only this newly-created child process group is owned.
                    stop_owned_child(process)
                    job.update(status='timeout' if isinstance(error, TimeoutError) else 'failed', error=str(error))
                    fail_owned_case(folder, job['status'], job['error'])
                    queue['halt_reason'] = job['error']
                    if isinstance(error, (KeyboardInterrupt, SystemExit)):
                        raise
                active = active_folder = active_job = None
            job.update(ended_utc=now(), elapsed_seconds=time.monotonic() - child_started)
            write_json(args.output / 'queue.json', queue)
            write_json(args.output / 'summary.json', collect(args.output, manifest, ROOT))
            print(json.dumps({'case_id': case['case_id'], 'status': job['status']}), flush=True)
            report_path = folder / 'report.json'
            detail = read_json(report_path) if report_path.is_file() else {}
            if stop_signal is not None:
                queue.setdefault('halt_reason', f'Queue cancelled by signal {stop_signal}')
            if queue.get('halt_reason') or should_halt_after_case(index, job, detail):
                queue.setdefault('halt_reason', 'Canary/resource qualification failed or OOM; repair before expansion')
                break
    except BaseException:
        queue['halt_reason'] = traceback.format_exc()
    finally:
        if active_job is not None:
            if active is not None:
                stop_owned_child(active)
            active_job.update(status='failed', error=queue.get('halt_reason', 'Queue interrupted'))
            active_job['ended_utc'] = now()
            fail_owned_case(active_folder, 'failed', active_job['error'])
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
        queue.update(ended_utc=now(), elapsed_seconds=time.monotonic() - started)
        write_json(args.output / 'queue.json', queue)
        summary = collect(args.output, manifest, ROOT)
        summary['queue'] = queue
        write_json(args.output / 'summary.json', summary)
    return int(summary['scientific_status'] == 'INCONCLUSIVE' or bool(queue.get('halt_reason')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('run', '_case', 'collect'):
        child = sub.add_parser(name)
        child.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
        child.add_argument('--output', type=Path, required=True)
        if name != 'collect':
            child.add_argument('--root', type=Path, required=True)
            child.add_argument('--device', default='cuda:0')
        if name == '_case':
            child.add_argument('--case-id', required=True)
    args = parser.parse_args()
    if args.command == 'collect':
        report = collect(args.output, load_manifest(args.manifest), ROOT)
        queue = args.output / 'queue.json'
        if queue.is_file():
            report['queue'] = read_json(queue)
        write_json(args.output / 'summary-collected.json', report)
        print(json.dumps({k: report[k] for k in ('scientific_status', 'n_planned_runs', 'n_unique_assets', 'run_status_counts')}))
        return int(report['scientific_status'] == 'INCONCLUSIVE')
    return run_case(args) if args.command == '_case' else run_round(args)


if __name__ == '__main__':
    raise SystemExit(main())
