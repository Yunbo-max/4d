"""Three-arm ActionBench qualification glue, generated for Local acceptance.

Imports the existing scorer without replacing metrics. Scores and freshly repeats
every arm, retains failures and input bindings, and never certifies a scientific
gate. `request`/`plan` prepare records; `score` runs only inside the admitted Local
scientific harness. No model inference, automatic queue or candidate method.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid

import numpy as np

import research_census_eval as census

ARMS = ('native', 'world_gaussian', 'body_gaussian')
METRICS = ('cd_3d', 'cd_4d', 'cd_motion')


def file_ref(root: Path, path: Path) -> dict:
    root, path = Path(root).resolve(), Path(path).resolve()
    relative = path.relative_to(root).as_posix()
    return {'path': relative, 'sha256': census.digest(path)}


def resolve_ref(root: Path, ref: dict) -> Path:
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
        raise ValueError('A pinned path/sha256 reference is required')
    relative = Path(ref['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Project-relative nonescaping paths required')
    root = Path(root).resolve()
    path = (root/relative).resolve()
    path.relative_to(root)
    if census.digest(path) != ref['sha256']:
        raise ValueError('Changed pinned file: '+ref['path'])
    return path


def canonical_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def check_manifest(manifest: dict, uid: str):
    expected = [{'case_id': uid+'-'+arm, 'uid': uid, 'case_dir': arm} for arm in ARMS]
    if manifest.get('cases') != expected:
        raise ValueError('Exactly the three declared arm paths/IDs are required')


def read_arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def check_arm_arrays(source: dict, arm: dict):
    required = {'vertices', 'faces', 'timesteps', 'frame_indices', 'query_vertex_ids'}
    if not required.issubset(source) or set(source) != set(arm):
        raise ValueError('Complete original array metadata is required')
    vertices = source['vertices']
    if vertices.ndim != 3 or vertices.shape[0] != 16 or vertices.shape[-1] != 3:
        raise ValueError('Exactly 16 full source frames required')
    if arm['vertices'].shape != vertices.shape or arm['vertices'].dtype != vertices.dtype:
        raise ValueError('Control vertex shape/dtype changed')
    for arrays in (source, arm):
        for value in arrays.values():
            if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
                raise ValueError('Numeric finite arrays required')
        for key, expected in (('frame_indices', np.arange(16)), ('timesteps', np.arange(16)),
                              ('query_vertex_ids', np.arange(vertices.shape[1]))):
            if not np.array_equal(arrays[key], expected):
                raise ValueError('Original timeline/vertex identity required')
        for key in ('frame_indices', 'query_vertex_ids'):
            if not np.issubdtype(arrays[key].dtype, np.integer):
                raise ValueError('Integer timeline/vertex indices required')
    if not np.array_equal(source['vertices'][0], arm['vertices'][0]):
        raise ValueError('First-frame anchor changed')
    for key in set(source)-{'vertices'}:
        if source[key].dtype != arm[key].dtype or not np.array_equal(source[key], arm[key]):
            raise ValueError('Original topology/metadata changed: '+key)


def make_request(root: Path, *, source_case: Path, controls_dir: Path,
                 ground_truth: Path, repo_root: Path, population: Path) -> dict:
    root = Path(root).resolve()
    source_case, controls_dir, ground_truth, repo_root, population = map(
        lambda p: Path(p).resolve(), (source_case, controls_dir, ground_truth, repo_root, population))
    report = census.read_json(source_case/'report.json')
    uid, seed = report.get('uid'), report.get('seed')
    if not isinstance(uid, str) or not uid or Path(uid).name != uid or uid in ('.', '..'):
        raise ValueError('Original native UID required')
    if isinstance(seed, bool) or not isinstance(seed, int): raise ValueError('Integer inference seed required')
    released = census.read_json(population)
    if released.get('dataset') != 'facebook/actionbench' or uid not in released.get('uids', []):
        raise ValueError('UID must belong to the retained released ActionBench population')
    if ground_truth.name != 'surfaces.npy' or ground_truth.parent.name != uid:
        raise ValueError('GT must be original UID/surfaces.npy')
    sequence_ref = file_ref(root, source_case/'sequence.npz')
    if report.get('status') != 'completed' or report.get('sha256', {}).get('sequence.npz') != sequence_ref['sha256']:
        raise ValueError('Completed original generator report must bind source bytes')
    manifest_ref = file_ref(root, controls_dir/'manifest.json')
    check_manifest(census.read_json(controls_dir/'manifest.json'), uid)
    summary = census.read_json(controls_dir/'controls.json')
    if summary.get('uid') != uid or summary.get('seed') != seed:
        raise ValueError('Control summary differs from original source identity')
    inputs = [sequence_ref, file_ref(root, source_case/'report.json'), manifest_ref,
              file_ref(root, controls_dir/'controls.json'), file_ref(root, ground_truth), file_ref(root, population)]
    arm_records = []
    for arm in ARMS:
        directory = controls_dir/arm
        arm_report_ref = file_ref(root, directory/'report.json')
        arm_report = census.read_json(directory/'report.json')
        if (arm_report.get('uid'), arm_report.get('seed'), arm_report.get('baseline_arm')) != (uid, seed, arm):
            raise ValueError('Control arm identity differs: '+arm)
        if arm_report.get('source_sequence_sha256') != sequence_ref['sha256'] or arm_report.get('source_report_sha256') != inputs[1]['sha256']:
            raise ValueError('Control report not bound to original generator: '+arm)
        if arm_report.get('implementation_sha256') != census.digest(Path(__file__).with_name('simple_mesh_controls.py')):
            raise ValueError('Adapter implementation changed after preparation')
        if arm_report.get('sigma_frames') != 1.:
            raise ValueError('This qualification revision freezes sigma=1 frame')
        inputs.append(arm_report_ref)
        seq_ref = None
        if arm_report.get('status') == 'completed':
            seq_ref = file_ref(root, directory/'sequence.npz')
            if seq_ref['sha256'] != arm_report.get('sha256', {}).get('sequence.npz'):
                raise ValueError('Control report does not bind current sequence: '+arm)
            inputs.append(seq_ref)
            if arm == 'body_gaussian':
                pose_ref = file_ref(root, directory/'poses.npz')
                if pose_ref['sha256'] != arm_report.get('sha256', {}).get('poses.npz'):
                    raise ValueError('Unbound body pose export')
                inputs.append(pose_ref)
        elif arm_report.get('status') != 'error':
            raise ValueError('Unknown control arm status')
        arm_records.append({'arm': arm, 'report_ref': arm_report_ref, 'sequence_ref': seq_ref,
                            'preparation_status': arm_report['status']})
    code_paths = [Path(__file__).resolve(), Path(census.__file__).resolve(),
                  Path(__file__).with_name('simple_mesh_controls.py'), Path(__file__).with_name('__init__.py')]
    code_paths += [repo_root/'actionbench'/name for name in census.OFFICIAL_FILES]
    request = {'kind': 'baseline-control-scoring-request', 'version': '1.0.0',
               'purpose': 'baseline-native-qualification', 'evidence_mode': 'developmental',
               'uid': uid, 'inference_seed': seed, 'scoring_seed': 44, 'arms': arm_records,
               'manifest_ref': manifest_ref, 'source_sequence_ref': sequence_ref,
               'source_report_ref': inputs[1], 'control_summary_ref': inputs[3],
               'body_pose_ref': next((r for r in inputs if r['path'] == (controls_dir/'body_gaussian/poses.npz').relative_to(root).as_posix()), None),
               'ground_truth_ref': file_ref(root, ground_truth), 'population_ref': file_ref(root, population),
               'repo_root': repo_root.relative_to(root).as_posix(),
               'input_refs': inputs, 'code_refs': [file_ref(root, path) for path in code_paths],
               'native_protocol': {**census.PROTOCOL, 'sampling_seed': 44},
               'sigma_frames': 1.,
               'replay': {'passes_per_arm': 2, 'absolute_tolerance': 0.,
                          'scope': 'Fresh scorer process per pass; not trusted host native-contract replay'},
               'native_contract_qualified': False, 'candidate_methods_tested': False}
    request['request_digest'] = canonical_digest(request)
    verify_request(root, request)
    return request


def verify_request(root: Path, request: dict):
    if request.get('request_digest') != canonical_digest({k: v for k, v in request.items() if k != 'request_digest'}):
        raise ValueError('Request digest mismatch')
    if request.get('kind') != 'baseline-control-scoring-request' or request.get('purpose') != 'baseline-native-qualification' or request.get('evidence_mode') != 'developmental':
        raise ValueError('Baseline development request required')
    if request.get('native_protocol') != {**census.PROTOCOL, 'sampling_seed': 44} or request.get('scoring_seed') != 44:
        raise ValueError('Official protocol/budgets may not be changed')
    if request.get('replay', {}).get('passes_per_arm') != 2 or request.get('replay', {}).get('absolute_tolerance') != 0.:
        raise ValueError('Two passes with exact reproducibility comparison required')
    if request.get('sigma_frames') != 1.: raise ValueError('Current sigma=1 freeze required')
    if [a['arm'] for a in request['arms']] != list(ARMS): raise ValueError('Missing/extra/reordered arms')
    require_dependency_closure(request)
    pinned = {r['path']: r for r in request['input_refs']+request['code_refs']}
    for ref in pinned.values(): resolve_ref(root, ref)
    required_code = [Path(__file__).resolve(), Path(census.__file__).resolve(),
                     Path(__file__).with_name('simple_mesh_controls.py'), Path(__file__).with_name('__init__.py')]
    repo = (Path(root)/request['repo_root']).resolve(); repo.relative_to(Path(root).resolve())
    required_code += [repo/'actionbench'/name for name in census.OFFICIAL_FILES]
    for path in required_code:
        ref = file_ref(root, path)
        if pinned.get(ref['path']) != ref: raise ValueError('Scorer/adapter code not pinned')
    for key in ('source_sequence_ref', 'source_report_ref', 'control_summary_ref', 'manifest_ref', 'ground_truth_ref', 'population_ref'):
        if pinned.get(request[key]['path']) != request[key]: raise ValueError('Unstaged request dependency')
    source_path = resolve_ref(root, request['source_sequence_ref'])
    source = read_arrays(source_path)
    check_arm_arrays(source, source)
    uid = request['uid']
    source_report_path = resolve_ref(root, request['source_report_ref'])
    source_report = census.read_json(source_report_path)
    if source_report_path != source_path.with_name('report.json') or source_report.get('status') != 'completed' or source_report.get('uid') != uid or source_report.get('seed') != request['inference_seed'] or source_report.get('sha256', {}).get('sequence.npz') != request['source_sequence_ref']['sha256']:
        raise ValueError('Original generator report does not bind this source/UID/seed')
    population = census.read_json(resolve_ref(root, request['population_ref']))
    if population.get('dataset') != 'facebook/actionbench' or uid not in population.get('uids', []):
        raise ValueError('Unreleased benchmark UID')
    manifest = census.read_json(resolve_ref(root, request['manifest_ref']))
    check_manifest(manifest, uid)
    control_root = resolve_ref(root, request['manifest_ref']).parent
    summary_path = resolve_ref(root, request['control_summary_ref'])
    summary = census.read_json(summary_path)
    if summary_path != control_root/'controls.json' or summary.get('uid') != uid or summary.get('seed') != request['inference_seed']:
        raise ValueError('Control summary identity mismatch')
    gt = resolve_ref(root, request['ground_truth_ref'])
    if gt.name != 'surfaces.npy' or gt.parent.name != uid: raise ValueError('GT UID changed')
    for entry in request['arms']:
        report_ref = entry['report_ref']
        if pinned.get(report_ref['path']) != report_ref: raise ValueError('Arm report not staged')
        report_path = resolve_ref(root, report_ref)
        if report_path != control_root/entry['arm']/'report.json': raise ValueError('Arm report path mismatch')
        report = census.read_json(report_path)
        if (report.get('uid'), report.get('seed'), report.get('baseline_arm')) != (uid, request['inference_seed'], entry['arm']):
            raise ValueError('Arm identity mismatch')
        if report.get('status') != entry['preparation_status']: raise ValueError('Arm status mismatch')
        if report.get('implementation_sha256') != census.digest(Path(__file__).with_name('simple_mesh_controls.py')) or report.get('sigma_frames') != request['sigma_frames']:
            raise ValueError('Adapter implementation/parameters not current')
        if report.get('source_sequence_sha256') != request['source_sequence_ref']['sha256']:
            raise ValueError('Control/source identity mismatch')
        if report.get('source_report_sha256') != request['source_report_ref']['sha256']:
            raise ValueError('Control/original report identity mismatch')
        if entry['sequence_ref'] is None:
            if report.get('status') != 'error': raise ValueError('Missing successful arm output')
            continue
        if report.get('status') != 'completed' or pinned.get(entry['sequence_ref']['path']) != entry['sequence_ref']:
            raise ValueError('Sequence not bound to completed arm')
        sequence = resolve_ref(root, entry['sequence_ref'])
        if sequence != report_path.with_name('sequence.npz') or entry['sequence_ref']['sha256'] != report.get('sha256', {}).get('sequence.npz'):
            raise ValueError('Arm sequence path/hash mismatch')
        check_arm_arrays(source, read_arrays(sequence))
        if entry['arm'] == 'body_gaussian':
            pose_ref = request.get('body_pose_ref')
            if not pose_ref or pinned.get(pose_ref['path']) != pose_ref:
                raise ValueError('Body pose dependency missing')
            if resolve_ref(root, pose_ref) != report_path.with_name('poses.npz') or pose_ref['sha256'] != report.get('sha256', {}).get('poses.npz'):
                raise ValueError('Body pose path/hash mismatch')
        if entry['arm'] == 'native' and entry['sequence_ref']['sha256'] != request['source_sequence_ref']['sha256']:
            raise ValueError('Native arm must be an exact source byte copy')
        census.load_arrays(sequence, gt)  # Admission only; no native metric computed.


def require_dependency_closure(request: dict):
    pinned = {ref['path']: ref for ref in request['input_refs']}
    fields = ('source_sequence_ref', 'source_report_ref', 'control_summary_ref',
              'manifest_ref', 'ground_truth_ref', 'population_ref')
    refs = [request.get(key) for key in fields]
    for arm in request['arms']:
        refs.append(arm.get('report_ref'))
        if arm.get('preparation_status') == 'completed': refs.append(arm.get('sequence_ref'))
        if arm.get('arm') == 'body_gaussian' and arm.get('preparation_status') == 'completed':
            refs.append(request.get('body_pose_ref'))
    if any(not ref or pinned.get(ref.get('path')) != ref for ref in refs):
        raise ValueError('Request dependency closure is incomplete')


def initial_arm_records() -> dict:
    return {arm: {'passes': [{'status': 'pending'}, {'status': 'pending'}],
                  'replay_comparison': {'status': 'unscored'}} for arm in ARMS}


def valid_score(row: dict) -> bool:
    return row.get('status') == 'success' and all(
        isinstance(row.get(key), (int, float)) and not isinstance(row[key], bool)
        and math.isfinite(row[key]) and row[key] >= 0 for key in METRICS)


def replay_comparison(first: dict, repeated: dict) -> dict:
    ok = valid_score(first) and valid_score(repeated)
    return {'status': ('identical' if all(first[k] == repeated[k] for k in METRICS) else 'mismatch') if ok else 'unscored',
            'absolute_differences': {k: abs(first[k]-repeated[k]) for k in METRICS} if ok else None,
            'native_contract_qualified': False}


def paired_readout(rows: dict) -> dict:
    failed = [arm for arm in ARMS if not valid_score(rows.get(arm, {}))]
    return {'n_assets': 1, 'n_arms': 3, 'n_complete_assets': int(not failed),
            'failed_or_missing_arms': failed,
            'complete_pair_deltas': {arm: {key: rows[arm][key]-rows['native'][key] for key in METRICS}
                                     for arm in ARMS[1:]} if not failed else {},
            'direction': 'control minus native; smaller native distances are better',
            'confidence_interval': None, 'candidate_methods_tested': False,
            'scope': 'One development asset/inference seed; descriptive, no population or method verdict'}


def bundle_refs(request: dict) -> list[dict]:
    excluded = request['ground_truth_ref']['path']
    refs = {ref['path']: ref for ref in request['input_refs']+request['code_refs'] if ref['path'] != excluded}
    return list(refs.values())


def copy_evidence_bundle(root: Path, request: dict, output: Path) -> list[dict]:
    copied, missing = [], []
    bundle = output/'bundle'; bundle.mkdir(exist_ok=True)
    for ref in bundle_refs(request):
        try:
            target = bundle/ref['path']
            if not target.exists():
                source = resolve_ref(root, ref)
                target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
            if census.digest(target) != ref['sha256']: raise ValueError('Evidence copy mismatch')
            copied.append(ref)
        except Exception as exc: missing.append({'ref': ref, 'error': f'{type(exc).__name__}: {exc}'})
    census.write_json(output/'bundle-manifest.json', {'files': copied, 'missing': missing,
        'excluded_ground_truth_ref': request['ground_truth_ref'],
        'reason': 'Original released GT stays at its dataset revision; no GT redistribution',
        'status': 'complete_transport' if not missing else 'incomplete_transport'})
    return missing


def scorer_command(root: Path, request: dict, arm: str, stage: Path, device: str) -> list[str]:
    if arm not in ARMS or device not in ('cuda', 'cuda:0'): raise ValueError('Undeclared arm/device')
    return [sys.executable, str(Path(census.__file__).resolve()),
        '--case-dir', str(resolve_ref(root, request['manifest_ref']).parent),
        '--gt-dir', str(resolve_ref(root, request['ground_truth_ref']).parent.parent),
        '--output', str(stage/'scores.json'), '--manifest', str(stage/'manifest.json'),
        '--repo-root', str((root/request['repo_root']).resolve()), '--device', device, '--seed', '44']


def contract_scorer_command(root: Path, request: dict) -> list[str]:
    """Frozen faithful-harness command; prediction JSON is the arm manifest."""
    command = scorer_command(root, request, 'native', root/'unused-output', 'cuda:0')
    command[command.index('--output')+1] = '{output}'
    command[command.index('--manifest')+1] = '{predictions}'
    return command


def check_contract_binding(root: Path, request: dict, contract: dict):
    expected_sampling = {'policy': 'official-actionbench-full-sequence', 'parameters': request['native_protocol']}
    if contract['sampling'] != expected_sampling:
        raise ValueError('Native contract sampling differs from the executed full scorer')
    for key in ('n_pts_chamfer', 'n_pts_icp', 'icp_initial_rotations', 'icp_iterations'):
        if contract['budget'].get(key) != request['native_protocol'][key]:
            raise ValueError('Native contract budget mismatch: '+key)
    if contract['budget'].get('frames') != 16: raise ValueError('Full 16-frame native budget required')
    expected_metrics = [{'name': k, 'output_path': ['cases', 0, k]} for k in METRICS]
    if contract['metrics'] != expected_metrics:
        raise ValueError('Contract must read the actual three official case metrics')
    scorer = contract['scorer']
    if scorer['kind'] != 'faithful_harness' or scorer['command'] != contract_scorer_command(root, request) or scorer['cwd'] != '.' or scorer['denominator_path'] != ['denominator', 'n_declared']:
        raise ValueError('Frozen faithful scorer command/denominator differs from execution')
    if scorer['output'] != {'format': 'json', 'source': 'file', 'path': '{output}'}:
        raise ValueError('Frozen scorer must consume the existing native JSON output')
    expected_prediction = {'format': 'json', 'records_path': ['cases'], 'id_path': ['uid']}
    if contract['prediction_format'] != expected_prediction:
        raise ValueError('Native prediction records must be the one-arm UID manifest')
    actual = {r['path']: r for r in request['code_refs']}
    required_scorer = [file_ref(root, Path(census.__file__).resolve())]
    required_scorer += [file_ref(root, root/request['repo_root']/'actionbench'/n) for n in census.OFFICIAL_FILES]
    declared = {r['path']: r for r in scorer['code_refs']}
    if any(declared.get(r['path']) != r for r in required_scorer) or any(actual.get(p) != r for p, r in declared.items()):
        raise ValueError('Executed official/census scorer code must match all frozen scorer refs')
    for role, arm in contract['arm_requirements'].items():
        if any(actual.get(r['path']) != r for r in arm['implementation_refs']):
            raise ValueError('Unstaged or mismatched implementation for role '+role)
        adapter_ref = file_ref(root, Path(__file__).with_name('simple_mesh_controls.py'))
        if adapter_ref not in arm['implementation_refs']:
            raise ValueError('Every baseline/control must bind its actual preparation adapter')


class DeviceSamples:
    """Sample the allocated physical device; observations are not an exact peak."""
    def __init__(self, device_uuid: str, output: Path):
        if not device_uuid.startswith('GPU-') or any(c in device_uuid for c in '\n\r, '):
            raise ValueError('Actual allocated physical GPU UUID required')
        self.device_uuid, self.output = device_uuid, output
        self.stop = threading.Event(); self.thread = None
        self.errors = []; self.memory = []

    def capture(self):
        command = ['nvidia-smi', '--id='+self.device_uuid,
            '--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader,nounits']
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
            values = [v.strip() for v in result.stdout.strip().split(',')]
            if len(values) != 5 or values[0] != self.device_uuid:
                raise ValueError('Allocated GPU telemetry identity mismatch')
            sample = {'observed_at': datetime.now(timezone.utc).isoformat(), 'uuid': values[0],
                      'name': values[1], 'total_mib': float(values[2]),
                      'used_mib': float(values[3]), 'utilization_percent': float(values[4])}
            if not all(math.isfinite(sample[k]) and sample[k] >= 0 for k in ('total_mib', 'used_mib', 'utilization_percent')):
                raise ValueError('Invalid device telemetry')
            self.memory.append(sample['used_mib'])
        except Exception as exc:
            sample = {'observed_at': datetime.now(timezone.utc).isoformat(), 'error': f'{type(exc).__name__}: {exc}'}
            self.errors.append(sample['error'])
        with self.output.open('a') as stream: stream.write(json.dumps(sample, allow_nan=False)+'\n')

    def poll(self):
        while not self.stop.wait(1.): self.capture()

    def start(self):
        self.capture()
        if self.errors: raise RuntimeError('Allocated GPU identity/telemetry must be available before scoring')
        self.thread = threading.Thread(target=self.poll, daemon=True); self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread is not None: self.thread.join(timeout=12)
        self.capture()


def score_request(root: Path, request_path: Path, output: Path, device: str, gpu_uuid: str):
    root, request_path, output = map(lambda p: Path(p).resolve(), (root, request_path, output))
    output.relative_to(root)
    request = census.read_json(request_path)
    verify_request(root, request)
    if device not in ('cuda', 'cuda:0'): raise ValueError('Single allocated CUDA device required')
    output.mkdir(parents=True, exist_ok=False)
    census.write_json(output/'request.json', request)
    started = time.monotonic()
    record = {'kind': 'baseline-control-scoring-record', 'version': '1.0.0',
              'uid': request['uid'], 'inference_seed': request['inference_seed'],
              'started_at': datetime.now(timezone.utc).isoformat(), 'execution_id': str(uuid.uuid4()),
              'status': 'running', 'arms': initial_arm_records(), 'native_contract_qualified': False,
              'candidate_methods_tested': False, 'request_digest': request['request_digest'],
              'resources': {}, 'timing_scope': 'Cached three-arm scoring and fresh repeated scoring plus packaging; excludes inference and control preparation'}
    record['readout'] = paired_readout({})
    record['qualification_outcome'] = 'inconclusive'
    record['transport_status'] = 'incomplete'
    census.write_json(output/'record.json', record)
    telemetry = DeviceSamples(gpu_uuid, output/'device-samples.jsonl')
    try:
        missing = copy_evidence_bundle(root, request, output)
        if missing: raise ValueError('Cannot retain full source evidence before native scoring')
        record['transport_status'] = 'pinned_inputs_copied_outputs_pending'
        census.write_json(output/'record.json', record)
        if os.environ.get('CUDA_VISIBLE_DEVICES') != gpu_uuid:
            raise ValueError('The outer harness must allocate exactly this GPU UUID')
        import torch
        torch.cuda.set_device(0)
        torch.cuda.synchronize()
        properties = torch.cuda.get_device_properties(0)
        record['resources'] = {'cuda_visible_devices': __import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
            'torch_device_name': properties.name, 'device_total_mib': properties.total_memory/2**20,
            'torch_version': torch.__version__, 'torch_cuda_version': torch.version.cuda,
            'python': platform.python_version(), 'packages': {}}
        telemetry.start()
        for package in ('numpy', 'trimesh', 'scipy', 'pytorch3d'):
            try: record['resources']['packages'][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError: record['resources']['packages'][package] = None
        for entry in request['arms']:
            arm = entry['arm']; runs = record['arms'][arm]['passes']
            for pass_index in range(2):
                stage = output/(arm+'-pass'+str(pass_index)); stage.mkdir()
                manifest = {'cases': [{'case_id': request['uid']+'-'+arm, 'uid': request['uid'], 'case_dir': arm}]}
                census.write_json(stage/'manifest.json', manifest)
                command = scorer_command(root, request, arm, stage, device)
                start = time.monotonic()
                exit_code = None
                try:
                    with (stage/'stdout.log').open('x') as stdout, (stage/'stderr.log').open('x') as stderr:
                        process = subprocess.run(command, cwd=root, stdout=stdout, stderr=stderr, check=False)
                    exit_code = process.returncode
                    torch.cuda.synchronize()
                    if not (stage/'scores.json').is_file(): raise ValueError('Native scorer exited without a score file')
                    raw = census.read_json(stage/'scores.json')
                    if len(raw.get('cases', [])) != 1 or raw['cases'][0].get('case_id') != request['uid']+'-'+arm or raw.get('denominator', {}).get('n_declared') != 1:
                        raise ValueError('Scorer output denominator/identity mismatch')
                    row = raw['cases'][0]
                    if exit_code != 0 and row.get('status') == 'success':
                        raise ValueError('Nonzero native scorer exit')
                    row_record = {key: row.get(key) for key in ('status', *METRICS, 'error')}
                except Exception as exc:
                    row_record = {'status': 'error', 'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()}
                row_record.update(exit_code=exit_code, elapsed_seconds=time.monotonic()-start,
                    command=command, scorer_output_ref=file_ref(root, stage/'scores.json') if (stage/'scores.json').is_file() else None)
                runs[pass_index] = row_record
                census.write_json(output/'record.json', record)
            record['arms'][arm]['replay_comparison'] = replay_comparison(runs[0], runs[1])
        verify_request(root, request)  # Detect input/code mutation across execution.
        record['readout'] = paired_readout({arm: data['passes'][0] for arm, data in record['arms'].items()})
        identical = all(data['replay_comparison']['status'] == 'identical' for data in record['arms'].values())
        record['status'] = 'execution_completed' if identical else 'inconclusive'
    except (Exception, KeyboardInterrupt) as exc:
        record.update(status='error', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    finally:
        if telemetry.thread is not None: telemetry.close()
        record['resources'].update(device_uuid=gpu_uuid,
            observed_device_memory_max_mib=max(telemetry.memory) if telemetry.memory else None,
            device_sample_count=len(telemetry.memory), device_sample_errors=telemetry.errors,
            whole_device_peak_mib=None,
            memory_scope='One-second physical-device samples including other processes; exact peak and full-generation peak remain unmeasured')
        record['readout'] = paired_readout({a: r['passes'][0] for a, r in record['arms'].items()})
        missing = copy_evidence_bundle(root, request, output)
        if missing: record.update(status='inconclusive', evidence_copy_failures=missing)
        record['transport_status'] = 'inputs_complete' if not missing else 'incomplete'
        record['qualification_outcome'] = 'pending_trusted_host_replay' if record['status'] == 'execution_completed' else 'inconclusive'
    record['completed_at'] = datetime.now(timezone.utc).isoformat()
    record['elapsed_seconds'] = time.monotonic()-started
    record['cost_usable_for_full_eight_hour_queue'] = False
    census.write_json(output/'record.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    prepare = sub.add_parser('request')
    prepare.add_argument('--root', type=Path, required=True)
    for name in ('source-case', 'controls-dir', 'ground-truth', 'repo-root', 'population', 'output'):
        prepare.add_argument('--'+name, type=Path, required=True)
    execute = sub.add_parser('score')
    execute.add_argument('--root', type=Path, required=True)
    execute.add_argument('--request', type=Path, required=True)
    execute.add_argument('--output', type=Path, required=True)
    execute.add_argument('--device', default='cuda:0')
    execute.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args()
    if args.operation == 'request':
        request = make_request(args.root, source_case=args.source_case, controls_dir=args.controls_dir,
            ground_truth=args.ground_truth, repo_root=args.repo_root, population=args.population)
        args.output.resolve().relative_to(args.root.resolve())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream: stream.write(json.dumps(request, indent=2, allow_nan=False)+'\n')
        print(json.dumps({'request': str(args.output), 'request_digest': request['request_digest'],
                          'execution_started': False, 'native_contract_qualified': False}))
        return 0
    record = score_request(args.root, args.request, args.output, args.device, args.gpu_uuid)
    print(json.dumps({'status': record['status'], 'native_contract_qualified': False}))
    return 0 if record['status'] == 'execution_completed' else 2


if __name__ == '__main__': raise SystemExit(main())
