"""Freeze C14's five logical native roles without scoring or selecting B*.

This module consumes already generated full-sequence artifacts.  It validates
their shared native identity and emits a content-addressed request that a
separate admitted official-scoring plan can consume.  It never generates a
mesh, chooses a comparator from outcomes, runs ActionBench, or advances a gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from research_math import corotational_residual_candidate as candidate_module
from research_math import simple_mesh_controls as controls_module


CANDIDATE_ID = '4d-math-20261006-c14'
ROLES = ('b0', 'b_star', 'world_gaussian', 'body_gaussian',
         'corotational_residual')
CONTROL_ROLES = ('world_gaussian', 'body_gaussian')
POSE_SOURCE = 'predicted_sequence_geometry_only'
POSE_FIT = ('uniform-vertex proper Kabsch to centered frame zero; '
            'frozen during repair')
REPAIR_OPERATOR = 'anchored_nonuniform_group_tv_first_difference'


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> dict:
    def object_without_duplicates(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('Duplicate JSON key: ' + key)
            value[key] = item
        return value
    value = json.loads(Path(path).read_text(),
                       object_pairs_hook=object_without_duplicates)
    if not isinstance(value, dict):
        raise ValueError('Expected JSON object: ' + str(path))
    return value


def canonical_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(',', ':'), allow_nan=False
    ).encode()).hexdigest()


def file_ref(root: Path, path: Path) -> dict:
    root, path = Path(root).resolve(), Path(path).resolve()
    return {'path': path.relative_to(root).as_posix(), 'sha256': digest(path)}


def resolve_ref(root: Path, ref: dict) -> Path:
    if (not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}
            or not isinstance(ref.get('sha256'), str)
            or len(ref['sha256']) != 64
            or any(character not in '0123456789abcdef'
                   for character in ref['sha256'])):
        raise ValueError('Exact path/sha256 reference required')
    relative = Path(ref['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Project-relative nonescaping reference required')
    root = Path(root).resolve()
    candidate = root / relative
    if candidate.is_symlink() or any(parent.is_symlink()
                                     for parent in candidate.parents
                                     if parent != root.parent):
        raise ValueError('Physical nonsymlink evidence file required')
    path = candidate.resolve()
    path.relative_to(root)
    if not path.is_file():
        raise ValueError('Physical evidence file required: ' + ref['path'])
    if digest(path) != ref['sha256']:
        raise ValueError('Changed pinned file: ' + ref['path'])
    return path


def _read_arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name].copy() for name in saved.files}


def _validate_source(arrays: dict) -> None:
    required = {'vertices', 'faces', 'timesteps', 'frame_indices',
                'query_vertex_ids'}
    if not required.issubset(arrays):
        raise ValueError('Complete native mesh arrays required')
    vertices = arrays['vertices']
    if (vertices.dtype != np.dtype(np.float32) or vertices.ndim != 3
            or vertices.shape[0] != 16 or vertices.shape[-1] != 3
            or vertices.shape[1] < 3):
        raise ValueError('Exactly 16 float32 full vertex frames required')
    faces = arrays['faces']
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or not np.issubdtype(faces.dtype, np.integer)
            or faces.min() < 0 or faces.max() >= vertices.shape[1]):
        raise ValueError('Valid shared triangle topology required')
    if (not np.issubdtype(arrays['frame_indices'].dtype, np.integer)
            or not np.array_equal(arrays['frame_indices'], np.arange(16))):
        raise ValueError('Original 16-frame order required')
    if (not np.issubdtype(arrays['query_vertex_ids'].dtype, np.integer)
            or not np.array_equal(arrays['query_vertex_ids'],
                                   np.arange(vertices.shape[1]))):
        raise ValueError('Original vertex identity required')
    times = arrays['timesteps']
    if (times.ndim != 1 or len(times) != 16
            or not np.issubdtype(times.dtype, np.number)):
        raise ValueError('One strictly increasing native timestamp per frame required')
    times64 = times.astype(np.float64)
    if not np.isfinite(times64).all() or np.any(np.diff(times64) <= 0):
        raise ValueError('One strictly increasing native timestamp per frame required')
    for name, array in arrays.items():
        if (not np.issubdtype(array.dtype, np.number)
                or not np.isfinite(array).all()):
            raise ValueError('Finite numeric native array required: ' + name)


def _validate_arm_arrays(source: dict, arm: dict) -> None:
    _validate_source(source)
    _validate_source(arm)
    if set(arm) != set(source):
        raise ValueError('Arm must retain every native array and no new arrays')
    if (arm['vertices'].shape != source['vertices'].shape
            or arm['vertices'].dtype != source['vertices'].dtype
            or not np.array_equal(arm['vertices'][0], source['vertices'][0])):
        raise ValueError('Arm changed shape, dtype, or exact frame-zero anchor')
    for name in set(source) - {'vertices'}:
        if (arm[name].dtype != source[name].dtype
                or not np.array_equal(arm[name], source[name])):
            raise ValueError('Arm changed native topology/identity/metadata: ' + name)


def _implementation_sha(report: dict, role: str) -> str:
    value = report.get('implementation_sha256')
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in '0123456789abcdef' for character in value)):
        raise ValueError('Exact implementation identity required: ' + role)
    expected = None
    if role in CONTROL_ROLES:
        expected = digest(Path(controls_module.__file__))
    elif role == 'corotational_residual':
        expected = digest(Path(candidate_module.__file__))
    if expected is not None and value != expected:
        raise ValueError('Arm implementation differs from current source: ' + role)
    return value


def _validate_control_report(report: dict, role: str) -> None:
    if report.get('baseline_arm') != role:
        raise ValueError('Control role identity mismatch: ' + role)
    sigma = report.get('sigma_frames')
    if (isinstance(sigma, bool) or not isinstance(sigma, (int, float))
            or not math.isfinite(sigma) or sigma <= 0):
        raise ValueError('Frozen positive Gaussian sigma required: ' + role)
    _implementation_sha(report, role)


def _validate_candidate_report(report: dict) -> None:
    parameters = report.get('parameters')
    expected_parameter_keys = {
        'weight', 'rho', 'absolute_tolerance', 'relative_tolerance',
        'max_iterations', 'timestamp_units',
    }
    if (report.get('candidate_id') != CANDIDATE_ID
            or report.get('method_id') != CANDIDATE_ID
            or report.get('arm_role') != candidate_module.ARM
            or report.get('pose_source') != POSE_SOURCE
            or report.get('pose_fit') != POSE_FIT
            or report.get('repair_operator') != REPAIR_OPERATOR
            or not isinstance(parameters, dict)
            or set(parameters) != expected_parameter_keys
            or parameters.get('timestamp_units') !=
            'supplied native sequence units; weight scales in those units'):
        raise ValueError('Exact C14 candidate construction required')
    for name in ('weight', 'rho', 'absolute_tolerance', 'relative_tolerance'):
        value = parameters.get(name)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError('Positive finite C14 parameter required: ' + name)
    iterations = parameters.get('max_iterations')
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError('Positive C14 max_iterations required')
    if report.get('status') == 'completed':
        solver = report.get('solver')
        if (not isinstance(solver, dict) or solver.get('operator') != REPAIR_OPERATOR
                or solver.get('converged') is not True
                or any(solver.get(name) != parameters[name] for name in (
                    'weight', 'rho', 'absolute_tolerance', 'relative_tolerance',
                    'max_iterations'))):
            raise ValueError('Converged solver certificate matching parameters required')
    _implementation_sha(report, 'corotational_residual')


def _validate_pose_arrays(path: Path, source_vertices: np.ndarray) -> dict:
    poses = _read_arrays(path)
    if set(poses) != {'rotation_rows', 'centroids', 'singular_values'}:
        raise ValueError('Exact body-Gaussian pose arrays required')
    expected = controls_module.smooth_body(source_vertices, 1.0)[1]
    # Sigma does not affect the fitted pose, so 1.0 is a harmless validation value.
    for name, shape in (('rotation_rows', (16, 3, 3)),
                        ('centroids', (16, 3)),
                        ('singular_values', (16, 3))):
        value = poses[name]
        if value.shape != shape or not np.issubdtype(value.dtype, np.floating):
            raise ValueError('Invalid body-Gaussian pose array: ' + name)
        if not np.isfinite(value).all() or not np.array_equal(value, expected[name]):
            raise ValueError('Body-Gaussian poses differ from current construction: ' + name)
    return poses


def _validate_candidate_certificate(path: Path, source: dict,
                                    repaired_vertices: np.ndarray,
                                    report: dict) -> None:
    certificate = _read_arrays(path)
    expected_keys = {
        'anchor', 'rotation_rows', 'centroids', 'singular_values', 'fit_rms',
        'observed_body_residual', 'repaired_body_residual',
    }
    if set(certificate) != expected_keys:
        raise ValueError('Exact C14 corotational certificate arrays required')
    factors = candidate_module.fit_frozen_rigid_factors(source['vertices'])
    expected = {
        'anchor': factors['anchor'], 'rotation_rows': factors['rotation_rows'],
        'centroids': factors['centroids'],
        'singular_values': factors['singular_values'], 'fit_rms': factors['fit_rms'],
        'observed_body_residual': factors['body_residual'],
    }
    for name, value in expected.items():
        actual = certificate[name]
        if (actual.dtype != value.dtype or actual.shape != value.shape
                or not np.isfinite(actual).all() or not np.array_equal(actual, value)):
            raise ValueError('C14 certificate differs from frozen pose fit: ' + name)
    repaired = certificate['repaired_body_residual']
    if (repaired.dtype != np.dtype(np.float64)
            or repaired.shape != factors['body_residual'].shape
            or not np.isfinite(repaired).all()
            or not np.array_equal(repaired[0], factors['body_residual'][0])):
        raise ValueError('Invalid anchored C14 repaired body residual')
    parameters = report['parameters']
    expected_repaired, expected_solver = candidate_module.repair_body_residual(
        factors['body_residual'], source['timesteps'],
        weight=parameters['weight'], rho=parameters['rho'],
        absolute_tolerance=parameters['absolute_tolerance'],
        relative_tolerance=parameters['relative_tolerance'],
        max_iterations=parameters['max_iterations'])
    if (not np.array_equal(repaired, expected_repaired)
            or report.get('solver') != expected_solver):
        raise ValueError('C14 residual/solver differs from the frozen optimizer')
    rebuilt = candidate_module.reconstruct_with_fixed_pose(
        factors['anchor'], repaired, factors['rotation_rows'], factors['centroids'])
    rebuilt[0] = source['vertices'][0].astype(np.float64)
    if not np.array_equal(rebuilt.astype(np.float32), repaired_vertices):
        raise ValueError('C14 certificate does not reconstruct scored sequence')


def _validate_freeze_core(freeze: dict) -> None:
    core = {key: value for key, value in freeze.items()
            if key != 'freeze_digest'}
    if freeze.get('freeze_digest') != canonical_digest(core):
        raise ValueError('Freeze digest mismatch')
    if (freeze.get('kind') != 'c14-native-comparison-freeze'
            or freeze.get('version') != 1
            or freeze.get('candidate_id') != CANDIDATE_ID
            or not isinstance(freeze.get('frozen_at'), str)
            or not freeze['frozen_at']):
        raise ValueError('Current frozen C14 comparison record required')
    try:
        frozen_at = datetime.fromisoformat(freeze['frozen_at'].replace('Z', '+00:00'))
    except ValueError as error:
        raise ValueError('Timezone-aware ISO frozen_at required') from error
    if frozen_at.tzinfo is None:
        raise ValueError('Timezone-aware ISO frozen_at required')
    if freeze.get('scoring_seed') != 44:
        raise ValueError('Official scoring seed 44 is fixed')
    if (freeze.get('primary_metric') != 'cd_motion'
            or freeze.get('guardrail_metrics') != ['cd_3d', 'cd_4d']):
        raise ValueError('C14 official primary/guardrail metrics are fixed')
    rows = freeze.get('roles')
    if (not isinstance(rows, list)
            or [row.get('role') if isinstance(row, dict) else None
                for row in rows] != list(ROLES)):
        raise ValueError('Exactly the five ordered roles must be frozen')
    if (not isinstance(freeze.get('uid'), str) or not freeze['uid']
            or Path(freeze['uid']).name != freeze['uid']
            or freeze['uid'] in ('.', '..')):
        raise ValueError('Frozen native UID required')
    if isinstance(freeze.get('inference_seed'), bool) or not isinstance(
            freeze.get('inference_seed'), int):
        raise ValueError('Frozen integer inference seed required')


def _report_identity(report: dict, freeze: dict, role: str) -> None:
    if (report.get('uid') != freeze['uid']
            or report.get('seed') != freeze['inference_seed']):
        raise ValueError('Arm UID/inference seed mismatch: ' + role)
    expected_source = freeze['source_sequence_ref']['sha256']
    expected_report = freeze['source_report_ref']['sha256']
    if role == 'b0':
        if (report.get('status') != 'completed'
                or report.get('sha256', {}).get('sequence.npz') != expected_source):
            raise ValueError('B0 must be the exact completed source sequence')
        return
    if (report.get('source_sequence_sha256') != expected_source
            or report.get('source_report_sha256') != expected_report):
        raise ValueError('Arm does not bind the same source bytes: ' + role)
    if role == 'world_gaussian' and report.get('baseline_arm') != 'world_gaussian':
        raise ValueError('Gaussian role must be the world Gaussian construction')
    if (role == 'body_gaussian'
            and report.get('baseline_arm') != 'body_gaussian'):
        raise ValueError('Body Gaussian role identity mismatch')
    if role == 'corotational_residual' and (
            report.get('candidate_id') != CANDIDATE_ID
            or report.get('arm_role') != candidate_module.ARM):
        raise ValueError('Corotational role must be the C14 candidate construction')


def make_request(root: Path, *, freeze_path: Path, _verify: bool = True) -> dict:
    """Validate a prospective freeze and emit a hash-bound scoring request."""
    root, freeze_path = Path(root).resolve(), Path(freeze_path).resolve()
    freeze_path.relative_to(root)
    freeze = read_json(freeze_path)
    _validate_freeze_core(freeze)
    source_path = resolve_ref(root, freeze['source_sequence_ref'])
    source_report_path = resolve_ref(root, freeze['source_report_ref'])
    source_report = read_json(source_report_path)
    if (source_path.name != 'sequence.npz'
            or source_report_path != source_path.with_name('report.json')):
        raise ValueError('Source report and sequence must share one case directory')
    _report_identity(source_report, freeze, 'b0')
    source_arrays = _read_arrays(source_path)
    _validate_source(source_arrays)
    pinned = [file_ref(root, freeze_path), freeze['source_sequence_ref'],
              freeze['source_report_ref'], freeze['b_star_decision_ref']]
    decision = read_json(resolve_ref(root, freeze['b_star_decision_ref']))
    decision_core = {key: value for key, value in decision.items()
                     if key != 'decision_digest'}
    decision_fields = {
        'kind', 'version', 'candidate_id', 'uid', 'inference_seed',
        'decided_at', 'selected_role', 'selected_method_id',
        'selected_without_c14_native_outcomes', 'selection_basis_refs',
        'decision_digest',
    }
    if (set(decision) != decision_fields
            or decision.get('kind') != 'c14-b-star-decision'
            or decision.get('version') != 1
            or decision.get('candidate_id') != CANDIDATE_ID
            or decision.get('uid') != freeze['uid']
            or decision.get('inference_seed') != freeze['inference_seed']
            or decision.get('decision_digest') != canonical_digest(decision_core)
            or decision.get('selected_without_c14_native_outcomes') is not True):
        raise ValueError('B* must be prospectively selected without C14 native outcomes')
    basis_refs = decision.get('selection_basis_refs')
    if not isinstance(basis_refs, list) or not basis_refs:
        raise ValueError('B* requires nonempty pinned prospective selection basis')
    try:
        decided_at = datetime.fromisoformat(
            decision['decided_at'].replace('Z', '+00:00'))
    except (KeyError, AttributeError, ValueError) as error:
        raise ValueError('Timezone-aware B* decided_at required') from error
    if decided_at.tzinfo is None:
        raise ValueError('Timezone-aware B* decided_at required')
    freeze_time = datetime.fromisoformat(freeze['frozen_at'].replace('Z', '+00:00'))
    if decided_at > freeze_time:
        raise ValueError('B* decision must precede or equal the comparison freeze')
    for ref in basis_refs:
        resolve_ref(root, ref)
        pinned.append(ref)
    if any(key in decision for key in ('observed_scores', 'metrics', 'results')):
        raise ValueError('B* decision may not embed outcome-derived selection data')
    physical = {}
    normalized = []
    for row in freeze['roles']:
        role = row['role']
        if not isinstance(row.get('method_id'), str) or not row['method_id']:
            raise ValueError('Nonempty frozen method_id required: ' + role)
        alias = row.get('alias_of')
        if alias is not None:
            if role != 'b_star' or alias not in ('b0', 'world_gaussian',
                                                  'body_gaussian'):
                raise ValueError('B* alias must name a frozen simple control role')
            if set(row) != {'role', 'method_id', 'alias_of'}:
                raise ValueError('Alias must not carry competing physical refs')
            normalized.append({'role': role, 'method_id': row['method_id'],
                               'alias_of': alias})
            if decision.get('selected_role') != alias:
                raise ValueError('B* decision must name the exact aliased role')
            if decision.get('selected_method_id') != row['method_id']:
                raise ValueError('B* decision must name the exact frozen method')
            continue
        expected_keys = {'role', 'method_id', 'report_ref', 'sequence_ref'}
        if role == 'b0':
            expected_keys.add('implementation_ref')
        if role == 'body_gaussian':
            expected_keys.add('poses_ref')
        if role == 'corotational_residual':
            expected_keys.add('certificate_ref')
        if set(row) != expected_keys:
            raise ValueError('Physical role needs exact report/sequence refs: ' + role)
        report_path = resolve_ref(root, row['report_ref'])
        if report_path.name != 'report.json':
            raise ValueError('Arm report must be report.json: ' + role)
        report = read_json(report_path)
        _report_identity(report, freeze, role)
        b0_implementation_ref = None
        if role == 'b0':
            b0_implementation = resolve_ref(root, row['implementation_ref'])
            b0_implementation_ref = row['implementation_ref']
            pinned.append(b0_implementation_ref)
            reported_implementation = report.get('implementation_sha256')
            if (reported_implementation is not None
                    and reported_implementation != digest(b0_implementation)):
                raise ValueError('B0 report and frozen implementation differ')
        elif role in CONTROL_ROLES:
            _validate_control_report(report, role)
        elif role == 'corotational_residual':
            _validate_candidate_report(report)
            if row['method_id'] != CANDIDATE_ID:
                raise ValueError('C14 candidate method_id is fixed')
        elif role == 'b_star':
            _implementation_sha(report, role)
        if role == 'b_star' and (
                decision.get('selected_role') != 'b_star'
                or decision.get('selected_method_id') != row['method_id']
                or report.get('method_id') != row['method_id']):
            raise ValueError('Physical B* identity must match its prospective decision')
        if report.get('status') == 'error':
            if (role == 'b0' or row['sequence_ref'] is not None
                    or row.get('certificate_ref') is not None
                    or row.get('poses_ref') is not None):
                raise ValueError('Failed non-B0 arm must not claim a sequence: ' + role)
            error = report.get('error')
            exception_type = report.get('exception_type')
            if (not isinstance(error, str) or not error.strip()
                    or len(error) > 4096 or not isinstance(exception_type, str)
                    or not exception_type.strip() or len(exception_type) > 256):
                raise ValueError('Bounded nonempty failure evidence required: ' + role)
            physical[role] = None
            pinned.append(row['report_ref'])
            normalized.append({
                'role': role, 'method_id': row['method_id'],
                'report_ref': row['report_ref'], 'sequence_ref': None,
                'preparation_status': 'error', 'case_id': None,
                'preparation_error': error,
                'preparation_exception_type': exception_type,
                'implementation_sha256': report['implementation_sha256'],
                **({'sigma_frames': report['sigma_frames']}
                   if role in CONTROL_ROLES else {}),
            })
            continue
        if report.get('status') != 'completed' or row['sequence_ref'] is None:
            raise ValueError('Known completed/error preparation status required: ' + role)
        if role == 'b0' and (row['report_ref'] != freeze['source_report_ref']
                             or row['sequence_ref'] != freeze['source_sequence_ref']):
            raise ValueError('B0 refs must be the exact source refs')
        sequence_path = resolve_ref(root, row['sequence_ref'])
        if sequence_path != report_path.with_name('sequence.npz'):
            raise ValueError('Arm report/sequence must share one directory: ' + role)
        if report.get('sha256', {}).get('sequence.npz') != row['sequence_ref']['sha256']:
            raise ValueError('Arm report does not bind current sequence: ' + role)
        arm_arrays = _read_arrays(sequence_path)
        _validate_arm_arrays(source_arrays, arm_arrays)
        if role == 'world_gaussian':
            expected_vertices = controls_module.smooth_world(
                source_arrays['vertices'], report['sigma_frames'])
            if not np.array_equal(arm_arrays['vertices'], expected_vertices):
                raise ValueError('World-Gaussian vertices differ from current construction')
        poses_ref = None
        if role == 'body_gaussian':
            poses_path = resolve_ref(root, row['poses_ref'])
            if (poses_path != report_path.with_name('poses.npz')
                    or report.get('sha256', {}).get('poses.npz')
                    != row['poses_ref']['sha256']):
                raise ValueError('Body-Gaussian pose evidence is missing or stale')
            _validate_pose_arrays(poses_path, source_arrays['vertices'])
            expected_vertices, _ = controls_module.smooth_body(
                source_arrays['vertices'], report['sigma_frames'])
            if not np.array_equal(arm_arrays['vertices'], expected_vertices):
                raise ValueError('Body-Gaussian vertices differ from current construction')
            poses_ref = row['poses_ref']
            pinned.append(poses_ref)
        certificate_ref = None
        if role == 'corotational_residual':
            certificate = resolve_ref(root, row['certificate_ref'])
            if (certificate != report_path.with_name('certificate.npz')
                    or report.get('sha256', {}).get('certificate.npz')
                    != row['certificate_ref']['sha256']):
                raise ValueError('C14 corotational certificate is missing or stale')
            certificate_ref = row['certificate_ref']
            _validate_candidate_certificate(
                certificate, source_arrays, arm_arrays['vertices'], report)
            pinned.append(certificate_ref)
        physical[role] = {
            'role': role, 'method_id': row['method_id'],
            'report_ref': row['report_ref'], 'sequence_ref': row['sequence_ref'],
            'case_id': freeze['uid'] + '-' + role,
            'preparation_status': 'completed',
            'implementation_sha256': (b0_implementation_ref['sha256']
                                      if role == 'b0'
                                      else report['implementation_sha256']),
        }
        if b0_implementation_ref is not None:
            physical[role]['implementation_ref'] = b0_implementation_ref
        if role in CONTROL_ROLES:
            physical[role]['sigma_frames'] = report['sigma_frames']
        if poses_ref is not None:
            physical[role]['poses_ref'] = poses_ref
        if certificate_ref is not None:
            physical[role]['certificate_ref'] = certificate_ref
        pinned.extend([row['report_ref'], row['sequence_ref']])
        normalized.append(dict(physical[role]))
    gaussian_sigmas = [next(row for row in normalized if row['role'] == role)[
        'sigma_frames'] for role in CONTROL_ROLES]
    if gaussian_sigmas[0] != gaussian_sigmas[1]:
        raise ValueError('World/body Gaussian controls require one fair frozen sigma')
    # Every byte-identical completed arm is one physical measurement.  Preserve
    # all five logical roles, but score each unique sequence exactly once.
    canonical_by_sha = {}
    scoring_cases = []
    for role in ROLES:
        item = physical.get(role)
        if item is None:
            continue
        sequence_sha = item['sequence_ref']['sha256']
        canonical = canonical_by_sha.get(sequence_sha)
        if canonical is None:
            canonical_by_sha[sequence_sha] = item
            scoring_cases.append(item)
        else:
            item['case_id'] = canonical['case_id']
            normalized_row = next(row for row in normalized if row['role'] == role)
            normalized_row['case_id'] = canonical['case_id']
    for row in normalized:
        if row.get('alias_of'):
            if row['alias_of'] not in physical:
                raise ValueError('B* alias target must be a declared physical role')
            target = physical[row['alias_of']]
            target_row = next(item for item in normalized
                              if item['role'] == row['alias_of'])
            if row['method_id'] != target_row['method_id']:
                raise ValueError('B* alias method_id must match its physical target')
            row['case_id'] = target['case_id'] if target is not None else None
            row['preparation_status'] = ('completed' if target is not None else 'error')
            if target is None:
                row['preparation_error'] = target_row.get('preparation_error')
    role_to_case = {row['role']: row['case_id'] for row in normalized}
    logical_denominator = {
        'n_roles': len(ROLES),
        'roles': [{'role': row['role'],
                   'preparation_status': row['preparation_status'],
                   'case_id': row['case_id']}
                  for row in normalized],
        'failure_policy': ('Every frozen role remains in the denominator; '
                           'missing/error metrics are never zero-imputed.'),
    }
    unique = {ref['path']: ref for ref in pinned}
    request = {
        'kind': 'c14-native-comparison-request', 'version': 1,
        'candidate_id': CANDIDATE_ID, 'uid': freeze['uid'],
        'inference_seed': freeze['inference_seed'], 'scoring_seed': 44,
        'primary_metric': 'cd_motion',
        'guardrail_metrics': ['cd_3d', 'cd_4d'],
        'roles': normalized, 'role_to_case': role_to_case,
        'logical_denominator': logical_denominator,
        'scoring_cases': scoring_cases,
        'freeze_ref': file_ref(root, freeze_path),
        'b_star_decision_ref': freeze['b_star_decision_ref'],
        'source_sequence_ref': freeze['source_sequence_ref'],
        'source_report_ref': freeze['source_report_ref'],
        'input_refs': list(unique.values()),
        'generated_unexecuted': True, 'native_qualified': False,
        'scientific_verdict': 'not_computed', 'dispatch_ready': False,
    }
    request['request_digest'] = canonical_digest(request)
    if _verify:
        verify_request(root, request)
    return request


def verify_request(root: Path, request: dict) -> None:
    core = {key: value for key, value in request.items()
            if key != 'request_digest'}
    if request.get('request_digest') != canonical_digest(core):
        raise ValueError('Request digest mismatch')
    if (request.get('kind') != 'c14-native-comparison-request'
            or request.get('candidate_id') != CANDIDATE_ID
            or request.get('scoring_seed') != 44
            or request.get('generated_unexecuted') is not True
            or request.get('native_qualified') is not False
            or request.get('dispatch_ready') is not False):
        raise ValueError('Unexecuted C14 request scope required')
    if [row.get('role') for row in request.get('roles', [])] != list(ROLES):
        raise ValueError('Exactly the five ordered logical roles required')
    if set(request.get('role_to_case', {})) != set(ROLES):
        raise ValueError('Every logical role must map to a scoring case')
    if len({row.get('case_id') for row in request.get('scoring_cases', [])}) != len(
            request.get('scoring_cases', [])):
        raise ValueError('Physical scoring case IDs must be unique')
    refs = request.get('input_refs')
    if not isinstance(refs, list):
        raise ValueError('Pinned input closure required')
    pinned = {ref['path']: ref for ref in refs}
    for ref in refs:
        resolve_ref(root, ref)
    for key in ('freeze_ref', 'b_star_decision_ref', 'source_sequence_ref',
                'source_report_ref'):
        if pinned.get(request[key]['path']) != request[key]:
            raise ValueError('Request dependency closure missing: ' + key)
    freeze = read_json(resolve_ref(root, request['freeze_ref']))
    if freeze.get('freeze_digest') != canonical_digest(
            {key: value for key, value in freeze.items()
             if key != 'freeze_digest'}):
        raise ValueError('Current freeze digest mismatch')
    for case in request['scoring_cases']:
        keys = ['report_ref', 'sequence_ref']
        if case['role'] == 'b0':
            keys.append('implementation_ref')
        if case['role'] == 'body_gaussian':
            keys.append('poses_ref')
        if case['role'] == 'corotational_residual':
            keys.append('certificate_ref')
        for key in keys:
            if pinned.get(case[key]['path']) != case[key]:
                raise ValueError('Physical case dependency not pinned')
    expected = make_request(root, freeze_path=resolve_ref(root, request['freeze_ref']),
                            _verify=False)
    if request != expected:
        raise ValueError('Request differs from the current frozen role construction')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('request',))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = make_request(args.root, freeze_path=args.freeze)
    output = args.output.resolve(); output.relative_to(args.root.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(json.dumps(request, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'request': str(output),
                      'request_digest': request['request_digest'],
                      'execution_started': False, 'native_qualified': False,
                      'dispatch_ready': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
