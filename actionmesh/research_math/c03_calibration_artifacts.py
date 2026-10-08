"""C03 development-fit artifact boundary; generated, unexecuted source.

This consumes an explicitly prepared pointwise development label bank. It does
not construct or qualify material correspondence, run a model or score a mesh.
The missing native label-bank producer remains a delivery gap. Hashes and split
checks establish byte identity/isolation only, never scientific admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from research_math.self_map_candidate import digest, load, physical, write
from research_math.correlated_calibration import (
    CalibrationFailure, fit_affine, validate_fit, apply_frozen_fits, unit_c01,
)

CANDIDATE_ID = '4d-math-20261006-c03'
FIT_ROLES = {
    'intercept_squared': ('intercept', 'squared'),
    'intercept_unsquared': ('intercept', 'smoothed_unsquared'),
    'diagonal_squared': ('diagonal', 'squared'),
    'diagonal_unsquared': ('diagonal', 'smoothed_unsquared'),
    'full_squared': ('full', 'squared'),
    'full_smoothed_unsquared': ('full', 'smoothed_unsquared'),
}
SCOPE = {'native_qualified': False, 'scientific_admission': False,
         'local_method_verified': False, 'candidate_methods_tested': False}


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _strings(values, name):
    if (not isinstance(values, list) or not values
            or any(not isinstance(x, str) or not x for x in values)
            or values != sorted(set(values))):
        raise ValueError(name + ' must be nonempty, sorted unique strings')
    return set(values)


def validate_policy(policy):
    """Fixed-design metadata checks; no claim that declared provenance is true."""
    expected = {'kind', 'version', 'data_sha256', 'development_uids',
        'confirmation_uids', 'uid_to_family', 'coordinate_policy',
        'correspondence_evidence', 'parameters', 'generation_seed', 'frame_indices'}
    if (set(policy) != expected or policy['kind'] != 'c03-development-calibration-policy'
            or type(policy['version']) is not int or policy['version'] != 1):
        raise ValueError('Exact C03 development policy schema required')
    if (policy['coordinate_policy'] != 'plain_pointwise_no_asset_specific_icp'
            or type(policy['generation_seed']) is not int or policy['generation_seed'] != 42
            or policy['frame_indices'] != list(range(16))):
        raise ValueError('Only seed42/full16 plain pointwise specialization is implemented')
    dev = _strings(policy['development_uids'], 'development_uids')
    confirm = _strings(policy['confirmation_uids'], 'confirmation_uids')
    families = policy['uid_to_family']
    if (dev & confirm or not isinstance(families, dict) or set(families) != dev | confirm
            or any(not isinstance(v, str) or not v for v in families.values())
            or {families[u] for u in dev} & {families[u] for u in confirm}):
        raise ValueError('Complete disjoint UID and family partitions required')
    refs = policy['correspondence_evidence']
    if not isinstance(refs, dict) or not refs:
        raise ValueError('Retained correspondence evidence bytes required; declaration is not qualification')
    for value in [policy['data_sha256'], *refs.values()]:
        if (not isinstance(value, str) or len(value) != 64
                or any(c not in '0123456789abcdef' for c in value)):
            raise ValueError('Canonical SHA256 required')
    from research_math.self_map_candidate import relative_name
    for name in refs:
        relative_name(name)
    params = policy['parameters']
    if set(params) != {'ridge', 'tau', 'max_iterations', 'gradient_tolerance', 'objective_tolerance'}:
        raise ValueError('All common estimator settings must be explicitly frozen')
    if type(params['max_iterations']) is not int or params['max_iterations'] < 1:
        raise ValueError('Positive integer iteration budget required')
    for key in ('ridge', 'tau', 'gradient_tolerance', 'objective_tolerance'):
        value = params[key]
        if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
            raise ValueError('Finite positive common parameter required: ' + key)
    return policy


def load_development_bank(data_path, policy):
    validate_policy(policy)
    if digest(physical(data_path)) != policy['data_sha256']:
        raise ValueError('Development data identity differs')
    with np.load(data_path, allow_pickle=False) as bank:
        if set(bank.files) != {'reference_residual', 'labeled_error', 'weights', 'uids'}:
            raise ValueError('Exact development bank fields required')
        values = {k: bank[k] for k in bank.files}
    r, e, w, uids = (values[k] for k in
        ('reference_residual', 'labeled_error', 'weights', 'uids'))
    if (r.ndim != 2 or r.shape[1:] != (3,) or len(r) == 0
            or e.shape != (15, len(r), 3) or w.shape != (len(r),)
            or uids.shape != (len(r),) or uids.dtype.kind != 'U'
            or any(x.dtype.kind != 'f' or not np.isfinite(x).all() for x in (r, e, w))
            or np.any(w < 0) or not np.isfinite(w.sum()) or w.sum() <= 0):
        raise ValueError('Finite aligned residual/error/weight arrays and Unicode UID rows required')
    if set(uids.tolist()) != set(policy['development_uids']):
        raise ValueError('Bank must contain exactly development UIDs; confirmation rows prohibited')
    if any(w[uids == uid].sum() <= 0 for uid in policy['development_uids']):
        raise ValueError('Each declared development UID must have positive weight')
    return values


def fit_bundle(data_path, policy_path, evidence_files, output):
    """Harness inner task. Failure of one fit retains all other attempts."""
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {'candidate_id': CANDIDATE_ID, 'status': 'error', **SCOPE,
              'source_delivery_status': 'generated_unexecuted_at_authoring',
              'scope': 'unqualified development pointwise surrogate; no native scores'}
    try:
        policy = validate_policy(load(policy_path))
        if set(evidence_files) != set(policy['correspondence_evidence']):
            raise ValueError('Exact explicit correspondence evidence closure required')
        for name, path in evidence_files.items():
            if digest(physical(path)) != policy['correspondence_evidence'][name]:
                raise ValueError('Correspondence evidence changed: ' + name)
        values = load_development_bank(data_path, policy)
        policy_hash = digest(policy_path)
        result.update(policy_sha256=policy_hash, data_sha256=policy['data_sha256'],
                      policy=policy, role_results={})
        # One failure is not substituted with another estimator or silently retried.
        for role, (mode, loss) in FIT_ROLES.items():
            fits, failures = [], []
            for frame in range(15):
                try:
                    fits.append(fit_affine(values['reference_residual'],
                        values['labeled_error'][frame], values['weights'],
                        mode=mode, loss=loss, **policy['parameters']))
                except (CalibrationFailure, ValueError, np.linalg.LinAlgError) as error:
                    fits.append(None)
                    failures.append({'frame_index': frame + 1,
                        'exception_type': type(error).__name__, 'error': str(error),
                        'diagnostics': getattr(error, 'diagnostics', None)})
            result['role_results'][role] = {'status': 'failed' if failures else 'completed',
                'fits': fits, 'failures': failures}
        result['status'] = ('completed' if all(x['status'] == 'completed'
                            for x in result['role_results'].values()) else 'incomplete')
        # Recheck every externally supplied byte after fitting to detect mutation.
        if digest(data_path) != policy['data_sha256'] or digest(policy_path) != policy_hash:
            raise ValueError('Development input changed during fitting')
        for name, path in evidence_files.items():
            if digest(path) != policy['correspondence_evidence'][name]:
                raise ValueError('Correspondence input changed during fitting')
    except Exception as error:
        result.update(status='error', exception_type=type(error).__name__, error=str(error))
        result.pop('role_results', None)
    finally:
        result['elapsed_seconds'] = time.monotonic() - started
        result['bundle_digest'] = canonical_digest(result)
        write(output / 'fit-bundle.json', result)
    return result


def validate_bundle(bundle):
    if (bundle.get('candidate_id') != CANDIDATE_ID
            or bundle.get('status') not in ('completed', 'incomplete')
            or any(bundle.get(k) is not v for k, v in SCOPE.items())
            or canonical_digest({k: v for k, v in bundle.items() if k != 'bundle_digest'})
                != bundle.get('bundle_digest')):
        raise ValueError('Unmodified completed/incomplete unqualified fit bundle required')
    policy = validate_policy(bundle['policy'])
    if (bundle.get('data_sha256') != policy['data_sha256']
            or set(bundle['role_results']) != set(FIT_ROLES)):
        raise ValueError('Fit bundle must retain all six matched roles')
    any_failed = False
    for role, (mode, loss) in FIT_ROLES.items():
        row = bundle['role_results'][role]
        if row.get('status') not in ('completed', 'failed') or len(row['fits']) != 15:
            raise ValueError('Complete per-frame fit inventory required')
        absent = [i + 1 for i, fit in enumerate(row['fits']) if fit is None]
        if (len(row['failures']) != len(absent)
                or [f['frame_index'] for f in row['failures']] != absent
                or (row['status'] == 'failed') != bool(absent)):
            raise ValueError('Retain exact failed-frame denominator')
        any_failed |= bool(absent)
        for fit in row['fits']:
            if fit is not None:
                validated = validate_fit(fit)
                if (validated['mode'] != mode or validated['loss'] != loss
                        or any(validated[k] != policy['parameters'][k]
                               for k in ('ridge', 'tau'))
                        or any(validated['diagnostics'][k] != policy['parameters'][k]
                               for k in ('max_iterations', 'gradient_tolerance',
                                         'objective_tolerance'))):
                    raise ValueError('Role or frozen parameter identity differs')
    if bundle['status'] != ('incomplete' if any_failed else 'completed'):
        raise ValueError('Bundle terminal state differs from fit inventory')
    return bundle


def apply_bundle_arrays(raw16, self_residual, anchor, bundle, *, uid, family):
    """Label-free application API; never accepts target labels or refits."""
    bundle = validate_bundle(bundle)
    policy = bundle['policy']
    if (uid not in policy['confirmation_uids']
            or policy['uid_to_family'].get(uid) != family):
        raise ValueError('Application must belong to the exact frozen confirmation partition')
    arrays, failures = {}, {}
    try:
        arrays['unit_C01'] = unit_c01(raw16, self_residual, anchor)
    except (ValueError, FloatingPointError) as error:
        failures['unit_C01'] = [{'stage': 'application',
            'exception_type': type(error).__name__, 'error': str(error)}]
    for role, row in bundle['role_results'].items():
        if row['status'] == 'failed':
            failures[role] = row['failures']
        else:
            try:
                arrays[role] = apply_frozen_fits(raw16, self_residual, anchor, row['fits'])
            except (ValueError, FloatingPointError) as error:
                failures[role] = [{'stage': 'application',
                    'exception_type': type(error).__name__, 'error': str(error)}]
    return arrays, failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'policy', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--evidence-file', nargs=2, action='append', required=True)
    args = parser.parse_args(argv)
    evidence = dict(args.evidence_file)
    if len(evidence) != len(args.evidence_file):
        parser.error('Duplicate evidence names forbidden')
    result = fit_bundle(args.data, args.policy, evidence, args.output)
    print(json.dumps({'status': result['status'], **SCOPE}))
    return 0 if result['status'] == 'completed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
