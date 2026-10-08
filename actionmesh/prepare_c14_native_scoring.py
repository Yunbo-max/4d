"""Build the single-use admitted C14 official-scoring harness plan.

The builder fails closed unless the current research-autopilot protocol verifies,
C14 is design-verified at the dispatch boundary, the frozen native contract names
the exact five logical roles, and the current Local runtime/GPU identity matches.
It emits plans only; it never runs the scorer or resumes a stopped GPU campaign.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys

if "scoring" not in globals():
    from research_math import c14_native_scoring as scoring


CANDIDATE_ID = '4d-math-20261006-c14'
ROLE_FOR_CONTRACT_ARM = {
    'treatment': 'corotational_residual',
    'baseline': 'b_star',
    'b0': 'b0',
    'world_gaussian': 'world_gaussian',
    'body_gaussian': 'body_gaussian',
}
PROFILE = 'c14'
TASK_ID = 'c14-five-role-official-scoring'
AUTHORIZATION_SCOPE = 'single_c14_scoring_attempt'
INPUT_NAMESPACE = 'c14'
OUTPUT_DIRECTORY = 'c14-scoring-output'
SCORING_MODULE = 'research_math.c14_native_scoring'
COMPARISON_SOURCE = 'actionmesh/research_math/c14_native_comparison.py'
SCORING_SOURCE = 'actionmesh/research_math/c14_native_scoring.py'
LAUNCHER_SOURCE = 'actionmesh/launch_c14_native_scoring.py'
EXTRA_CODE_SOURCES = ('actionmesh/prepare_c14_native_scoring.py',)
LAUNCH_TICKET_KIND = 'c14-staged-launch-ticket'
AUTHORIZATION_KIND = 'c14-gpu-resume-authorization'
RESERVATION_KIND = 'c14-gpu-resume-authorization-reservation'
CONSUMPTION_KIND = 'c14-gpu-resume-authorization-consumption'
FAMILY_SPLIT_KIND = 'c14-family-split'
FAMILY_ASSIGNMENTS_KIND = 'c14-family-assignments'
FAMILY_DERIVATION_KIND = 'c14-family-derivation'
FAMILY_REVIEW_KIND = 'c14-family-derivation-review'
CRITERIA_KIND = 'c14-outcome-criteria'
ANALYSIS_KIND = 'c14-g01-analysis-plan'
ADMISSION_KIND = 'c14-scientific-dispatch-admission'
CONTRACT_CONTRASTS = {
    'treatment': 'corotational_residual',
    'baseline': 'b_star',
    'controls': ['b0', 'world_gaussian', 'body_gaussian'],
}
CONTRACT_ARM_NAMES = {
    'treatment': 'corotational_residual', 'baseline': 'b_star',
    'b0': 'b0', 'world_gaussian': 'world_gaussian',
    'body_gaussian': 'body_gaussian',
}
PRIMARY_METRIC = 'cd_motion'
GUARDRAIL_METRICS = ('cd_3d', 'cd_4d')


def canonical_record_digest(value: dict, digest_key: str = 'admission_digest') -> str:
    core = {key: item for key, item in value.items() if key != digest_key}
    return hashlib.sha256(json.dumps(
        core, sort_keys=True, separators=(',', ':'), allow_nan=False
    ).encode()).hexdigest()


def analysis_protocol_core_digest(protocol: dict) -> str:
    """Acyclic analysis binding; the final protocol still hashes the analysis ref.

    Author the protocol core first, then the analysis with this digest, then
    pin analysis_plan_ref and freeze the final protocol.  Downstream criteria
    and GPU authorization bind that final protocol_digest without a back-edge.
    Only the analysis reference and the two finalization fields are excluded;
    criteria, identities, sample closure and method discovery remain covered.
    """
    core = {key: value for key, value in protocol.items()
            if key not in {"analysis_plan_ref", "protocol_digest", "frozen_at"}}
    return hashlib.sha256(json.dumps(
        core, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def _timezone(value, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except (AttributeError, ValueError) as error:
        raise ValueError('Timezone-aware ' + label + ' required') from error
    if parsed.tzinfo is None:
        raise ValueError('Timezone-aware ' + label + ' required')
    return parsed


def require_admission_core(admission: dict, *, uid: str,
                           benchmark_revision: str, comparison_digest: str,
                           protocol_ref: dict, method_batch_ref: dict) -> None:
    refs = ('natural_gate_0_ref', 'importance_decision_ref', 'family_split_ref',
            'outcome_criteria_ref', 'gpu_resume_authorization_ref')
    required = {
        'kind', 'version', 'candidate_id', 'uid', 'benchmark_revision',
        'comparison_digest', 'protocol_ref', 'method_batch_ref', *refs,
        'status', 'admitted_at', 'admission_digest',
    }
    if (set(admission) != required
            or admission.get('kind') != ADMISSION_KIND
            or admission.get('version') != 1
            or admission.get('candidate_id') != CANDIDATE_ID
            or admission.get('uid') != uid
            or admission.get('benchmark_revision') != benchmark_revision
            or admission.get('comparison_digest') != comparison_digest
            or admission.get('protocol_ref') != protocol_ref
            or admission.get('method_batch_ref') != method_batch_ref
            or admission.get('status') != 'admitted'
            or admission.get('admission_digest') != canonical_record_digest(admission)):
        raise ValueError('Exact hash-bound C14 scientific admission required')
    _timezone(admission['admitted_at'], 'admitted_at')
    for key in refs:
        ref = admission.get(key)
        if (not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}):
            raise ValueError('Exact admission evidence reference required: ' + key)


def bind_contract_arms(contract: dict, comparison: dict,
                       root: Path | None = None) -> list[dict]:
    rows = comparison.get('roles')
    by_role = {row.get('role'): row for row in rows or []
               if isinstance(row, dict)}
    if set(by_role) != set(ROLE_FOR_CONTRACT_ARM.values()):
        raise ValueError('Frozen comparison lacks exact five logical roles')
    arms = contract['arm_requirements']
    refs = []
    for arm_role, comparison_role in ROLE_FOR_CONTRACT_ARM.items():
        arm = arms[arm_role]
        row = by_role[comparison_role]
        identity_row = by_role[row['alias_of']] if row.get('alias_of') else row
        implementation_refs = arm.get('implementation_refs')
        if (arm.get('name') != comparison_role
                or arm.get('revision') != row.get('method_id')
                or not isinstance(implementation_refs, list)
                or not implementation_refs):
            raise ValueError('Contract arm differs from frozen method identity: '
                             + arm_role)
        for ref in implementation_refs:
            if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
                raise ValueError('Exact arm implementation reference required')
            refs.append(ref)
        implementation_sha = identity_row.get('implementation_sha256')
        if (not isinstance(implementation_sha, str)
                or implementation_sha not in {
                    ref['sha256'] for ref in implementation_refs}):
            raise ValueError('Contract implementation differs from frozen arm: '
                             + arm_role)
        if root is not None and identity_row.get('report_ref') is not None:
            report = scoring.read_json(scoring.resolve_ref(
                root, identity_row['report_ref']))
            if report.get('implementation_sha256') != implementation_sha:
                raise ValueError('Frozen arm/report implementation mismatch: '
                                 + arm_role)
    alias = by_role['b_star'].get('alias_of')
    if alias is not None:
        target_arm = next(key for key, role in ROLE_FOR_CONTRACT_ARM.items()
                          if role == alias)
        if arms['baseline']['implementation_refs'] != arms[target_arm][
                'implementation_refs']:
            raise ValueError('Aliased B* must bind the target implementation')
    return list(_unique_refs(refs, 'arm implementation').values())


def _record(root: Path, ref: dict, *, kind: str, version=1,
            digest_key='digest') -> dict:
    value = scoring.read_json(scoring.resolve_ref(root, ref))
    if (value.get('kind') != kind or value.get('version') != version
            or value.get(digest_key) != canonical_record_digest(value, digest_key)):
        raise ValueError('Invalid hash-bound admission record: ' + kind)
    return value


def _nested_refs(value) -> list[dict]:
    if isinstance(value, dict):
        if set(value) == {'path', 'sha256'}:
            return [value]
        refs = []
        for item in value.values():
            refs.extend(_nested_refs(item))
        return refs
    if isinstance(value, list):
        refs = []
        for item in value:
            refs.extend(_nested_refs(item))
        return refs
    return []


def _unique_refs(refs: list[dict], label: str) -> dict[str, dict]:
    unique = {}
    for ref in refs:
        if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
            raise ValueError('Exact ' + label + ' reference required')
        prior = unique.get(ref['path'])
        if prior is not None and prior != ref:
            raise ValueError('Conflicting ' + label + ' references: ' + ref['path'])
        unique[ref['path']] = ref
    return unique


def require_dispatch_admission(root: Path, admission_path: Path, *, request: dict,
                               protocol_ref: dict, protocol: dict,
                               request_ref: dict, method_batch_ref: dict,
                               environment_ref: dict, run_id: str,
                               group: str, contract_digest: str,
                               gpu_uuid: str, wall_seconds: int,
                               core) -> tuple[dict, list[dict], dict, dict]:
    admission_ref = scoring.file_ref(root, admission_path)
    scoring.resolve_ref(root, admission_ref)
    admission = scoring.read_json(admission_path)
    require_admission_core(
        admission, uid=request['uid'],
        benchmark_revision=request['benchmark_revision'],
        comparison_digest=request['comparison_ref']['sha256'],
        protocol_ref=protocol_ref, method_batch_ref=method_batch_ref)
    gate = scoring.read_json(scoring.resolve_ref(
        root, admission['natural_gate_0_ref']))
    core.verify_ref(root, admission['natural_gate_0_ref'], 'natural-gate-0')
    if (gate.get('schema_id') != 'natural-gate-0'
            or gate.get('schema_version') != '1.0.0'
            or gate.get('outcome') != 'PASS'):
        raise ValueError('Natural Gate 0 PASS required')
    importance = scoring.read_json(scoring.resolve_ref(
        root, admission['importance_decision_ref']))
    core.verify_ref(root, admission['importance_decision_ref'],
                    'importance-decision')
    if (importance.get('schema_id') != 'importance-decision'
            or importance.get('schema_version') != '1.0.0'
            or importance.get('outcome') not in ('CLEAR', 'CONCURRENT')
            or importance.get('natural_gate_0_ref') != admission[
                'natural_gate_0_ref']
            or importance.get('parent_problem_ref') != gate.get(
                'parent_problem_ref')):
        raise ValueError('Compatible importance/collision decision required')
    family = _record(root, admission['family_split_ref'],
                     kind=FAMILY_SPLIT_KIND, digest_key='split_digest')
    family_keys = {
        'kind', 'version', 'candidate_id', 'benchmark_revision',
        'development_ids', 'confirmation_ids', 'family_by_uid',
        'family_assignments_ref', 'frozen_at', 'split_digest',
    }
    development = family.get('development_ids')
    confirmation = family.get('confirmation_ids')
    family_by_uid = family.get('family_by_uid')
    if (set(family) != family_keys
            or family.get('candidate_id') != CANDIDATE_ID
            or family.get('benchmark_revision') != request['benchmark_revision']
            or not isinstance(development, list)
            or not isinstance(confirmation, list)
            or not development or not confirmation
            or len(set(development)) != len(development)
            or len(set(confirmation)) != len(confirmation)
            or set(development) & set(confirmation)
            or request['uid'] not in confirmation
            or not isinstance(family_by_uid, dict)
            or set(family_by_uid) != set(development) | set(confirmation)
            or not all(isinstance(item, str) and item
                       for item in family_by_uid.values())
            or ({family_by_uid[item] for item in development}
                & {family_by_uid[item] for item in confirmation})):
        raise ValueError('Frozen independent-family confirmation split required')
    _timezone(family['frozen_at'], 'family split frozen_at')
    assignments = _record(root, family['family_assignments_ref'],
                          kind=FAMILY_ASSIGNMENTS_KIND,
                          digest_key='assignment_digest')
    assignment_keys = {
        'kind', 'version', 'candidate_id', 'benchmark_revision',
        'family_by_uid', 'source_refs', 'derivation_ref', 'assignment_rule',
        'review_ref', 'assigned_at', 'assignment_digest',
    }
    if (set(assignments) != assignment_keys
            or assignments.get('candidate_id') != CANDIDATE_ID
            or assignments.get('benchmark_revision') != request['benchmark_revision']
            or assignments.get('family_by_uid') != family_by_uid
            or not isinstance(assignments.get('source_refs'), list)
            or not assignments['source_refs']
            or not isinstance(assignments.get('assignment_rule'), str)
            or not assignments['assignment_rule']
            or not isinstance(assignments.get('review_ref'), dict)):
        raise ValueError('Pinned source-derived family assignments required')
    _timezone(assignments['assigned_at'], 'family assignments assigned_at')
    for ref in assignments['source_refs']:
        scoring.resolve_ref(root, ref)
    scoring.resolve_ref(root, assignments['review_ref'])
    derivation = _record(root, assignments['derivation_ref'],
                         kind=FAMILY_DERIVATION_KIND,
                         digest_key='derivation_digest')
    derivation_keys = {
        'kind', 'version', 'candidate_id', 'benchmark_revision',
        'family_by_uid', 'source_refs', 'assignment_rule', 'reviewed',
        'derived_at', 'derivation_digest',
    }
    if (set(derivation) != derivation_keys
            or derivation.get('candidate_id') != CANDIDATE_ID
            or derivation.get('benchmark_revision') != request['benchmark_revision']
            or derivation.get('family_by_uid') != family_by_uid
            or derivation.get('source_refs') != assignments['source_refs']
            or derivation.get('assignment_rule') != assignments['assignment_rule']
            or derivation.get('reviewed') is not True):
        raise ValueError('Reviewed family derivation must reproduce family map')
    _timezone(derivation['derived_at'], 'family derivation derived_at')
    review = _record(root, assignments['review_ref'],
                     kind=FAMILY_REVIEW_KIND,
                     digest_key='review_digest')
    review_keys = {
        'kind', 'version', 'candidate_id', 'derivation_ref', 'outcome',
        'checks', 'reviewer', 'reviewed_at', 'review_digest',
    }
    if (set(review) != review_keys
            or review.get('candidate_id') != CANDIDATE_ID
            or review.get('derivation_ref') != assignments['derivation_ref']
            or review.get('outcome') != 'verified'
            or not isinstance(review.get('checks'), list)
            or set(review.get('checks', [])) != {
                'source_identity', 'mapping_reproduction', 'split_independence'}
            or not isinstance(review.get('reviewer'), str)
            or not review['reviewer']):
        raise ValueError('Independent verified family-derivation review required')
    _timezone(review['reviewed_at'], 'family review reviewed_at')
    criteria = _record(root, admission['outcome_criteria_ref'],
                       kind=CRITERIA_KIND,
                       digest_key='criteria_digest')
    criteria_keys = {
        'kind', 'version', 'candidate_id', 'comparison_digest',
        'protocol_digest', 'analysis_plan_ref', 'primary_metric',
        'guardrail_metrics', 'min_effect',
        'noninferiority_margins', 'independent_unit', 'frozen_at',
        'criteria_digest',
    }
    margins = criteria.get('noninferiority_margins')
    if (set(criteria) != criteria_keys
            or criteria.get('candidate_id') != CANDIDATE_ID
            or criteria.get('comparison_digest') != request['comparison_ref'][
                'sha256']
            or criteria.get('protocol_digest') != protocol.get('protocol_digest')
            or criteria.get('analysis_plan_ref') != protocol.get('analysis_plan_ref')
            or criteria.get('primary_metric') != PRIMARY_METRIC
            or criteria.get('guardrail_metrics') != list(GUARDRAIL_METRICS)
            or criteria.get('independent_unit') != 'asset_family'
            or isinstance(criteria.get('min_effect'), bool)
            or not isinstance(criteria.get('min_effect'), (int, float))
            or not math.isfinite(criteria['min_effect'])
            or criteria['min_effect'] <= 0
            or not isinstance(margins, dict)
            or set(margins) != set(GUARDRAIL_METRICS)
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or value < 0
                   for value in margins.values())):
        raise ValueError('Prospectively frozen numeric C14 criteria required')
    _timezone(criteria['frozen_at'], 'outcome criteria frozen_at')
    analysis_ref = criteria['analysis_plan_ref']
    analysis = _record(root, analysis_ref, kind=ANALYSIS_KIND,
                       digest_key='plan_digest')
    analysis_keys = {
        'kind', 'version', 'candidate_id', 'protocol_core_digest',
        'primary_metric', 'direction', 'min_effect', 'guardrail_margins',
        'independent_unit', 'failure_policy', 'paired_statistics',
        'multiplicity', 'sensitivity', 'development_confirmation',
        'protocol_criteria', 'frozen_at', 'plan_digest',
    }
    paired = analysis.get('paired_statistics')
    multiplicity = analysis.get('multiplicity')
    sensitivity = analysis.get('sensitivity')
    boundary = analysis.get('development_confirmation')
    protocol_criteria = protocol.get('criteria')
    if (set(analysis) != analysis_keys
            or analysis.get('candidate_id') != CANDIDATE_ID
            or analysis.get('protocol_core_digest') != analysis_protocol_core_digest(protocol)
            or analysis.get('primary_metric') != PRIMARY_METRIC
            or analysis.get('direction') != 'minimize'
            or analysis.get('min_effect') != criteria['min_effect']
            or analysis.get('guardrail_margins') != margins
            or analysis.get('independent_unit') != 'asset_family'
            or analysis.get('failure_policy') !=
            'all_frozen_roles_and_assets_in_denominator_no_zero_imputation'
            or not isinstance(paired, dict)
            or paired.get('unit') != 'asset_family'
            or paired.get('pairing') != 'same_uid_same_source'
            or not isinstance(paired.get('method'), str) or not paired['method']
            or isinstance(paired.get('confidence_level'), bool)
            or not isinstance(paired.get('confidence_level'), (int, float))
            or not 0 < paired['confidence_level'] < 1
            or not isinstance(multiplicity, dict)
            or not isinstance(multiplicity.get('method'), str)
            or not multiplicity['method']
            or set(multiplicity.get('family', [])) !=
            {PRIMARY_METRIC, *GUARDRAIL_METRICS}
            or not isinstance(sensitivity, list) or not sensitivity
            or any(not isinstance(row, dict)
                   or not isinstance(row.get('name'), str)
                   or not isinstance(row.get('values'), list)
                   or len(row['values']) < 2 for row in sensitivity)
            or not isinstance(boundary, dict)
            or boundary.get('development_ids') != development
            or boundary.get('confirmation_ids') != confirmation
            or boundary.get('confirmation_locked') is not True
            or not isinstance(protocol_criteria, list)
            or not protocol_criteria
            or analysis.get('protocol_criteria') != protocol_criteria):
        raise ValueError('Complete protocol-bound C14 G01 analysis plan required')
    by_metric = {row.get('metric'): row for row in protocol_criteria
                 if isinstance(row, dict)}
    if (set(by_metric) != {PRIMARY_METRIC, *GUARDRAIL_METRICS}
            or by_metric[PRIMARY_METRIC].get('direction') != 'minimize'
            or by_metric[PRIMARY_METRIC].get('min_effect') != criteria['min_effect']
            or by_metric[PRIMARY_METRIC].get('paired_statistics') != paired
            or by_metric[PRIMARY_METRIC].get('multiplicity') != multiplicity
            or any(by_metric[name].get('direction') != 'minimize'
                   or by_metric[name].get('noninferiority_margin') != margins[name]
                   or by_metric[name].get('paired_statistics') != paired
                   or by_metric[name].get('multiplicity') != multiplicity
                   for name in GUARDRAIL_METRICS)):
        raise ValueError('Protocol criteria contradict the frozen G01 analysis')
    _timezone(analysis['frozen_at'], 'G01 analysis frozen_at')
    authorization = _record(root, admission['gpu_resume_authorization_ref'],
                            kind=AUTHORIZATION_KIND,
                            digest_key='authorization_digest')
    authorization_keys = {
        'kind', 'version', 'candidate_id', 'uid', 'gpu_uuid',
        'protocol_digest', 'comparison_digest', 'request_ref',
        'method_batch_ref', 'environment_ref', 'run_id', 'task_id',
        'group', 'contract_digest',
        'scope', 'max_gpu_task_seconds', 'status', 'authorized_at',
        'expires_at', 'authorization_nonce', 'stop_acknowledgement',
        'authorization_digest',
    }
    if (set(authorization) != authorization_keys
            or authorization.get('candidate_id') != CANDIDATE_ID
            or authorization.get('uid') != request['uid']
            or authorization.get('gpu_uuid') != gpu_uuid
            or authorization.get('protocol_digest') != protocol.get(
                'protocol_digest')
            or authorization.get('comparison_digest') != request[
                'comparison_ref']['sha256']
            or authorization.get('request_ref') != request_ref
            or authorization.get('method_batch_ref') != method_batch_ref
            or authorization.get('environment_ref') != environment_ref
            or authorization.get('run_id') != run_id
            or authorization.get('task_id') != TASK_ID
            or authorization.get('group') != group
            or authorization.get('contract_digest') != contract_digest
            or authorization.get('scope') != AUTHORIZATION_SCOPE
            or authorization.get('stop_acknowledgement') !=
            'explicit_resume_for_exact_attempt'
            or not isinstance(authorization.get('authorization_nonce'), str)
            or len(authorization['authorization_nonce']) != 64
            or any(character not in '0123456789abcdef'
                   for character in authorization['authorization_nonce'])
            or authorization.get('status') != 'authorized'
            or isinstance(authorization.get('max_gpu_task_seconds'), bool)
            or not isinstance(authorization.get('max_gpu_task_seconds'), int)
            or authorization.get('max_gpu_task_seconds') < wall_seconds):
        raise ValueError('Explicit current GPU-resume authorization required')
    authorized_at = _timezone(authorization.get('authorized_at'), 'authorized_at')
    expires_at = _timezone(authorization.get('expires_at'), 'expires_at')
    now = datetime.now(timezone.utc)
    if authorized_at > now or authorized_at > expires_at or now > expires_at:
        raise ValueError('Current unexpired GPU-resume authorization required')
    for frozen_time, label in ((family['frozen_at'], 'family split'),
                               (assignments['assigned_at'], 'family assignments'),
                               (derivation['derived_at'], 'family derivation'),
                               (review['reviewed_at'], 'family review'),
                               (criteria['frozen_at'], 'outcome criteria'),
                               (analysis['frozen_at'], 'G01 analysis'),
                               (protocol.get('frozen_at'), 'protocol')):
        if _timezone(frozen_time, label + ' frozen_at') > authorized_at:
            raise ValueError(label + ' must be frozen before authorization')
    direct = [admission[key] for key in (
        'natural_gate_0_ref', 'importance_decision_ref', 'family_split_ref',
        'outcome_criteria_ref', 'gpu_resume_authorization_ref')]
    closure = (direct + _nested_refs(gate) + _nested_refs(importance)
               + [family['family_assignments_ref'], assignments['derivation_ref'],
                  assignments['review_ref'], analysis_ref]
               + assignments['source_refs'])
    unique = _unique_refs(closure, 'admission evidence')
    for ref in unique.values():
        scoring.resolve_ref(root, ref)
    return (admission_ref, list(unique.values()),
            admission['gpu_resume_authorization_ref'], authorization)


def _authorization_identity(authorization_ref: dict, authorization: dict, *,
                            request_ref: dict, method_batch_ref: dict,
                            environment_ref: dict, run_id: str, group: str,
                            contract_digest: str) -> dict:
    return {
        'authorization_ref': authorization_ref,
        'authorization_nonce': authorization['authorization_nonce'],
        'request_ref': request_ref, 'method_batch_ref': method_batch_ref,
        'environment_ref': environment_ref, 'run_id': run_id,
        'task_id': TASK_ID,
        'group': group, 'contract_digest': contract_digest,
    }


def authorization_consumption_path(root: Path, authorization_ref: dict) -> Path:
    return (Path(root)/'inputs'/INPUT_NAMESPACE/'authorization-consumption'/
            (authorization_ref['sha256'] + '.json'))


def reserve_authorization(root: Path, authorization_ref: dict,
                          authorization: dict, *, request_ref: dict,
                          method_batch_ref: dict, environment_ref: dict,
                          run_id: str, group: str,
                          contract_digest: str) -> dict:
    """Atomically reserve one authorization; exact interrupted work may resume."""
    directory = Path(root)/'inputs'/INPUT_NAMESPACE/'authorization-consumption'
    directory.mkdir(parents=True, exist_ok=True)
    final_path = authorization_consumption_path(root, authorization_ref)
    if final_path.exists():
        raise FileExistsError(str(final_path))
    path = directory/(authorization_ref['sha256'] + '.reservation.json')
    identity = _authorization_identity(
        authorization_ref, authorization, request_ref=request_ref,
        method_batch_ref=method_batch_ref, environment_ref=environment_ref,
        run_id=run_id, group=group, contract_digest=contract_digest)
    value = {
        'kind': RESERVATION_KIND, 'version': 1,
        **identity, 'state': 'reserved',
        'reserved_at': datetime.now(timezone.utc).isoformat(),
    }
    value['reservation_digest'] = canonical_record_digest(
        value, 'reservation_digest')
    try:
        with path.open('x') as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
    except FileExistsError:
        existing = scoring.read_json(path)
        expected_keys = set(value)
        if (set(existing) != expected_keys
                or existing.get('kind') != value['kind']
                or existing.get('version') != 1
                or existing.get('state') != 'reserved'
                or any(existing.get(key) != item for key, item in identity.items())
                or existing.get('reservation_digest') != canonical_record_digest(
                    existing, 'reservation_digest')):
            raise ValueError('Authorization is reserved by different work')
    return scoring.file_ref(root, path)


def issue_launch_ticket(root: Path, authorization_ref: dict,
                        authorization: dict, *, reservation_ref: dict,
                        request_ref: dict, method_batch_ref: dict,
                        environment_ref: dict, run_id: str, group: str,
                        contract_digest: str, gpu_uuid: str) -> dict:
    """Create the immutable non-circular ticket that the runtime will stage."""
    directory = Path(root)/'inputs'/INPUT_NAMESPACE/'authorization-consumption'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/(authorization_ref['sha256'] + '.launch-ticket.json')
    value = {
        'kind': LAUNCH_TICKET_KIND, 'version': 1,
        **_authorization_identity(
            authorization_ref, authorization, request_ref=request_ref,
            method_batch_ref=method_batch_ref, environment_ref=environment_ref,
            run_id=run_id, group=group, contract_digest=contract_digest),
        'reservation_ref': reservation_ref, 'gpu_uuid': gpu_uuid,
        'expected_output': 'actionmesh/' + OUTPUT_DIRECTORY,
        'execution_contract': {
            'purpose': 'scientific',
            'evidence_mode': 'prospective_confirmatory',
            'max_attempts': 1, 'max_confirmation_trials': 1,
            'max_retries_per_trial': 0,
        },
        'issued_at': datetime.now(timezone.utc).isoformat(),
        'expires_at': authorization['expires_at'],
    }
    value['ticket_digest'] = canonical_record_digest(value, 'ticket_digest')
    try:
        with path.open('x') as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
    except FileExistsError:
        existing = scoring.read_json(path)
        stable_keys = set(value) - {'issued_at', 'ticket_digest'}
        if (set(existing) != set(value)
                or any(existing.get(key) != value[key] for key in stable_keys)
                or existing.get('ticket_digest') != canonical_record_digest(
                    existing, 'ticket_digest')):
            raise ValueError('C14 launch ticket belongs to different work')
    return scoring.file_ref(root, path)


def consume_authorization(root: Path, authorization_ref: dict,
                          authorization: dict, *, reservation_ref: dict,
                          launch_ticket_ref: dict,
                          request_ref: dict, method_batch_ref: dict,
                          environment_ref: dict, run_id: str, group: str,
                          contract_digest: str, native_plan_ref: dict,
                          native_plan_digest: str, harness_plan_ref: dict,
                          harness_plan_digest: str, expected_command: list[str],
                          native_validator, harness_validator) -> dict:
    directory = Path(root)/'inputs'/INPUT_NAMESPACE/'authorization-consumption'
    directory.mkdir(parents=True, exist_ok=True)
    path = authorization_consumption_path(root, authorization_ref)
    identity = _authorization_identity(
        authorization_ref, authorization, request_ref=request_ref,
        method_batch_ref=method_batch_ref, environment_ref=environment_ref,
        run_id=run_id, group=group, contract_digest=contract_digest)
    reservation = scoring.read_json(scoring.resolve_ref(root, reservation_ref))
    launch_ticket = scoring.read_json(scoring.resolve_ref(root, launch_ticket_ref))
    if (reservation.get('kind') != RESERVATION_KIND
            or reservation.get('version') != 1
            or reservation.get('state') != 'reserved'
            or any(reservation.get(key) != item for key, item in identity.items())
            or reservation.get('reservation_digest') != canonical_record_digest(
                reservation, 'reservation_digest')):
        raise ValueError('Exact recoverable authorization reservation required')
    if (launch_ticket.get('kind') != LAUNCH_TICKET_KIND
            or launch_ticket.get('authorization_ref') != authorization_ref
            or launch_ticket.get('reservation_ref') != reservation_ref
            or launch_ticket.get('request_ref') != request_ref
            or launch_ticket.get('run_id') != run_id
            or launch_ticket.get('ticket_digest') != canonical_record_digest(
                launch_ticket, 'ticket_digest')):
        raise ValueError('Exact staged C14 launch ticket required')
    native_path = scoring.resolve_ref(root, native_plan_ref)
    harness_path = scoring.resolve_ref(root, harness_plan_ref)
    native_plan = scoring.read_json(native_path)
    harness_plan = scoring.read_json(harness_path)
    for value, label in ((native_plan_digest, 'native'),
                         (harness_plan_digest, 'harness')):
        if (not isinstance(value, str) or len(value) != 64
                or any(character not in '0123456789abcdef'
                       for character in value)):
            raise ValueError('Exact ' + label + ' plan digest required')
    native_validator.validate_plan(root, native_plan)
    harness_validate = getattr(harness_validator, 'validate_plan', None)
    if not callable(harness_validate):
        raise ValueError('Installed harness plan validator required')
    harness_validate(root, harness_plan)
    tasks = harness_plan.get('tasks')
    jobs = native_plan.get('jobs')
    declared_inputs = {(ref.get('path'), ref.get('sha256'))
                       for ref in jobs[0].get('input_refs', [])} \
        if isinstance(jobs, list) and len(jobs) == 1 else set()
    required_staged = {
        (ref['path'], ref['sha256']) for ref in
        (request_ref, authorization_ref, reservation_ref, launch_ticket_ref)}
    if (native_plan.get('plan_digest') != native_plan_digest
            or native_validator.plan_digest(native_plan) != native_plan_digest
            or harness_plan.get('plan_digest') != harness_plan_digest
            or canonical_record_digest(harness_plan, 'plan_digest')
            != harness_plan_digest
            or native_plan.get('run_id') != run_id
            or not isinstance(jobs, list) or len(jobs) != 1
            or jobs[0].get('trial_id') != TASK_ID
            or jobs[0].get('group') != group
            or jobs[0].get('command') != expected_command
            or not required_staged.issubset(declared_inputs)
            or harness_plan.get('batch_id') != run_id
            or not isinstance(tasks, list) or len(tasks) != 1
            or tasks[0].get('task_id') != TASK_ID
            or tasks[0].get('plan_ref') != native_plan_ref):
        raise ValueError('Exact validated C14 native/harness plan pair required')
    value = {
        'kind': CONSUMPTION_KIND, 'version': 1,
        **identity, 'reservation_ref': reservation_ref,
        'launch_ticket_ref': launch_ticket_ref,
        'native_plan_ref': native_plan_ref,
        'native_plan_digest': native_plan_digest,
        'harness_plan_ref': harness_plan_ref,
        'harness_plan_digest': harness_plan_digest,
        'consumed_at': datetime.now(timezone.utc).isoformat(),
    }
    value['consumption_digest'] = canonical_record_digest(
        value, 'consumption_digest')
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
    return scoring.file_ref(root, path)


def validate_final_consumption(root: Path, consumption_path: Path, *,
                               expected_harness_digest: str,
                               native_validator, harness_validator) -> tuple:
    """Validate the controller-side final receipt and exact installed plans."""
    root, consumption_path = Path(root).resolve(), Path(consumption_path).resolve()
    consumption_path.relative_to(root)
    consumption = scoring.read_json(consumption_path)
    required = {
        'kind', 'version', 'authorization_ref', 'authorization_nonce',
        'request_ref', 'method_batch_ref', 'environment_ref', 'run_id',
        'task_id', 'group', 'contract_digest', 'reservation_ref',
        'launch_ticket_ref', 'native_plan_ref', 'native_plan_digest',
        'harness_plan_ref', 'harness_plan_digest', 'consumed_at',
        'consumption_digest',
    }
    if (set(consumption) != required
            or consumption.get('kind') !=
            CONSUMPTION_KIND
            or consumption.get('version') != 1
            or consumption.get('harness_plan_digest') != expected_harness_digest
            or consumption.get('consumption_digest') != canonical_record_digest(
                consumption, 'consumption_digest')):
        raise ValueError('Exact finalized C14 consumption receipt required')
    canonical_path = authorization_consumption_path(
        root, consumption['authorization_ref']).resolve()
    if consumption_path != canonical_path:
        raise ValueError('Canonical C14 consumption path required')
    authorization = _record(
        root, consumption['authorization_ref'],
        kind=AUTHORIZATION_KIND,
        digest_key='authorization_digest')
    reservation = scoring.read_json(scoring.resolve_ref(
        root, consumption['reservation_ref']))
    ticket = scoring.read_json(scoring.resolve_ref(
        root, consumption['launch_ticket_ref']))
    identity_keys = (
        'authorization_ref', 'authorization_nonce', 'request_ref',
        'method_batch_ref', 'environment_ref', 'run_id', 'task_id', 'group',
        'contract_digest')
    if (any(consumption.get(key) != ticket.get(key) for key in identity_keys)
            or any(consumption.get(key) != reservation.get(key)
                   for key in identity_keys)
            or authorization.get('authorization_nonce') != consumption.get(
                'authorization_nonce')
            or authorization.get('request_ref') != consumption.get('request_ref')
            or authorization.get('method_batch_ref') != consumption.get(
                'method_batch_ref')
            or authorization.get('environment_ref') != consumption.get(
                'environment_ref')
            or authorization.get('run_id') != consumption.get('run_id')
            or authorization.get('task_id') != consumption.get('task_id')
            or authorization.get('group') != consumption.get('group')
            or authorization.get('contract_digest') != consumption.get(
                'contract_digest')
            or authorization.get('status') != 'authorized'
            or authorization.get('scope') != AUTHORIZATION_SCOPE
            or reservation.get('kind') !=
            RESERVATION_KIND
            or reservation.get('state') != 'reserved'
            or reservation.get('reservation_digest') != canonical_record_digest(
                reservation, 'reservation_digest')
            or ticket.get('kind') != LAUNCH_TICKET_KIND
            or ticket.get('reservation_ref') != consumption['reservation_ref']
            or ticket.get('gpu_uuid') != authorization.get('gpu_uuid')
            or ticket.get('expected_output') !=
            'actionmesh/' + OUTPUT_DIRECTORY
            or ticket.get('execution_contract') != {
                'purpose': 'scientific',
                'evidence_mode': 'prospective_confirmatory',
                'max_attempts': 1, 'max_confirmation_trials': 1,
                'max_retries_per_trial': 0}
            or ticket.get('expires_at') != authorization.get('expires_at')
            or ticket.get('ticket_digest') != canonical_record_digest(
                ticket, 'ticket_digest')):
        raise ValueError('C14 final receipt identity closure mismatch')
    authorized_at = _timezone(authorization.get('authorized_at'), 'authorized_at')
    expires_at = _timezone(authorization.get('expires_at'), 'expires_at')
    consumed_at = _timezone(consumption.get('consumed_at'), 'consumed_at')
    if (authorized_at > consumed_at or consumed_at > expires_at
            or datetime.now(timezone.utc) > expires_at):
        raise ValueError('Finalized C14 authorization is expired or reordered')
    native_path = scoring.resolve_ref(root, consumption['native_plan_ref'])
    harness_path = scoring.resolve_ref(root, consumption['harness_plan_ref'])
    native_plan = scoring.read_json(native_path)
    harness_plan = scoring.read_json(harness_path)
    native_validator.validate_plan(root, native_plan)
    harness_validator.validate_plan(root, harness_plan)
    jobs, tasks = native_plan.get('jobs'), harness_plan.get('tasks')
    declared = {(ref.get('path'), ref.get('sha256'))
                for ref in jobs[0].get('input_refs', [])} \
        if isinstance(jobs, list) and len(jobs) == 1 else set()
    closure = {
        (consumption[key]['path'], consumption[key]['sha256'])
        for key in ('request_ref', 'authorization_ref', 'reservation_ref',
                    'launch_ticket_ref')}
    expected_limits = {
        'max_attempts': 1, 'max_development_trials': 0,
        'max_confirmation_trials': 1, 'max_retries_per_trial': 0,
    }
    job = jobs[0] if isinstance(jobs, list) and len(jobs) == 1 else {}
    command = job.get('command')
    expected_flags = {
        '--root': '..',
        '--request': str(scoring.resolve_ref(root, consumption['request_ref'])),
        '--output': OUTPUT_DIRECTORY,
        '--gpu-uuid': authorization['gpu_uuid'],
        '--launch-ticket': str(scoring.resolve_ref(
            root, consumption['launch_ticket_ref'])),
    }
    command_bound = (isinstance(command, list) and len(command) >= 4
                     and command[1:4] == [
                         '-m', SCORING_MODULE, 'score'])
    if command_bound:
        for flag, value in expected_flags.items():
            if command.count(flag) != 1 or command.index(flag) + 1 >= len(command) \
                    or command[command.index(flag) + 1] != value:
                command_bound = False
                break
    if (native_plan.get('plan_digest') != consumption['native_plan_digest']
            or native_validator.plan_digest(native_plan) != consumption[
                'native_plan_digest']
            or harness_plan.get('plan_digest') != expected_harness_digest
            or harness_validator.plan_digest(harness_plan) != expected_harness_digest
            or native_plan.get('run_id') != consumption['run_id']
            or native_plan.get('purpose') != 'scientific'
            or native_plan.get('evidence_mode') != 'prospective_confirmatory'
            or any(native_plan.get('limits', {}).get(key) != value
                   for key, value in expected_limits.items())
            or job.get('trial_id') != consumption['task_id']
            or job.get('group') != consumption['group']
            or job.get('arm_role') != 'treatment'
            or not command_bound
            or not closure.issubset(declared)
            or not isinstance(tasks, list) or len(tasks) != 1
            or tasks[0].get('task_id') != consumption['task_id']
            or tasks[0].get('plan_ref') != consumption['native_plan_ref']):
        raise ValueError('Final C14 consumption differs from installed plan pair')
    for key in ('authorization_ref', 'reservation_ref', 'launch_ticket_ref',
                'request_ref'):
        scoring.resolve_ref(root, consumption[key])
    return consumption, native_plan, harness_plan


def scientific_job_contract(protocol: dict, *, wall_seconds: int):
    evidence_mode = protocol.get('evidence_mode')
    if evidence_mode != 'prospective_confirmatory':
        raise ValueError('C14 native scoring requires prospective confirmation')
    return ({'arm_role': 'treatment'}, {
        'max_attempts': 1, 'max_development_trials': 0,
        'max_confirmation_trials': 1, 'max_retries_per_trial': 0,
        'wall_time_seconds': wall_seconds,
        'attempt_timeout_seconds': wall_seconds,
    })


def contains_ref(value, wanted):
    if isinstance(value, dict):
        return value == wanted or any(contains_ref(item, wanted)
                                      for item in value.values())
    if isinstance(value, list):
        return any(contains_ref(item, wanted) for item in value)
    return False


def require_method_boundary(report: dict) -> None:
    boundary = report.get('workflow_boundary') if isinstance(report, dict) else None
    if (not isinstance(boundary, dict)
            or boundary.get('action') != 'dispatch'
            or boundary.get('candidate_id') != CANDIDATE_ID
            or boundary.get('required_stage') != 'design_verified'
            or boundary.get('ready') is not True):
        raise ValueError('Current C14 design-verified dispatch boundary required')


def require_c14_contract(contract: dict, *, benchmark_revision: str) -> None:
    expected_contrasts = CONTRACT_CONTRASTS
    expected_names = CONTRACT_ARM_NAMES
    arms = contract.get('arm_requirements')
    if (contract.get('benchmark_id') != 'facebook/actionbench'
            or contract.get('benchmark_revision') != benchmark_revision
            or contract.get('primary_metric') != PRIMARY_METRIC
            or [row.get('name') for row in contract.get('metrics', [])]
            != list(scoring.METRICS)
            or contract.get('contrasts') != expected_contrasts
            or not isinstance(arms, dict) or set(arms) != set(expected_names)
            or any(arms[role].get('name') != name
                   for role, name in expected_names.items())):
        raise ValueError('Exact C14 five-role native contract required')
    scorer = contract.get('scorer')
    if not isinstance(scorer, dict) or scorer.get('kind') != 'official':
        raise ValueError('C14 requires the official ActionBench scorer')


def scoring_outputs() -> list[str]:
    return [
        'actionmesh/' + OUTPUT_DIRECTORY + '/result.json',
        'actionmesh/' + OUTPUT_DIRECTORY + '/raw-manifest.json',
        'actionmesh/' + OUTPUT_DIRECTORY + '/raw-evidence.tar',
    ]


def _runtime(root: Path, environment_path: Path, gpu_uuid: str) -> tuple[dict, dict]:
    from prepare_actionbench_full128_window import validate_environment_closure
    canonical = Path(root).resolve()/'inputs/native-runtime/environment.json'
    if Path(environment_path).resolve() != canonical:
        raise ValueError('Canonical current native environment path required')
    environment_ref = scoring.file_ref(root, environment_path)
    environment = scoring.read_json(environment_path)
    validate_environment_closure(root, environment_path, environment, gpu_uuid)
    packages = ('numpy', 'torch', 'trimesh', 'scipy', 'pytorch3d')
    for name in packages:
        if environment.get('packages', {}).get(name) != importlib.metadata.version(name):
            raise ValueError('Current installed package differs from runtime lock: ' + name)
    dependency_refs = environment.get('dependency_lock_refs')
    if not isinstance(dependency_refs, list) or not dependency_refs:
        raise ValueError('Retained native dependency lock files required')
    for ref in dependency_refs:
        scoring.resolve_ref(root, ref)
    return environment_ref, environment


def build_plans(root: Path, *, request_path: Path, protocol_path: Path,
                method_batch_path: Path, environment_path: Path,
                admission_path: Path,
                skill_dir: Path, plan_dir: Path, run_id: str, group: str,
                gpu_uuid: str, wall_seconds: int, ram_mib: int,
                cpu_cores: int):
    root = Path(root).resolve()
    request_path, protocol_path, method_batch_path, environment_path, admission_path = map(
        lambda item: Path(item).resolve(),
        (request_path, protocol_path, method_batch_path, environment_path,
         admission_path))
    plan_dir = Path(plan_dir).resolve(); plan_dir.relative_to(root)
    if not 301 <= wall_seconds <= scoring.MAX_TOTAL_SECONDS:
        raise ValueError('C14 scoring wall budget must be 301..27000 seconds')
    if ram_mib < 1 or cpu_cores < 1:
        raise ValueError('Explicit positive admitted host RAM/CPU required')
    if (not gpu_uuid.startswith('GPU-')
            or any(character in gpu_uuid for character in '\n\r, ')):
        raise ValueError('One actual physical GPU UUID required')
    scripts = Path(skill_dir).resolve()/'scripts'
    if not (scripts/'run_harness.py').is_file():
        raise ValueError('Complete installed research-autopilot skill required')
    sys.path.insert(0, str(scripts))
    import _autoresearch as core
    import _native_eval as evaluation
    import run_experiments as native
    import run_harness as harness
    import verify_methods

    request = scoring.read_json(request_path)
    comparison = scoring.verify_scoring_request(root, request)
    request_ref = scoring.file_ref(root, request_path)
    if request['timeout_seconds_per_case'] * len(comparison['scoring_cases']) > wall_seconds - 300:
        raise ValueError('Plan must reserve at least 300 seconds for validation/collection')
    environment_ref, environment = _runtime(root, environment_path, gpu_uuid)
    protocol = scoring.read_json(protocol_path)
    evaluation.verify_protocol(root, protocol)
    protocol_ref = scoring.file_ref(root, protocol_path)
    discovery = protocol.get('method_discovery')
    if (not isinstance(discovery, dict)
            or discovery.get('candidate_id') != CANDIDATE_ID):
        raise ValueError('Frozen C14 method-discovery protocol required')
    method_batch = core.load_file(method_batch_path)
    method_batch_ref = scoring.file_ref(root, method_batch_path)
    method_report = verify_methods.before_action(
        root, method_batch, 'dispatch', CANDIDATE_ID)
    require_method_boundary(method_report)
    contract = evaluation.contract_for_group(protocol, group)
    contract_digest = hashlib.sha256(json.dumps(
        contract, sort_keys=True, separators=(',', ':'), allow_nan=False
    ).encode()).hexdigest()
    require_c14_contract(contract,
                         benchmark_revision=request['benchmark_revision'])
    implementation_refs = bind_contract_arms(contract, comparison, root)
    for ref in implementation_refs:
        scoring.resolve_ref(root, ref)
    base_sample_ref = contract['sample_manifest_ref']
    scoring.resolve_ref(root, base_sample_ref)
    sample_ref = contract.get('selection', {}).get(
        'selected_manifest_ref', base_sample_ref)
    sample = scoring.read_json(scoring.resolve_ref(root, sample_ref))
    if (sample.get('sample_ids') != [request['uid']]
            or sample.get('denominator') != 1
            or sample.get('predictions_per_sample') != 1):
        raise ValueError('Frozen one-UID C14 sample manifest required')
    labels_ref = contract['labels_or_tests_ref']
    if labels_ref != request['ground_truth_ref']:
        labels = scoring.read_json(scoring.resolve_ref(root, labels_ref))
        if not contains_ref(labels, request['ground_truth_ref']):
            raise ValueError('Frozen labels must include exact released ground truth')
    scorer_refs = {ref['path']: ref for ref in
                   contract['scorer']['source_refs'] + contract['scorer']['code_refs']}
    for ref in (request['adapter_ref'], request['deterministic_entry_ref'],
                *request['official_source_refs']):
        if scorer_refs.get(ref['path']) != ref:
            raise ValueError('Native contract omits executed official scorer source')
    if (not protocol.get('frozen_at')
            or protocol.get('protocol_digest') != core.protocol_hash(protocol)):
        raise ValueError('Frozen reviewed C14 protocol required')
    if (request['inference_seed'] not in protocol['seed_policy']['seeds']
            or group not in protocol['required_groups']):
        raise ValueError('C14 request lies outside the frozen protocol inventory')
    admission_ref, admission_evidence_refs, authorization_ref, authorization = \
        require_dispatch_admission(
        root, admission_path, request=request, protocol_ref=protocol_ref,
        protocol=protocol, request_ref=request_ref,
        method_batch_ref=method_batch_ref, environment_ref=environment_ref,
        run_id=run_id, group=group, contract_digest=contract_digest,
        gpu_uuid=gpu_uuid, wall_seconds=wall_seconds,
        core=core)
    job_contract, limits = scientific_job_contract(
        protocol, wall_seconds=wall_seconds)

    comparison_inputs = comparison['input_refs']
    inputs = [
        request_ref, request['comparison_ref'],
        request['ground_truth_ref'], request['population_ref'],
        request['dataset_admission_ref'], request['dataset_semantics_ref'],
        protocol_ref, method_batch_ref, admission_ref, *admission_evidence_refs,
        base_sample_ref, sample_ref, labels_ref, environment_ref,
        *environment['dependency_lock_refs'], *comparison_inputs,
    ]
    code_refs = [
        scoring.file_ref(root, root/'actionmesh/research_math/__init__.py'),
        scoring.file_ref(root, root/COMPARISON_SOURCE),
        scoring.file_ref(root, root/SCORING_SOURCE),
        scoring.file_ref(root, root/LAUNCHER_SOURCE),
        *(scoring.file_ref(root, root/path) for path in EXTRA_CODE_SOURCES),
        request['adapter_ref'], request['deterministic_entry_ref'],
        scoring.file_ref(root, root/'actionmesh/research_math/deterministic_knn.py'),
        *request['official_source_refs'], *implementation_refs,
    ]
    unique_inputs = _unique_refs(inputs, 'plan input')
    unique_code = _unique_refs(code_refs, 'plan code')
    native_path = plan_dir/'native.json'
    harness_path = plan_dir/'harness.json'
    ticket_path = (root/'inputs'/INPUT_NAMESPACE/'authorization-consumption'/
                   (authorization_ref['sha256'] + '.launch-ticket.json'))
    command = [
        sys.executable, '-m', SCORING_MODULE, 'score',
        '--root', '..', '--request', str(request_path),
        '--output', OUTPUT_DIRECTORY, '--gpu-uuid', gpu_uuid,
        '--launch-ticket', str(ticket_path.resolve()),
    ]
    revision = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=root, check=True,
        capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(
        ['git', 'diff', 'HEAD', '--binary'], cwd=root, check=True,
        capture_output=True).stdout
    untracked = subprocess.run(
        ['git', 'ls-files', '--others', '--exclude-standard'], cwd=root,
        check=True, capture_output=True, text=True).stdout.splitlines()
    if any(path.endswith('.py') for path in untracked):
        raise ValueError('Commit actual Python sources before Local acceptance')
    reservation_ref = reserve_authorization(
        root, authorization_ref, authorization, request_ref=request_ref,
        method_batch_ref=method_batch_ref, environment_ref=environment_ref,
        run_id=run_id, group=group, contract_digest=contract_digest)
    launch_ticket_ref = issue_launch_ticket(
        root, authorization_ref, authorization,
        reservation_ref=reservation_ref, request_ref=request_ref,
        method_batch_ref=method_batch_ref, environment_ref=environment_ref,
        run_id=run_id, group=group, contract_digest=contract_digest,
        gpu_uuid=gpu_uuid)
    unique_inputs[reservation_ref['path']] = reservation_ref
    unique_inputs[launch_ticket_ref['path']] = launch_ticket_ref
    plan_dir.mkdir(parents=True, exist_ok=True)
    allowed = {'dirty.patch', 'native.json', 'harness.json'}
    if any(path.name not in allowed for path in plan_dir.iterdir()):
        raise ValueError('C14 plan recovery directory contains unknown files')
    dirty_path = plan_dir/'dirty.patch'
    if dirty_path.exists():
        if dirty_path.is_symlink() or not dirty_path.is_file() \
                or dirty_path.read_bytes() != dirty:
            raise ValueError('Interrupted C14 plan has a different dirty patch')
    else:
        with dirty_path.open('xb') as stream:
            stream.write(dirty)
    unique_inputs[scoring.file_ref(root, dirty_path)['path']] = scoring.file_ref(
        root, dirty_path)
    plan = native.make_plan(
        root, run_id=run_id, purpose='scientific',
        evidence_mode=protocol['evidence_mode'],
        protocol_ref=protocol_ref,
        jobs=[{
            'trial_id': TASK_ID,
            'command': command, 'cwd': 'actionmesh',
            'input_refs': list(unique_inputs.values()),
            'code_refs': list(unique_code.values()),
            'output_paths': scoring_outputs(),
            'seed': request['inference_seed'], 'group': group,
            **job_contract,
        }],
        provenance={
            'git_revision': revision,
            'git_refs': [scoring.file_ref(root, dirty_path)],
            'model_revision': ('No model loaded by scoring; all five role '
                               'artifacts are frozen inputs'),
            'data_revision': request['benchmark_revision'],
            'environment_digest': environment_ref['sha256'],
            'environment_refs': [environment_ref,
                                 *environment['dependency_lock_refs']],
            'method_verification_ref': scoring.file_ref(root, method_batch_path),
            'scientific_admission_ref': admission_ref,
        },
        limits=limits)
    if native_path.exists():
        if scoring.read_json(native_path) != plan:
            raise ValueError('Interrupted C14 native plan differs from exact recovery')
    else:
        with native_path.open('x') as stream:
            stream.write(json.dumps(plan, indent=2, allow_nan=False) + '\n')
    outer = harness.make_plan(
        root, batch_id=run_id,
        tasks=[{
            'task_id': TASK_ID,
            'idea_id': CANDIDATE_ID, 'depends_on': [], 'priority': 1,
            'plan_ref': scoring.file_ref(root, native_path),
            'resources': {
                'cpu_cores': cpu_cores, 'ram_mib': ram_mib, 'gpu_count': 1,
                'gpu_peak_mib': None, 'allow_gpu_share': False,
                'memory_profile_ref': None,
                'exclusive_keys': [TASK_ID],
            },
        }],
        limits={
            'total_wall_seconds': wall_seconds,
            'window_seconds': wall_seconds,
            'max_parallel_tasks': 1, 'cpu_cores': cpu_cores,
            'ram_mib': ram_mib, 'max_gpu_task_seconds': wall_seconds,
        },
        gpus={'uuids': [gpu_uuid], 'safety_margin_mib': 1024,
              'max_tasks_per_gpu': 1})
    if harness_path.exists():
        if scoring.read_json(harness_path) != outer:
            raise ValueError('Interrupted C14 harness plan differs from exact recovery')
    else:
        with harness_path.open('x') as stream:
            stream.write(json.dumps(outer, indent=2, allow_nan=False) + '\n')
    consume_authorization(
        root, authorization_ref, authorization, reservation_ref=reservation_ref,
        launch_ticket_ref=launch_ticket_ref,
        request_ref=request_ref,
        method_batch_ref=method_batch_ref, environment_ref=environment_ref,
        run_id=run_id, group=group, contract_digest=contract_digest,
        native_plan_ref=scoring.file_ref(root, native_path),
        native_plan_digest=plan['plan_digest'],
        harness_plan_ref=scoring.file_ref(root, harness_path),
        harness_plan_digest=outer['plan_digest'], expected_command=command,
        native_validator=native, harness_validator=harness)
    return plan, outer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'request', 'protocol', 'method-batch', 'environment',
                 'admission', 'skill-dir', 'plan-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('run-id', 'group', 'gpu-uuid'):
        parser.add_argument('--' + name, required=True)
    for name in ('wall-seconds', 'ram-mib', 'cpu-cores'):
        parser.add_argument('--' + name, type=int, required=True)
    args = parser.parse_args()
    _, outer = build_plans(
        args.root, request_path=args.request, protocol_path=args.protocol,
        method_batch_path=args.method_batch, environment_path=args.environment,
        admission_path=args.admission,
        skill_dir=args.skill_dir, plan_dir=args.plan_dir, run_id=args.run_id,
        group=args.group, gpu_uuid=args.gpu_uuid,
        wall_seconds=args.wall_seconds, ram_mib=args.ram_mib,
        cpu_cores=args.cpu_cores)
    print(json.dumps({
        'plan': str(args.plan_dir.resolve()/'harness.json'),
        'approved_plan_digest': outer['plan_digest'],
        'execution_started': False, 'candidate_id': CANDIDATE_ID,
        'native_qualified': False, 'scientific_verdict': 'not_computed',
    }))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
