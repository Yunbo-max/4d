"""Build the single-use admitted C13 official-scoring harness plan.

The builder fails closed unless the current research-autopilot protocol verifies,
C13 is design-verified at the dispatch boundary, the frozen native contract names
the exact five logical roles, and the current Local runtime/GPU identity matches.
It emits plans only; it never runs the scorer or resumes a stopped GPU campaign.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys

from research_math import c13_native_scoring as scoring
from prepare_actionbench_full128_window import validate_environment_closure


CANDIDATE_ID = '4d-math-20261006-c13'
ROLE_FOR_CONTRACT_ARM = {
    'treatment': 'group_acceleration',
    'baseline': 'b_star',
    'b0': 'b0',
    'gaussian': 'gaussian',
    'quadratic_acceleration': 'quadratic_acceleration',
}


def canonical_record_digest(value: dict, digest_key: str = 'admission_digest') -> str:
    core = {key: item for key, item in value.items() if key != digest_key}
    return hashlib.sha256(json.dumps(
        core, sort_keys=True, separators=(',', ':'), allow_nan=False
    ).encode()).hexdigest()


def _timezone(value, label: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except (AttributeError, ValueError) as error:
        raise ValueError('Timezone-aware ' + label + ' required') from error
    if parsed.tzinfo is None:
        raise ValueError('Timezone-aware ' + label + ' required')


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
            or admission.get('kind') != 'c13-scientific-dispatch-admission'
            or admission.get('version') != 1
            or admission.get('candidate_id') != CANDIDATE_ID
            or admission.get('uid') != uid
            or admission.get('benchmark_revision') != benchmark_revision
            or admission.get('comparison_digest') != comparison_digest
            or admission.get('protocol_ref') != protocol_ref
            or admission.get('method_batch_ref') != method_batch_ref
            or admission.get('status') != 'admitted'
            or admission.get('admission_digest') != canonical_record_digest(admission)):
        raise ValueError('Exact hash-bound C13 scientific admission required')
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
        if root is not None and row.get('report_ref') is not None:
            report = json.loads(scoring.resolve_ref(root, row['report_ref']).read_text())
            implementation_sha = report.get('implementation_sha256')
            if (implementation_sha is not None
                    and implementation_sha not in {
                        ref['sha256'] for ref in implementation_refs}):
                raise ValueError('Contract implementation differs from arm report: '
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
    value = json.loads(scoring.resolve_ref(root, ref).read_text())
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
                               method_batch_ref: dict, gpu_uuid: str,
                               wall_seconds: int) -> tuple[dict, list[dict]]:
    admission_ref = scoring.file_ref(root, admission_path)
    scoring.resolve_ref(root, admission_ref)
    admission = json.loads(admission_path.read_text())
    require_admission_core(
        admission, uid=request['uid'],
        benchmark_revision=request['benchmark_revision'],
        comparison_digest=request['comparison_ref']['sha256'],
        protocol_ref=protocol_ref, method_batch_ref=method_batch_ref)
    gate = json.loads(scoring.resolve_ref(
        root, admission['natural_gate_0_ref']).read_text())
    if (gate.get('schema_id') != 'natural-gate-0'
            or gate.get('schema_version') != '1.0.0'
            or gate.get('outcome') != 'PASS'):
        raise ValueError('Natural Gate 0 PASS required')
    importance = json.loads(scoring.resolve_ref(
        root, admission['importance_decision_ref']).read_text())
    if (importance.get('schema_id') != 'importance-decision'
            or importance.get('schema_version') != '1.0.0'
            or importance.get('outcome') not in ('CLEAR', 'CONCURRENT')
            or importance.get('natural_gate_0_ref') != admission[
                'natural_gate_0_ref']
            or importance.get('parent_problem_ref') != gate.get(
                'parent_problem_ref')):
        raise ValueError('Compatible importance/collision decision required')
    family = _record(root, admission['family_split_ref'],
                     kind='c13-family-split', digest_key='split_digest')
    family_keys = {
        'kind', 'version', 'candidate_id', 'benchmark_revision',
        'development_ids', 'confirmation_ids', 'family_by_uid', 'frozen_at',
        'split_digest',
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
    criteria = _record(root, admission['outcome_criteria_ref'],
                       kind='c13-outcome-criteria',
                       digest_key='criteria_digest')
    criteria_keys = {
        'kind', 'version', 'candidate_id', 'comparison_digest',
        'primary_metric', 'guardrail_metrics', 'min_effect',
        'noninferiority_margins', 'independent_unit', 'frozen_at',
        'criteria_digest',
    }
    margins = criteria.get('noninferiority_margins')
    if (set(criteria) != criteria_keys
            or criteria.get('candidate_id') != CANDIDATE_ID
            or criteria.get('comparison_digest') != request['comparison_ref'][
                'sha256']
            or criteria.get('primary_metric') != 'cd_motion'
            or criteria.get('guardrail_metrics') != ['cd_3d', 'cd_4d']
            or criteria.get('independent_unit') != 'asset_family'
            or isinstance(criteria.get('min_effect'), bool)
            or not isinstance(criteria.get('min_effect'), (int, float))
            or not math.isfinite(criteria['min_effect'])
            or criteria['min_effect'] <= 0
            or not isinstance(margins, dict)
            or set(margins) != {'cd_3d', 'cd_4d'}
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or value < 0
                   for value in margins.values())):
        raise ValueError('Prospectively frozen numeric C13 criteria required')
    _timezone(criteria['frozen_at'], 'outcome criteria frozen_at')
    authorization = _record(root, admission['gpu_resume_authorization_ref'],
                            kind='c13-gpu-resume-authorization',
                            digest_key='authorization_digest')
    authorization_keys = {
        'kind', 'version', 'candidate_id', 'uid', 'gpu_uuid',
        'protocol_digest', 'comparison_digest', 'scope',
        'max_gpu_task_seconds', 'status', 'authorized_at',
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
            or authorization.get('scope') != 'single_c13_scoring_attempt'
            or authorization.get('status') != 'authorized'
            or isinstance(authorization.get('max_gpu_task_seconds'), bool)
            or not isinstance(authorization.get('max_gpu_task_seconds'), int)
            or authorization.get('max_gpu_task_seconds') < wall_seconds):
        raise ValueError('Explicit current GPU-resume authorization required')
    _timezone(authorization.get('authorized_at'), 'authorized_at')
    direct = [admission[key] for key in (
        'natural_gate_0_ref', 'importance_decision_ref', 'family_split_ref',
        'outcome_criteria_ref', 'gpu_resume_authorization_ref')]
    closure = direct + _nested_refs(gate) + _nested_refs(importance)
    unique = _unique_refs(closure, 'admission evidence')
    for ref in unique.values():
        scoring.resolve_ref(root, ref)
    return admission_ref, list(unique.values())


def scientific_job_contract(protocol: dict, *, wall_seconds: int):
    evidence_mode = protocol.get('evidence_mode')
    if evidence_mode not in ('prospective_confirmatory', 'retrospective_audit'):
        raise ValueError('C13 native scoring requires a frozen confirmation protocol')
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
        raise ValueError('Current C13 design-verified dispatch boundary required')


def require_c13_contract(contract: dict, *, benchmark_revision: str) -> None:
    expected_contrasts = {
        'treatment': 'group_acceleration',
        'baseline': 'b_star',
        'controls': ['b0', 'gaussian', 'quadratic_acceleration'],
    }
    expected_names = {
        'treatment': 'group_acceleration',
        'baseline': 'b_star',
        'b0': 'b0',
        'gaussian': 'gaussian',
        'quadratic_acceleration': 'quadratic_acceleration',
    }
    arms = contract.get('arm_requirements')
    if (contract.get('benchmark_id') != 'facebook/actionbench'
            or contract.get('benchmark_revision') != benchmark_revision
            or contract.get('primary_metric') != 'cd_motion'
            or [row.get('name') for row in contract.get('metrics', [])]
            != list(scoring.METRICS)
            or contract.get('contrasts') != expected_contrasts
            or not isinstance(arms, dict) or set(arms) != set(expected_names)
            or any(arms[role].get('name') != name
                   for role, name in expected_names.items())):
        raise ValueError('Exact C13 five-role native contract required')
    scorer = contract.get('scorer')
    if not isinstance(scorer, dict) or scorer.get('kind') != 'official':
        raise ValueError('C13 requires the official ActionBench scorer')


def scoring_outputs() -> list[str]:
    return [
        'actionmesh/c13-scoring-output/result.json',
        'actionmesh/c13-scoring-output/raw-manifest.json',
        'actionmesh/c13-scoring-output/raw-evidence.tar',
    ]


def _runtime(root: Path, environment_path: Path, gpu_uuid: str) -> tuple[dict, dict]:
    canonical = Path(root).resolve()/'inputs/native-runtime/environment.json'
    if Path(environment_path).resolve() != canonical:
        raise ValueError('Canonical current native environment path required')
    environment_ref = scoring.file_ref(root, environment_path)
    environment = json.loads(environment_path.read_text())
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
        raise ValueError('C13 scoring wall budget must be 301..27000 seconds')
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

    request = json.loads(request_path.read_text())
    comparison = scoring.verify_scoring_request(root, request)
    if request['timeout_seconds_per_case'] * len(comparison['scoring_cases']) > wall_seconds - 300:
        raise ValueError('Plan must reserve at least 300 seconds for validation/collection')
    environment_ref, environment = _runtime(root, environment_path, gpu_uuid)
    protocol = json.loads(protocol_path.read_text())
    evaluation.verify_protocol(root, protocol)
    protocol_ref = scoring.file_ref(root, protocol_path)
    discovery = protocol.get('method_discovery')
    if (not isinstance(discovery, dict)
            or discovery.get('candidate_id') != CANDIDATE_ID):
        raise ValueError('Frozen C13 method-discovery protocol required')
    method_batch = core.load_file(method_batch_path)
    method_batch_ref = scoring.file_ref(root, method_batch_path)
    method_report = verify_methods.before_action(
        root, method_batch, 'dispatch', CANDIDATE_ID)
    require_method_boundary(method_report)
    contract = evaluation.contract_for_group(protocol, group)
    require_c13_contract(contract,
                         benchmark_revision=request['benchmark_revision'])
    implementation_refs = bind_contract_arms(contract, comparison, root)
    for ref in implementation_refs:
        scoring.resolve_ref(root, ref)
    base_sample_ref = contract['sample_manifest_ref']
    scoring.resolve_ref(root, base_sample_ref)
    sample_ref = contract.get('selection', {}).get(
        'selected_manifest_ref', base_sample_ref)
    sample = json.loads(scoring.resolve_ref(root, sample_ref).read_text())
    if (sample.get('sample_ids') != [request['uid']]
            or sample.get('denominator') != 1
            or sample.get('predictions_per_sample') != 1):
        raise ValueError('Frozen one-UID C13 sample manifest required')
    labels_ref = contract['labels_or_tests_ref']
    if labels_ref != request['ground_truth_ref']:
        labels = json.loads(scoring.resolve_ref(root, labels_ref).read_text())
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
        raise ValueError('Frozen reviewed C13 protocol required')
    if (request['inference_seed'] not in protocol['seed_policy']['seeds']
            or group not in protocol['required_groups']):
        raise ValueError('C13 request lies outside the frozen protocol inventory')
    admission_ref, admission_evidence_refs = require_dispatch_admission(
        root, admission_path, request=request, protocol_ref=protocol_ref,
        protocol=protocol, method_batch_ref=method_batch_ref,
        gpu_uuid=gpu_uuid, wall_seconds=wall_seconds)
    job_contract, limits = scientific_job_contract(
        protocol, wall_seconds=wall_seconds)

    comparison_inputs = comparison['input_refs']
    inputs = [
        scoring.file_ref(root, request_path), request['comparison_ref'],
        request['ground_truth_ref'], request['population_ref'],
        protocol_ref, method_batch_ref, admission_ref, *admission_evidence_refs,
        base_sample_ref, sample_ref, labels_ref, environment_ref,
        *environment['dependency_lock_refs'], *comparison_inputs,
    ]
    code_refs = [
        scoring.file_ref(root, root/'actionmesh/research_math/__init__.py'),
        scoring.file_ref(root, root/'actionmesh/research_math/c13_native_comparison.py'),
        scoring.file_ref(root, root/'actionmesh/research_math/c13_native_scoring.py'),
        request['adapter_ref'], request['deterministic_entry_ref'],
        *request['official_source_refs'], *implementation_refs,
    ]
    unique_inputs = _unique_refs(inputs, 'plan input')
    unique_code = _unique_refs(code_refs, 'plan code')
    command = [
        sys.executable, '-m', 'research_math.c13_native_scoring', 'score',
        '--root', '..', '--request', str(request_path),
        '--output', 'c13-scoring-output', '--gpu-uuid', gpu_uuid,
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
    plan_dir.mkdir(parents=True, exist_ok=False)
    dirty_path = plan_dir/'dirty.patch'; dirty_path.write_bytes(dirty)
    unique_inputs[scoring.file_ref(root, dirty_path)['path']] = scoring.file_ref(
        root, dirty_path)
    plan = native.make_plan(
        root, run_id=run_id, purpose='scientific',
        evidence_mode=protocol['evidence_mode'],
        protocol_ref=protocol_ref,
        jobs=[{
            'trial_id': 'c13-five-role-official-scoring',
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
    native_path = plan_dir/'native.json'
    native_path.write_text(json.dumps(plan, indent=2, allow_nan=False) + '\n')
    outer = harness.make_plan(
        root, batch_id=run_id,
        tasks=[{
            'task_id': 'c13-five-role-official-scoring',
            'idea_id': CANDIDATE_ID, 'depends_on': [], 'priority': 1,
            'plan_ref': scoring.file_ref(root, native_path),
            'resources': {
                'cpu_cores': cpu_cores, 'ram_mib': ram_mib, 'gpu_count': 1,
                'gpu_peak_mib': None, 'allow_gpu_share': False,
                'memory_profile_ref': None,
                'exclusive_keys': ['c13-five-role-official-scoring'],
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
    (plan_dir/'harness.json').write_text(
        json.dumps(outer, indent=2, allow_nan=False) + '\n')
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
