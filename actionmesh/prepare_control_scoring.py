"""Build an admitted one-asset scientific qualification plan, never execute it.

Requires Local's source-backed native qualification protocol, real request and
actual allocated physical GPU UUID. Absence of those inputs is a named blocker;
this builder does not invent scientific gates/thresholds or disguise evaluation
as engineering. New source is generated_unexecuted until Local acceptance.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import subprocess

from research_math.control_scoring import ARMS, check_contract_binding, file_ref, verify_request


def contains_ref(value, wanted):
    if isinstance(value, dict):
        return value == wanted or any(contains_ref(v, wanted) for v in value.values())
    if isinstance(value, list): return any(contains_ref(v, wanted) for v in value)
    return False


def build_plans(root: Path, *, request_path: Path, protocol_path: Path, skill_dir: Path,
                environment_path: Path, plan_dir: Path, run_id: str, group: str, gpu_uuid: str,
                wall_seconds: int, ram_mib: int, cpu_cores: int):
    root = root.resolve(); request_path = request_path.resolve(); protocol_path = protocol_path.resolve()
    plan_dir = plan_dir.resolve(); plan_dir.relative_to(root)
    if not 1 <= wall_seconds <= 27000: raise ValueError('Finite limit <=27000; reserve 1800/28800 for collection')
    if ram_mib < 1 or cpu_cores < 1: raise ValueError('Explicit positive admitted host RAM/CPU required')
    if not gpu_uuid.startswith('GPU-') or any(c in gpu_uuid for c in '\n\r, '):
        raise ValueError('One actual physical GPU UUID required')
    scripts = skill_dir.resolve()/'scripts'
    if not (scripts/'run_harness.py').is_file(): raise ValueError('Complete installed skill required')
    sys.path.insert(0, str(scripts))
    import run_experiments as native
    import run_harness as harness
    import _native_eval as evaluation
    import _autoresearch as core
    request = json.loads(request_path.read_text()); verify_request(root, request)
    environment_ref = file_ref(root, environment_path)
    environment = json.loads(environment_path.read_text())
    if environment.get('python_executable') != sys.executable or environment.get('gpu_uuid') != gpu_uuid or environment.get('execution_mode') != 'native_host':
        raise ValueError('Actual Local interpreter/native runtime/GPU identity required')
    for name in ('numpy', 'torch', 'trimesh', 'scipy', 'pytorch3d'):
        if environment.get('packages', {}).get(name) != importlib.metadata.version(name):
            raise ValueError('Current installed package differs from runtime lock: '+name)
    dependency_refs = environment.get('dependency_lock_refs')
    if not isinstance(dependency_refs, list) or not dependency_refs:
        raise ValueError('Retained native dependency lock files required')
    from research_math.control_scoring import resolve_ref
    for ref in dependency_refs: resolve_ref(root, ref)
    protocol = json.loads(protocol_path.read_text())
    if protocol.get('method_discovery'):
        raise ValueError('This preparation is baseline-only; cannot dispatch a candidate method')
    evaluation.verify_protocol(root, protocol)  # Current native contract; no green booleans substituted.
    contract = evaluation.contract_for_group(protocol, group)
    check_contract_binding(root, request, contract)
    selected_ref = contract.get('selection', {}).get('selected_manifest_ref', contract['sample_manifest_ref'])
    sample_manifest = json.loads(resolve_ref(root, selected_ref).read_text())
    if sample_manifest.get('sample_ids') != [request['uid']] or sample_manifest.get('denominator') != 1 or sample_manifest.get('predictions_per_sample') != 1:
        raise ValueError('One native asset per arm; three arms are not three samples')
    labels_ref = contract['labels_or_tests_ref']
    if labels_ref != request['ground_truth_ref']:
        labels = json.loads(resolve_ref(root, labels_ref).read_text())
        if not contains_ref(labels, request['ground_truth_ref']):
            raise ValueError('Frozen label inventory must include exact original GT bytes')
    population = json.loads(resolve_ref(root, request['population_ref']).read_text())
    if (contract['benchmark_id'], contract['benchmark_revision']) != ('facebook/actionbench', population['revision']):
        raise ValueError('Protocol/request benchmark version mismatch')
    if not protocol.get('frozen_at') or protocol.get('protocol_digest') != core.protocol_hash(protocol):
        raise ValueError('Local must freeze and review the native qualification protocol first')
    if request['inference_seed'] not in protocol['seed_policy']['seeds'] or group not in protocol['required_groups']:
        raise ValueError('Request outside the accepted native inventory')
    # Bind Local's explicit scientific arm names to the ordinary three-arm identities.
    requirements = contract['arm_requirements']
    expected_roles = {'baseline': 'native', 'treatment': 'world_gaussian', 'body_gaussian': 'body_gaussian'}
    if set(requirements) != set(expected_roles) or any(requirements[role]['name'] != name for role, name in expected_roles.items()):
        raise ValueError('Required scientific roles: baseline=native, treatment=world_gaussian, control=body_gaussian')
    inputs = request['input_refs']+[file_ref(root, request_path), file_ref(root, protocol_path),
        contract['sample_manifest_ref'], contract['labels_or_tests_ref'], environment_ref]+dependency_refs
    if contract.get('selection'): inputs.append(contract['selection']['selected_manifest_ref'])
    unique = {ref['path']: ref for ref in inputs}
    command = [sys.executable, '-m', 'research_math.control_scoring', 'score', '--root', '..',
               '--request', str(request_path), '--output', 'scoring-output', '--device', 'cuda:0', '--gpu-uuid', gpu_uuid]
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', 'diff', 'HEAD', '--binary'], cwd=root, check=True, capture_output=True).stdout
    untracked = subprocess.run(['git', 'ls-files', '--others', '--exclude-standard'], cwd=root, check=True, capture_output=True, text=True).stdout.splitlines()
    # Generated inputs may be untracked; any executable Python source must be committed before acceptance.
    if any(path.endswith('.py') for path in untracked):
        raise ValueError('Commit actual Python sources before Local acceptance; preserve other untracked data')
    plan_dir.mkdir(parents=True, exist_ok=False)
    dirty_path = plan_dir/'dirty.patch'; dirty_path.write_bytes(dirty)
    inputs.append(file_ref(root, dirty_path)); unique = {ref['path']: ref for ref in inputs}
    plan = native.make_plan(root, run_id=run_id, purpose='scientific', evidence_mode='developmental',
        protocol_ref=file_ref(root, protocol_path),
        jobs=[{'trial_id': 'score-three-arms-and-repeat', 'command': command, 'cwd': 'actionmesh',
               'input_refs': list(unique.values()), 'code_refs': request['code_refs'],
               'output_paths': ['actionmesh/scoring-output/record.json', 'actionmesh/scoring-output/request.json',
                                'actionmesh/scoring-output/bundle-manifest.json', 'actionmesh/scoring-output/device-samples.jsonl'],
               'seed': request['inference_seed'], 'group': group, 'arm_role': 'baseline'}],
        provenance={'git_revision': revision, 'git_refs': [file_ref(root, dirty_path)],
                    'model_revision': 'No model loaded; original generator report retained',
                    'data_revision': request['ground_truth_ref']['sha256'],
                    'environment_digest': environment_ref['sha256'], 'environment_refs': [environment_ref]+dependency_refs},
        limits={'max_attempts': 1, 'max_development_trials': 1, 'max_confirmation_trials': 0,
                'max_retries_per_trial': 0, 'wall_time_seconds': wall_seconds,
                'attempt_timeout_seconds': wall_seconds})
    # Include every per-arm raw output and copied source, not just a summary receipt.
    for arm in ARMS:
        for index in range(2):
            for name in ('manifest.json', 'scores.json', 'stdout.log', 'stderr.log'):
                plan['jobs'][0]['output_paths'].append(f'actionmesh/scoring-output/{arm}-pass{index}/{name}')
    from research_math.control_scoring import bundle_refs
    plan['jobs'][0]['output_paths'] += ['actionmesh/scoring-output/bundle/'+r['path'] for r in bundle_refs(request)]
    plan['plan_digest'] = native.plan_digest(plan); native.validate_plan(root, plan)
    native_path = plan_dir/'native.json'; native_path.write_text(json.dumps(plan, indent=2)+'\n')
    outer = harness.make_plan(root, batch_id=run_id,
        tasks=[{'task_id': 'native-baseline-qualification', 'idea_id': 'baseline-qualification', 'depends_on': [],
                'priority': 1, 'plan_ref': file_ref(root, native_path),
                'resources': {'cpu_cores': cpu_cores, 'ram_mib': ram_mib, 'gpu_count': 1,
                              'gpu_peak_mib': None, 'allow_gpu_share': False, 'memory_profile_ref': None,
                              'exclusive_keys': ['native-baseline-qualification']}}],
        limits={'total_wall_seconds': wall_seconds, 'window_seconds': wall_seconds,
                'max_parallel_tasks': 1, 'cpu_cores': cpu_cores, 'ram_mib': ram_mib,
                'max_gpu_task_seconds': wall_seconds},
        gpus={'uuids': [gpu_uuid], 'safety_margin_mib': 1024, 'max_tasks_per_gpu': 1})
    (plan_dir/'harness.json').write_text(json.dumps(outer, indent=2)+'\n')
    return outer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'request', 'protocol', 'environment', 'skill-dir', 'plan-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('run-id', 'group', 'gpu-uuid'): parser.add_argument('--'+name, required=True)
    for name in ('wall-seconds', 'ram-mib', 'cpu-cores'): parser.add_argument('--'+name, type=int, required=True)
    args = parser.parse_args()
    plan = build_plans(args.root, request_path=args.request, protocol_path=args.protocol,
        skill_dir=args.skill_dir, environment_path=args.environment, plan_dir=args.plan_dir, run_id=args.run_id, group=args.group,
        gpu_uuid=args.gpu_uuid, wall_seconds=args.wall_seconds, ram_mib=args.ram_mib, cpu_cores=args.cpu_cores)
    print(json.dumps({'plan': str(args.plan_dir/'harness.json'), 'approved_plan_digest': plan['plan_digest'],
                      'execution_started': False, 'native_contract_qualified': False}))
    return 0


if __name__ == '__main__': raise SystemExit(main())
