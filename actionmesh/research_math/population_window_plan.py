"""Emit bounded fixed-order population plans; execute no scientific workload.

The manifests phase composes existing CPU input admission. After receipt-bound
manifest promotion, the gpu phase composes the unchanged complete-unit runner.
Every unit remains an engineering baseline observation, never a candidate result.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from research_math.control_scoring import file_ref
from research_math.complete_unit_contract import validate_generation_profile


def window_indices(total, start, count, unit_seconds):
    if (any(type(v) is not int for v in (total, start, count, unit_seconds)) or
            total < 1 or not 0 <= start < total or count < 1 or unit_seconds < 60):
        raise ValueError('Positive bounded contiguous population window required')
    indices = list(range(start, min(total, start + count)))
    if len(indices) * unit_seconds > 28800 - 1800:
        raise ValueError('Window exceeds eight hours minus collection reserve')
    return indices


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def build(args):
    # Delayed imports keep parser-only software acceptance independent of
    # controller scripts that are not part of a native experiment's code closure.
    from prepare_actionbench_unit_manifest import build_plan as manifest_plan
    from research_math.complete_unit_plan import build_plan as complete_plan
    root = args.root.resolve()
    plan_dir = args.plan_dir.resolve()
    campaign_dir = args.campaign_dir.resolve()
    for path in (plan_dir, campaign_dir):
        path.relative_to(root)
    if plan_dir.exists():
        raise FileExistsError('Preserve existing window plan')
    population = json.loads(args.population.read_text())
    uids = population['uids']
    if len(uids) != 128 or uids != sorted(set(uids)):
        raise ValueError('Exact ordered 128-member released population required')
    indices = window_indices(len(uids), args.start, args.count, args.unit_seconds)
    base = json.loads(args.contract.read_text())
    validate_generation_profile(base['generation'])
    profile_ref = file_ref(root, args.contract)
    scripts = args.skill_dir.resolve() / 'scripts'
    sys.path.insert(0, str(scripts))
    import run_harness as harness
    campaign_path = campaign_dir / 'campaign.json'
    expected = {
        'kind': 'actionbench-population-engineering-window', 'version': 1,
        'population_ref': file_ref(root, args.population), 'uids': uids,
        'indices': indices, 'generation_profile_ref': profile_ref,
        'snapshot_admission_ref': file_ref(root, args.snapshot_admission),
        'dataset_semantics_ref': file_ref(root, args.dataset_semantics),
        'unit_seconds': args.unit_seconds, 'scoring_seed': 44,
        'window_seconds': 28800, 'collection_reserve_seconds': 1800,
        'selection_policy': 'canonical full-population index; no outcome-based exclusions',
        'scientific_effect_qualification': False, 'candidate_methods_tested': False,
    }
    if args.phase == 'manifests':
        if campaign_dir.exists():
            raise FileExistsError('Preserve prior population campaign')
        write_new(campaign_path, expected)
        for index in indices:
            child = copy.deepcopy(base)
            child.update(kind='actionbench-current-release-population-unit-contract',
                         parent_profile_ref=profile_ref,
                         scope='One fixed-index member of current-release full-population engineering baseline')
            child['calibration_unit'] = {
                'uid': uids[index], 'population_index': index,
                'selection_rule': expected['selection_policy'],
                'role': 'full-population engineering baseline; not a candidate experiment',
                'independent_scientific_sample': False,
            }
            child['generation']['input'] = 'data/' + uids[index] + '/imgs containing exactly 00.png through 15.png'
            write_new(campaign_dir / f'{index:03d}' / 'contract.json', child)
    elif json.loads(campaign_path.read_text()) != expected:
        raise ValueError('Frozen population campaign changed')
    tasks = []
    for index in indices:
        unit_dir = campaign_dir / f'{index:03d}'
        subdir = plan_dir / f'{index:03d}'
        run_id = args.run_id + f'-unit-{index:03d}'
        if args.phase == 'manifests':
            manifest_plan(root, skill_dir=args.skill_dir,
                contract_path=unit_dir / 'contract.json', population_path=args.population,
                snapshot_admission_path=args.snapshot_admission,
                dataset_semantics_path=args.dataset_semantics, source_root=args.source_root,
                dataset_root=args.dataset_root, plan_dir=subdir, run_id=run_id,
                wall_seconds=120, ram_mib=2048, cpu_cores=1)
        else:
            manifest_path = unit_dir / 'unit-manifest.json'
            manifest = json.loads(manifest_path.read_text())
            if manifest['calibration_unit']['uid'] != uids[index]:
                raise ValueError('Promoted manifest has another UID')
            complete_plan(SimpleNamespace(
                root=root, plan_dir=subdir, skill_dir=args.skill_dir, run_id=run_id,
                contract=unit_dir / 'contract.json', population=args.population,
                snapshot_contract=args.snapshot_contract,
                snapshot_admission=args.snapshot_admission,
                dataset_semantics=args.dataset_semantics, unit_manifest=manifest_path,
                environment=args.environment, source_root=args.source_root,
                dataset_root=args.dataset_root, weights_root=args.weights_root,
                gpu_uuid=args.gpu_uuid, wall_seconds=args.unit_seconds))
        subplan = json.loads((subdir / 'harness.json').read_text())
        task = subplan['tasks'][0]
        task.update(task_id=f'population-{index:03d}', priority=128-index)
        tasks.append(task)
    gpu = args.phase == 'gpu'
    # Independent units retain their failures and cannot block later population
    # members. No implicit retry or removal from the final denominator occurs.
    limits = {
        'total_wall_seconds': 28800 if gpu else 120 * len(indices) + 300,
        'window_seconds': 3600 if gpu else 120 * len(indices) + 300,
        'max_parallel_tasks': 1, 'cpu_cores': 8 if gpu else 1,
        'ram_mib': 32768 if gpu else 2048,
        'max_gpu_task_seconds': len(indices) * args.unit_seconds if gpu else 0,
    }
    options = {'gpus': {'uuids': [args.gpu_uuid], 'safety_margin_mib': 1024,
                        'max_tasks_per_gpu': 1}} if gpu else {}
    outer = harness.make_plan(root, batch_id=args.run_id, tasks=tasks, limits=limits, **options)
    write_new(plan_dir / 'harness.json', outer)
    return {'plan': str(plan_dir / 'harness.json'), 'approved_plan_digest': outer['plan_digest'],
            'execution_started': False, 'indices': indices, 'phase': args.phase}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('manifests', 'gpu'), required=True)
    for name in ('root', 'plan-dir', 'campaign-dir', 'skill-dir', 'contract', 'population',
                 'snapshot-contract', 'snapshot-admission', 'dataset-semantics',
                 'environment', 'source-root', 'dataset-root', 'weights-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--count', type=int, default=9)
    parser.add_argument('--unit-seconds', type=int, default=2700)
    print(json.dumps(build(parser.parse_args())))


if __name__ == '__main__':
    main()
