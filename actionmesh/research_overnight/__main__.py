"""Prepare now; run a serial native-benchmark census for at most eight hours."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys

from .engine import (Lease, Unit, Window, atomic_json, execute, fingerprint,
                     initialize_queue, record_preflight, run_queue, utc)
from .preparation import find_root, prepare_generation, prepare_perception
from .protocol import ACTIONMESH, cohort_manifest, frozen_protocol, resource_admission, verify_records
from .reporting import collect
from .resources import GPUMonitor, doctor, gpu_snapshot, require_available


def plan() -> dict:
    return {'generation': cohort_manifest(),
            'perception': {'status': 'optional; needs prepared published QA/video data and local Qwen2-VL-7B'},
            'resources': {'gpu_count': 1, 'gpu_workers': 1, 'hours_max': 8,
                          'initial_generation_complete_pair_estimate_seconds': 2400,
                          'throughput_basis': 'historical native generation ~650s; new scoring/overhead unmeasured',
                          'saving_reserve_seconds': 120},
            'intended_london_window': ['2026-10-05T23:00:00+01:00', '2026-10-06T07:00:00+01:00'],
            'execution_deadline': 'eight hours from actual launch, persistent on resume',
            'resource_admission': resource_admission('generation'),
            'scientific_gate_pass': False}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('plan', help='Show frozen cohort and scope without CUDA/models')
    for name in ('prepare', 'doctor', 'run', 'report'):
        child = sub.add_parser(name)
        child.add_argument('--root', type=Path, help='Existing ActionMesh asset root; overrides ACTIONMESH_WORKDIR')
        child.add_argument('--output', type=Path, default=ACTIONMESH.parent / 'results/overnight-2080ti-20261005')
        child.add_argument('--suite', choices=('generation', 'perception', 'both'), default='generation')
        child.add_argument('--gpu-index', default='0', help='Physical nvidia-smi GPU index')
        if name == 'prepare':
            child.add_argument('--assets', type=int, default=16)
            child.add_argument('--data-root', type=Path)
            child.add_argument('--offline', action='store_true')
            child.add_argument('--qa-json', type=Path)
            child.add_argument('--qa-videos', type=Path)
            child.add_argument('--qwen-model', type=Path)
            child.add_argument('--questions', type=int, default=48)
        if name == 'run':
            child.add_argument('--hours', type=float, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    invocation_start = utc()
    if args.command == 'plan':
        print(json.dumps(plan(), indent=2))
        return 0
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.command == 'report':
        print(json.dumps(collect(output), indent=2))
        return 0
    root = find_root(args.root) if args.suite in ('generation', 'both') else (args.root or ACTIONMESH).expanduser().resolve()
    if args.command == 'prepare':
        with Lease(output / 'runner.lock'):
            if (output / 'window.json').exists():
                raise ValueError('A window already exists; prepare must not mutate its frozen inputs')
            if args.suite in ('generation', 'both'):
                prepared = prepare_generation(root, output, args.assets, args.data_root, args.offline)
                print(f"Prepared {len(prepared['cohort']['uids'])} native ActionBench assets.", flush=True)
            if args.suite in ('perception', 'both'):
                if not all((args.qa_json, args.qa_videos, args.qwen_model)):
                    raise ValueError('Optional QA needs --qa-json, --qa-videos and --qwen-model; no automatic smaller-model substitution')
                qa = prepare_perception(args.qa_json, args.qa_videos, args.qwen_model, args.questions, output)
                print(f"Prepared {len(qa['question_ids'])} native QA questions.", flush=True)
        print('Data preparation complete. Run doctor before 23:00; preparation has not consumed the execution window.', flush=True)
        return 0
    prepared = json.loads((output / 'prepared-generation.json').read_text()) if args.suite in ('generation', 'both') else None
    qa = json.loads((output / 'prepared-perception.json').read_text()) if args.suite in ('perception', 'both') else None
    if args.command == 'doctor':
        return 0 if doctor(root, args.suite, args.gpu_index, output / 'doctor.json', {'generation': prepared, 'perception': qa})['status'] == 'ready_for_baseline_calibration' else 2
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    old_term = signal.signal(signal.SIGTERM, interrupted)
    try:
        return run(args, root, output, prepared, qa, invocation_start)
    except KeyboardInterrupt:
        return 130
    finally:
        signal.signal(signal.SIGTERM, old_term)


def make_units(output: Path, prepared: dict | None, qa: dict | None, signature: str, gpu_index: str) -> list[Unit]:
    units, generation_units, qa_units = [], [], []
    if prepared:
        for uid in prepared['cohort']['uids']:
            unit_id = 'gen-' + uid
            directory = output / 'units' / unit_id
            argv = [sys.executable, '-u', '-m', 'research_overnight.generation', '--protocol', str(output / 'protocol.json'),
                    '--uid', uid, '--output', str(directory), '--fingerprint', signature, '--gpu-index', gpu_index]
            generation_units.append(Unit(unit_id, 'generation', argv, 2400, directory / 'receipt.json'))
    if qa:
        for qid in qa['question_ids']:
            unit_id = 'qa-' + hashlib.sha256(qid.encode()).hexdigest()[:16]
            directory = output / 'units' / unit_id
            argv = [sys.executable, '-u', '-m', 'research_overnight.perception', '--protocol', str(output / 'protocol.json'),
                    '--question-id', qid, '--output', str(directory), '--fingerprint', signature]
            qa_units.append(Unit(unit_id, 'perception', argv, 180, directory / 'receipt.json'))
    for i in range(max(len(generation_units), len(qa_units))):
        if i < len(generation_units):
            units.append(generation_units[i])
        if i < len(qa_units):
            units.append(qa_units[i])
    return units


def run(args, root: Path, output: Path, prepared: dict | None, qa: dict | None, invocation_start) -> int:
    with Lease(output / 'runner.lock'):
        snapshot = gpu_snapshot(args.gpu_index)
        require_available(snapshot, args.suite)
        with Lease(Path('/tmp') / ('4d-overnight-' + snapshot['uuid'] + '.lock')):
            protocol = frozen_protocol(prepared, args.suite, root, qa)
            protocol['selected_gpu'] = {'uuid': snapshot['uuid'], 'physical_index': snapshot['physical_index'],
                                        'name': snapshot['name'], 'total_mib': snapshot['total_mib']}
            signature = fingerprint(protocol)
            window = Window.open(output / 'window.json', args.hours, signature, now=invocation_start)
            atomic_json(output / 'protocol.json', protocol)
            units = make_units(output, prepared, qa, signature, args.gpu_index)
            state = initialize_queue(units, output)
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=snapshot['uuid'], HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       HF_DATASETS_OFFLINE='1', PYTHONUNBUFFERED='1')
            env['PYTHONPATH'] = str(ACTIONMESH) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
            try:
                if window.remaining() <= 120:
                    run_queue(units, output, window, env=env)
                    print('Original execution deadline reached; no new task started.', flush=True)
                    return 0
                if prepared:
                    verify_records(prepared['data_records'])
                if qa:
                    verify_records(qa['input_records'])
                preflight_command = [sys.executable, '-m', 'research_overnight', 'doctor', '--root', str(root),
                             '--suite', args.suite, '--gpu-index', args.gpu_index, '--output', str(output)]
                log = output / 'logs' / f"preflight__attempt{len(state.get('preflight_attempts', [])) + 1}.log"
                def preflight_started(identity):
                    state['active_process'] = identity
                    atomic_json(output / 'queue.json', state)
                try:
                    check = execute(preflight_command, log, min(600, window.remaining() - 120), env=env,
                                    heartbeat=window.checkpoint, on_start=preflight_started)
                except KeyboardInterrupt:
                    check = json.loads(log.with_suffix('.execution.json').read_text())
                    record_preflight(output, state, {**check, 'log': str(log)})
                    raise
                record_preflight(output, state, {**check, 'log': str(log)})
                if check['status'] != 'completed':
                    print(log.read_text(errors='replace')[-12000:])
                    return 2
                (output / 'preflight.json').write_bytes((output / 'doctor.json').read_bytes())
                state = run_queue(units, output, window, env=env,
                                  monitor=GPUMonitor(args.gpu_index, output / 'gpu-telemetry.jsonl'))
                return 1 if state['status'] == 'resource_stop' else 0
            finally:
                report = collect(output)
                print(json.dumps({'saved_report': str(output / 'REPORT.md'), 'queue_status': report['queue_status'],
                                  'native_pairs_complete': report['generation']['complete_pairs']}), flush=True)

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(2)
