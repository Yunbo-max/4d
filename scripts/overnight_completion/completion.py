"""Finish the already frozen census after its original supervisor releases it.

This operational addendum changes scheduling only. It never edits the frozen
worker package, scientific protocol, original deadline, or completed artifacts.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

from research_overnight.engine import (Lease, atomic_json, digest, execute,
    fingerprint, initialize_queue, process_identity, still_running, utc, valid_receipt)


@contextmanager
def wait_for_runner(path, notify, poll_seconds=15):
    while True:
        lease = Lease(path)
        try:
            lease.__enter__()
        except RuntimeError:
            notify(status='waiting_for_original_runner')
            time.sleep(poll_seconds)
            continue
        break
    try:
        yield lease
    finally:
        lease.__exit__(None, None, None)


def verify_authorization(output, authorization):
    protocol = json.loads((output / 'protocol.json').read_text())
    signature = fingerprint(protocol)
    original = json.loads((output / 'window.json').read_text())
    if signature != original['fingerprint'] or signature != authorization['fingerprint']:
        raise ValueError('Original protocol fingerprint changed')
    if authorization['mode'] != 'until_inventory_complete':
        raise ValueError('Expected explicit completion scheduling override')
    ids = ['gen-' + uid for uid in protocol['generation']['cohort']['uids']]
    if authorization['unit_ids'] != ids:
        raise ValueError('Authorized frozen inventory changed')
    if protocol['suite'] != 'generation' or protocol.get('perception'):
        raise ValueError('This addendum only covers the frozen generation census')
    return protocol, signature


def finish_units(units, output, signature, *, env, monitor=None,
                 notify=lambda **values: None, poll_seconds=1):
    """Run every missing unit with no elapsed-time deadline; preserve failures."""
    state = initialize_queue(units, output)
    state['completion_mode'] = 'until_inventory_complete'
    last_heartbeat = [0.0]

    def save():
        state['counts'] = {'declared': len(units),
            'complete': sum(r['status'] == 'complete' for r in state['units'].values())}
        state['ended_or_saved_utc'] = utc().isoformat()
        state.pop('remaining_seconds', None)
        atomic_json(output / 'queue.json', state)

    def heartbeat():
        if time.monotonic() - last_heartbeat[0] >= 15:
            last_heartbeat[0] = time.monotonic()
            notify(status='running_to_completion', active_unit=state.get('active_unit'),
                   complete=sum(r['status'] == 'complete' for r in state['units'].values()))

    try:
        for unit in units:
            record = state['units'][unit.unit_id]
            if record['status'] == 'complete' or unit.receipt.exists():
                if not valid_receipt(unit.receipt, signature):
                    raise ValueError('Corrupt or incompatible completed receipt: ' + unit.unit_id)
                record.update(status='complete', elapsed_seconds=record.get('elapsed_seconds', 0))
                save()
                continue
            while len(record['attempts']) < 2:
                if monitor:
                    monitor()
                attempt = len(record['attempts']) + 1
                log = output / 'logs' / f'{unit.unit_id}__attempt{attempt}.log'
                if log.exists():
                    raise ValueError('Unreconciled attempt log already exists: ' + str(log))
                state.update(status='running', active_unit=unit.unit_id)
                record['status'] = 'running'
                record['attempts'].append({'attempt': attempt, 'log': str(log), 'status': 'running'})
                save()

                def on_start(identity):
                    state['active_process'] = identity
                    save()

                try:
                    result = execute(unit.argv, log, float('inf'), env=env,
                        monitor=monitor, on_start=on_start, heartbeat=heartbeat,
                        poll_seconds=poll_seconds)
                except BaseException as exc:
                    execution = log.with_suffix('.execution.json')
                    result = json.loads(execution.read_text()) if execution.exists() else {
                        'status': 'interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'launch_error',
                        'error': f'{type(exc).__name__}: {exc}', 'elapsed_seconds': 0}
                    record['attempts'][-1].update(result)
                    record['status'] = result['status']
                    raise
                record['attempts'][-1].update(result)
                state['active_process'] = None
                record['elapsed_seconds'] = result['elapsed_seconds']
                if result['status'] == 'completed' and valid_receipt(unit.receipt, signature):
                    record['status'] = 'complete'
                    save()
                    break
                record['status'] = result['status'] if result['status'] != 'completed' else 'invalid_receipt'
                save()
                if unit.receipt.exists():
                    raise ValueError('Worker produced an invalid receipt: ' + unit.unit_id)
                tail = log.read_text(errors='replace')[-16000:].lower()
                if result['status'] == 'resource_error' or any(t in tail for t in (
                    'out of memory', 'cuda error', 'native scorer unavailable', 'no space left on device')):
                    state['status'] = 'resource_stop'
                    return state
            if record['status'] != 'complete':
                record['status'] = 'failed_attempt_limit'
                save()
        complete = sum(r['status'] == 'complete' for r in state['units'].values())
        state['status'] = 'inventory_complete' if complete == len(units) else 'incomplete_requires_debug'
        return state
    except (KeyboardInterrupt, SystemExit):
        state['status'] = 'interrupted'
        raise
    except Exception:
        state['status'] = 'invalid_resume_or_execution'
        raise
    finally:
        state['active_process'] = None
        state['active_unit'] = None
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--authorization', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    authorization = json.loads(args.authorization.read_text())
    if digest(Path(__file__)) != authorization['wrapper_sha256']:
        raise ValueError('Continuation script changed since authorization was recorded')
    protocol, signature = verify_authorization(output, authorization)
    record = {'process': process_identity(os.getpid()), 'started_utc': utc().isoformat(),
        'fingerprint': signature, 'mode': 'until_inventory_complete',
        'authorization_sha256': digest(args.authorization), 'wrapper_sha256': digest(Path(__file__)),
        'declared_assets': len(authorization['unit_ids']), 'total_time_limit_seconds': None}

    def notify(**values):
        record.update(values, heartbeat_utc=utc().isoformat())
        atomic_json(output / 'completion-status.json', record)
        print(json.dumps(record), flush=True)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    from research_overnight.__main__ import make_units
    from research_overnight.protocol import ACTIONMESH, verify_records
    from research_overnight.reporting import collect
    from research_overnight.resources import GPUMonitor, gpu_snapshot, require_available

    with Lease(output / 'completion.lock'):
        try:
            with wait_for_runner(output / 'runner.lock', notify):
                protocol, signature = verify_authorization(output, authorization)
                previous_queue = json.loads((output / 'queue.json').read_text())
                if still_running(previous_queue.get('active_process')):
                    raise RuntimeError('The previous worker is still running; refusing overlapping preflight')
                gpu = protocol['selected_gpu']
                index = gpu['physical_index']
                units = make_units(output, protocol['generation'], None, signature, index)
                stamp = utc().strftime('%Y%m%dT%H%M%S%fZ')
                backup = output / 'completion-handoffs' / stamp
                backup.mkdir(parents=True)
                for name in ('queue.json','window.json','summary.json','REPORT.md'):
                    if (output / name).exists():
                        shutil.copy2(output / name, backup / name)
                notify(status='checking_handoff', handoff=str(backup))
                verify_records(protocol['runner_sources'])
                verify_records(protocol['native_source']['native_files_sha256'])
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=gpu['uuid'], HF_HUB_OFFLINE='1',
                    TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1', PYTHONUNBUFFERED='1',
                    PYTHONPATH=str(ACTIONMESH))
                if not all(valid_receipt(u.receipt, signature) for u in units):
                    with Lease(Path('/tmp') / ('4d-overnight-' + gpu['uuid'] + '.lock')):
                        snapshot = gpu_snapshot(index)
                        if snapshot['uuid'] != gpu['uuid']:
                            raise ValueError('Frozen physical GPU UUID changed')
                        require_available(snapshot, 'generation')
                        verify_records(protocol['generation']['data_records'])
                        verify_records(protocol['generation']['weight_records'])
                        preflight = execute([sys.executable, '-m', 'research_overnight', 'doctor',
                            '--root', protocol['root'], '--output', str(output), '--gpu-index', index],
                            output / 'logs' / f'completion-preflight-{stamp}.log', 600, env=env,
                            heartbeat=lambda: None)
                        if preflight['status'] != 'completed':
                            raise RuntimeError('Continuation preflight failed; see saved execution log')
                        gpu_monitor = GPUMonitor(index, output / 'gpu-telemetry.jsonl')

                        def monitor():
                            gpu_monitor()
                            if shutil.disk_usage(output).free < 512 * 1024**2:
                                raise RuntimeError('Free output disk below 512 MiB')

                        state = finish_units(units, output, signature, env=env, monitor=monitor, notify=notify)
                else:
                    state = finish_units(units, output, signature, env=env, notify=notify)
                report = collect(output)
                complete = report['generation']['complete_pairs']
                status = 'complete' if complete == len(units) else state['status']
                notify(status=status, complete=complete, finished_utc=utc().isoformat())
                (output / 'COMPLETION.md').write_text(
                    '# Frozen census completion\n\n'
                    'The user authorized finishing all 16 frozen assets even beyond eight hours.\n'
                    'Original protocol.json and window.json are retained as historical records;\n'
                    'the original total time limit is superseded by completion-authorization.json.\n'
                    'Only scheduling changed. All generation, evaluation, seeds and inputs are unchanged.\n\n'
                    f'Validated complete pairs: {complete}/{len(units)}. Status: {status}.\n'
                    'See summary.json for complete metrics and missing/failed assets.\n'
                    'The legacy REPORT.md heading refers to the original scheduler.\n')
                return 0 if status == 'complete' else 2
        except BaseException as exc:
            notify(status='interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'requires_debug',
                   error=f'{type(exc).__name__}: {exc}')
            raise


if __name__ == '__main__':
    raise SystemExit(main())
