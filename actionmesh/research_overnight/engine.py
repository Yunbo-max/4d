"""Serial, recoverable process supervisor with a persistent elapsed-time limit."""
from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from typing import Callable


def utc() -> datetime:
    return datetime.now(timezone.utc)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class Lease(AbstractContextManager):
    """An advisory OS lock: a second runner fails instead of duplicating work."""
    def __init__(self, path: Path):
        self.path = path
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('a+')
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.stream.close()
            raise RuntimeError(f'A runner already holds {self.path}') from None
        self.stream.seek(0)
        self.stream.truncate()
        self.stream.write(json.dumps({'pid': os.getpid(), 'host': socket.gethostname(), 'utc': utc().isoformat()}))
        self.stream.flush()
        return self

    def __exit__(self, *args):
        if self.stream:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            self.stream.close()


class Window:
    def __init__(self, path: Path, state: dict, now: datetime):
        self.path, self.state = path, state
        self.deadline = datetime.fromisoformat(state['deadline_utc'])
        self.watermark = max(now, datetime.fromisoformat(state['watermark_utc']))
        self._remaining_at_open = max(0.0, (self.deadline - self.watermark).total_seconds())
        self._opened_monotonic = time.monotonic()

    @classmethod
    def open(cls, path: Path, hours: float, signature: str, now: datetime | None = None):
        if not math.isfinite(hours) or not 0 < hours <= 8:
            raise ValueError('hours must be finite and in (0,8]')
        now = now or utc()
        if now.tzinfo is None:
            raise ValueError('An aware UTC timestamp is required')
        if path.exists():
            state = json.loads(path.read_text())
            if state['fingerprint'] != signature:
                raise ValueError('Resume fingerprint changed; use a new output directory')
            if state['hours'] != hours:
                raise ValueError('Resume hours changed; the original deadline is binding')
        else:
            state = {'hours': hours, 'fingerprint': signature, 'started_utc': now.isoformat(),
                     'deadline_utc': (now + timedelta(hours=hours)).isoformat(),
                     'watermark_utc': now.isoformat()}
            atomic_json(path, state)
        return cls(path, state, now)

    def remaining(self, now: datetime | None = None) -> float:
        observed = max(now or utc(), self.watermark)
        wall_remaining = (self.deadline - observed).total_seconds()
        monotonic_remaining = self._remaining_at_open - (time.monotonic() - self._opened_monotonic)
        return max(0.0, min(wall_remaining, monotonic_remaining))

    def checkpoint(self, now: datetime | None = None) -> None:
        elapsed = time.monotonic() - self._opened_monotonic
        self.watermark = max(self.watermark, now or utc(),
                             self.deadline - timedelta(seconds=max(0, self._remaining_at_open - elapsed)))
        self.state['watermark_utc'] = self.watermark.isoformat()
        atomic_json(self.path, self.state)


def stop_group(process: subprocess.Popen, grace_seconds: float = 2.0) -> None:
    # The leader can have exited while its descendants still hold files/CUDA.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=grace_seconds)


def process_identity(pid: int) -> dict:
    record = {'pid': pid, 'start_ticks': None, 'host': socket.gethostname(), 'boot_id': None}
    try:
        record['boot_id'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        # Some hosted execution tools expose another PID namespace's /proc.
        # Do not misidentify an unrelated process as this subprocess.
        if int(Path('/proc/self/stat').read_text().split(' ', 1)[0]) == os.getpid():
            stat = (Path('/proc') / str(pid) / 'stat').read_text()
            record['start_ticks'] = stat[stat.rfind(')') + 2:].split()[19]
    except (OSError, ValueError, IndexError):
        pass  # A short-lived child may already have exited.
    return record


def still_running(identity: dict | None) -> bool:
    if not identity:
        return False
    try:
        if identity.get('host') != socket.gethostname():
            raise RuntimeError('Previous worker is on another host; reconcile it before resuming')
        actual = process_identity(identity['pid'])
        if identity.get('start_ticks') is not None:
            return actual == identity
        os.kill(identity['pid'], 0)
        return True  # Without /proc identity, conservatively reject a live PID.
    except (OSError, KeyError, IndexError):
        return False


def execute(argv: list[str], log_path: Path, timeout_seconds: float, *,
            env: dict | None = None, cwd: Path | None = None,
            monitor: Callable | None = None, on_start: Callable | None = None,
            poll_seconds: float = 1.0, heartbeat: Callable | None = None) -> dict:
    if timeout_seconds <= 0:
        return {'status': 'timeout', 'returncode': None, 'elapsed_seconds': 0.0}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    state = {'argv': argv, 'started_utc': utc().isoformat(), 'status': 'running'}
    with log_path.open('w') as stream:
        process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT,
                                   env=env, cwd=cwd, start_new_session=True)
        state['process'] = process_identity(process.pid)
        try:
            if on_start:
                on_start(state['process'])
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= timeout_seconds:
                    state['status'] = 'timeout'
                    stop_group(process)
                    break
                if monitor:
                    monitor()
                if heartbeat:
                    heartbeat()
                time.sleep(min(poll_seconds, max(0, timeout_seconds - elapsed)))
            if state['status'] == 'running':
                state['status'] = 'completed' if process.returncode == 0 else 'failed'
        except (KeyboardInterrupt, SystemExit):
            state['status'] = 'interrupted'
            stop_group(process)
            raise
        except Exception as exc:
            state.update(status='resource_error', error=f'{type(exc).__name__}: {exc}')
            stop_group(process)
        finally:
            stop_group(process)  # Also terminate descendants of an exited leader.
            state.update(returncode=process.returncode, elapsed_seconds=time.monotonic() - started,
                         ended_utc=utc().isoformat())
            atomic_json(log_path.with_suffix('.execution.json'), state)
    return state


def write_receipt(path: Path, signature: str, artifacts: list[Path], **extra) -> None:
    root = path.parent.resolve()
    records = {}
    for artifact in artifacts:
        relative = artifact.resolve().relative_to(root)
        if not artifact.is_file():
            raise ValueError(f'Missing artifact: {artifact}')
        records[str(relative)] = {'sha256': digest(artifact), 'bytes': artifact.stat().st_size}
    if not records:
        raise ValueError('A completed receipt needs artifacts')
    atomic_json(path, {'status': 'complete', 'fingerprint': signature,
                       'artifacts': records, 'verified_utc': utc().isoformat(), **extra})


def valid_receipt(path: Path, signature: str) -> bool:
    try:
        receipt = json.loads(path.read_text())
        if receipt['status'] != 'complete' or receipt['fingerprint'] != signature or not receipt['artifacts']:
            return False
        for relative, record in receipt['artifacts'].items():
            artifact = (path.parent / relative).resolve()
            artifact.relative_to(path.parent.resolve())
            if artifact.stat().st_size != record['bytes'] or digest(artifact) != record['sha256']:
                return False
        return True
    except (OSError, KeyError, ValueError, TypeError):
        return False


@dataclass(frozen=True)
class Unit:
    unit_id: str
    family: str
    argv: list[str]
    estimate_seconds: float
    receipt: Path


def initialize_queue(units: list[Unit], output: Path) -> dict:
    """Freeze the complete inventory before preflight without discarding resume history."""
    state_path = output / 'queue.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if still_running(state.get('active_process')):
            raise RuntimeError('The previous worker is already running; do not relaunch it')
        if state['unit_ids'] != [u.unit_id for u in units]:
            raise ValueError('Frozen unit inventory changed')
    else:
        state = {'unit_ids': [u.unit_id for u in units], 'units': {}, 'status': 'running'}
    state['active_process'] = None
    state['status'] = 'running'
    for unit in units:
        state['units'].setdefault(unit.unit_id, {'family': unit.family, 'status': 'queued', 'attempts': []})
    atomic_json(state_path, state)
    return state


def record_preflight(output: Path, state: dict, result: dict) -> None:
    state.setdefault('preflight_attempts', []).append(result)
    state['status'] = 'ready' if result['status'] == 'completed' else 'preflight_blocked'
    state['active_process'] = None
    atomic_json(output / 'queue.json', state)


def run_queue(units: list[Unit], output: Path, window: Window, *, env: dict,
              monitor: Callable | None = None, reserve_seconds: float = 120) -> dict:
    state_path = output / 'queue.json'
    state = initialize_queue(units, output)
    timings = {}
    for record in state['units'].values():
        if record['status'] == 'complete':
            timings.setdefault(record['family'], []).append(record['elapsed_seconds'])
    last_notice = [0.0]

    def heartbeat():
        window.checkpoint()
        if time.monotonic() - last_notice[0] >= 30:
            print(json.dumps({'utc': utc().isoformat(), 'remaining_seconds': round(window.remaining()),
                              'active_unit': state.get('active_unit')}), flush=True)
            last_notice[0] = time.monotonic()

    try:
        for unit in units:
            record = state['units'].setdefault(unit.unit_id, {'family': unit.family, 'status': 'queued', 'attempts': []})
            if record['status'] == 'complete':
                if not valid_receipt(unit.receipt, window.state['fingerprint']):
                    raise ValueError('Completed receipt changed: ' + unit.unit_id)
                continue
            estimate = max(unit.estimate_seconds, max(timings.get(unit.family, [0])) * 1.35)
            record['admission_estimate_seconds'] = estimate
            if unit.receipt.exists():
                if valid_receipt(unit.receipt, window.state['fingerprint']):
                    # Worker completed before the supervisor's last save.
                    record.update(status='complete', elapsed_seconds=record.get('elapsed_seconds', estimate))
                    continue
                raise ValueError('Corrupt or incompatible worker receipt: ' + unit.unit_id)
            if len(record['attempts']) >= 2:
                record['status'] = 'failed_attempt_limit'
                continue
            if window.remaining() < estimate + reserve_seconds:
                record['status'] = 'pending_budget'
                state['status'] = 'budget_boundary'
                break
            state['active_unit'] = unit.unit_id
            attempt = len(record['attempts']) + 1
            log = output / 'logs' / f'{unit.unit_id}__attempt{attempt}.log'
            record['status'] = 'running'
            record['attempts'].append({'attempt': attempt, 'log': str(log), 'status': 'running'})
            atomic_json(state_path, state)

            def on_start(identity):
                state['active_process'] = identity
                atomic_json(state_path, state)

            result = execute(unit.argv, log, window.remaining() - reserve_seconds,
                             env=env, monitor=monitor, on_start=on_start, heartbeat=heartbeat)
            record['attempts'][-1].update(result)
            state['active_process'] = None
            record['elapsed_seconds'] = result['elapsed_seconds']
            if result['status'] == 'completed' and valid_receipt(unit.receipt, window.state['fingerprint']):
                record['status'] = 'complete'
                timings.setdefault(unit.family, []).append(result['elapsed_seconds'])
            else:
                record['status'] = result['status'] if result['status'] != 'completed' else 'invalid_receipt'
                tail = log.read_text(errors='replace')[-16000:]
                resource_failure = result['status'] in ('timeout', 'resource_error') or any(
                    phrase in tail.lower() for phrase in ('out of memory', 'cuda error', 'native scorer unavailable'))
                if resource_failure:
                    state['status'] = 'resource_stop' if result['status'] != 'timeout' else 'budget_boundary'
                    atomic_json(state_path, state)
                    break
            window.checkpoint()
            atomic_json(state_path, state)
        else:
            state['status'] = 'inventory_finished'
    except (KeyboardInterrupt, SystemExit):
        state['status'] = 'interrupted'
        if state.get('active_unit'):
            current = state['units'][state['active_unit']]
            current['status'] = 'interrupted'
            if current['attempts'] and current['attempts'][-1]['status'] == 'running':
                attempt = current['attempts'][-1]
                try:
                    attempt.update(json.loads(Path(attempt['log']).with_suffix('.execution.json').read_text()))
                except (OSError, ValueError):
                    attempt['status'] = 'interrupted'
        raise
    except Exception:
        state['status'] = 'invalid_resume_or_execution'
        raise
    finally:
        state['active_process'] = None
        state['ended_or_saved_utc'] = utc().isoformat()
        state['remaining_seconds'] = window.remaining()
        state['counts'] = {'declared': len(units), 'complete': sum(r['status'] == 'complete' for r in state['units'].values())}
        window.checkpoint()
        atomic_json(state_path, state)
    return state
