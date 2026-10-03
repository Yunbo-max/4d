"""Bounded, resource-aware GPU queue. Existing run directories are never reused."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


DEFAULT_PEAK_MIB = 10209


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def admitted(free_mib, active, candidate_peak, reserve_mib, max_concurrency, initial_free_mib=None):
    """Reserve peaks not yet materialized; free memory already accounts for usage."""
    if len(active) >= max_concurrency:
        return False
    # nvidia-smi's device and process queries are not atomic. This second bound
    # prevents stale per-process attribution from creating imaginary capacity.
    declared = sum(max(j['estimated_peak_mib'], j.get('gpu_memory_mib', 0)) for j in active)
    if initial_free_mib is not None and declared + candidate_peak + reserve_mib > initial_free_mib:
        return False
    unrealized = sum(max(0, j['estimated_peak_mib'] - j.get('gpu_memory_mib', 0))
                     for j in active)
    return free_mib - unrealized - candidate_peak >= reserve_mib


def gpu_snapshot(gpu_index):
    def query(arguments):
        raw = subprocess.check_output(['nvidia-smi', *arguments, '--format=csv,noheader,nounits'],
                                      text=True, timeout=10)
        return list(csv.reader(io.StringIO(raw)))
    rows = query(['--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu'])
    row = next((r for r in rows if int(r[0]) == gpu_index), None)
    if row is None:
        raise RuntimeError('Requested GPU index not present: ' + str(gpu_index))
    snapshot = dict(utc=utc(), gpu_index=gpu_index, uuid=row[1].strip(),
                    total_mib=int(row[2]), used_mib=int(row[3]), free_mib=int(row[4]),
                    utilization_percent=int(row[5]), processes=[])
    for values in query(['--query-compute-apps=gpu_uuid,pid,used_gpu_memory']):
        if len(values) < 3 or values[0].strip() != snapshot['uuid']:
            continue
        memory = values[2].strip()
        snapshot['processes'].append({'pid': int(values[1]),
                                      'memory_mib': int(memory) if memory.isdigit() else None})
    return snapshot


def attribute(snapshot, active):
    """Attribute physical device memory by our explicitly created process groups."""
    groups = {job['pid']: job for job in active}
    for job in active:
        job['gpu_memory_mib'] = 0
    foreign = []
    for process in snapshot['processes']:
        try:
            group = os.getpgid(process['pid'])
        except (ProcessLookupError, PermissionError):
            group = None
        owner = groups.get(group)
        process['owner_job_id'] = owner['id'] if owner else None
        if owner is None:
            foreign.append(process['pid'])
        elif process['memory_mib'] is not None:
            owner['gpu_memory_mib'] += process['memory_mib']
    return foreign


def stop_owned(process, grace_seconds=5):
    """Only signal the new-session process group created for this child."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    # The direct child may exit before its descendants; clean the entire owned group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return process.wait()


def load_jobs(manifest, project_root):
    document = json.loads(manifest.read_text())
    jobs = document.get('jobs')
    if not isinstance(jobs, list) or not jobs:
        raise ValueError('Manifest needs a nonempty jobs array')
    ids, outputs = set(), set()
    result = []
    for source in jobs:
        identifier, argv = source.get('id'), source.get('argv')
        if (not isinstance(identifier, str) or not identifier or
                any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in identifier)):
            raise ValueError('Job id must contain only letters, digits, underscore or dash')
        if identifier in ids:
            raise ValueError('Duplicate job id: ' + identifier)
        if not isinstance(argv, list) or not argv or any(not isinstance(x, str) or not x for x in argv):
            raise ValueError('Job argv must be an explicit nonempty string array')
        if not Path(argv[0]).is_absolute() or not Path(argv[0]).is_file():
            raise ValueError('Freeze argv[0] to an existing absolute executable path')
        if not isinstance(source.get('output'), str) or not source['output']:
            raise ValueError('Every job needs an output path')
        output = Path(source['output'])
        output = (project_root / output).resolve() if not output.is_absolute() else output.resolve()
        if output in outputs:
            raise ValueError('Duplicate job output: ' + str(output))
        if output.exists():
            raise FileExistsError('Existing job output requires reconciliation: ' + str(output))
        peak = int(source.get('estimated_peak_mib', DEFAULT_PEAK_MIB))
        timeout = float(source.get('timeout_seconds', 900))
        if peak <= 0 or not 0 < timeout <= 900:
            raise ValueError('Peak must be positive and job timeout must be in (0, 900] seconds')
        dependencies = source.get('depends_on', [])
        if not isinstance(dependencies, list) or any(not isinstance(d, str) for d in dependencies):
            raise ValueError('depends_on must be an array of job ids')
        ids.add(identifier)
        outputs.add(output)
        result.append(dict(id=identifier, argv=argv, output=str(output),
                           estimated_peak_mib=peak, timeout_seconds=timeout,
                           depends_on=dependencies, status='pending'))
    by_id = {job['id']: job for job in result}
    visited, visiting = set(), set()

    def visit(identifier):
        if identifier not in by_id:
            raise ValueError('Unknown dependency: ' + identifier)
        if identifier in visiting:
            raise ValueError('Dependency cycle involving: ' + identifier)
        if identifier in visited:
            return
        visiting.add(identifier)
        for dependency in by_id[identifier]['depends_on']:
            visit(dependency)
        visiting.remove(identifier)
        visited.add(identifier)
    for identifier in by_id:
        visit(identifier)
    return result


def eligible_jobs(jobs):
    """Propagate failed prerequisites without dropping jobs from the denominator."""
    by_id = {job['id']: job for job in jobs}
    changed = True
    while changed:
        changed = False
        for job in jobs:
            if job['status'] != 'pending':
                continue
            failed = [identifier for identifier in job['depends_on']
                      if by_id[identifier]['status'] not in ('pending', 'starting', 'running', 'completed')]
            if failed:
                job.update(status='dependency_failed', failed_dependencies=failed, finished_utc=utc())
                changed = True
    return [job for job in jobs if job['status'] == 'pending'
            and all(by_id[d]['status'] == 'completed' for d in job['depends_on'])]


def run(args, snapshot_fn=gpu_snapshot, sample_interval=2):
    started = time.monotonic()
    project_root = args.project_root.resolve(strict=True)
    jobs = load_jobs(args.manifest, project_root)
    # mkdir is the inter-process claim: refuse both completed and interrupted runs.
    args.output.mkdir(parents=True, exist_ok=False)
    state_path = args.output / 'queue.json'
    report = dict(status='initializing', started_utc=utc(), queue_pid=os.getpid(),
                  project_root=str(project_root), maximum_seconds=args.max_seconds,
                  max_concurrency=args.max_concurrency, reserve_mib=args.reserve_mib,
                  manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  jobs=jobs, events=[])
    atomic_json(args.output / 'manifest.json', json.loads(args.manifest.read_text()))
    active = []
    stop_reason = None
    received_signal = []
    old_handlers = {}

    def save():
        report['elapsed_seconds'] = time.monotonic() - started
        atomic_json(state_path, report)

    def request_stop(signum, _frame):
        received_signal.append(signum)

    def finish(job, status, code):
        job['status'] = status
        job['exit_code'] = code
        job['finished_utc'] = utc()
        job['elapsed_seconds'] = time.monotonic() - runtime[job['id']]['start']
        runtime[job['id']]['log'].close()
        active.remove(job)
        print('END', job['id'], status, 'exit', code, flush=True)
        save()

    runtime = {}
    save()
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, request_stop)
        with (args.output / 'gpu-samples.jsonl').open('x', buffering=1) as samples:
            snapshot = snapshot_fn(args.gpu_index)
            samples.write(json.dumps(snapshot) + '\n')
            initial_free_mib = snapshot['free_mib']
            report['initial_gpu_free_mib'] = initial_free_mib
            if snapshot['processes']:
                stop_reason = 'external_gpu_busy_initially'
                report['events'].append(dict(utc=utc(), reason=stop_reason,
                                             pids=[p['pid'] for p in snapshot['processes']]))
            elif any(j['estimated_peak_mib'] + args.reserve_mib > snapshot['total_mib'] for j in jobs):
                stop_reason = 'declared_peak_exceeds_device_budget'
            report['status'] = 'running' if not stop_reason else stop_reason
            save()
            while not stop_reason:
                now = time.monotonic()
                if received_signal:
                    stop_reason = 'interrupted_signal_' + str(received_signal[0])
                    break
                if now - started >= args.max_seconds:
                    stop_reason = 'budget_exhausted'
                    break
                for job in list(active):
                    child = runtime[job['id']]['process']
                    code = child.poll()
                    if code is not None:
                        job['output_exists'] = Path(job['output']).exists()
                        status = ('completed' if job['output_exists'] else 'missing_output') if code == 0 else 'failed'
                        finish(job, status, code)
                    elif now >= runtime[job['id']]['deadline']:
                        code = stop_owned(child)
                        finish(job, 'timeout', code)
                pending = eligible_jobs(jobs)
                if not any(j['status'] == 'pending' for j in jobs) and not active:
                    break
                snapshot = snapshot_fn(args.gpu_index)
                foreign = attribute(snapshot, active)
                samples.write(json.dumps(snapshot) + '\n')
                report['last_gpu_sample'] = snapshot
                for job in active:
                    job['observed_gpu_peak_mib'] = max(job.get('observed_gpu_peak_mib', 0), job['gpu_memory_mib'])
                # Admission in a batch uses updated active reservations after every spawn.
                if not foreign:
                    for job in pending:
                        if not admitted(snapshot['free_mib'], active, job['estimated_peak_mib'],
                                        args.reserve_mib, args.max_concurrency, initial_free_mib):
                            continue
                        if Path(job['output']).exists():
                            job.update(status='refused_existing_output', finished_utc=utc(), exit_code=None)
                            save()
                            continue
                        remaining = args.max_seconds - (time.monotonic() - started)
                        if remaining <= 0 or received_signal:
                            break
                        timeout = min(job['timeout_seconds'], remaining)
                        job.update(status='starting', started_utc=utc(), effective_timeout_seconds=timeout,
                                   launch_gpu_free_mib=snapshot['free_mib'], gpu_memory_mib=0)
                        save()  # An interrupted launch is visible even before we have a PID.
                        log = (args.output / (job['id'] + '.log')).open('x')
                        try:
                            env = os.environ.copy()
                            env['CUDA_VISIBLE_DEVICES'] = snapshot['uuid']
                            child = subprocess.Popen(job['argv'], cwd=project_root, env=env,
                                                     stdout=log, stderr=subprocess.STDOUT,
                                                     start_new_session=True)
                        except Exception as exc:
                            log.close()
                            job.update(status='launch_failed', error=repr(exc), exit_code=None, finished_utc=utc())
                            save()
                            continue
                        launch = time.monotonic()
                        runtime[job['id']] = dict(process=child, start=launch, deadline=launch + timeout, log=log)
                        job.update(status='running', pid=child.pid, process_group_id=child.pid)
                        active.append(job)
                        print('START', job['id'], 'pid', child.pid, flush=True)
                        save()
                save()
                now = time.monotonic()
                next_deadline = min([runtime[j['id']]['deadline'] for j in active] + [started + args.max_seconds])
                time.sleep(max(0, min(sample_interval, next_deadline - now)))
    except BaseException as exc:
        stop_reason = 'runner_error'
        report['error'] = repr(exc)
    finally:
        for job in list(active):
            code = stop_owned(runtime[job['id']]['process'])
            finish(job, 'cancelled_' + (stop_reason or 'runner_exit'), code)
        for job in jobs:
            if job['status'] == 'pending':
                job.update(status='not_started', reason=stop_reason or 'runner_exit')
        report['status'] = stop_reason or ('completed' if all(j['status'] == 'completed' for j in jobs)
                                         else 'completed_with_failures')
        report['finished_utc'] = utc()
        save()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    print(json.dumps({'status': report['status'], 'state': str(state_path)}), flush=True)
    return 0 if report['status'] == 'completed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--max-concurrency', type=int, default=2)
    parser.add_argument('--reserve-mib', type=int, default=2048)
    parser.add_argument('--max-seconds', type=float, default=5400)
    parser.add_argument('--gpu-index', type=int, default=0)
    args = parser.parse_args()
    if args.max_concurrency < 1 or args.reserve_mib < 0 or args.max_seconds <= 0:
        parser.error('Concurrency and duration must be positive; reserve must be nonnegative')
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
