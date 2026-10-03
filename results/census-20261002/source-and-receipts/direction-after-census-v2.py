"""One bounded, idempotent, post-census diagnostic batch; no model changes."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import signal
import subprocess
import time

ROOT = Path('/root/rivermind-data/actionmesh-repro')
SOURCE = ROOT / 'research/census-20261002'
OUTPUT = ROOT / 'outputs/census-20261002'
PYTHON = ROOT / 'inference-env/bin/python'
RECEIPT = SOURCE / 'direction-after-census-v2.json'
MANNEQUIN = '000-037_1358c424008a43cbaa35eba5e58551ac__seed42__attempt2'
BAT = '000-043_061697e330d44524bd11f8cf95772e2d__seed42'


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save():
    state['updated_utc'] = utc()
    temp = RECEIPT.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2) + '\n')
    temp.replace(RECEIPT)


def gpu():
    free, used = map(int, subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.free,memory.used',
         '--format=csv,noheader,nounits'], text=True).splitlines()[0].split(','))
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                   '--format=csv,noheader,nounits'], text=True)
    processes = {int(a): int(b) for a, b in (line.split(',') for line in raw.splitlines() if line.strip())}
    return free, used, processes


def job(name, argv, output, gpu_job):
    if output.exists():
        raise FileExistsError('Reconcile existing output instead of duplicating: ' + str(output))
    free, used, processes = gpu()
    if gpu_job and (processes or free < 8192):
        raise RuntimeError('Expected drained GPU before diagnostic admission')
    environment = dict(os.environ, OMP_NUM_THREADS='2' if gpu_job else '1',
                       MKL_NUM_THREADS='2' if gpu_job else '1', OPENBLAS_NUM_THREADS='2' if gpu_job else '1')
    if not gpu_job:
        environment['CUDA_VISIBLE_DEVICES'] = ''
    with (SOURCE / (name + '.log')).open('xb') as log:
        process = subprocess.Popen([str(x) for x in argv], env=environment,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    row = dict(name=name, pid=process.pid, argv=[str(x) for x in argv], status='running',
               started_utc=utc(), deadline_seconds=300, gpu_job=gpu_job,
               observed_process_peak_mib=0, observed_card_peak_mib=used)
    state['jobs'].append(row)
    state['status'] = 'running'
    save()
    started = time.monotonic()
    while process.poll() is None:
        free, used, processes = gpu()
        row['observed_card_peak_mib'] = max(used, row['observed_card_peak_mib'])
        row['observed_process_peak_mib'] = max(processes.get(process.pid, 0), row['observed_process_peak_mib'])
        reason = 'deadline' if time.monotonic() - started >= 300 else None
        if gpu_job and (free < 2048 or processes.get(process.pid, 0) > 6144):
            reason = 'resource_limit'
        if reason:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            row['termination_reason'] = reason
            break
        save()
        time.sleep(1)
    row.update(status='completed' if process.returncode == 0 else 'failed',
               exit_code=process.returncode, elapsed_seconds=time.monotonic()-started, finished_utc=utc())
    save()
    if process.returncode:
        raise RuntimeError('Stop after failed bounded job: ' + name)
    report = json.loads((output / 'report.json').read_text())
    if report['status'] != 'completed':
        raise RuntimeError('Exit zero without completed report: ' + name)


if RECEIPT.exists():
    raise FileExistsError('Existing batch identity; inspect receipt and actual PID')
state = dict(status='waiting_for_census', started_utc=utc(), batch_pid=os.getpid(),
             script_sha256=digest(Path(__file__)), wait_limit_seconds=1200, jobs=[])
try:
    manifest = json.loads((SOURCE / 'direction-after-census-v2-freeze.json').read_text())
    for filename, expected in manifest['source_sha256'].items():
        if digest(SOURCE / filename) != expected:
            raise ValueError('Frozen source mismatch: ' + filename)
    state['freeze_sha256'] = digest(SOURCE / 'direction-after-census-v2-freeze.json')
    save()
    waiting = time.monotonic()
    while True:
        queue = json.loads((OUTPUT / 'queue-main/queue.json').read_text())
        if any(j['status'] in ('failed', 'timeout', 'timed_out') for j in queue['jobs']):
            raise RuntimeError('Existing census has a failed child; inspect before adding GPU work')
        free, used, processes = gpu()
        if queue['status'] != 'running' and not processes:
            if not all(j['status'] == 'completed' for j in queue['jobs']):
                raise RuntimeError('Census ended with unfinished children; inspect before adding GPU work')
            break
        if time.monotonic() - waiting >= 1200:
            raise TimeoutError('Census did not drain within bounded admission wait')
        state['wait_seconds'] = time.monotonic() - waiting
        save()
        time.sleep(5)
    state['census_terminal_status'] = queue['status']
    state['census_queue_sha256'] = digest(OUTPUT / 'queue-main/queue.json')
    job('time-direction-mannequin-resume-v2', [PYTHON, '-u', SOURCE / 'research_census_time_direction_resume.py',
        '--root', ROOT, '--case-dir', OUTPUT / 'cases' / MANNEQUIN,
        '--source-attempt', OUTPUT / 'time-direction/mannequin-seed42-v1',
        '--timeout-receipt', SOURCE / 'time-direction-launch-v1.json',
        '--output', OUTPUT / 'time-direction/mannequin-seed42-v2', '--max-seconds', '300'],
        OUTPUT / 'time-direction/mannequin-seed42-v2', True)
    job('time-direction-bat-v1', [PYTHON, '-u', SOURCE / 'research_census_time_direction.py',
        '--root', ROOT, '--case-dir', OUTPUT / 'cases' / BAT,
        '--output', OUTPUT / 'time-direction/bat-seed42-v1', '--max-seconds', '300'],
        OUTPUT / 'time-direction/bat-seed42-v1', True)
    for label, case, version in [('mannequin', MANNEQUIN, 'v2'), ('bat', BAT, 'v1')]:
        output = OUTPUT / 'time-direction-eval' / (label + '-seed42-' + version)
        job('time-direction-eval-' + label + '-' + version, [PYTHON, '-u', SOURCE / 'research_census_time_direction_eval.py',
            '--root', ROOT, '--case-dir', OUTPUT / 'cases' / case,
            '--direction-dir', OUTPUT / 'time-direction' / (label + '-seed42-' + version),
            '--stage-gt-dir', OUTPUT / 'stage-gt' / (label + '-seed42-v1'),
            '--native-cache-dir', OUTPUT / 'motion-controls' / (label + '-seed42-v1'),
            '--gt-dir', ROOT / 'data/actionbench-census-20261002/data',
            '--output', output, '--max-seconds', '300'], output, False)
    state['status'] = 'completed'
except Exception as exc:
    state.update(status='stopped', error=repr(exc))
finally:
    save()
