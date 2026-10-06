"""One-off launch of the user's explicitly authorized completion continuation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from research_overnight.engine import Lease, atomic_json, digest, fingerprint, process_identity, still_running, utc

OUTPUT = Path('/root/rivermind-data/actionmesh-repro/outputs/overnight-2080ti-20261005')
PACKAGE_PARENT = Path('/root/rivermind-data/actionmesh-repro/research/overnight-20261005-3fef350/actionmesh')
EXPECTED = '8f6dd5a5db9c6e01bed16570b3e1911180d91d093858a86b8073b450b1ec38ee'


def main():
    with Lease(OUTPUT / 'completion-launch.lock'):
        for name in ('completion-launch.json', 'completion-status.json'):
            if (OUTPUT / name).exists():
                previous = json.loads((OUTPUT / name).read_text())
                raise RuntimeError(f'Existing {name}; reconcile before relaunch. alive={still_running(previous.get("process"))}')
        script = OUTPUT / 'completion.py'
        if digest(script) != EXPECTED:
            raise ValueError('Uploaded wrapper does not match the tested local file')
        protocol = json.loads((OUTPUT / 'protocol.json').read_text())
        window = json.loads((OUTPUT / 'window.json').read_text())
        signature = fingerprint(protocol)
        if signature != window['fingerprint']:
            raise ValueError('Protocol fingerprint changed')
        ids = ['gen-' + uid for uid in protocol['generation']['cohort']['uids']]
        if len(ids) != 16:
            raise ValueError('Expected the frozen 16-asset census')
        authorization = {
            'recorded_utc': utc().isoformat(), 'mode': 'until_inventory_complete',
            'user_instructions': ['我没说停就不停哪怕是过了8小时', '跑完全部吧'],
            'scope': 'same 16 assets, native generation and both official scoring arms; scheduling only',
            'fingerprint': signature, 'unit_ids': ids, 'wrapper_sha256': EXPECTED,
            'supersedes': {'original_total_hours': window['hours'], 'original_deadline_utc': window['deadline_utc']},
            'total_time_limit_seconds': None, 'gpu_workers': 1, 'maximum_attempts_per_asset': 2,
            'exception_handling': 'failed receipts/resource faults remain incomplete and require debugging',
            'scientific_protocol_changed': False, 'candidate_methods_enabled': False,
            'software_tests': {'local_passed':9, 'remote_passed':9, 'skipped':0}}
        path = OUTPUT / 'completion-authorization.json'
        if path.exists():
            raise RuntimeError('Existing authorization record; refusing overwrite')
        atomic_json(path, authorization)
        env = os.environ.copy()
        env.update(PYTHONPATH=str(PACKAGE_PARENT), PYTHONUNBUFFERED='1')
        with (OUTPUT / 'completion-console.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-u', str(script), '--output', str(OUTPUT),
                '--authorization', str(path)], cwd=OUTPUT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        launch = {'process': process_identity(child.pid), 'launched_utc': utc().isoformat(),
                  'script_sha256': EXPECTED, 'output':str(OUTPUT), 'mode':'until_inventory_complete'}
        atomic_json(OUTPUT / 'completion-launch.json', launch)
        for _ in range(100):
            if child.poll() is not None:
                raise RuntimeError('Continuation exited before handshake: ' + (OUTPUT / 'completion-console.log').read_text()[-4000:])
            if (OUTPUT / 'completion-status.json').exists():
                record = json.loads((OUTPUT / 'completion-status.json').read_text())
                if record['process'] != launch['process']:
                    raise RuntimeError('Continuation handshake identity differs')
                print(json.dumps(record, indent=2))
                return
            time.sleep(.1)
        raise RuntimeError('Continuation handshake missing; inspect existing launched PID before retry')


if __name__ == '__main__':
    main()
