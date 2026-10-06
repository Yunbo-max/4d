"""Read-only status and artifact verification for the frozen remote census."""
import json
from pathlib import Path
import tarfile

from research_overnight.engine import atomic_json, digest, fingerprint, still_running, utc, valid_receipt
from research_overnight.protocol import verify_records
from research_overnight.resources import gpu_snapshot

output = Path('/root/rivermind-data/actionmesh-repro/outputs/overnight-2080ti-20261005')
protocol = json.loads((output / 'protocol.json').read_text())
window = json.loads((output / 'window.json').read_text())
signature = fingerprint(protocol)
assert signature == window['fingerprint']
verify_records(protocol['runner_sources'])
queue = json.loads((output / 'queue.json').read_text())
guardian = json.loads((output / 'completion-status.json').read_text())
done = []
for uid in protocol['generation']['cohort']['uids']:
    folder = output / 'units' / ('gen-' + uid)
    if (folder / 'receipt.json').exists() and valid_receipt(folder / 'receipt.json', signature):
        done.append({'uid':uid, 'pair':json.loads((folder / 'pair.json').read_text())})
record = {'checked_utc':utc().isoformat(), 'declared_assets':16, 'verified_complete_pairs':len(done),
    'queue_status':queue['status'], 'active_unit':queue.get('active_unit'),
    'active_worker_alive':still_running(queue.get('active_process')),
    'completion_supervisor_alive':still_running(guardian['process']),
    'completion_supervisor_status':guardian['status'], 'fingerprint':signature,
    'frozen_runner_sources_verified':True, 'gpu':gpu_snapshot('0'), 'completed_pairs':done}
atomic_json(output / 'completion-verification.json', record)
names = ['protocol.json','window.json','queue.json','completion-authorization.json',
    'completion-launch.json','completion-status.json','completion-verification.json',
    'completion-software-tests.log']
archive = output / 'completion-checkpoint-20261005.tar.gz'
with tarfile.open(archive, 'w:gz') as tar:
    for name in names:
        tar.add(output / name, arcname=name)
    for item in done:
        folder = output / 'units' / ('gen-' + item['uid'])
        for name in ('pair.json','receipt.json'):
            tar.add(folder / name, arcname=str((folder / name).relative_to(output)))
print(json.dumps({**record,'checkpoint_archive_sha256':digest(archive)},indent=2))
