"""Read-only 20-minute resource/status journal for an authorized harness run.

This monitor never launches, retries, stops or edits an experiment. It is not an
AI debugger. A human/agent must inspect recorded failures and perform repairs.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if Path(args.run_id).name != args.run_id:
        parser.error('One run identifier required')
    output = args.output.resolve()
    output.relative_to(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    state_path = root / 'runs/harness' / args.run_id / 'state.json'
    deadline = time.monotonic() + 28800
    with output.open('x', buffering=1) as stream:
        while True:
            row = {'observed_at': datetime.now(timezone.utc).isoformat(),
                   'run_id': args.run_id, 'free_disk_bytes': shutil.disk_usage(root).free}
            try:
                state = json.loads(state_path.read_text())
                row['status'] = state['status']
                row['tasks'] = {key: {field: value[field] for field in
                    ('status', 'reason_code', 'started_at', 'completed_at') if field in value}
                    for key, value in state['tasks'].items()}
                row['failed_tasks'] = [key for key, value in row['tasks'].items()
                                       if value['status'] in ('failed', 'unknown', 'blocked')]
            except (OSError, ValueError, KeyError) as error:
                row.update(status='unavailable', error=str(error))
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=uuid,utilization.gpu,memory.used,memory.total',
                    '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
                row['gpu'] = result.stdout.strip()
                row['gpu_query_exit_code'] = result.returncode
                if result.returncode:
                    row['gpu_query_error'] = result.stderr[-2000:]
            except (OSError, subprocess.TimeoutExpired) as error:
                row['gpu_query_error'] = str(error)
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            if row['status'] in ('completed', 'failed', 'budget_exhausted', 'reconciliation_required'):
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(1200, remaining))


if __name__ == '__main__':
    main()
