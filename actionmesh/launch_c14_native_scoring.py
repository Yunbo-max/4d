"""Launch finalized C14 scoring through the installed harness exactly once.

This is a controller-side authorization gate, not another executor.  It validates
the final consumption and both installed plans, acquires a stable single-owner
lock, records/reuses one exact claim, then delegates to run_harness.run_harness.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys

from prepare_c14_native_scoring import (
    CANDIDATE_ID, canonical_record_digest, validate_final_consumption)
from research_math import c14_native_scoring as scoring

PROFILE_LABEL = 'C14'
CLAIM_KIND = 'c14-launch-claim'
ENV_PREFIX = 'C14'


def claim_launch(root: Path, consumption_path: Path, consumption: dict,
                 *, expected_harness_digest: str):
    root, consumption_path = Path(root).resolve(), Path(consumption_path).resolve()
    directory = consumption_path.parent
    stem = consumption['authorization_ref']['sha256']
    lock_path = directory/(stem + '.launch.lock')
    lock_path.touch(exist_ok=True)
    lock_stream = lock_path.open('r+')
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_stream.close()
        raise RuntimeError(PROFILE_LABEL + ' launch already has a live controller owner')
    claim_path = directory/(stem + '.launch-claim.json')
    identity = {
        'kind': CLAIM_KIND, 'version': 1,
        'consumption_ref': scoring.file_ref(root, consumption_path),
        'authorization_ref': consumption['authorization_ref'],
        'run_id': consumption['run_id'], 'task_id': consumption['task_id'],
        'native_plan_ref': consumption['native_plan_ref'],
        'native_plan_digest': consumption['native_plan_digest'],
        'harness_plan_ref': consumption['harness_plan_ref'],
        'harness_plan_digest': expected_harness_digest,
    }
    value = {**identity, 'claimed_at': datetime.now(timezone.utc).isoformat()}
    value['claim_digest'] = canonical_record_digest(value, 'claim_digest')
    try:
        with claim_path.open('x') as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
    except FileExistsError:
        existing = scoring.read_json(claim_path)
        if (set(existing) != set(value)
                or any(existing.get(key) != item for key, item in identity.items())
                or existing.get('claim_digest') != canonical_record_digest(
                    existing, 'claim_digest')):
            lock_stream.close()
            raise ValueError('Existing ' + PROFILE_LABEL + ' launch claim has different identity')
    return lock_stream, scoring.file_ref(root, claim_path)


def launch(root: Path, consumption_path: Path, *, skill_dir: Path,
           approved_plan_digest: str, stop_after_report: bool = False):
    root = Path(root).resolve()
    scripts = Path(skill_dir).resolve()/'scripts'
    if not (scripts/'run_harness.py').is_file():
        raise ValueError('Complete installed research-autopilot skill required')
    sys.path.insert(0, str(scripts))
    import run_experiments as native
    import run_harness as harness

    consumption, _, harness_plan = validate_final_consumption(
        root, consumption_path,
        expected_harness_digest=approved_plan_digest,
        native_validator=native, harness_validator=harness)
    lock_stream, claim_ref = claim_launch(
        root, consumption_path, consumption,
        expected_harness_digest=approved_plan_digest)
    injected = {
        ENV_PREFIX + '_CONTROLLER_ROOT': str(root),
        ENV_PREFIX + '_LAUNCH_CLAIM_PATH': str((root/claim_ref['path']).resolve()),
        ENV_PREFIX + '_LAUNCH_CLAIM_SHA256': claim_ref['sha256'],
        ENV_PREFIX + '_CONSUMPTION_PATH': str(Path(consumption_path).resolve()),
    }
    previous = {key: os.environ.get(key) for key in injected}
    os.environ.update(injected)
    try:
        result = harness.run_harness(
            root, harness_plan,
            authorizer=lambda scope: (
                scope.get('plan_digest') == approved_plan_digest
                and scope.get('plan') == harness_plan),
            stop_after_report=stop_after_report)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        lock_stream.close()
    return {'candidate_id': CANDIDATE_ID, 'claim_ref': claim_ref,
            'harness_plan_digest': approved_plan_digest,
            'harness_result': result, 'gate_advanced': False,
            'scientific_verdict': 'not_computed'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--consumption', type=Path, required=True)
    parser.add_argument('--skill-dir', type=Path, required=True)
    parser.add_argument('--approved-plan-digest', required=True)
    parser.add_argument('--stop-after-report', action='store_true')
    args = parser.parse_args()
    print(json.dumps(launch(
        args.root, args.consumption, skill_dir=args.skill_dir,
        approved_plan_digest=args.approved_plan_digest,
        stop_after_report=args.stop_after_report), allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
