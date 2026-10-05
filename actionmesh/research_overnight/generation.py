"""Unmodified ActionMesh plus stationary anchor, both measured by ActionBench."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .engine import atomic_json, digest, valid_receipt, write_receipt
from .protocol import ACTIONMESH, verify_records


def stationary_control(source: Path, output: Path) -> None:
    import numpy as np
    with np.load(source, allow_pickle=False) as saved:
        vertices, faces = saved['vertices'], saved['faces']
        indices = saved['frame_indices']
        if vertices.ndim != 3 or vertices.shape[0] != 16 or vertices.shape[-1] != 3:
            raise ValueError('Require the full native 16-frame sequence')
        if not np.array_equal(indices, np.arange(16)) or not np.isfinite(vertices).all():
            raise ValueError('Native timeline or finite vertices are invalid')
        static = np.repeat(vertices[:1], 16, axis=0)
        np.savez(output, vertices=static, faces=faces, frame_indices=indices,
                 timesteps=np.arange(16, dtype=np.float32))


def phase(unit: Path, name: str, signature: str, run) -> dict:
    receipt = unit / f'{name}.receipt.json'
    phase_signature = signature + ':' + name
    if receipt.exists():
        if not valid_receipt(receipt, phase_signature):
            raise ValueError('Partial phase receipt changed: ' + str(receipt))
        return json.loads(receipt.read_text())['result']
    # Interrupted/error directories are retained; never overwrite an attempt.
    attempt = len(list(unit.glob(name + '__attempt*'))) + 1
    directory = unit / f'{name}__attempt{attempt}'
    result, artifacts = run(directory)
    write_receipt(receipt, phase_signature, artifacts, result=result)
    return result


def evaluation_failure(report_path: Path) -> str:
    """Separate a broken native backend from a retained per-asset data/mesh error."""
    try:
        report = json.loads(report_path.read_text())
    except (OSError, ValueError):
        return 'Native scorer unavailable: no readable evaluator report; inspect score.log'
    if report.get('backend_error'):
        return 'Native scorer unavailable: ' + str(report['backend_error'])
    errors = [str(case.get('error', '')) + '\n' + str(case.get('traceback', ''))
              for case in report.get('cases', []) if case.get('status') == 'error']
    if not errors:
        return 'Native scorer unavailable: nonzero exit without recorded case error; inspect score.log'
    return 'Native asset evaluation failed; case retained: ' + '\n'.join(errors)


def worker(protocol_path: Path, uid: str, unit: Path, signature: str, gpu_index: str) -> None:
    import numpy as np
    protocol = json.loads(protocol_path.read_text())
    manifest = protocol['generation']['cohort']
    if uid not in manifest['uids']:
        raise ValueError('UID absent from frozen native cohort')
    root = Path(protocol['root'])
    data_root = Path(protocol['generation']['data_root'])
    relevant = {k: v for k, v in protocol['generation']['data_records'].items() if uid in Path(k).parts}
    verify_records(relevant)
    verify_records(protocol['runner_sources'])
    verify_records(protocol['native_source']['native_files_sha256'])
    verify_records(protocol['generation']['weight_records'])
    unit.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def generate(directory):
        command = [sys.executable, '-u', str(ACTIONMESH / 'research_census_case.py'),
                   '--root', str(root), '--input', str(data_root / 'data' / uid / 'imgs'),
                   '--uid', uid, '--seed', '42', '--output', str(directory), '--gpu-index', gpu_index]
        with (unit / 'generation-command.log').open('a') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            print((unit / 'generation-command.log').read_text(errors='replace')[-12000:], flush=True)
            raise RuntimeError('Native generation failed; failed attempt retained')
        report_path = directory / 'report.json'
        report = json.loads(report_path.read_text())
        for key, value in {'status': 'completed', 'uid': uid, 'seed': 42, 'frames': 16,
                           'stage0_steps': 100, 'stage1_steps': 30, 'guidance_scale': 7.5}.items():
            if report.get(key) != value:
                raise ValueError('Effective native generation parameter differs: ' + key)
        artifacts = [report_path]
        for name, expected in report['sha256'].items():
            path = directory / name
            if digest(path) != expected:
                raise ValueError('Generation artifact hash differs: ' + name)
            artifacts.append(path)
        return {'directory': str(directory), 'report': report}, artifacts

    native = phase(unit, 'generation', signature, generate)
    native_dir = Path(native['directory'])

    def static(directory):
        directory.mkdir()
        stationary_control(native_dir / 'sequence.npz', directory / 'sequence.npz')
        report = {'status': 'completed', 'uid': uid, 'seed': 42, 'frames': 16,
                  'role': 'stationary anchor comparison, no input motion',
                  'source_sequence_sha256': digest(native_dir / 'sequence.npz')}
        atomic_json(directory / 'report.json', report)
        with np.load(native_dir / 'sequence.npz') as original, np.load(directory / 'sequence.npz') as control:
            if not np.array_equal(control['faces'], original['faces']) or not np.array_equal(control['vertices'][0], original['vertices'][0]):
                raise ValueError('Stationary control changed the common anchor/topology')
        return {'directory': str(directory)}, [directory / 'sequence.npz', directory / 'report.json']

    control = phase(unit, 'stationary', signature, static)

    def score(name, case_directory):
        def scoring(directory):
            directory.mkdir()
            manifest_path = directory / 'manifest.json'
            atomic_json(manifest_path, {'cases': [{'case_id': name, 'uid': uid, 'case_dir': str(case_directory)}]})
            output = directory / 'metrics.json'
            command = [sys.executable, '-u', str(ACTIONMESH / 'research_census_eval.py'),
                       '--case-dir', str(unit), '--manifest', str(manifest_path),
                       '--gt-dir', str(data_root / 'data'), '--repo-root', str(root / 'repo'),
                       '--device', 'cuda', '--seed', '44', '--output', str(output)]
            with (directory / 'score.log').open('w') as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                print((directory / 'score.log').read_text(errors='replace')[-12000:], flush=True)
                raise RuntimeError(evaluation_failure(output))
            report = json.loads(output.read_text())
            cases = report['cases']
            if len(cases) != 1 or cases[0]['status'] != 'success' or cases[0]['uid'] != uid:
                raise ValueError('Native evaluation incomplete or mismatched')
            return {'metrics': {k: cases[0][k] for k in ('cd_3d', 'cd_4d', 'cd_motion')},
                    'native_report': str(output)}, [output, manifest_path, directory / 'score.log']
        return phase(unit, name, signature, scoring)

    native_score = score('native_score', native_dir)
    static_score = score('stationary_score', Path(control['directory']))
    verify_records(relevant)
    verify_records(protocol['runner_sources'])
    verify_records(protocol['native_source']['native_files_sha256'])
    verify_records(protocol['generation']['weight_records'])
    pair = {'uid': uid, 'independent_unit': 'asset', 'scope': 'native baseline/stationary comparison',
            'native': native_score['metrics'], 'stationary': static_score['metrics'],
            'native_minus_stationary': {k: native_score['metrics'][k] - static_score['metrics'][k] for k in native_score['metrics']},
            'verification': {'full_native_scoring': True, 'same_anchor': True, 'input/source_hashes_checked': True,
                             'candidate_method_tested': False, 'formal_gate_pass': False},
            'elapsed_this_attempt_seconds': time.monotonic() - started}
    atomic_json(unit / 'pair.json', pair)
    artifacts = [unit / 'pair.json', *unit.glob('*.receipt.json')]
    # Include raw products, not just checksums-of-checksums, in the final receipt.
    for partial in list(unit.glob('*.receipt.json')):
        partial_data = json.loads(partial.read_text())
        artifacts.extend(unit / name for name in partial_data['artifacts'])
    write_receipt(unit / 'receipt.json', signature, list(dict.fromkeys(artifacts)), uid=uid, family='generation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--uid', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fingerprint', required=True)
    parser.add_argument('--gpu-index', default='0')
    args = parser.parse_args()
    worker(args.protocol.resolve(), args.uid, args.output.resolve(), args.fingerprint, args.gpu_index)

if __name__ == '__main__':
    main()
