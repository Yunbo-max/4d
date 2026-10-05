"""Pinned native assets, prospective cohort selection and code/input provenance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .engine import atomic_json, digest, fingerprint

PACKAGE = Path(__file__).resolve().parent
ACTIONMESH = PACKAGE.parent
ASSETS = PACKAGE / 'assets'
SALT = '4d-native-census-20261005-v1'
INSPECTED_UIDS = (
    '000-037_1358c424008a43cbaa35eba5e58551ac',
    '000-043_061697e330d44524bd11f8cf95772e2d',
    '000-073_1008f82a1e4b4a5ea61e878803b997d0',
    '000-049_f50b8e5f13e2498c907cbbe20bf7cf95',
    '000-078_b0b646eb731848f7837ab8257dfc658c',
    '000-067_09a05166420843fab423047a799c87da',
    '000-046_3bb11febd0624d7f8b30f491fb317d58',
    '000-014_6cf7de7d14064b23be17811d19a63f13',
)


def resource_admission(suite: str) -> dict:
    qa = suite in ('perception', 'both')
    return {'minimum_total_mib': 18000 if qa else 11000,
            'minimum_free_mib': 18000 if qa else 10769,
            'generation_historical_peak_mib': 10257, 'generation_margin_mib': 512,
            'basis': 'single historical native case; first complete unit is still required for capacity calibration',
            'fit_guaranteed': False}


def generation_cohort(population: list[str], count: int) -> list[str]:
    eligible = sorted(set(population) - set(INSPECTED_UIDS))
    if not 1 <= count <= len(eligible):
        raise ValueError('Invalid generation cohort size')
    return sorted(eligible, key=lambda uid: hashlib.sha256(f'{SALT}:{uid}'.encode()).hexdigest())[:count]


def cohort_manifest(count: int = 16) -> dict:
    population = json.loads((ASSETS / 'actionbench_population.json').read_text())
    if len(population['uids']) != 128:
        raise ValueError('Require the complete pinned 128-asset ActionBench population')
    return {
        'schema_version': 1, 'dataset': population['dataset'], 'revision': population['revision'],
        'uids': generation_cohort(population['uids'], count), 'excluded_inspected_uids': list(INSPECTED_UIDS),
        'selection': 'salted sha256 order before outputs; fixed finite development census', 'salt': SALT,
        'source_population_sha256': digest(ASSETS / 'actionbench_population.json'),
        'native_inference': {'frames': 16, 'stage0_steps': 100, 'stageI_steps': 30, 'cfg': 7.5,
                             'seed': 42, 'low_ram': True, 'dtype': 'float16', 'weights_frozen': True},
        'native_evaluation': {'scorer': 'official ActionBench compute_chamfer_3d_4d', 'frames': 16,
                              'sampling_seed': 44, 'surface_samples': 100000, 'icp_samples': 10000,
                              'icp_rotations': 24, 'icp_iterations': 200,
                              'metrics': ['cd_3d', 'cd_4d', 'cd_motion']},
        'scope': 'baseline/evaluator qualification and development census',
        'formal_gate_pass': False, 'new_candidate_methods_enabled': False,
        'pending_gates': ['Natural Gate 0', 'functional collision audit', 'IPCG', 'Gate A'],
    }


def verify_source(root: Path) -> dict:
    expected = json.loads((ASSETS / 'source_manifest.json').read_text())
    records = {}
    for entry in expected['files']:
        path = ACTIONMESH / entry['path'] if entry['path'].startswith('research_') else root / 'repo' / entry['path']
        data = path.read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if blob != entry['git_blob_sha']:
            raise ValueError(f'Pinned native source differs: {path}; reconcile code before execution')
        records[str(path)] = digest(path)
    return {'repository': expected['source_repository'], 'commit': expected['source_commit'],
            'native_files_sha256': records}


def runner_sources() -> dict:
    paths = [*PACKAGE.glob('*.py'), *ASSETS.glob('*')]
    paths += [ACTIONMESH / name for name in ('research_census_case.py', 'research_census_eval.py', 'research_three_ideas.py')]
    return {str(path): digest(path) for path in sorted(paths) if path.is_file()}


def verify_records(records: dict) -> None:
    for name, record in records.items():
        path = Path(name)
        expected = record if isinstance(record, str) else record['sha256']
        if not path.is_file() or digest(path) != expected:
            raise ValueError('Frozen input/source bytes changed: ' + str(path))


def data_records(data_root: Path, manifest: dict) -> dict:
    import numpy as np
    records = {}
    for uid in manifest['uids']:
        directory = data_root / 'data' / uid
        paths = [directory / 'camera.json', directory / 'surfaces.npy']
        paths += [directory / 'imgs' / f'{index:02d}.png' for index in range(16)]
        actual_png = sorted((directory / 'imgs').glob('*.png'))
        if len(actual_png) != 16:
            raise ValueError('Require exactly 16 native frames: ' + uid)
        gt = np.load(directory / 'surfaces.npy', mmap_mode='r', allow_pickle=False)
        if gt.shape != (16, 100000, 6) or not np.issubdtype(gt.dtype, np.floating):
            raise ValueError(f'Native tracked point-cloud shape/dtype differs for {uid}: {gt.shape}/{gt.dtype}')
        for path in paths:
            if not path.is_file() or path.with_name(path.name + '.aria2').exists():
                raise ValueError('Incomplete native data: ' + str(path))
            records[str(path.resolve())] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    return records


def frozen_protocol(prepared: dict, suite: str, root: Path, qa: dict | None) -> dict:
    if prepared:
        verify_records(prepared['weight_records'])
    return {'schema_version': 1, 'suite': suite, 'generation': prepared,
            'root': str(root), 'perception': qa,
            'runner_sources': runner_sources(),
            'native_source': verify_source(root) if suite in ('generation', 'both') else None,
            'resources': {'gpu_count': 1, 'gpu_workers': 1, 'hours_max': 8,
                          'initial_generation_pair_estimate_seconds': 2400,
                          'estimate_update': 'max(initial,1.35*largest_observed_complete_unit)',
                          'saving_reserve_seconds': 120},
            'resource_admission': resource_admission(suite),
            'claims': {'candidate_efficacy': False, 'novelty_pass': False, 'Gate_A_pass': False}}
