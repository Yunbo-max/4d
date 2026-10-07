"""Capture installed Conda/package identities inside an admitted CPU harness task.

Metadata capture only. No imports of torch, CUDA initialization, native scoring,
credential/environment dump, package installation, or scientific qualification.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re
import sys

REQUIRED_PACKAGES = ('numpy', 'torch', 'trimesh', 'scipy', 'pytorch3d')


def package_inventory():
    search_order = {}
    for index, entry in enumerate(sys.path):
        try:
            search_order.setdefault(Path(entry or '.').resolve(), index)
        except OSError:
            continue
    packages = {}
    locations = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get('Name')
        version = distribution.version
        if not isinstance(name, str) or not isinstance(version, str) or not version:
            raise ValueError('Installed distribution has missing name/version metadata')
        name = re.sub(r'[-_.]+', '-', name).lower()
        if not re.fullmatch(r'[a-z0-9][a-z0-9-]*', name) or any(c in version for c in '\n\r'):
            raise ValueError('Invalid installed package identity')
        try:
            location = Path(distribution.locate_file('')).resolve()
        except (AttributeError, OSError, TypeError):
            location = None
        order = search_order.get(location)
        if name not in packages:
            packages[name], locations[name] = version, (location, order)
            continue
        if packages[name] == version:
            continue
        previous_location, previous_order = locations[name]
        # A system-site-packages venv can intentionally shadow a Conda package.
        # Record the version Python resolves first, but reject ambiguous duplicates.
        if (location is None or previous_location is None or order is None or
                previous_order is None or order == previous_order):
            raise ValueError('Conflicting installed package versions: '+name)
        if order < previous_order:
            packages[name], locations[name] = version, (location, order)
    missing = set(REQUIRED_PACKAGES)-packages.keys()
    if missing:
        raise ValueError('Required native dependencies absent: '+', '.join(sorted(missing)))
    return dict(sorted(packages.items()))


def conda_prefix():
    for candidate in (Path(sys.prefix), Path(sys.base_prefix)):
        if (candidate/'conda-meta').is_dir():
            return candidate.resolve()
    raise ValueError('Use the existing Conda interpreter with retained conda-meta records')


def conda_inventory():
    directory = conda_prefix()/'conda-meta'
    records = sorted(directory.glob('*.json'))
    if not records:
        raise ValueError('Use the existing Conda interpreter with retained conda-meta records')
    packages = []
    for path in records:
        record = json.loads(path.read_text())
        entry = {}
        for key in ('name', 'version', 'build', 'subdir'):
            value = record.get(key)
            if not isinstance(value, str) or not value or any(c in value for c in '\n\r'):
                raise ValueError('Incomplete Conda package identity: '+path.name)
            entry[key] = value
        packages.append(entry)
    # URLs, channels, tokens and unrelated metadata never enter the export.
    return sorted(packages, key=lambda p: (p['name'], p['version'], p['build'], p['subdir']))


def capture(root: Path, output: Path, gpu_uuid: str):
    root, output = Path(root).resolve(), Path(output).resolve()
    relative = output.relative_to(root)
    if not gpu_uuid.startswith('GPU-') or any(c in gpu_uuid for c in '\n\r, '):
        raise ValueError('One controller-observed physical GPU UUID required')
    if output.exists(): raise FileExistsError('Preserve the existing runtime capture: '+str(output))
    packages = package_inventory()
    conda_packages = conda_inventory()
    lock = {'kind': 'installed-native-dependency-inventory', 'version': '1.0.0',
            'python_executable': sys.executable, 'python_version': sys.version,
            'python_prefix': sys.prefix, 'conda_prefix': str(conda_prefix()),
            'packages': packages, 'conda_packages': conda_packages,
            'scope': 'Observed package identities; not an upstream artifact checksum or reproducible solver lock'}
    lock_bytes = (json.dumps(lock, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    lock_path = relative/'dependencies.json'
    environment = {'execution_mode': 'native_host', 'python_executable': sys.executable,
        'python_version': sys.version, 'python_prefix': sys.prefix,
        'conda_prefix': str(conda_prefix()), 'gpu_uuid': gpu_uuid,
        'packages': {name: packages[name] for name in REQUIRED_PACKAGES},
        'dependency_lock_refs': [{'path': lock_path.as_posix(),
                                 'sha256': hashlib.sha256(lock_bytes).hexdigest()}],
        'captured_at': datetime.now(timezone.utc).isoformat(),
        'gpu_identity_source': 'Controller-supplied; retain separate current host inventory',
        'gpu_identity_verified': False, 'native_contract_qualified': False,
        'scope': 'Installed environment metadata only; no CUDA, compatibility or scorer parity established'}
    output.mkdir(parents=True, exist_ok=False)
    (output/'dependencies.json').write_bytes(lock_bytes)
    (output/'environment.json').write_text(json.dumps(environment, indent=2, allow_nan=False)+'\n')
    return environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args()
    environment = capture(args.root, args.output, args.gpu_uuid)
    print(json.dumps({'environment': str(args.output/'environment.json'),
                      'dependency_lock_refs': environment['dependency_lock_refs'],
                      'native_contract_qualified': False, 'gpu_identity_verified': False}))
    return 0


if __name__ == '__main__': raise SystemExit(main())
