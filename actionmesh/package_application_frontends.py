"""Append verified frontend evidence to three already-exported application cases.

Runs on CPU using only the standard library. No inference, network access,
weight modification, or overwriting of previously added evidence directories.
Existing backend SHA256SUMS files are verified before augmentation and then
regenerated. Prompt embeddings are verified but omitted; their hashes remain.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time


CASES = ('text-octopus', 'image-mushroom', 'mesh-panda')
STAGES = ('encode-text', 'video')
FRONT_RECORDS = ('config.json', 'report.json', 'environment.json', 'gpu.json',
                 'validation.json', 'denoising-steps.jsonl') + tuple(
    stage + suffix for stage in STAGES
    for suffix in ('-report.json', '-supervisor.json', '.log', '-gpu.csv'))
ANCHOR_RECORDS = ('anchor.glb', 'foreground.png', 'condition.png', 'input.json',
                  'weights-verified.json', 'command.json', 'config.json',
                  'environment.json', 'stages.json', 'report.json', 'status.json')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_file(path):
    require(path.is_file() and not path.is_symlink(), f'Missing file or symlink: {path}')
    return path


def in_root(root, relative):
    path = root / relative
    require(not Path(relative).is_absolute() and path.resolve().is_relative_to(root),
            f'Path escapes root: {relative}')
    return path


def verify_checksum_file(folder):
    checksum_file = checked_file(folder / 'SHA256SUMS')
    entries = []
    for line in checksum_file.read_text().splitlines():
        expected, name = line.split('  ', 1)
        path = checked_file(in_root(folder, name))
        require(len(expected) == 64 and sha256(path) == expected, f'Backend checksum mismatch: {path}')
        entries.append(name)
    require(entries and len(entries) == len(set(entries)), f'Empty or duplicate checksum list: {folder}')
    return len(entries)


def checksum_tree(folder):
    lines = []
    for path in sorted(folder.rglob('*')):
        require(not path.is_symlink(), f'Unexpected symlink in publication: {path}')
        if path.is_file() and path.name != 'SHA256SUMS':
            lines.append(f'{sha256(path)}  {path.relative_to(folder)}\n')
    return ''.join(lines)


def validate_frontend(folder, smoke=False):
    config, report = read_json(folder / 'config.json'), read_json(folder / 'report.json')
    expected_status = 'runtime_smoke_completed' if smoke else 'technical_checks_passed'
    require(report.get('status') == expected_status, f'Frontend has not passed: {folder}')
    require(bool(config.get('smoke')) == smoke, f'Unexpected smoke configuration: {folder}')
    frames, steps, resolution = (17, 2, 256) if smoke else (33, 50, 512)
    for key, expected in [('frames', frames), ('steps', steps), ('width', resolution), ('height', resolution)]:
        require(config.get(key) == expected, f'Unexpected {key}: {folder}')
    validation = read_json(folder / 'validation.json')
    checks = validation.get('checks', {})
    require(checks and all(value is True for value in checks.values()), f'Frontend validation failed: {folder}')
    require(validation.get('mp4_readback_frames') == frames and
            validation.get('frame_shape') == [frames, resolution, resolution, 3],
            f'Frontend frame count/shape mismatch: {folder}')
    require(report.get('validation') == validation, f'Frontend report/validation mismatch: {folder}')
    for stage in STAGES:
        record = read_json(folder / f'{stage}-report.json')
        require(record.get('status') == 'completed', f'Incomplete {stage}: {folder}')
        require(read_json(folder / f'{stage}-supervisor.json').get('return_code') == 0,
                f'Nonzero {stage} return code: {folder}')
        if stage == 'video':
            result = record.get('result', {})
            require(result.get('denoising_steps_checked') == steps and
                    result.get('all_denoising_latents_finite') is True,
                    f'Unverified video denoising steps: {folder}')
    for name in ('generated.mp4', 'preview.gif', 'prompt-embeddings.safetensors'):
        artifact = report.get('artifacts', {}).get(name, {})
        path = checked_file(folder / name)
        require(artifact.get('bytes') == path.stat().st_size and artifact.get('sha256') == sha256(path),
                f'Frontend artifact checksum mismatch: {path}')
    for name in FRONT_RECORDS:
        checked_file(folder / name)
    expected_files = [f'{i:03d}.png' for i in range(frames)]
    actual_files = sorted(path.name for path in (folder / 'frames').glob('*.png'))
    require(actual_files == expected_files, f'Frontend PNG sequence mismatch: {folder}')
    for name in expected_files:
        checked_file(folder / 'frames' / name)
    return config, report


def plan_package(root, cases_path, publication):
    cases_doc = read_json(cases_path)
    require(set(cases_doc.get('cases', {})) == set(CASES), 'Expected exactly the three defined application cases')
    runs = root / 'outputs/applications-20261002'
    plan, additions, omissions = [], [], {}

    def add(source, relative):
        plan.append((checked_file(source), Path(relative)))

    for case_name in CASES:
        case = cases_doc['cases'][case_name]
        destination = publication / case_name
        require(destination.is_dir(), f'Backend publication is missing: {destination}')
        verify_checksum_file(destination)
        report = read_json(destination / 'report.json')
        require(report.get('status') == 'verified' and report.get('frames') == 16 and
                report.get('animations_in_glb', 0) > 0, f'Backend geometry/animation is unverified: {case_name}')
        require(read_json(destination / 'status.json').get('status') == 'verified' and
                read_json(destination / 'export.json').get('status') == 'verified',
                f'Backend/export is incomplete: {case_name}')
        frontend = runs / (case_name + '-video')
        config, front_report = validate_frontend(frontend)
        require(config.get('prompt') == case['prompt'] and config.get('mode') == case['mode'],
                f'Frontend does not match fixed case prompt/mode: {case_name}')
        for key in ('seed', 'height', 'width', 'frames', 'steps', 'guidance_scale'):
            require(config.get(key) == cases_doc[key], f'Frontend {key} differs from cases: {case_name}')
        video_hash = front_report['artifacts']['generated.mp4']['sha256']
        require(read_json(destination / 'input.json').get('source_sha256') == video_hash,
                f'Backend did not consume this frontend video: {case_name}')
        require(sha256(checked_file(destination / 'source-video.mp4')) == video_hash,
                f'Published backend source video differs: {case_name}')
        additions.append(Path(case_name) / 'frontend')
        for name in FRONT_RECORDS + ('generated.mp4', 'preview.gif'):
            add(frontend / name, Path(case_name) / 'frontend' / name)
        for path in sorted((frontend / 'frames').glob('*.png')):
            add(path, Path(case_name) / 'frontend/frames' / path.name)
        omissions[case_name] = {'prompt-embeddings.safetensors': {
            **front_report['artifacts']['prompt-embeddings.safetensors'],
            'reason': 'Verified before packaging; omitted to keep portable evidence small. Original report retains this hash.'}}
        if case['mode'] == 'image':
            source_image = checked_file(in_root(root, case['image']))
            require(config.get('image_sha256') == sha256(source_image), f'Conditioning image changed: {case_name}')
            add(frontend / 'input.png', Path(case_name) / 'frontend/input.png')
            additions.append(Path(case_name) / 'conditioning-render')
            for name in ('render.json', 'view_0.png', 'view_1.png', 'view_2.png', 'view_3.png'):
                add(source_image.parent / name, Path(case_name) / 'conditioning-render' / name)
            mesh = checked_file(in_root(root, case['mesh']))
            mesh_hash = sha256(mesh)
            require(read_json(destination / 'mesh-input.json').get('sha256') == mesh_hash and
                    sha256(checked_file(destination / 'input-mesh.glb')) == mesh_hash,
                    f'Backend mesh differs from fixed case: {case_name}')
        else:
            require(config.get('image') is None, 'Text case unexpectedly used an image')

    mushroom = cases_doc['cases']['image-mushroom']
    original = checked_file(in_root(root, mushroom['original_image']))
    original_record = original.with_suffix('.json')
    require(read_json(original_record).get('sha256') == sha256(original), 'Original mushroom image provenance mismatch')
    additions.append(Path('image-mushroom/original-input'))
    add(original, 'image-mushroom/original-input/mushroom.png')
    add(original_record, 'image-mushroom/original-input/mushroom.json')
    anchor = in_root(root, mushroom['mesh']).parent
    anchor_report = read_json(anchor / 'report.json')
    require(anchor_report.get('status') == 'verified_static_anchor' and
            read_json(anchor / 'status.json').get('status') == 'verified_static_anchor', 'Static anchor is not verified')
    require(anchor_report.get('source_sha256') == sha256(original) and
            anchor_report.get('anchor_sha256') == sha256(anchor / 'anchor.glb') and
            anchor_report.get('condition_sha256') == sha256(anchor / 'condition.png'), 'Static anchor provenance mismatch')
    additions.append(Path('image-mushroom/static-anchor'))
    for name in ANCHOR_RECORDS:
        add(anchor / name, Path('image-mushroom/static-anchor') / name)
    for source in sorted(anchor.glob('source-image.*')):
        add(source, Path('image-mushroom/static-anchor') / source.name)

    additions.append(Path('common'))
    add(cases_path, 'common/application-cases.json')
    audit_path = root / 'manifests/wan-weights-audit.json'
    ledger_path = root / 'manifests/wan-weights-verified.json'
    manifest_path = root / 'video-weights-manifest.json'
    audit = read_json(audit_path)
    require(audit.get('status') == 'verified' and
            audit.get('download_ledger_sha256') == sha256(ledger_path) and
            audit.get('manifest_sha256') == sha256(manifest_path), 'Wan audit is unverified or provenance changed')
    for path in (audit_path, ledger_path, manifest_path):
        add(path, Path('common/verification') / path.name)
    smoke_dirs = sorted(runs.glob('*-video-smoke'))
    require(len(smoke_dirs) == 2, f'Expected two preserved smoke runs; found {len(smoke_dirs)}')
    for smoke in smoke_dirs:
        validate_frontend(smoke, smoke=True)
        for name in FRONT_RECORDS:
            add(smoke / name, Path('common/smoke') / smoke.name / name)
    for relative in additions:
        require(not (publication / relative).exists(), f'Refusing to overwrite prior evidence: {publication / relative}')
    targets = [relative for _, relative in plan]
    require(len(targets) == len(set(targets)), 'Duplicate package targets')
    return plan, additions, omissions


def package(root, cases_path, publication):
    root, cases_path, publication = root.resolve(), cases_path.resolve(), publication.resolve()
    require(publication.is_dir(), 'Publication directory must already exist after backend exports')
    plan, additions, omissions = plan_package(root, cases_path, publication)
    provenance = {'status': 'verified_frontend_evidence_packaged',
                  'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  'source_root': str(root), 'cases_sha256': sha256(cases_path),
                  'validation_scope': 'Verified existing backend exports, frontend reports, exact source-video/mesh linkage, artifact SHA256s and audit provenance. Does not repeat inference or independently evaluate prompt fidelity.',
                  'omitted_artifacts': omissions, 'copied_files': []}
    # Build and hash all additions before exposing them in the publication tree.
    with tempfile.TemporaryDirectory(prefix='.frontend-package-', dir=publication.parent) as temp:
        staging = Path(temp)
        for source, relative in plan:
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = sha256(source)
            shutil.copy2(source, destination)
            require(sha256(destination) == digest, f'Copy checksum mismatch: {source}')
            provenance['copied_files'].append({'source': str(source), 'destination': str(relative),
                                               'bytes': destination.stat().st_size, 'sha256': digest})
        (staging / 'common/packaging.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + '\n')
        for relative in additions:
            require(not (publication / relative).exists(), f'Destination appeared during packaging: {relative}')
        for relative in additions:
            (staging / relative).rename(publication / relative)
    for folder in [publication / case for case in CASES] + [publication]:
        temporary = folder / '.SHA256SUMS.tmp'
        require(not temporary.exists(), f'Temporary checksum path already exists: {temporary}')
        contents = checksum_tree(folder)
        temporary.write_text(contents)
        temporary.replace(folder / 'SHA256SUMS')
    return {'status': 'verified_frontend_evidence_packaged', 'publication': str(publication),
            'copied_files': len(plan), 'cases': list(CASES)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cases', type=Path, help='Default: ROOT/application-cases.json')
    parser.add_argument('--publication', type=Path, help='Default: ROOT/publication/applications-20261002')
    parser.add_argument('--check-only', action='store_true', help='Validate everything without copying or changing checksums')
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    cases_path = (args.cases or root / 'application-cases.json').expanduser().resolve()
    publication = (args.publication or root / 'publication/applications-20261002').expanduser().resolve()
    if args.check_only:
        plan, _, _ = plan_package(root, cases_path, publication)
        print(json.dumps({'status': 'preflight_passed', 'copy_files': len(plan)}))
    else:
        print(json.dumps(package(root, cases_path, publication)))


if __name__ == '__main__':
    main()
