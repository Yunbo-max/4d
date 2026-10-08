"""Build a CPU-only plan that validates and safely extracts native context."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def file_ref(root, path):
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return {'path': relative, 'sha256': digest.hexdigest()}


def build_plans(root: Path, *, bundle_root: Path, run_id: str, plan_dir: Path,
                wall_seconds: int, expected_result_sha256: str,
                expected_manifest_sha256: str, expected_archive_sha256: str,
                expected_uid: str, expected_gpu_uuid: str,
                expected_generation_identity_sha256: str,
                expected_source_time_query: bool, max_files: int,
                max_unpacked_bytes: int, max_member_bytes: int,
                max_archive_bytes: int, max_metadata_bytes: int):
    import run_experiments as native
    import run_harness as harness
    from research_math.native_context_delivery import validate_bundle

    root, bundle_root, plan_dir = Path(root).resolve(), Path(bundle_root).resolve(), Path(plan_dir).resolve()
    bundle_root.relative_to(root); plan_dir.relative_to(root)
    if wall_seconds < 1 or wall_seconds > 1800:
        raise ValueError('CPU bundle-consumption wall budget must be 1..1800 seconds')
    result_path, manifest_path, archive_path = (
        bundle_root/'result.json', bundle_root/'raw-manifest.json',
        bundle_root/'raw-evidence.tar')
    validated = validate_bundle(result_path, manifest_path, archive_path,
        expected_result_sha256=expected_result_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_archive_sha256=expected_archive_sha256,
        expected_uid=expected_uid, expected_gpu_uuid=expected_gpu_uuid,
        expected_generation_identity_sha256=expected_generation_identity_sha256,
        expected_source_time_query=expected_source_time_query,
        max_files=max_files, max_unpacked_bytes=max_unpacked_bytes,
        max_member_bytes=max_member_bytes, max_archive_bytes=max_archive_bytes,
        max_metadata_bytes=max_metadata_bytes)
    names = ('result.json', 'raw-manifest.json', 'raw-evidence.tar')
    inputs = [file_ref(root, bundle_root / name) for name in names]
    code = [file_ref(root, root/'actionmesh/research_math'/name)
            for name in ('__init__.py', 'native_context_delivery.py')]
    code.append(file_ref(root, root/'actionmesh/prepare_native_context_consumption.py'))
    outputs = ['actionmesh/context-consumed/raw/' + row['path'] for row in validated['files']]
    outputs += ['actionmesh/context-consumed/' + name for name in
                ('bundle-result.json', 'bundle-manifest.json', 'bundle-consumption.json')]
    command = [sys.executable, '-m', 'research_math.native_context_delivery',
        '--result', str(result_path), '--manifest', str(manifest_path),
        '--archive', str(archive_path), '--output', 'context-consumed',
        '--expected-result-sha256', expected_result_sha256,
        '--expected-manifest-sha256', expected_manifest_sha256,
        '--expected-archive-sha256', expected_archive_sha256,
        '--expected-uid', expected_uid, '--expected-gpu-uuid', expected_gpu_uuid,
        '--expected-generation-identity-sha256', expected_generation_identity_sha256,
        '--expected-source-time-query', str(expected_source_time_query).lower(),
        '--max-files', str(max_files), '--max-unpacked-bytes', str(max_unpacked_bytes),
        '--max-member-bytes', str(max_member_bytes),
        '--max-archive-bytes', str(max_archive_bytes),
        '--max-metadata-bytes', str(max_metadata_bytes)]
    environment_digest = hashlib.sha256(sys.version.encode()).hexdigest()
    git_revision = subprocess.check_output(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    plan = native.make_plan(root, run_id=run_id, purpose='engineering', evidence_mode='developmental',
        jobs=[{'trial_id': 'consume-native-context', 'command': command, 'cwd': 'actionmesh',
               'input_refs': inputs, 'code_refs': code, 'output_paths': outputs,
               'seed': 0, 'group': 'engineering',
               'arm_role': 'paired-context-consumer-no-candidate'}],
        provenance={'git_revision': git_revision,
                    'model_revision': 'none; no model loaded',
                    'data_revision': expected_generation_identity_sha256,
                    'environment_digest': environment_digest},
        limits={'max_attempts': 1, 'max_development_trials': 1, 'max_confirmation_trials': 0,
                'max_retries_per_trial': 0, 'wall_time_seconds': wall_seconds,
                'attempt_timeout_seconds': wall_seconds})
    plan_dir.mkdir(parents=True, exist_ok=False)
    native_path = plan_dir/'native.json'; native_path.write_text(json.dumps(plan, indent=2)+'\n')
    outer = harness.make_plan(root, batch_id=run_id, tasks=[{
        'task_id': 'consume-native-context', 'idea_id': 'native-context-transport',
        'depends_on': [], 'priority': 1, 'plan_ref': file_ref(root, native_path),
        'resources': {'cpu_cores': 1, 'ram_mib': 2048, 'gpu_count': 0,
                      'gpu_peak_mib': None, 'allow_gpu_share': False,
                      'memory_profile_ref': None, 'exclusive_keys': []}}],
        limits={'total_wall_seconds': wall_seconds + 60, 'window_seconds': wall_seconds + 60,
                'max_parallel_tasks': 1, 'cpu_cores': 1, 'ram_mib': 2048,
                'max_gpu_task_seconds': 0})
    (plan_dir/'harness.json').write_text(json.dumps(outer, indent=2)+'\n')
    return plan, outer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'skill-dir', 'bundle-root', 'plan-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--wall-seconds', type=int, default=900)
    parser.add_argument('--expected-result-sha256', required=True)
    parser.add_argument('--expected-manifest-sha256', required=True)
    parser.add_argument('--expected-archive-sha256', required=True)
    parser.add_argument('--expected-uid', required=True)
    parser.add_argument('--expected-gpu-uuid', required=True)
    parser.add_argument('--expected-generation-identity-sha256', required=True)
    parser.add_argument('--expected-source-time-query', choices=('true', 'false'), required=True)
    parser.add_argument('--max-files', type=int, required=True)
    parser.add_argument('--max-unpacked-bytes', type=int, required=True)
    parser.add_argument('--max-member-bytes', type=int, required=True)
    parser.add_argument('--max-archive-bytes', type=int, required=True)
    parser.add_argument('--max-metadata-bytes', type=int, required=True)
    args = parser.parse_args()
    scripts = args.skill_dir.resolve()/'scripts'
    if not (scripts/'run_harness.py').is_file():
        parser.error('Complete installed research-autopilot skill required')
    sys.path.insert(0, str(scripts))
    _, outer = build_plans(args.root, bundle_root=args.bundle_root,
        run_id=args.run_id, plan_dir=args.plan_dir, wall_seconds=args.wall_seconds,
        expected_result_sha256=args.expected_result_sha256,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_archive_sha256=args.expected_archive_sha256,
        expected_uid=args.expected_uid, expected_gpu_uuid=args.expected_gpu_uuid,
        expected_generation_identity_sha256=args.expected_generation_identity_sha256,
        expected_source_time_query=args.expected_source_time_query == 'true',
        max_files=args.max_files, max_unpacked_bytes=args.max_unpacked_bytes,
        max_member_bytes=args.max_member_bytes,
        max_archive_bytes=args.max_archive_bytes,
        max_metadata_bytes=args.max_metadata_bytes)
    print(json.dumps({'plan': str(args.plan_dir.resolve()/'harness.json'),
                      'approved_plan_digest': outer['plan_digest'],
                      'execution_started': False, 'native_context_qualified': False,
                      'candidate_methods_tested': False}))


if __name__ == '__main__': main()
