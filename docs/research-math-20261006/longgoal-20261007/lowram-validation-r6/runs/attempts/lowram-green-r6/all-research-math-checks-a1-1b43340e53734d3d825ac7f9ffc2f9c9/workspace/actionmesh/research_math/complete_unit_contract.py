"""Fixed default-release generation and complete three-arm result checks.

No fallback, scoring threshold, or scientific qualification is implied here.
"""
import math
from pathlib import Path
import sys

ARMS = ('native', 'world_gaussian', 'body_gaussian')
METRICS = ('cd_3d', 'cd_4d', 'cd_motion')


REPAIR_PARENT_RECEIPT = '7b4223a3fdcf4a738c1cec646ecb8f8172ece850a46aa8936aa96d8626cd62f9'


def validate_generation_profile(generation: dict) -> str:
    profile = generation.get('runtime_profile', 'default')
    if profile not in ('default', 'fp16-lowram-v1'):
        raise ValueError('Unknown runtime profile')
    repair = profile == 'fp16-lowram-v1'
    expected = {'seed': 42, 'fast': False, 'low_ram': repair,
                'dtype': 'float16' if repair else 'bfloat16',
                'config': 'actionmesh/configs/actionmesh_lowram.yaml' if repair
                          else 'actionmesh/configs/actionmesh.yaml'}
    if any(generation.get(key) != value for key, value in expected.items()):
        raise ValueError('Runtime configuration differs from explicit profile')
    if repair and generation.get('repair_parent_receipt_sha256') != REPAIR_PARENT_RECEIPT:
        raise ValueError('Explicit repair must bind the retained default OOM receipt')
    return profile


def generation_argv(source: Path, frames: Path, output: Path, *, profile='default') -> list[str]:
    if profile not in ('default', 'fp16-lowram-v1'):
        raise ValueError('Unknown runtime profile')
    command = [sys.executable, '-u', str(source/'inference/video_to_animated_mesh.py'),
            '--input', str(frames), '--output_dir', str(output), '--seed', '42',
            '--dtype', 'float16' if profile == 'fp16-lowram-v1' else 'bfloat16',
            '--stage_0_steps', '100', '--stage_1_steps', '30',
            '--face_decimation', '40000', '--floaters_threshold', '0.02',
            '--guidance_scales', '7.5', '--anchor_idx', '0']
    if profile == 'fp16-lowram-v1': command.append('--low_ram')
    return command


def require_three_scores(report: dict, uid: str) -> dict:
    rows = report.get('cases')
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError('Exactly three retained official score rows required')
    expected = {uid+'-'+arm: arm for arm in ARMS}
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Malformed official score row')
        arm = expected.get(row.get('case_id'))
        if (arm is None or arm in result or row.get('uid') != uid
                or row.get('status') != 'success' or row.get('n_frames') != 16):
            raise ValueError('Missing, duplicated, failed or incomplete official arm')
        values = {}
        for metric in METRICS:
            value = row.get(metric)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError('Invalid official metric: '+metric)
            values[metric] = float(value)
        result[arm] = values
    return result
