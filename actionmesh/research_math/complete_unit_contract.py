"""Fixed default-release generation and complete three-arm result checks.

No fallback, scoring threshold, or scientific qualification is implied here.
"""
import math
from pathlib import Path
import sys

ARMS = ('native', 'world_gaussian', 'body_gaussian')
METRICS = ('cd_3d', 'cd_4d', 'cd_motion')


def generation_argv(source: Path, frames: Path, output: Path) -> list[str]:
    return [sys.executable, '-u', str(source/'inference/video_to_animated_mesh.py'),
            '--input', str(frames), '--output_dir', str(output), '--seed', '42',
            '--dtype', 'bfloat16', '--stage_0_steps', '100', '--stage_1_steps', '30',
            '--face_decimation', '40000', '--floaters_threshold', '0.02',
            '--guidance_scales', '7.5', '--anchor_idx', '0']


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
