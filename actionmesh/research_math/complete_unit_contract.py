"""Fixed default-release generation and complete three-arm result checks.

No fallback, scoring threshold, or scientific qualification is implied here.
"""
import math
from pathlib import Path
import sys

from official_actionbench_adapter import OFFICIAL_FILES

ARMS = ('native', 'world_gaussian', 'body_gaussian')
METRICS = ('cd_3d', 'cd_4d', 'cd_motion')


def complete_unit_output_paths(uid: str,
                               prefix: str = 'actionmesh/unit-output') -> list[str]:
    """Return the exhaustive successful-unit output closure for the receipt.

    These are the deterministic files emitted by the frozen generator, simple
    controls and official adapter.  Failed attempts remain failed attempts and
    may retain only a prefix of this inventory; a completed native receipt must
    bind every entry rather than trusting the summary's nested hash list alone.
    """
    if not isinstance(uid, str) or not uid or Path(uid).name != uid or uid in ('.', '..'):
        raise ValueError('Canonical ActionBench UID required')
    root = Path(prefix)
    paths = [
        root/'result.json', root/'device-samples.jsonl', root/'host-samples.jsonl',
        root/'revalidated-unit-manifest.json',
        root/'generation.stdout.log', root/'generation.stderr.log',
        root/'generation.execution.json',
        root/'official-scoring.stdout.log', root/'official-scoring.stderr.log',
        root/'official-scoring.execution.json', root/'official-scores.json',
        root/'final-integrity/revalidated-unit-manifest.json',
    ]
    native = root/'native-generation'
    paths += [native/'deformations_vertices.npy', native/'deformations_faces.npy',
              native/'sequence.npz', native/'report.json', native/'grid_normal.mp4']
    paths += [native/f'mesh_{index:02d}.glb' for index in range(16)]
    controls = root/'controls'
    paths += [controls/'manifest.json', controls/'controls.json']
    for arm in ARMS:
        paths += [controls/arm/'sequence.npz', controls/arm/'report.json']
    paths.append(controls/'body_gaussian/poses.npz')
    official = root/'official-scores.json.official'
    paths += [official/'official-source'/name for name in OFFICIAL_FILES]
    for arm in ARMS:
        case = official/(uid+'-'+arm)
        paths += [case/'execution.json', case/'export-manifest.json',
                  case/'official.csv', case/'official.summary.json',
                  case/'official.backend.json', case/'stdout.log', case/'stderr.log']
        paths += [case/'predictions'/uid/f'mesh_{index:05d}.glb'
                  for index in range(16)]
    rendered = [path.as_posix() for path in paths]
    if len(rendered) != len(set(rendered)):
        raise AssertionError('Complete-unit output inventory contains duplicates')
    return rendered


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
