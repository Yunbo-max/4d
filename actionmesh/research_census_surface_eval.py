"""CPU-only, GT-isolated evaluation of the four frozen three-frame surface arms.

Reuse the EXACT saved anchor matrix; never align or optimize a prediction here.
Frames 8 and 15 define the primary geometry endpoint. Frame 0 is a no-change
control. No CD-M, temporal interpolation, or 16-frame benchmark claim is made.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from research_census_eval import digest, load_arrays, load_official
from research_census_stage_gt import distance
from research_three_ideas import json_write

FRAMES = (0, 8, 15)
ARMS = ('identity', 'closest_surface', 'normal_only', 'arap')
POINTS = 100000


def read(path):
    return json.loads(path.read_text())


def bitwise_equal(first, second):
    return (first.shape == second.shape and first.dtype == second.dtype and
            np.ascontiguousarray(first).tobytes() == np.ascontiguousarray(second).tobytes())


def require_hash(path, expected, inventory):
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError('Missing source SHA-256: ' + str(path))
    actual = digest(path)
    if actual != expected:
        raise ValueError('Source SHA-256 mismatch: ' + str(path))
    inventory[str(path)] = actual
    return actual


def verify_inventory(declared, inventory):
    if not isinstance(declared, dict) or not declared:
        raise ValueError('Missing declared source file inventory')
    for filename, expected in declared.items():
        require_hash(Path(filename), expected, inventory)


def validate_variant(sequence, original_vertices, original_faces, identity=False):
    with np.load(sequence, allow_pickle=False) as saved:
        vertices, faces = saved['vertices'].copy(), saved['faces'].copy()
        frames = saved['frame_indices'].copy()
    expected = original_vertices[list(FRAMES)]
    if vertices.shape != expected.shape or not np.issubdtype(vertices.dtype, np.floating):
        raise ValueError('Expected three [0,8,15] vertex frames with unchanged vertex count')
    if not np.array_equal(frames, np.asarray(FRAMES)):
        raise ValueError('Variant frame indices must be exactly [0,8,15]')
    if not np.isfinite(vertices).all():
        raise ValueError('Nonfinite variant vertices')
    if not bitwise_equal(faces, original_faces):
        raise ValueError('Variant topology differs from native faces')
    if not bitwise_equal(vertices[0], original_vertices[0]):
        raise ValueError('Variant anchor is not bitwise unchanged')
    if identity and not bitwise_equal(vertices, expected):
        raise ValueError('Identity arm differs from untouched native frames')
    return vertices, faces


def quantiles(values):
    values = np.asarray(values)
    if not len(values):
        return None
    return {str(q): float(np.quantile(values, q)) for q in (0, .01, .05, .5, .95, .99, 1)}


def geometry_diagnostics(vertices, reference, faces, anchor_diagonal):
    """Geometric damage against the original SAME-TIME mesh, not anchor pose."""
    if not np.array_equal(vertices.shape, reference.shape) or anchor_diagonal <= 0:
        raise ValueError('Invalid geometry diagnostic reference')
    vertices, reference = vertices.astype(np.float64), reference.astype(np.float64)
    target_triangles, source_triangles = vertices[faces], reference[faces]
    normals = np.cross(target_triangles[:, 1] - target_triangles[:, 0], target_triangles[:, 2] - target_triangles[:, 0])
    original_normals = np.cross(source_triangles[:, 1] - source_triangles[:, 0], source_triangles[:, 2] - source_triangles[:, 0])
    area2, original_area2 = np.linalg.norm(normals, axis=-1), np.linalg.norm(original_normals, axis=-1)
    epsilon_area2 = 1e-12 * anchor_diagonal ** 2
    valid_original, valid_current = original_area2 > epsilon_area2, area2 > epsilon_area2
    comparable = valid_original & valid_current
    cosine = (normals[comparable] * original_normals[comparable]).sum(-1) / (area2[comparable] * original_area2[comparable])
    edges = np.unique(np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1), axis=0)
    original_lengths = np.linalg.norm(reference[edges[:, 0]] - reference[edges[:, 1]], axis=-1)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=-1)
    valid_edges = original_lengths > 1e-12 * anchor_diagonal
    displacement = np.linalg.norm(vertices - reference, axis=-1)
    radius = .02 * anchor_diagonal
    return {
        'reference': 'same-time unmodified Stage-II vertices and fixed faces',
        'anchor_diagonal': anchor_diagonal, 'triangle_area2_threshold': epsilon_area2,
        'reference_degenerate_triangles': int((~valid_original).sum()),
        'degenerate_triangles': int((~valid_current).sum()),
        'new_degenerate_triangles': int((valid_original & ~valid_current).sum()),
        'total_area': float(.5 * area2.sum()), 'reference_total_area': float(.5 * original_area2.sum()),
        'total_area_ratio': float(area2.sum() / original_area2.sum()),
        'triangle_area_ratio_quantiles': quantiles(area2[valid_original] / original_area2[valid_original]),
        'new_area_collapse_below_10_percent': int(np.count_nonzero(area2[valid_original] / original_area2[valid_original] < .1)),
        'normal_comparable_triangles': int(comparable.sum()),
        'normal_orientation_reversals': int(np.count_nonzero(cosine < 0)),
        'normal_cosine_quantiles': quantiles(cosine),
        'unique_edges': int(len(edges)), 'reference_zero_length_edges': int((~valid_edges).sum()),
        'edge_length_ratio_quantiles': quantiles(lengths[valid_edges] / original_lengths[valid_edges]),
        'displacement_quantiles': quantiles(displacement),
        'displacement_over_anchor_diagonal_quantiles': quantiles(displacement / anchor_diagonal),
        'fixed_trust_radius': radius,
        'trust_radius_violations': int(np.count_nonzero(displacement > radius + 1e-7 * anchor_diagonal)),
        'at_trust_radius_fraction': float(np.mean(displacement >= radius - 1e-7 * anchor_diagonal)),
        'topology_unchanged': True,
        'interpretation': 'Fixed topology does not prevent crossings/folds; normal sign and area are diagnostics, not complete self-intersection tests.'}


def provenance(args, inventory):
    generation = read(args.case_dir / 'report.json')
    baseline = read(args.baseline_dir / 'report.json')
    stage = read(args.stage_gt_dir / 'report.json')
    stage_provenance = read(args.stage_gt_dir / 'provenance.json')
    if generation.get('status') != 'completed' or stage.get('status') != 'completed':
        raise ValueError('Native case and saved Stage-GT diagnostic must be completed')
    identity = (generation['uid'], generation['seed'])
    if (baseline.get('uid'), baseline.get('seed')) != identity or (stage.get('uid'), stage.get('seed')) != identity:
        raise ValueError('Native case, baseline, and Stage-GT UID/seed must match')
    for directory in (args.case_dir, args.baseline_dir, args.stage_gt_dir):
        path = directory / 'report.json'
        inventory[str(path)] = digest(path)
    source_case = baseline.get('source_case', {})
    require_hash(args.case_dir / 'sequence.npz', source_case.get('sequence_sha256'), inventory)
    require_hash(args.case_dir / 'report.json', source_case.get('report_sha256'), inventory)
    for filename in ('sequence.npz', 'denoised.npz'):
        require_hash(args.case_dir / filename, generation.get('sha256', {}).get(filename), inventory)
    verify_inventory(baseline.get('source_files_sha256'), inventory)
    verify_inventory(stage_provenance.get('source_files_sha256'), inventory)
    for frame in FRAMES:
        target_hashes = []
        for declared in (baseline['source_files_sha256'], stage_provenance['source_files_sha256']):
            matches = [value for name, value in declared.items()
                       if Path(name).name == 'stageI-raw-mesh.npz' and Path(name).parent.name == f'frame_{frame:02d}']
            if len(matches) != 1:
                raise ValueError('Missing or ambiguous Stage-I target provenance at frame ' + str(frame))
            target_hashes.append(matches[0])
        if target_hashes[0] != target_hashes[1]:
            raise ValueError('Fitting and Stage-GT use different Stage-I surfaces at frame ' + str(frame))
    # The prior diagnostic must attest to this exact native sequence/report, not just a similar UID.
    stage_sources = stage_provenance['source_files_sha256']
    sequence_paths = [Path(p) for p in stage_sources if Path(p).name == 'sequence.npz']
    if len(sequence_paths) != 1:
        raise ValueError('Ambiguous saved Stage-GT native sequence provenance')
    for name in ('sequence.npz', 'report.json'):
        old_path = str(sequence_paths[0].parent / name)
        if stage_sources.get(old_path) != inventory[str(args.case_dir / name)]:
            raise ValueError('Saved alignment belongs to a different native source: ' + name)
    transform_path = args.stage_gt_dir / 'shared-anchor-transform.npy'
    require_hash(transform_path, stage.get('shared_matrix_sha256'), inventory)
    require_hash(transform_path, stage_provenance.get('shared_matrix_sha256'), inventory)
    require_hash(args.stage_gt_dir / 'protocol.json', stage_provenance.get('protocol_sha256'), inventory)
    inventory[str(args.stage_gt_dir / 'provenance.json')] = digest(args.stage_gt_dir / 'provenance.json')
    gt_path = args.gt_dir / identity[0] / 'surfaces.npy'
    gt_hashes = [h for p, h in stage_sources.items() if Path(p).name == 'surfaces.npy']
    if len(gt_hashes) != 1:
        raise ValueError('Ambiguous saved Stage-GT ground truth provenance')
    require_hash(gt_path, gt_hashes[0], inventory)
    vertices, faces, gt = load_arrays(args.case_dir / 'sequence.npz', gt_path)
    if len(vertices) != 16 or generation.get('frames') != 16:
        raise ValueError('Expected original complete 16-frame native source')
    matrix = np.load(transform_path, allow_pickle=False)
    if matrix.shape != (1, 4, 4) or not np.isfinite(matrix).all():
        raise ValueError('Invalid saved anchor transform')
    return generation, baseline, stage, stage_provenance, vertices, faces, gt, matrix


def evaluate(args, state):
    import torch
    import trimesh
    torch.set_num_threads(1)
    inventory = state['source_files_sha256']
    generation, baseline, stage, stage_provenance, original, faces, gt, matrix = provenance(args, inventory)
    state.update(uid=generation['uid'], seed=generation['seed'], baseline_status=baseline.get('status'),
                 baseline_report_sha256=digest(args.baseline_dir / 'report.json'))
    _, official = load_official(args.root / 'repo', cpu_rng_fix=True)
    if official['sha256'] != stage_provenance['official']['sha256']:
        raise ValueError('Official evaluator source differs from saved Stage-GT diagnostic')
    state['official'] = official
    state['packages'] = {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'torch', 'trimesh', 'pytorch3d')}
    from sample_mesh import sample_points
    from chamfer import compute_chamfer_score
    from pytorch3d.transforms import Transform3d
    transform = Transform3d(matrix=torch.from_numpy(matrix), device='cpu')
    diagonal = float(np.linalg.norm(np.ptp(original[0].astype(np.float64), axis=0)))
    if not math.isfinite(diagonal) or diagonal <= 0:
        raise ValueError('Invalid anchor diagonal')
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    adjacency = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                           shape=(original.shape[1], original.shape[1]))
    state['fixed_topology_components'] = {
        'count_including_isolated_vertices': int(connected_components(adjacency, directed=False, return_labels=False)),
        'isolated_unreferenced_vertices': int(original.shape[1] - len(np.unique(faces))),
        'scope': 'Identical graph for every accepted arm; does not detect geometric self-intersections.'}
    stage_frames = {row['frame']: row for row in stage['frames']}
    if set(stage_frames) != set(FRAMES):
        raise ValueError('Stage-GT diagnostic must contain exactly frames [0,8,15]')
    stage_clouds = {}
    for frame in FRAMES:
        path = args.stage_gt_dir / f'frame_{frame:02d}' / 'clouds.npz'
        require_hash(path, stage_frames[frame].get('clouds_sha256'), inventory)
        with np.load(path, allow_pickle=False) as cloud:
            stage_clouds[frame] = {k: cloud[k].copy() for k in ('stageI_raw', 'stageII_raw', 'stageII_aligned', 'ground_truth')}
            if int(cloud['frame']) != frame or int(cloud['surface_seed']) != 44 + frame:
                raise ValueError('Stage-GT cloud frame/sampling mismatch')
        if not np.array_equal(stage_clouds[frame]['ground_truth'], gt[frame]):
            raise ValueError('Cached Stage-GT points differ from the supplied ground truth')
    for arm in ARMS:
        row = state['arms'][arm]
        folder = args.baseline_dir / 'variants' / arm
        try:
            report_path = folder / 'report.json'
            arm_report = read(report_path)
            inventory[str(report_path)] = digest(report_path)
            row.update(source_status=arm_report.get('status'), solver_qualified=arm_report.get('solver_qualified'),
                       source_frame_reports=arm_report.get('frame_reports'), source_error=arm_report.get('error'))
            if baseline.get('variants', {}).get(arm) != arm_report:
                raise ValueError('Top-level and per-arm reports disagree')
            if arm_report.get('status') != 'completed':
                row['status'] = 'not_evaluated_incomplete_arm'
                continue
            if (arm_report.get('uid'), arm_report.get('seed'), arm_report.get('variant')) != (generation['uid'], generation['seed'], arm):
                raise ValueError('Arm UID/seed/variant identity mismatch')
            if not isinstance(arm_report.get('solver_qualified'), bool):
                raise ValueError('Missing explicit solver qualification')
            sequence = folder / 'sequence.npz'
            require_hash(sequence, arm_report.get('sequence_sha256'), inventory)
            vertices, actual_faces = validate_variant(sequence, original, faces, identity=arm == 'identity')
            frame_reports = arm_report.get('frame_reports', [])
            if [r.get('frame') for r in frame_reports] != list(FRAMES):
                raise ValueError('Incomplete or reordered per-frame solver reports')
            if arm_report['solver_qualified'] != all(r.get('qualified') is True for r in frame_reports):
                raise ValueError('Arm solver qualifier contradicts per-frame qualification')
            for index, frame_report in enumerate(frame_reports):
                frame = frame_report['frame']
                frame_artifact = folder / f'frame_{frame:02d}.npz'
                require_hash(frame_artifact, frame_report.get('frame_artifact_sha256'), inventory)
                with np.load(frame_artifact, allow_pickle=False) as saved:
                    if (int(saved['frame']) != frame or not bitwise_equal(saved['vertices'], vertices[index])
                            or not bitwise_equal(saved['faces'], faces)):
                        raise ValueError('Solver frame artifact differs from scored sequence')
            row.update(anchor_bitwise_unchanged=True, faces_bitwise_unchanged=True)
            for i, frame in enumerate(FRAMES):
                frame_dir = args.output / arm / f'frame_{frame:02d}'
                frame_dir.mkdir(parents=True, exist_ok=False)
                if frame == 0 and arm != 'identity' and state['arms']['identity']['status'] == 'evaluated':
                    metric = deepcopy(state['arms']['identity']['frames'][0])
                    metric.update(reused_identity_anchor_metrics=True,
                                  reason='Same bitwise anchor/faces and fixed sampling/transform; no repeated score computation')
                    metric.pop('samples_sha256', None)
                else:
                    mesh = trimesh.Trimesh(vertices[i], actual_faces, process=False)
                    sampled = sample_points(mesh, POINTS, seed=44 + frame)
                    if sampled.device.type != 'cpu':
                        raise ValueError('Surface evaluator must remain CPU-only')
                    aligned = transform.transform_points(sampled[None])[0].numpy()
                    native = sampled.numpy()
                    directional = distance(aligned, gt[frame], 44, 45)
                    scalar = float(compute_chamfer_score(torch.from_numpy(aligned), torch.from_numpy(gt[frame]), n=10000, seed=44))
                    if not math.isclose(directional['sum_unsquared'], scalar, rel_tol=1e-10, abs_tol=1e-10):
                        raise ValueError('Directional CD disagrees with official scalar')
                    metric = {'frame': frame, 'surface_seed': 44 + frame,
                              'aligned_to_gt': directional, 'official_cd_scalar_crosscheck': scalar,
                              'stageI_surface_agreement_raw': distance(native, stage_clouds[frame]['stageI_raw'], 44, 45),
                              'geometry': geometry_diagnostics(vertices[i], original[frame], faces, diagonal)}
                    np.savez_compressed(frame_dir / 'sampled-clouds.npz', native=native, aligned=aligned,
                                        frame=np.asarray(frame), surface_seed=np.asarray(44 + frame))
                    metric['samples_sha256'] = digest(frame_dir / 'sampled-clouds.npz')
                    if arm == 'identity':
                        error = float(np.max(np.abs(aligned - stage_clouds[frame]['stageII_aligned'])))
                        expected = stage_frames[frame]['stageII_aligned_to_gt']['sum_unsquared']
                        metric['identity_vs_saved_stageII_cd_difference'] = scalar - expected
                        metric['identity_vs_saved_stageII_aligned_sample_max_difference'] = error
                        if not np.array_equal(native, stage_clouds[frame]['stageII_raw']) or error > 2e-6 or abs(scalar - expected) > 1e-8:
                            raise ValueError('Untouched identity does not reproduce saved Stage-II diagnostic')
                json_write(frame_dir / 'metrics.json', metric)
                row['frames'].append(metric)
            row['primary_mean_frames_8_15_cd'] = float(np.mean([r['aligned_to_gt']['sum_unsquared'] for r in row['frames'] if r['frame'] in (8, 15)]))
            row['moving_frames_mean_stageI_sampled_surface_cd'] = float(np.mean([
                r['stageI_surface_agreement_raw']['sum_unsquared'] for r in row['frames'] if r['frame'] in (8, 15)]))
            row['status'] = 'evaluated'
        except Exception as exc:
            row.update(status='evaluation_error', error=repr(exc), traceback=traceback.format_exc())
        finally:
            json_write(args.output / 'report.json', state)
    identity = state['arms']['identity']
    if identity['status'] == 'evaluated':
        reference = {r['frame']: r for r in identity['frames']}
        for arm, row in state['arms'].items():
            if row['status'] != 'evaluated':
                continue
            for frame in row['frames']:
                base = reference[frame['frame']]['aligned_to_gt']['sum_unsquared']
                current = frame['aligned_to_gt']['sum_unsquared']
                frame['effect_vs_identity'] = {'cd_delta': current - base,
                    'relative_cd_improvement': (base - current) / base if base else None}
                base_agreement = reference[frame['frame']]['stageI_surface_agreement_raw']['sum_unsquared']
                current_agreement = frame['stageI_surface_agreement_raw']['sum_unsquared']
                frame['effect_vs_identity'].update(
                    stageI_sampled_cd_delta=current_agreement - base_agreement,
                    stageI_sampled_cd_relative_reduction=(base_agreement - current_agreement) / base_agreement if base_agreement else None)
                json_write(args.output / arm / f"frame_{frame['frame']:02d}" / 'metrics.json', frame)
            base = identity['primary_mean_frames_8_15_cd']
            current = row['primary_mean_frames_8_15_cd']
            row['primary_effect_vs_identity'] = {'cd_delta': current - base,
                'relative_cd_improvement': (base - current) / base if base else None}
            row['unchanged_anchor_cd_control'] = row['frames'][0]['aligned_to_gt'] == reference[0]['aligned_to_gt']
    state['counts'] = {'planned_arms': 4, 'planned_frames_per_arm': 3,
                      'evaluated_arms': sum(r['status'] == 'evaluated' for r in state['arms'].values()),
                      'solver_qualified_evaluated_arms': sum(r['status'] == 'evaluated' and r.get('solver_qualified') is True for r in state['arms'].values())}
    # No source is allowed to change while it is being used as evaluation evidence.
    for filename, expected in list(inventory.items()):
        require_hash(Path(filename), expected, {})
    state['source_hashes_unchanged_after_evaluation'] = True
    state['status'] = 'completed' if state['counts']['evaluated_arms'] == 4 else 'completed_with_incomplete_arms'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'case-dir', 'baseline-dir', 'stage-gt-dir', 'gt-dir', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    for key, value in vars(args).items():
        setattr(args, key, value.expanduser().resolve())
    if any(args.output == path or path in args.output.parents for path in (args.case_dir, args.baseline_dir, args.stage_gt_dir, args.gt_dir)):
        parser.error('Evaluation output must be separate from every immutable input directory')
    args.output.mkdir(parents=True, exist_ok=False)
    # No CUDA operation is required by sampling, fixed-transform application or KDTree CD.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    started = time.monotonic()
    state = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
             'source_files_sha256': {}, 'evaluator_sha256': digest(Path(__file__)),
             'helper_sha256': {name: digest(Path(__file__).with_name(name)) for name in ('research_census_stage_gt.py', 'research_census_eval.py')},
             'protocol': {'frames': list(FRAMES), 'primary_frames': [8, 15], 'surface_points': POINTS,
                          'surface_seed': '44 + actual frame', 'directional_query_points': 10000,
                          'directional_query_seeds': [44, 45], 'alignment': 'EXACT saved anchor matrix; no fitting',
                          'device': 'CPU', 'ground_truth_used_only_for_evaluation': True,
                          'material_motion_metric_computed': False, 'interpolation_performed': False,
                          'selected_geometry_diagnostic_not_full_16_frame_benchmark': True,
                          'all_arms_retained': True, 'unqualified_solver_results_not_discarded': True},
             'arms': {arm: {'status': 'pending', 'frames': []} for arm in ARMS}}
    json_write(args.output / 'command.json', {'argv': sys.argv, 'paths': {k: str(v) for k, v in vars(args).items()}})
    try:
        evaluate(args, state)
    except Exception as exc:
        state.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        for row in state['arms'].values():
            if row['status'] == 'pending':
                row.update(status='blocked_input_validation', reason=repr(exc))
    finally:
        state['elapsed_seconds'] = time.monotonic() - started
        json_write(args.output / 'report.json', state)
    print(json.dumps({'status': state['status'], 'output': str(args.output), 'counts': state.get('counts')}), flush=True)
    return int(state['status'] != 'completed')


if __name__ == '__main__':
    raise SystemExit(main())
