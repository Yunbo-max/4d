"""Descriptive paired diagnostics, not GT accuracy, for native 4D pilot outputs."""
import argparse
import json
from pathlib import Path
import numpy as np


def load(folder):
    candidates = [folder / 'deformations.npz', folder / 'geometry.npz']
    for p in candidates:
        if p.exists():
            with np.load(p, allow_pickle=False) as d:
                return d['vertices'].astype(np.float64), d['faces']
    return (np.load(folder / 'deformations_vertices.npy', allow_pickle=False).astype(np.float64),
            np.load(folder / 'deformations_faces.npy', allow_pickle=False))


def diagnostics(x, faces, reference):
    if x.shape != reference.shape or not np.isfinite(x).all():
        raise ValueError('Shape mismatch or nonfinite geometry')
    scale = np.linalg.norm(np.ptp(reference[0], axis=0))
    if scale <= 0:
        raise ValueError('Degenerate anchor')
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    lengths = np.linalg.norm(x[:, edges[:, 0]] - x[:, edges[:, 1]], axis=-1)
    valid = lengths[0] > 1e-5 * scale
    strain = np.abs(np.log((lengths[:, valid] + 1e-12) / (lengths[:1, valid] + 1e-12)))
    speed = np.linalg.norm(np.diff(x, axis=0), axis=-1)
    ref_speed = np.linalg.norm(np.diff(reference, axis=0), axis=-1)
    acceleration = np.linalg.norm(np.diff(x, n=2, axis=0), axis=-1)
    displacement = np.linalg.norm(x - x[:1], axis=-1)
    return {
        'frames': int(len(x)), 'vertices': int(x.shape[1]), 'triangles': int(len(faces)),
        'finite': True, 'fixed_reference_scale': float(scale),
        'anchor_max_abs_difference': float(np.max(np.abs(x[0] - reference[0]))),
        'same_index_mean_distance_to_unmodified_over_scale': float(np.linalg.norm(x-reference, axis=-1).mean()/scale),
        'mean_displacement_from_anchor_over_scale': float(displacement.mean()/scale),
        'mean_frame_speed_over_scale': float(speed.mean()/scale),
        'frame_speed_retention_vs_unmodified': float(speed.mean()/max(ref_speed.mean(),1e-12)),
        'mean_frame_acceleration_over_scale': float(acceleration.mean()/scale),
        'edge_absolute_log_strain_p95': float(np.quantile(strain, 0.95)),
        'edge_absolute_log_strain_mean': float(strain.mean()),
        'bbox_diagonal_max_over_anchor': float(np.max(np.linalg.norm(np.ptp(x, axis=1), axis=-1))/scale),
        'frame_centroid_distance_to_unmodified_over_scale': float(np.linalg.norm(x.mean(axis=1)-reference.mean(axis=1),axis=-1).mean()/scale),
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--baseline', default='baseline')
    p.add_argument('--variants', nargs='+', default=['baseline','cfg_cap','cfg_global','moment_energy','rms_energy'])
    args=p.parse_args()
    ref, faces=load(args.root / args.baseline)
    result={'scope':'Descriptive model-frame-step diagnostics on one subject/seed; no GT accuracy, calibration, human preference or cross-model claim.', 'cases':{}}
    for name in args.variants:
        try:
            x,f=load(args.root/name)
            if not np.array_equal(f,faces):
                raise ValueError('Topology differs from paired reference')
            result['cases'][name]={'status':'measured', **diagnostics(x,f,ref)}
        except Exception as exc:
            result['cases'][name]={'status':'missing_or_failed','error':str(exc)}
    (args.root/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__=='__main__':
    main()
