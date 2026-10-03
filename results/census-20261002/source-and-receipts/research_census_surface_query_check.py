"""CPU-only qualification for the v2 owned exact triangle-query cache.

This catches changed tie selection, accidental orphan-vertex guidance, stale
geometry, or an optimization that still builds a KDTree for every query batch.
No GT or alignment is read. Use --actual-data with the same case/probe paths.
"""
from __future__ import annotations
import argparse
import importlib
import json
from pathlib import Path
import sys
import time


def paired_query(v2, target, points, check=lambda stage: None):
    import numpy as np
    import trimesh
    cache = target._census_exact_query
    before = cache.statistics().copy()
    start = time.monotonic()
    old = [[], [], []]
    for offset in range(0, len(points), v2.QUERY_BATCH):
        check('qualification_original_query')
        result = trimesh.proximity.closest_point(target, points[offset:offset + v2.QUERY_BATCH])
        for values, piece in zip(old, result):
            values.append(piece)
    old = tuple(np.concatenate(parts) for parts in old)
    original_seconds = time.monotonic() - start
    start = time.monotonic()
    new = v2.closest(target, points, check)
    cached_seconds = time.monotonic() - start
    labels = ('closest_points', 'distances', 'triangle_ids')
    for label, reference, actual in zip(labels, old, new):
        if not np.array_equal(reference, actual):
            raise AssertionError(f'Original/cached {label} differ; no approximation accepted')
    # Candidate order matters for original closest_point's two-face tie branch.
    subset = points[:min(len(points), 32)]
    original_candidates = trimesh.proximity.nearby_faces(target, subset)
    cached_candidates = cache.nearby_faces(target, subset)
    if not all(np.array_equal(a, b) for a, b in zip(original_candidates, cached_candidates)):
        raise AssertionError('Candidate order differs')
    after = cache.statistics()
    expected_calls = (len(points) + v2.QUERY_BATCH - 1) // v2.QUERY_BATCH + 1
    assert after['kdtree_builds'] == before['kdtree_builds'] == 1
    assert after['cached_constructor_calls'] - before['cached_constructor_calls'] == expected_calls
    return {'n_points': len(points), 'bitwise_equal_q_distance_face_ids': True,
            'candidate_order_equal_first_32': True, 'kdtree_builds': after['kdtree_builds'],
            'original_seconds': original_seconds, 'cached_seconds': cached_seconds,
            'scope': 'Same exact geometry query; timing is a CPU sample, not full-run speedup'}


def fixtures(v2):
    import numpy as np
    import trimesh
    # Two parallel surfaces plus duplicated opposite-winding triangles exercise
    # equidistance, face-normal tie resolution, on-surface/edge/vertex queries.
    vertices = np.array([[0.,0.,-1.],[1.,0.,-1.],[0.,1.,-1.],
                         [0.,0.,1.],[1.,0.,1.],[0.,1.,1.],
                         [.2,.2,0.]])  # Last vertex is deliberately unreferenced.
    faces = np.array([[0,1,2],[3,4,5],[2,1,0],[5,4,3]], dtype=np.int64)
    target = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    target._census_exact_query = v2.OwnedExactQuery(target)
    points = np.tile(np.array([[.2,.2,0.],[.2,.2,.4],[0,0,0],[.5,.5,0],
                               [.1,.1,1.],[0,0,1.],[2,2,.3],[-1,-1,-.2]]), (40,1))
    module_nearby = trimesh.proximity.nearby_faces
    module_closest = trimesh.proximity.closest_point
    row = paired_query(v2, target, points)
    result = v2.closest(target, points[:1])
    assert result[1][0] == 1.0, 'Orphan vertex must not be a zero-distance surface'
    assert abs(result[0][0,2]) == 1.0
    assert trimesh.proximity.nearby_faces is module_nearby
    assert trimesh.proximity.closest_point is module_closest
    # Cache ownership must reject a different mesh and replaced arrays, not
    # silently return answers against a stale tree.
    other = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    try:
        target._census_exact_query.nearby_faces(other, points[:1])
    except (ValueError, RuntimeError):
        pass
    else:
        raise AssertionError('Cache accepted a different target')
    target.vertices = np.asarray(target.vertices).copy() + [0,0,.1]
    try:
        target._census_exact_query.closest_point(target, points[:1])
    except (ValueError, RuntimeError):
        pass
    else:
        raise AssertionError('Cache accepted mutated target geometry')
    return {'status':'passed', 'fixtures':row, 'ties_edges_vertices_orphans':True,
            'module_globals_unmodified':True, 'foreign_or_mutated_target_rejected':True}


def actual_data(v2, args, check):
    import numpy as np
    vertices, faces, targets, metadata, sources, uid, seed = v2.validate_sources(args)
    rows = []
    for frame in v2.FRAMES:
        check(f'qualification_frame{frame}')
        count = min(args.points, len(vertices[frame]))
        indices = np.linspace(0, len(vertices[frame])-1, count, dtype=np.int64)
        row = paired_query(v2, targets[frame], vertices[frame,indices].astype(np.float64), check)
        row.update(frame=frame, selection='evenly spaced original vertex indices, no GT',
                   selected_vertex_indices=indices.tolist(), cache=targets[frame]._census_exact_query.provenance())
        rows.append(row)
    return {'status':'passed','uid':uid,'seed':seed,'actual_data':rows,
            'source_files_sha256':sources,'gpu_used':False,'gt_used':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-module',default='research_census_surface_baseline_v2')
    parser.add_argument('--extra-python',type=Path)
    parser.add_argument('--actual-data',action='store_true')
    parser.add_argument('--case-dir',type=Path);parser.add_argument('--probe-dir',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--points',type=int,default=256)
    parser.add_argument('--max-seconds',type=float,default=180.)
    args=parser.parse_args()
    if args.extra_python:sys.path.insert(0,str(args.extra_python.expanduser().resolve()))
    if args.points<1 or not 0<args.max_seconds<float('inf'):parser.error('Invalid point/time budget')
    if args.actual_data and (args.case_dir is None or args.probe_dir is None):parser.error('Actual-data paths required')
    if args.output and args.output.exists():raise FileExistsError(args.output)
    started=time.monotonic()
    def check(stage):
        if time.monotonic()-started>args.max_seconds:raise TimeoutError(f'Qualification timeout at {stage}')
    v2=importlib.import_module(args.baseline_module)
    report={'fixtures':fixtures(v2)}
    if args.actual_data:report['actual']=actual_data(v2,args,check)
    report.update(status='passed',elapsed_seconds=time.monotonic()-started,
                  script_sha256=v2.digest(Path(__file__)),baseline_sha256=v2.digest(Path(v2.__file__)))
    if args.output:v2.write_json(args.output,report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
