"""CPU-only fixed-topology output fitting, v2 exact-query cache optimization.

No GT, camera, alignment matrix, model weights or GPU is used. Four fixed arms,
three fixed frame indices, unchanged anchor, exact triangle queries, new output.
V1 geometry algorithms, query batches, strengths and solver limits are unchanged.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback

FRAMES = (0, 8, 15)
ARMS = ('identity', 'closest_surface', 'normal_only', 'arap')
QUERY_BATCH = 128
OUTER = 5
LOCAL_GLOBAL = 5
GLOBAL_MAX = 200
GLOBAL_TOL = 1e-7
V1_SOURCE_SHA256 = '6c64929d3a9380bbf3ecf4e130ddd7a4a1418628855eb05e6c5009ec3dd5b4ff'


class OwnedExactQuery:
    """Reuse one immutable target KDTree without changing Trimesh's algorithm.

    Both official functions retain their original bytecode. Each gets an owned
    globals dictionary; only nearby_faces's cKDTree constructor and
    closest_point's nearby_faces binding change. No module monkeypatch occurs.
    All Rtree candidate order, tolerances, distance and face tie logic remain
    installed Trimesh code, including intersection_v and its native fallback.
    """
    def __init__(self, mesh):
        import inspect
        import types
        import numpy as np
        import trimesh
        original_nearby = trimesh.proximity.nearby_faces
        original_closest = trimesh.proximity.closest_point
        if (not isinstance(original_nearby, types.FunctionType) or
                not isinstance(original_closest, types.FunctionType) or
                'cKDTree' not in original_nearby.__code__.co_names or
                'nearby_faces' not in original_closest.__code__.co_names):
            raise RuntimeError('Unsupported Trimesh proximity binding; cache requires qualification')
        self._mesh = mesh
        # Materialize lazy mesh properties before locking geometry arrays.
        _ = mesh.triangles_tree, mesh.triangles, mesh.face_normals
        self._vertices, self._faces = mesh.vertices, mesh.faces
        self._vertices.flags.writeable = False
        self._faces.flags.writeable = False
        referenced = mesh.vertices[mesh.referenced_vertices]
        self._tree = original_nearby.__globals__['cKDTree'](referenced)
        self._tree_data_shape = referenced.shape
        self._tree_data_dtype = referenced.dtype
        self._factory_calls = 0
        self._query_calls = 0
        self._sources = {
            'trimesh_version': trimesh.__version__,
            'nearby_faces_signature': str(inspect.signature(original_nearby)),
            'closest_point_signature': str(inspect.signature(original_closest)),
            'nearby_faces_source_sha256': hashlib.sha256(inspect.getsource(original_nearby).encode()).hexdigest(),
            'closest_point_source_sha256': hashlib.sha256(inspect.getsource(original_closest).encode()).hexdigest(),
            'candidate_search': 'Original nearby_faces bytecode with owned cached cKDTree constructor',
            'narrow_phase_and_tie_resolution': 'Original closest_point bytecode, unchanged',
            'referenced_vertex_order': 'mesh.vertices[mesh.referenced_vertices], original order',
            'module_globals_mutated': False,
            'tol_merge': float(original_nearby.__globals__['tol'].merge),
        }
        def owned_copy(function, bindings):
            namespace = function.__globals__.copy()
            namespace.update(bindings)
            result = types.FunctionType(function.__code__, namespace, function.__name__,
                                        function.__defaults__, function.__closure__)
            result.__kwdefaults__ = function.__kwdefaults__
            return result
        def cached_constructor(data, *args, **kwargs):
            # Native 5.1.0 calls cKDTree(data) without options. Do not silently
            # reuse a differently configured tree if a dependency changes.
            if args or kwargs or data.shape != self._tree_data_shape or data.dtype != self._tree_data_dtype:
                raise RuntimeError('Trimesh KDTree constructor contract changed')
            self._factory_calls += 1
            return self._tree
        self._nearby = owned_copy(original_nearby, {'cKDTree': cached_constructor})
        self._closest = owned_copy(original_closest, {'nearby_faces': self.nearby_faces})

    def _check_owned(self, mesh):
        if (mesh is not self._mesh or mesh.vertices is not self._vertices or
                mesh.faces is not self._faces or self._vertices.flags.writeable or
                self._faces.flags.writeable):
            raise ValueError('Exact-query cache requires its original immutable target geometry')

    def nearby_faces(self, mesh, points):
        self._check_owned(mesh)
        return self._nearby(mesh, points)

    def closest_point(self, mesh, points):
        self._check_owned(mesh)
        self._query_calls += 1
        return self._closest(mesh, points)

    def statistics(self):
        return {'kdtree_builds': 1, 'cached_constructor_calls': self._factory_calls,
                'closest_point_calls': self._query_calls,
                'kdtree_referenced_vertices': self._tree_data_shape[0]}

    def provenance(self):
        return {**self._sources, **self.statistics()}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def check_mesh(v, f):
    import numpy as np
    if v.ndim != 2 or v.shape[1] != 3 or not len(v) or not np.isfinite(v).all():
        raise ValueError('Expected finite nonempty mesh vertices[N,3]')
    if f.ndim != 2 or f.shape[1] != 3 or not len(f) or not np.issubdtype(f.dtype, np.integer):
        raise ValueError('Expected nonempty integer triangle indices[F,3]')
    if f.min() < 0 or f.max() >= len(v):
        raise ValueError('Face indices out of bounds')


def triangle_target(vertices, faces):
    """Reindex used vertices only; preserve every triangle/position and face order."""
    import numpy as np
    import trimesh
    check_mesh(vertices, faces)
    used, remapped = np.unique(faces, return_inverse=True)
    target = trimesh.Trimesh(vertices=vertices[used].copy(), faces=remapped.reshape(faces.shape), process=False)
    if not np.isfinite(target.area_faces).all() or target.area <= 0:
        raise ValueError('Target surface area must be positive and finite')
    _ = target.triangles_tree  # Exact-query dependency checked before optimization.
    target._census_exact_query = OwnedExactQuery(target)
    return target, {'raw_vertices': len(vertices), 'referenced_vertices': len(used),
                    'faces': len(faces), 'geometry_cleanup': False, 'only_unused_indices_removed': True,
                    'exact_query_cache': target._census_exact_query.provenance()}


def closest(target, points, check=lambda stage: None):
    import numpy as np
    qs, ds, ids = [], [], []
    for start in range(0, len(points), QUERY_BATCH):
        check('closest_triangle_batch')
        q, d, f = target._census_exact_query.closest_point(target, points[start:start + QUERY_BATCH])
        if not np.isfinite(q).all() or not np.isfinite(d).all() or (f < 0).any():
            raise ValueError('Exact triangle query returned invalid values')
        qs.append(q); ds.append(d); ids.append(f)
    return np.concatenate(qs), np.concatenate(ds), np.concatenate(ids)


def trust_project(y, x, radius):
    import numpy as np
    delta = y - x
    lengths = np.linalg.norm(delta, axis=1)
    scales = np.minimum(1., radius / np.maximum(lengths, 1e-30))
    return x + delta * scales[:, None]


def graph_data(x, faces):
    import numpy as np
    from scipy.sparse import coo_matrix, diags
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges.sort(axis=1)
    edges = np.unique(edges[edges[:, 0] != edges[:, 1]], axis=0)
    if not len(edges):
        raise ValueError('No valid ARAP graph edges')
    i, j = edges.T
    adjacency = coo_matrix((np.ones(2 * len(edges)), (np.r_[i, j], np.r_[j, i])), shape=(len(x), len(x))).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    laplacian = diags(degree) - adjacency
    return {'i': i, 'j': j, 'reference_edges': x[i] - x[j], 'laplacian': laplacian,
            'degree_max': float(degree.max()), 'n': len(x), 'e': len(edges)}


def local_rotations(y, graph):
    import numpy as np
    i, j, ref = graph['i'], graph['j'], graph['reference_edges']
    dy = y[i] - y[j]
    covariance = np.zeros((len(y), 3, 3), dtype=np.float64)
    products = dy[:, :, None] * ref[:, None, :]
    np.add.at(covariance, i, products)
    np.add.at(covariance, j, products)
    u, _, vh = np.linalg.svd(covariance)
    signs = np.ones((len(y), 3), dtype=np.float64)
    signs[:, 2] = np.linalg.det(u @ vh)
    return (u * signs[:, None, :]) @ vh


def rotated_edges(rotations, graph):
    import numpy as np
    ref = graph['reference_edges']
    return (np.einsum('eij,ej->ei', rotations[graph['i']], ref),
            np.einsum('eij,ej->ei', rotations[graph['j']], ref))


def energy(y, q, rotations, graph, diagonal):
    import numpy as np
    a, b = rotated_edges(rotations, graph)
    dy = y[graph['i']] - y[graph['j']]
    data = np.mean(np.sum((y - q) ** 2, axis=1)) / diagonal**2
    arap = .5 * np.mean(np.sum((dy - a) ** 2, axis=1) + np.sum((dy - b) ** 2, axis=1)) / diagonal**2
    return {'data': float(data), 'arap': float(arap), 'total': float(data + arap)}


def constrained_global(y, x, q, rotations, graph, diagonal, radius, check):
    """Projected gradient for the actual fixed-R convex trust-ball problem."""
    import numpy as np
    n, e = graph['n'], graph['e']
    a, b = rotated_edges(rotations, graph)
    rhs = np.zeros_like(y)
    np.add.at(rhs, graph['i'], a + b)
    np.add.at(rhs, graph['j'], -(a + b))
    # Hessian of unnormalized energy: (2/N)I+(2/E)L.
    # lambda_max(L)<=2*max_degree, so this is a valid Lipschitz bound.
    lipschitz = 2. / n + 4. * graph['degree_max'] / e
    before = energy(y, q, rotations, graph, diagonal)['total']
    max_rise, converged, residual = 0., False, None
    previous = before
    for iteration in range(GLOBAL_MAX):
        check('arap_projected_global')
        gradient = (2. / n) * (y - q) + (2. / e) * graph['laplacian'].dot(y) - rhs / e
        proposal = trust_project(y - gradient / lipschitz, x, radius)
        residual = float(np.sqrt(np.mean(np.sum((proposal - y)**2, axis=1))) / diagonal)
        current = energy(proposal, q, rotations, graph, diagonal)['total']
        max_rise = max(max_rise, current - previous)
        y, previous = proposal, current
        if residual <= GLOBAL_TOL:
            converged = True
            break
    # Re-evaluate the projected-gradient mapping at the returned point.
    gradient = (2. / n) * (y - q) + (2. / e) * graph['laplacian'].dot(y) - rhs / e
    mapping = trust_project(y - gradient / lipschitz, x, radius) - y
    residual = float(np.sqrt(np.mean(np.sum(mapping**2, axis=1))) / diagonal)
    return y, {'iterations': iteration + 1, 'converged': converged and residual <= GLOBAL_TOL,
               'projected_step_rms_over_D': residual, 'tolerance': GLOBAL_TOL,
               'lipschitz_bound_unnormalized_energy': lipschitz,
               'energy_before': before, 'energy_after': previous, 'max_energy_increase': max_rise,
               'descent': max_rise <= 1e-12}


def arap_correct(x, faces, target, diagonal, radius, check):
    import numpy as np
    graph = graph_data(x, faces)
    y = x.copy()
    logs, outer_logs = [], []
    for outer in range(OUTER):
        q, _, _ = closest(target, y, check)
        rotations = local_rotations(y, graph)
        outer_before = energy(y, q, rotations, graph, diagonal)
        for local_global in range(LOCAL_GLOBAL):
            check('arap_local_rotation')
            old_energy = energy(y, q, rotations, graph, diagonal)['total']
            rotations = local_rotations(y, graph)
            rotated_energy = energy(y, q, rotations, graph, diagonal)['total']
            y, info = constrained_global(y, x, q, rotations, graph, diagonal, radius, check)
            info.update(outer=outer, local_global=local_global, energy_before_local=old_energy,
                        energy_after_local=rotated_energy, local_descent=rotated_energy <= old_energy + 1e-12)
            logs.append(info)
        q_new, _, _ = closest(target, y, check)
        rotations = local_rotations(y, graph)
        after = energy(y, q_new, rotations, graph, diagonal)
        change = abs(after['total'] - outer_before['total']) / max(abs(outer_before['total']), 1e-12)
        outer_logs.append({'outer': outer, 'energy_before': outer_before, 'energy_after_refreshed_targets': after,
                           'relative_energy_change': change, 'descent': after['total'] <= outer_before['total'] + 1e-12})
    qualified = all(r['converged'] and r['descent'] and r['local_descent'] for r in logs)
    qualified = qualified and all(r['descent'] for r in outer_logs) and outer_logs[-1]['relative_energy_change'] <= 1e-3
    return y, {'global_solves': logs, 'outer_iterations': outer_logs,
               'numerically_qualified': bool(qualified), 'final_outer_change_tolerance': 1e-3,
               'qualification_scope': 'Projected convex subproblems converge with descent; final outer change small. Global nonconvex optimum is not certified.'}, {'rotations': rotations}


def correct(arm, x, faces, target, diagonal, check):
    import numpy as np
    import trimesh
    radius = .02 * diagonal
    x = np.asarray(x, dtype=np.float64)
    debug, solver = {}, {'numerically_qualified': True}
    if arm == 'identity':
        return x.copy(), solver, debug
    if arm == 'closest_surface':
        q, _, _ = closest(target, x, check)
        return trust_project(q, x, radius), solver, debug
    if arm == 'normal_only':
        normals = np.asarray(trimesh.Trimesh(x, faces, process=False).vertex_normals, dtype=np.float64)
        lengths = np.linalg.norm(normals, axis=1)
        if not np.isfinite(normals).all():
            raise ValueError('Native-frame normals nonfinite')
        normals = normals / np.maximum(lengths[:, None], 1e-30)
        scalar = np.zeros(len(x))
        for iteration in range(5):
            y = x + scalar[:, None] * normals
            q, _, _ = closest(target, y, check)
            scalar = np.clip(scalar + np.sum((q - y) * normals, axis=1), -radius, radius)
        debug['fixed_native_normals'] = normals
        solver['zero_native_normals'] = int(np.sum(lengths <= 1e-12))
        return x + scalar[:, None] * normals, solver, debug
    if arm == 'arap':
        return arap_correct(x, faces, target, diagonal, radius, check)
    raise ValueError(f'Unknown arm: {arm}')


def geometry_report(original, changed, faces, target, diagonal, check):
    import numpy as np
    q, distances, triangles = closest(target, changed, check)
    base = original[faces]
    end = changed[faces]
    n0 = np.cross(base[:, 1] - base[:, 0], base[:, 2] - base[:, 0])
    n1 = np.cross(end[:, 1] - end[:, 0], end[:, 2] - end[:, 0])
    area0, area1 = np.linalg.norm(n0, axis=1), np.linalg.norm(n1, axis=1)
    threshold = 1e-12 * diagonal**2
    valid = (area0 > threshold) & (area1 > threshold)
    displacement = np.linalg.norm(changed - original, axis=1)
    graph = graph_data(np.asarray(original, dtype=np.float64), faces)
    old_lengths = np.linalg.norm(original[graph['i']] - original[graph['j']], axis=1)
    new_lengths = np.linalg.norm(changed[graph['i']] - changed[graph['j']], axis=1)
    ratios = new_lengths / np.maximum(old_lengths, 1e-12 * diagonal)
    report = {'mean_surface_distance': float(distances.mean()),
        'p95_surface_distance': float(np.quantile(distances, .95)),
        'mean_squared_surface_distance_over_D2': float(np.mean(distances**2) / diagonal**2),
        'max_displacement_over_D': float(displacement.max() / diagonal),
        'trust_violation_over_D': float(max(0., displacement.max() / diagonal - .02)),
        'trust_boundary_fraction': float(np.mean(displacement >= .02 * diagonal * (1 - 1e-5))),
        'baseline_degenerate_faces': int(np.sum(area0 <= threshold)),
        'result_degenerate_faces': int(np.sum(area1 <= threshold)),
        'new_degenerate_faces': int(np.sum((area0 > threshold) & (area1 <= threshold))),
        'orientation_reversal_count_relative_native': int(np.sum(valid & (np.sum(n0 * n1, axis=1) < 0))),
        'edge_length_ratio_quantiles_01_50_99': np.quantile(ratios, [.01, .5, .99]).tolist(),
        'finite': bool(np.isfinite(changed).all()), 'surface_query': 'exact closest triangle via Trimesh/Rtree'}
    return report, {'closest_surface_points': q, 'target_triangle_ids': triangles}


def validate_sources(args):
    import numpy as np
    generation = json.loads((args.case_dir / 'report.json').read_text())
    probe = json.loads((args.probe_dir / 'report.json').read_text())
    provenance = json.loads((args.probe_dir / 'provenance.json').read_text())
    if generation.get('status') != 'completed' or probe.get('status') != 'completed':
        raise ValueError('Source generation and shape probe must be completed')
    uid, seed = generation['uid'], generation['seed']
    if (probe.get('uid'), probe.get('seed')) != (uid, seed):
        raise ValueError('Probe/source UID or seed mismatch')
    sources = {}
    for name in ('sequence.npz', 'denoised.npz', 'report.json'):
        actual = digest(args.case_dir / name)
        if actual != provenance.get('source_hashes', {}).get(name):
            raise ValueError(f'Probe/source hash mismatch: {name}')
        if name != 'report.json' and actual != generation.get('sha256', {}).get(name):
            raise ValueError(f'Generation/source hash mismatch: {name}')
        sources[str(args.case_dir / name)] = actual
    with np.load(args.case_dir / 'sequence.npz', allow_pickle=False) as data:
        vertices, faces = data['vertices'].copy(), data['faces'].copy()
        frame_indices = data['frame_indices'].copy()
    if vertices.ndim != 3 or vertices.shape[0] != 16 or not np.issubdtype(vertices.dtype, np.floating):
        raise ValueError('Expected original 16-frame floating-point vertices')
    if not np.array_equal(frame_indices, np.arange(16)):
        raise ValueError('Source timeline must be original frames0..15')
    for t in FRAMES:
        check_mesh(vertices[t], faces)
    targets, target_metadata = {}, {}
    for t in FRAMES:
        path = args.probe_dir / f'frame_{t:02d}/stageI-raw-mesh.npz'
        measurement_path = path.with_name('measurement.json')
        measurement = json.loads(measurement_path.read_text())
        if measurement.get('sha256', {}).get(path.name) != digest(path) or measurement.get('frame') != t:
            raise ValueError(f'Probe geometry identity/hash mismatch: frame{t}')
        with np.load(path, allow_pickle=False) as raw:
            if int(raw['frame']) != t:
                raise ValueError('Probe raw frame identity differs')
            targets[t], target_metadata[t] = triangle_target(raw['vertices'], raw['faces'])
        sources[str(path)] = digest(path)
        sources[str(measurement_path)] = digest(measurement_path)
    for name in ('report.json', 'provenance.json'):
        sources[str(args.probe_dir / name)] = digest(args.probe_dir / name)
    return vertices, faces, targets, target_metadata, sources, uid, seed


def protocol():
    return {'frames': list(FRAMES), 'arms': list(ARMS), 'anchor_bitwise_unchanged': True,
        'implementation_version': 2, 'v1_source_sha256': V1_SOURCE_SHA256,
        'v2_change': 'Cache one identical referenced-vertex KDTree per immutable target; original Trimesh bytecode/candidate order/ties retained. No geometry/strength/iteration/tolerance changes.',
        'amendment': 'Frozen before correction: exact triangle queries replace area-sampled approximate targets/density controls after isolated Rtree1.4.1 installation.',
        'trust_radius': '0.02 * original anchor bbox diagonal', 'gt_guidance': False,
        'camera_or_alignment_guidance': False, 'gpu_used': False,
        'targets': 'All referenced triangles of native Stage-I flash surfaces; no decimation or component removal',
        'closest_query_batch_size': QUERY_BATCH,
        'normal_only': '5 scalar updates along fixed original-frame vertex normals; cumulative trust scalar clipped',
        'arap_outer_iterations': OUTER, 'arap_local_global_per_outer': LOCAL_GLOBAL,
        'arap_energy': 'E=(1/(N D^2))*sum_i||Y_i-q_i||^2 + (1/(2 E D^2))*sum_{undirected(i,j)}(||Y_i-Y_j-R_i(X_i-X_j)||^2+||Y_i-Y_j-R_j(X_i-X_j)||^2)',
        'arap_weights': 'Uniform undirected edges; coefficients1 for data and ARAP; reference X is native mesh at the SAME time',
        'arap_constraint': 'For every vertex ||Y_i-X_i||<=0.02D',
        'arap_global_solver': 'Projected gradient on fixed-R convex subproblem, step1/(2/N+4maxdegree/E); true trust-ball projection each iterate, not clipped unconstrained solve',
        'arap_global_max_iterations': GLOBAL_MAX, 'arap_projected_step_rms_over_D_tolerance': GLOBAL_TOL,
        'arap_qualification': 'Every inner solve converged and descended; every outer step descended; final outer relative energy change<=1e-3; no new degenerate faces/trust violations',
        'semantics': 'Output geometry baseline, not novel steering. Fixed topology does not prove material correspondence. Three frames cannot establish CD-M improvement.'}


def run(args, state, started):
    import numpy as np
    import scipy
    import trimesh
    import rtree
    last_write = [0.]
    def check(stage):
        elapsed = time.monotonic() - started
        if elapsed > args.max_seconds:
            raise TimeoutError(f'CPU baseline exceeded {args.max_seconds}s at {stage}; no geometry/iteration fallback')
        if elapsed - last_write[0] >= 5:
            state['stage'] = stage; state['elapsed_seconds'] = elapsed
            write_json(args.output / 'progress.json', state); last_write[0] = elapsed
    check('source_validation')
    vertices, faces, targets, target_info, sources, uid, seed = validate_sources(args)
    diagonal = float(np.linalg.norm(np.ptp(vertices[0].astype(np.float64), axis=0)))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError('Anchor bounding-box diagonal must be positive')
    state.update(uid=uid, seed=seed, source_files_sha256=sources,
        source_case={'dir': str(args.case_dir), 'sequence_sha256': digest(args.case_dir / 'sequence.npz'), 'report_sha256': digest(args.case_dir / 'report.json')},
        source_probe={'dir': str(args.probe_dir), 'report_sha256': digest(args.probe_dir / 'report.json')},
        anchor_bbox_diagonal=diagonal, target_metadata=target_info,
        packages={'numpy': np.__version__, 'scipy': scipy.__version__, 'trimesh': trimesh.__version__, 'rtree': rtree.__version__})
    write_json(args.output / 'source-provenance.json', {k: state[k] for k in ('uid','seed','source_files_sha256','source_case','source_probe','target_metadata','packages')})
    for arm in ARMS:
        out = args.output / 'variants' / arm
        report = {'status': 'running', 'variant': arm, 'uid': uid, 'seed': seed,
                  'solver_qualified': False, 'frame_reports': [], 'started_utc': now()}
        state['variants'][arm] = report
        write_json(out / 'report.json', report)
        arm_started = time.monotonic()
        try:
            changed = vertices[list(FRAMES)].copy()  # Preserves source dtype and anchor bytes.
            for index, frame in enumerate(FRAMES):
                check(f'{arm}_frame{frame}')
                debug = {}
                if frame == 0 or arm == 'identity':
                    corrected, solver = vertices[frame].copy(), {'numerically_qualified': True, 'bypass': 'exact original bytes'}
                else:
                    corrected, solver, debug = correct(arm, vertices[frame], faces, targets[frame], diagonal, check)
                    corrected = corrected.astype(vertices.dtype)
                check_mesh(corrected, faces)
                changed[index] = corrected
                geometry, nearest = geometry_report(vertices[frame].astype(np.float64), corrected.astype(np.float64), faces, targets[frame], diagonal, check)
                qualified = bool(solver['numerically_qualified'] and geometry['trust_violation_over_D'] <= 1e-6 and geometry['new_degenerate_faces'] == 0)
                if frame == 0 and not np.array_equal(corrected, vertices[0]):
                    raise ValueError('Anchor changed')
                np.savez_compressed(out / f'frame_{frame:02d}.npz', vertices=corrected, faces=faces,
                    frame=np.asarray(frame, dtype=np.int64), displacement=corrected.astype(np.float64) - vertices[frame], **debug, **nearest)
                row = {'frame': frame, 'solver': solver, 'geometry': geometry, 'qualified': qualified,
                       'frame_artifact_sha256': digest(out / f'frame_{frame:02d}.npz')}
                report['frame_reports'].append(row)
                write_json(out / 'report.json', report)
            if changed.dtype != vertices.dtype or not np.array_equal(changed[0], vertices[0]):
                raise ValueError('Output dtype or anchor identity changed')
            if arm == 'identity' and not np.array_equal(changed, vertices[list(FRAMES)]):
                raise ValueError('Identity arm changed positions')
            np.savez_compressed(out / 'sequence.npz', vertices=changed, faces=faces,
                                frame_indices=np.asarray(FRAMES, dtype=np.int64))
            report.update(status='completed', sequence_sha256=digest(out / 'sequence.npz'),
                          solver_qualified=all(r['qualified'] for r in report['frame_reports']),
                          anchor_bitwise_held=True, vertex_dtype=str(changed.dtype), geometry_only=True)
        except Exception as exc:
            report.update(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        report.update(ended_utc=now(), elapsed_seconds=time.monotonic() - arm_started)
        write_json(out / 'report.json', report)
        write_json(args.output / 'report.json', state)
    successes = sum(r['status'] == 'completed' for r in state['variants'].values())
    state.update(status='completed' if successes == len(ARMS) else 'partial' if successes else 'failed',
                 n_declared_arms=len(ARMS), n_completed_arms=successes,
                 n_qualified_arms=sum(bool(r.get('solver_qualified')) for r in state['variants'].values()),
                 exact_query_caches={str(frame):target._census_exact_query.provenance() for frame,target in targets.items()})


def self_test():
    import numpy as np
    triangle = np.array([[0.,0.,1.],[1.,0.,1.],[0.,1.,1.],[0.,0.,0.]])
    tri_faces = np.array([[0,1,2]], dtype=np.int64)
    target, info = triangle_target(triangle, tri_faces)
    q, d, ids = closest(target, np.array([[0.,0.,0.]]))
    assert info['referenced_vertices'] == 3 and np.allclose(q[0],[0,0,1]) and d[0] == 1 and ids[0] == 0
    x = np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]])
    faces = np.array([[0,1,2],[0,2,3]], dtype=np.int64)
    graph = graph_data(x, faces); diagonal = math.sqrt(2)
    rotation = np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    y = x @ rotation.T + [4.,-2.,3.]
    rotations = local_rotations(y, graph)
    assert energy(y,y,rotations,graph,diagonal)['arap'] < 1e-25
    projected = trust_project(x + 100, x, .02 * diagonal)
    assert np.max(np.linalg.norm(projected-x,axis=1)) <= .02 * diagonal + 1e-14
    plane, _ = triangle_target(x + [0,0,.01], faces)
    for arm in ARMS:
        corrected, solver, _ = correct(arm, x, faces, plane, diagonal, lambda stage: None)
        assert np.isfinite(corrected).all()
        assert np.max(np.linalg.norm(corrected-x,axis=1)) <= .02 * diagonal + 1e-12
        if arm != 'identity':
            assert np.max(np.abs(corrected[:,2] - .01)) < 1e-5
        if arm == 'arap':
            assert all(s['descent'] and s['converged'] and s['local_descent'] for s in solver['global_solves'])
    source = np.stack([x.astype(np.float32)] * 16)
    result = source[list(FRAMES)].copy()
    result[1] = projected.astype(np.float32)
    assert result.dtype == source.dtype and result[0].tobytes() == source[0].tobytes()
    return {'status':'passed','rigid_rotation_translation_arap_energy_zero':True,
            'unreferenced_grid_vertex_excluded_from_exact_triangle_query':True,
            'trust_bound_enforced':True,'translated_plane_all_corrections':True,
            'projected_global_descent_and_convergence':True,'anchor_dtype_and_bytes_preserved':True,
            'gpu_used':False,'scope':'CPU analytic geometry fixtures, not generated-case efficacy'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/root/rivermind-data/actionmesh-repro'))
    parser.add_argument('--case-dir',type=Path);parser.add_argument('--probe-dir',type=Path)
    parser.add_argument('--output',type=Path);parser.add_argument('--extra-python',type=Path)
    parser.add_argument('--max-seconds',type=float,default=600.)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.extra_python:
        sys.path.insert(0,str(args.extra_python.expanduser().resolve()))
    if args.self_test:
        print(json.dumps(self_test(),indent=2));return 0
    if any(getattr(args,k) is None for k in ('case_dir','probe_dir','output')):
        parser.error('--case-dir --probe-dir --output required')
    if not 0 < args.max_seconds < math.inf:
        parser.error('max-seconds must be finite and positive')
    for k in ('root','case_dir','probe_dir','output'):
        setattr(args,k,getattr(args,k).expanduser().resolve())
    args.output.mkdir(parents=True,exist_ok=False)
    state={'status':'running','started_utc':now(),'variants':{},'stage':'setup'}
    for arm in ARMS:
        out=args.output/'variants'/arm;out.mkdir(parents=True)
        report={'status':'pending','variant':arm,'solver_qualified':False}
        state['variants'][arm]=report;write_json(out/'report.json',report)
    started=time.monotonic()
    write_json(args.output/'command.json',{'argv':sys.argv,'script_sha256':digest(Path(__file__)),
        'started_utc':state['started_utc'],'max_seconds':args.max_seconds,'cpu_only':True})
    write_json(args.output/'protocol.json',protocol())
    try:
        run(args,state,started)
    except Exception as exc:
        state.update(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        for arm,report in state['variants'].items():
            if report['status'] in ('pending','running'):
                report.update(status='failed',error=f'Source/setup failure: {exc}')
                write_json(args.output/'variants'/arm/'report.json',report)
    finally:
        usage=resource.getrusage(resource.RUSAGE_SELF)
        state.update(ended_utc=now(),elapsed_seconds=time.monotonic()-started,
            resources={'cpu_user_seconds':usage.ru_utime,'cpu_system_seconds':usage.ru_stime,
                       'peak_rss_bytes':int(usage.ru_maxrss if sys.platform=='darwin' else usage.ru_maxrss*1024),'gpu_used':False})
        write_json(args.output/'report.json',state);write_json(args.output/'progress.json',state)
    return int(state['status']!='completed')


if __name__=='__main__':
    raise SystemExit(main())
