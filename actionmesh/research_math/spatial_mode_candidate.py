"""Deterministic mathematical core for the C05 spatial-mode candidate.

This module deliberately starts from *coordinate trajectories* produced by
repeated Stage-I draws with one fixed Stage-0 anchor/query/topology.  It does
not interpret latent attention as a material-coordinate posterior.  Empirical
cluster frequencies are positive unary weights, not calibrated probabilities.

The module has no model loader, benchmark adapter, file writer or dispatch
entry.  Those provenance and execution boundaries belong to the future C05
producer/acceptance chain.  In particular, a bank with one retained cluster is
valid data but is not evidence of the non-degenerate C05 premise.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


_FLOAT = np.float64
METHOD_IDS = {
    "localized_mean": "c05-control-localized-empirical-mean",
    "temperature_matched_mean": "c05-control-temperature-matched-mean",
    "surface_projected_mean": "c05-control-union-surface-projected-mean",
    "independent_top1": "c05-control-independent-empirical-top1",
    "joint_spatial_labels": "4d-math-20261006-c05",
}


def _finite_array(value, name: str, *, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=_FLOAT)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _positive(value, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def validate_native_arrays(
    sampled_surfaces: np.ndarray,
    *,
    anchor_vertices: np.ndarray,
    faces: np.ndarray,
    target_frame_indices: np.ndarray,
    target_timesteps: np.ndarray,
    query_vertex_ids: np.ndarray,
    draw_seeds: Sequence[int],
) -> dict[str, np.ndarray | tuple[int, ...]]:
    """Validate one fixed-topology same-anchor stochastic coordinate bank.

    This is an in-memory mathematical boundary, not a provenance validator.
    A future artifact materializer must still rehash the retained native
    captures and prove that the first branch reproduces B0.
    """

    surfaces = _finite_array(sampled_surfaces, "sampled surfaces", ndim=4)
    anchor = _finite_array(anchor_vertices, "anchor vertices", ndim=2)
    if (surfaces.shape[0] < 2 or surfaces.shape[1] < 1 or surfaces.shape[3] != 3 or
            anchor.ndim != 2 or anchor.shape != (surfaces.shape[2], 3)):
        raise ValueError("Expected surfaces [K,T,V,3] and one matching anchor [V,3]")
    if (len(draw_seeds) != surfaces.shape[0] or any(type(seed) is not int for seed in draw_seeds)
            or len(set(draw_seeds)) != len(draw_seeds)):
        raise ValueError("One distinct integer Stage-I seed per surface draw required")
    raw_faces = np.asarray(faces)
    if (raw_faces.ndim != 2 or raw_faces.shape[1] != 3 or not len(raw_faces) or
            not np.issubdtype(raw_faces.dtype, np.integer)):
        raise ValueError("Nonempty integer triangle topology required")
    topology = raw_faces.astype(np.int64, copy=True)
    if topology.min() < 0 or topology.max() >= len(anchor):
        raise ValueError("Face index outside the fixed anchor topology")
    frame_ids = np.asarray(target_frame_indices)
    times = _finite_array(target_timesteps, "target timesteps", ndim=1)
    vertex_ids = np.asarray(query_vertex_ids)
    if (frame_ids.ndim != 1 or not np.issubdtype(frame_ids.dtype, np.integer) or
            frame_ids.shape != (surfaces.shape[1],) or
            times.shape != frame_ids.shape or
            np.any(frame_ids[1:] <= frame_ids[:-1]) or np.any(times[1:] <= times[:-1])):
        raise ValueError("Strictly ordered complete target frame/time IDs required")
    if (vertex_ids.ndim != 1 or not np.issubdtype(vertex_ids.dtype, np.integer) or
            vertex_ids.shape != (surfaces.shape[2],) or len(np.unique(vertex_ids)) != len(vertex_ids)):
        raise ValueError("One unique material-query ID per anchor vertex required")
    result: dict[str, np.ndarray | tuple[int, ...]] = {
        "sampled_surfaces": surfaces.copy(),
        "anchor_vertices": anchor.copy(),
        "faces": topology,
        "target_frame_indices": frame_ids.astype(np.int64, copy=True),
        "target_timesteps": times.copy(),
        "query_vertex_ids": vertex_ids.astype(np.int64, copy=True),
        "draw_seeds": tuple(int(seed) for seed in draw_seeds),
    }
    for value in result.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return result


@dataclass(frozen=True)
class LandmarkModes:
    """Empirical modes for one material-query index.

    ``atoms[m]`` is an observed full trajectory from one draw (a medoid), never
    a cluster centroid. ``masses[m]`` is cluster_size / draw_count.
    """

    atoms: np.ndarray  # [M,T,3]
    masses: np.ndarray  # [M]
    medoid_draw_indices: tuple[int, ...]
    member_draw_indices: tuple[tuple[int, ...], ...]
    scale: float

    def __post_init__(self) -> None:
        atoms = _finite_array(self.atoms, "mode atoms", ndim=3)
        masses = _finite_array(self.masses, "mode masses", ndim=1)
        if atoms.shape[0] < 1 or atoms.shape[1] < 1 or atoms.shape[2] != 3:
            raise ValueError("Mode atoms must have shape [M,T,3] with M,T >= 1")
        if masses.shape != (atoms.shape[0],) or np.any(masses <= 0):
            raise ValueError("Every mode requires one positive mass")
        if not math.isclose(float(masses.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Mode masses must sum to one")
        if len(self.medoid_draw_indices) != atoms.shape[0]:
            raise ValueError("Each mode requires one medoid draw index")
        if len(self.member_draw_indices) != atoms.shape[0]:
            raise ValueError("Each mode requires a nonempty member set")
        flat: list[int] = []
        for medoid, members in zip(self.medoid_draw_indices, self.member_draw_indices):
            if not members or tuple(sorted(set(members))) != members or medoid not in members:
                raise ValueError("Canonical unique members containing their medoid required")
            flat.extend(members)
        if sorted(flat) != list(range(len(flat))):
            raise ValueError("Mode members must partition all draw indices")
        _positive(self.scale, "landmark scale")
        object.__setattr__(self, "atoms", atoms.copy())
        object.__setattr__(self, "masses", masses.copy())
        self.atoms.setflags(write=False)
        self.masses.setflags(write=False)


@dataclass(frozen=True)
class SpatialModeBank:
    """Variable-cardinality mode sets for fixed material landmarks."""

    draw_seeds: tuple[int, ...]
    modes: tuple[LandmarkModes, ...]
    cluster_radius: float

    def __post_init__(self) -> None:
        if len(self.draw_seeds) < 2 or len(set(self.draw_seeds)) != len(self.draw_seeds):
            raise ValueError("At least two distinct fixed draw seeds required")
        if tuple(sorted(self.draw_seeds)) != self.draw_seeds:
            raise ValueError("Draw seeds must be stored in canonical increasing order")
        if not self.modes:
            raise ValueError("At least one landmark is required")
        frames = self.modes[0].atoms.shape[1]
        for mode in self.modes:
            if mode.atoms.shape[1] != frames:
                raise ValueError("All landmarks must share one complete time axis")
            members = sum((list(row) for row in mode.member_draw_indices), [])
            if len(members) != len(self.draw_seeds):
                raise ValueError("Every landmark must retain every draw exactly once")
        _positive(self.cluster_radius, "cluster radius")

    @property
    def landmark_count(self) -> int:
        return len(self.modes)

    @property
    def frame_count(self) -> int:
        return int(self.modes[0].atoms.shape[1])


def normalized_mode_separations(bank: SpatialModeBank) -> np.ndarray:
    """Return the largest observed-medoid separation at each landmark.

    The distance is the full-trajectory RMS divided by the prospectively fixed
    landmark scale.  A one-mode landmark has separation zero.  This supplies a
    direct Natural-Gate witness: merely storing two cluster labels is not itself
    evidence that two retained coordinate atoms are separated.
    """

    values = np.zeros(bank.landmark_count, dtype=_FLOAT)
    for landmark, mode in enumerate(bank.modes):
        if len(mode.atoms) < 2:
            continue
        distances = _trajectory_distances(mode.atoms, mode.scale)
        values[landmark] = float(np.max(distances))
    values.setflags(write=False)
    return values


def _trajectory_distances(draws: np.ndarray, scale: float) -> np.ndarray:
    difference = draws[:, None] - draws[None, :]
    return np.sqrt(np.mean(np.sum(difference * difference, axis=-1), axis=-1)) / scale


def _complete_link_clusters(distances: np.ndarray, radius: float) -> list[tuple[int, ...]]:
    """Canonical deterministic complete-link agglomeration.

    A merge is legal only when *every* cross-cluster pair is within ``radius``.
    This prevents a single-link chain from turning separated endpoints into one
    purported mode.  Ties are resolved by the member-index tuples.
    """

    clusters: list[tuple[int, ...]] = [(index,) for index in range(len(distances))]
    while True:
        legal: list[tuple[float, tuple[int, ...], int, int]] = []
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                maximum = max(
                    float(distances[i, j])
                    for i in clusters[left]
                    for j in clusters[right]
                )
                merged = tuple(sorted(clusters[left] + clusters[right]))
                if maximum <= radius:
                    legal.append((maximum, merged, left, right))
        if not legal:
            break
        _, merged, left, right = min(legal, key=lambda row: (row[0], row[1]))
        clusters = [cluster for index, cluster in enumerate(clusters)
                    if index not in (left, right)] + [merged]
        clusters.sort()
    return clusters


def cluster_empirical_trajectories(
    draws: np.ndarray,
    *,
    draw_seeds: Sequence[int],
    landmark_scales: Sequence[float],
    cluster_radius: float,
) -> SpatialModeBank:
    """Cluster same-anchor coordinate draws into empirical medoid modes.

    Args:
        draws: ``[K,T,L,3]`` direct coordinate trajectories.  All draws must
            already have the same anchor query indices, topology and time IDs.
        draw_seeds: The distinct Stage-I seeds corresponding to axis K.
        landmark_scales: Positive, prospectively defined geometry scale per L.
        cluster_radius: Complete-link radius in scale-normalized trajectory RMS.
    """

    coordinates = _finite_array(draws, "coordinate draws", ndim=4)
    if coordinates.shape[0] < 2 or coordinates.shape[1] < 1 or coordinates.shape[2] < 1:
        raise ValueError("Coordinate draws must have shape [K,T,L,3], K >= 2")
    if coordinates.shape[3] != 3:
        raise ValueError("Coordinate draws require XYZ in the final axis")
    if len(draw_seeds) != coordinates.shape[0] or any(type(seed) is not int for seed in draw_seeds):
        raise ValueError("One integer seed per draw required")
    if len(set(draw_seeds)) != len(draw_seeds):
        raise ValueError("Draw seeds must be distinct")
    scales = _finite_array(landmark_scales, "landmark scales", ndim=1)
    if scales.shape != (coordinates.shape[2],) or np.any(scales <= 0):
        raise ValueError("One positive geometry scale per landmark required")
    radius = _positive(cluster_radius, "cluster radius")

    # A caller's draw ordering must not change clustering or tie-breaking.
    order = np.argsort(np.asarray(draw_seeds, dtype=np.int64), kind="stable")
    coordinates = coordinates[order]
    seeds = tuple(int(draw_seeds[index]) for index in order)
    landmarks: list[LandmarkModes] = []
    for landmark, scale in enumerate(scales):
        trajectories = coordinates[:, :, landmark, :]
        distances = _trajectory_distances(trajectories, float(scale))
        clusters = _complete_link_clusters(distances, radius)
        rows: list[tuple[int, tuple[int, ...]]] = []
        for members in clusters:
            # Medoid is an observed trajectory.  Equal sums use the lower seed.
            medoid = min(
                members,
                key=lambda index: (
                    float(sum(distances[index, other] ** 2 for other in members)),
                    seeds[index],
                ),
            )
            rows.append((medoid, members))
        # Largest empirical mass first; ties use the medoid seed.  Therefore a
        # top-1 reduction is deterministic and is always mode index zero.
        rows.sort(key=lambda row: (-len(row[1]), seeds[row[0]], row[1]))
        masses = np.asarray([len(members) / len(seeds) for _, members in rows])
        landmarks.append(LandmarkModes(
            atoms=np.stack([trajectories[medoid] for medoid, _ in rows]),
            masses=masses,
            medoid_draw_indices=tuple(medoid for medoid, _ in rows),
            member_draw_indices=tuple(tuple(members) for _, members in rows),
            scale=float(scale),
        ))
    return SpatialModeBank(draw_seeds=seeds, modes=tuple(landmarks), cluster_radius=radius)


def _stack_landmark_targets(targets: Sequence[np.ndarray], bank: SpatialModeBank) -> np.ndarray:
    if len(targets) != bank.landmark_count:
        raise ValueError("One target trajectory per landmark required")
    return np.stack(targets, axis=1)  # [T,L,3]


def independent_top1(bank: SpatialModeBank) -> tuple[np.ndarray, np.ndarray]:
    """Select the largest empirical cluster independently at every landmark."""

    labels = np.zeros(bank.landmark_count, dtype=np.int64)
    return _stack_landmark_targets([mode.atoms[0] for mode in bank.modes], bank), labels


def temperature_mean(bank: SpatialModeBank, temperature: float) -> np.ndarray:
    """Mean each coordinate bank after a declared mass-temperature change."""

    temperature = _positive(temperature, "temperature")
    targets: list[np.ndarray] = []
    for mode in bank.modes:
        log_weights = np.log(mode.masses) / temperature
        weights = np.exp(log_weights - log_weights.max())
        weights /= weights.sum()
        targets.append(np.tensordot(weights, mode.atoms, axes=(0, 0)))
    return _stack_landmark_targets(targets, bank)


def localized_mean(bank: SpatialModeBank, radius: float) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Mass-weighted mean in a fixed trajectory ball around empirical top-1."""

    radius = _positive(radius, "localized radius")
    targets: list[np.ndarray] = []
    supports: list[tuple[int, ...]] = []
    for mode in bank.modes:
        distance = np.sqrt(np.mean(np.sum((mode.atoms - mode.atoms[0]) ** 2, axis=-1), axis=-1))
        support = tuple(int(index) for index in np.flatnonzero(distance <= radius * mode.scale))
        if not support or support[0] != 0:
            raise AssertionError("Top-1 must belong to its localized support")
        weights = mode.masses[list(support)].copy()
        weights /= weights.sum()
        targets.append(np.tensordot(weights, mode.atoms[list(support)], axes=(0, 0)))
        supports.append(support)
    return _stack_landmark_targets(targets, bank), tuple(supports)


def select_label_trajectories(bank: SpatialModeBank, labels: Sequence[int]) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.shape != (bank.landmark_count,) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("One integer mode label per landmark required")
    targets = []
    for landmark, label in enumerate(labels):
        if label < 0 or label >= len(bank.modes[landmark].atoms):
            raise ValueError("Mode label outside the landmark bank")
        targets.append(bank.modes[landmark].atoms[int(label)])
    return _stack_landmark_targets(targets, bank)


def _canonical_graph(edges: np.ndarray, landmark_count: int) -> np.ndarray:
    raw = np.asarray(edges)
    if raw.ndim != 2 or raw.shape[1] != 2 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("Edges must be an integer [E,2] array")
    canonical = np.sort(raw.astype(np.int64, copy=False), axis=1)
    if (not len(canonical) or np.any(canonical[:, 0] == canonical[:, 1]) or
            canonical.min() < 0 or canonical.max() >= landmark_count):
        raise ValueError("Nonempty in-range non-self edges required")
    canonical = canonical[np.lexsort((canonical[:, 1], canonical[:, 0]))]
    if len(np.unique(canonical, axis=0)) != len(canonical):
        raise ValueError("Duplicate graph edges are forbidden")
    reached = {0}
    while True:
        expanded = reached | {int(v) for edge in canonical for v in edge if any(u in reached for u in edge)}
        if expanded == reached:
            break
        reached = expanded
    if reached != set(range(landmark_count)):
        raise ValueError("Connected landmark graph required")
    return canonical


def _validated_energy_inputs(
    bank: SpatialModeBank,
    edges: np.ndarray,
    anchor_vertices: np.ndarray,
    rotations: np.ndarray,
    time_weights: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    graph = _canonical_graph(edges, bank.landmark_count)
    anchor = _finite_array(anchor_vertices, "anchor vertices", ndim=2)
    if anchor.shape != (bank.landmark_count, 3):
        raise ValueError("Anchor vertices must have shape [L,3]")
    rotation = _finite_array(rotations, "rotation hints", ndim=4)
    if rotation.shape != (len(graph), bank.frame_count, 3, 3):
        raise ValueError("Rotation hints must have shape [E,T,3,3]")
    eye = np.eye(3)
    gram = np.matmul(np.swapaxes(rotation, -1, -2), rotation)
    if (np.max(np.abs(gram - eye)) > 1e-6 or
            np.max(np.abs(np.linalg.det(rotation) - 1.0)) > 1e-6):
        raise ValueError("Proper orthogonal rotation hints required")
    weights = _finite_array(time_weights, "time weights", ndim=1)
    if weights.shape != (bank.frame_count,) or np.any(weights <= 0):
        raise ValueError("One positive weight per complete target frame required")
    weights = weights / weights.sum()
    return graph, anchor, rotation, weights


def spatial_mode_energy(
    bank: SpatialModeBank,
    labels: Sequence[int],
    *,
    edges: np.ndarray,
    anchor_vertices: np.ndarray,
    rotations: np.ndarray,
    time_weights: Sequence[float],
    unary_weight: float = 1.0,
    spatial_weight: float = 1.0,
) -> dict[str, float]:
    """Evaluate the explicit full-trajectory C05 labeling energy."""

    unary_weight = _positive(unary_weight, "unary weight")
    spatial_weight = _positive(spatial_weight, "spatial weight")
    graph, anchor, rotation, weights = _validated_energy_inputs(
        bank, edges, anchor_vertices, rotations, time_weights)
    label_array = np.asarray(labels)
    selected = select_label_trajectories(bank, label_array)
    unary = 0.0
    for landmark, label in enumerate(label_array):
        unary -= math.log(float(bank.modes[landmark].masses[int(label)]))
    pairwise = 0.0
    for edge_index, (left, right) in enumerate(graph):
        reference = rotation[edge_index] @ (anchor[left] - anchor[right])
        residual = selected[:, left] - selected[:, right] - reference
        pairwise += float(np.sum(weights * np.sum(residual * residual, axis=-1)))
    return {
        "unary": unary_weight * unary,
        "pairwise": spatial_weight * pairwise,
        "total": unary_weight * unary + spatial_weight * pairwise,
    }


@dataclass(frozen=True)
class ICMRun:
    start_name: str
    initial_labels: tuple[int, ...]
    labels: tuple[int, ...]
    objective_trace: tuple[float, ...]
    sweeps: int
    converged: bool
    one_flip_residual: float
    energy: dict[str, float]


@dataclass(frozen=True)
class JointLabelResult:
    labels: tuple[int, ...]
    trajectories: np.ndarray
    energy: dict[str, float]
    one_flip_residual: float
    selected_start: str
    runs: tuple[ICMRun, ...]


def _local_energy(
    bank: SpatialModeBank,
    labels: np.ndarray,
    landmark: int,
    proposed: int,
    *,
    graph: np.ndarray,
    anchor: np.ndarray,
    rotation: np.ndarray,
    weights: np.ndarray,
    unary_weight: float,
    spatial_weight: float,
) -> float:
    value = -unary_weight * math.log(float(bank.modes[landmark].masses[proposed]))
    proposed_trajectory = bank.modes[landmark].atoms[proposed]
    for edge_index, (left, right) in enumerate(graph):
        if landmark not in (left, right):
            continue
        left_trajectory = (proposed_trajectory if left == landmark else
                           bank.modes[left].atoms[int(labels[left])])
        right_trajectory = (proposed_trajectory if right == landmark else
                            bank.modes[right].atoms[int(labels[right])])
        reference = rotation[edge_index] @ (anchor[left] - anchor[right])
        residual = left_trajectory - right_trajectory - reference
        value += spatial_weight * float(np.sum(weights * np.sum(residual * residual, axis=-1)))
    return value


def solve_joint_labels_icm(
    bank: SpatialModeBank,
    *,
    edges: np.ndarray,
    anchor_vertices: np.ndarray,
    rotations: np.ndarray,
    time_weights: Sequence[float],
    unary_weight: float = 1.0,
    spatial_weight: float = 1.0,
    max_sweeps: int = 100,
    additional_starts: Iterable[tuple[str, Sequence[int]]] = (),
    tolerance: float = 1e-12,
) -> JointLabelResult:
    """Deterministic multi-start ICM with an auditable local certificate.

    This is an approximate solver.  ``one_flip_residual`` is the maximum
    remaining single-landmark objective improvement; it is not a global gap.
    """

    unary_weight = _positive(unary_weight, "unary weight")
    spatial_weight = _positive(spatial_weight, "spatial weight")
    if type(max_sweeps) is not int or max_sweeps < 1:
        raise ValueError("max_sweeps must be a positive integer")
    if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    graph, anchor, rotation, weights = _validated_energy_inputs(
        bank, edges, anchor_vertices, rotations, time_weights)

    starts: list[tuple[str, tuple[int, ...]]] = [
        ("independent_top1", tuple(0 for _ in bank.modes))
    ]
    for rank in range(max(len(mode.atoms) for mode in bank.modes)):
        starts.append((f"rank_{rank}", tuple(min(rank, len(mode.atoms) - 1) for mode in bank.modes)))
    for name, labels in additional_starts:
        labels = tuple(int(label) for label in labels)
        if not name or len(labels) != bank.landmark_count:
            raise ValueError("Named complete additional starts required")
        starts.append((str(name), labels))
    # Preserve first occurrence of identical labels, while rejecting duplicate names.
    if len({name for name, _ in starts}) != len(starts):
        raise ValueError("ICM start names must be unique")
    unique: list[tuple[str, tuple[int, ...]]] = []
    seen: set[tuple[int, ...]] = set()
    for name, labels in starts:
        select_label_trajectories(bank, labels)
        if labels not in seen:
            unique.append((name, labels))
            seen.add(labels)

    runs: list[ICMRun] = []
    for name, initial in unique:
        labels = np.asarray(initial, dtype=np.int64)
        initial_energy = spatial_mode_energy(
            bank, labels, edges=graph, anchor_vertices=anchor, rotations=rotation,
            time_weights=weights, unary_weight=unary_weight, spatial_weight=spatial_weight)
        trace = [initial_energy["total"]]
        converged = False
        sweeps = 0
        for sweep in range(1, max_sweeps + 1):
            changed = False
            for landmark, mode in enumerate(bank.modes):
                current = int(labels[landmark])
                current_value = _local_energy(
                    bank, labels, landmark, current, graph=graph, anchor=anchor,
                    rotation=rotation, weights=weights, unary_weight=unary_weight,
                    spatial_weight=spatial_weight)
                candidate_values = [
                    _local_energy(
                        bank, labels, landmark, candidate, graph=graph, anchor=anchor,
                        rotation=rotation, weights=weights, unary_weight=unary_weight,
                        spatial_weight=spatial_weight)
                    for candidate in range(len(mode.atoms))
                ]
                best = min(range(len(candidate_values)), key=lambda index: (candidate_values[index], index))
                if candidate_values[best] < current_value - tolerance:
                    labels[landmark] = best
                    changed = True
            sweeps = sweep
            energy = spatial_mode_energy(
                bank, labels, edges=graph, anchor_vertices=anchor, rotations=rotation,
                time_weights=weights, unary_weight=unary_weight, spatial_weight=spatial_weight)
            if energy["total"] > trace[-1] + tolerance:
                raise AssertionError("ICM sweep increased the declared objective")
            trace.append(energy["total"])
            if not changed:
                converged = True
                break
        residual = 0.0
        for landmark, mode in enumerate(bank.modes):
            current = _local_energy(
                bank, labels, landmark, int(labels[landmark]), graph=graph, anchor=anchor,
                rotation=rotation, weights=weights, unary_weight=unary_weight,
                spatial_weight=spatial_weight)
            best = min(
                _local_energy(
                    bank, labels, landmark, candidate, graph=graph, anchor=anchor,
                    rotation=rotation, weights=weights, unary_weight=unary_weight,
                    spatial_weight=spatial_weight)
                for candidate in range(len(mode.atoms))
            )
            residual = max(residual, current - best)
        final_energy = spatial_mode_energy(
            bank, labels, edges=graph, anchor_vertices=anchor, rotations=rotation,
            time_weights=weights, unary_weight=unary_weight, spatial_weight=spatial_weight)
        runs.append(ICMRun(
            start_name=name,
            initial_labels=initial,
            labels=tuple(int(label) for label in labels),
            objective_trace=tuple(float(value) for value in trace),
            sweeps=sweeps,
            converged=converged,
            one_flip_residual=float(residual),
            energy=final_energy,
        ))
    selected = min(runs, key=lambda run: (run.energy["total"], run.start_name, run.labels))
    trajectories = select_label_trajectories(bank, selected.labels)
    trajectories.setflags(write=False)
    return JointLabelResult(
        labels=selected.labels,
        trajectories=trajectories,
        energy=selected.energy,
        one_flip_residual=selected.one_flip_residual,
        selected_start=selected.start_name,
        runs=tuple(runs),
    )


def _segment_projection(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = end - start
    denominator = np.sum(direction * direction, axis=1)
    numerator = np.sum((point[None] - start) * direction, axis=1)
    fraction = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    fraction = np.clip(fraction, 0.0, 1.0)
    return start + fraction[:, None] * direction, fraction


def _closest_on_triangle_chunk(point: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, int, np.ndarray, float]:
    """Closest point to one triangle chunk, with deterministic candidate ties."""

    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = b - a, c - a
    d00 = np.sum(ab * ab, axis=1)
    d01 = np.sum(ab * ac, axis=1)
    d11 = np.sum(ac * ac, axis=1)
    ap = point[None] - a
    d20 = np.sum(ap * ab, axis=1)
    d21 = np.sum(ap * ac, axis=1)
    denominator = d00 * d11 - d01 * d01
    v = np.divide(d11 * d20 - d01 * d21, denominator,
                  out=np.zeros_like(denominator), where=np.abs(denominator) > 1e-30)
    w = np.divide(d00 * d21 - d01 * d20, denominator,
                  out=np.zeros_like(denominator), where=np.abs(denominator) > 1e-30)
    u = 1.0 - v - w
    plane = u[:, None] * a + v[:, None] * b + w[:, None] * c
    valid_plane = ((u >= 0) & (v >= 0) & (w >= 0) & (np.abs(denominator) > 1e-30))

    ab_point, ab_t = _segment_projection(point, a, b)
    bc_point, bc_t = _segment_projection(point, b, c)
    ca_point, ca_t = _segment_projection(point, c, a)
    candidates = np.stack((plane, ab_point, bc_point, ca_point))  # [4,F,3]
    bary = np.stack((
        np.stack((u, v, w), axis=1),
        np.stack((1.0 - ab_t, ab_t, np.zeros_like(ab_t)), axis=1),
        np.stack((np.zeros_like(bc_t), 1.0 - bc_t, bc_t), axis=1),
        np.stack((ca_t, np.zeros_like(ca_t), 1.0 - ca_t), axis=1),
    ))
    squared = np.sum((candidates - point) ** 2, axis=-1)
    squared[0, ~valid_plane] = np.inf
    flat = int(np.argmin(squared))
    candidate_index, triangle_index = np.unravel_index(flat, squared.shape)
    return (candidates[candidate_index, triangle_index], int(triangle_index),
            bary[candidate_index, triangle_index], float(squared[candidate_index, triangle_index]))


def project_to_sampled_surface_union(
    points: np.ndarray,
    sampled_surfaces: np.ndarray,
    faces: np.ndarray,
    *,
    face_chunk_size: int = 4096,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Project sparse trajectories to the union of sampled model surfaces.

    ``points`` has shape ``[T,L,3]`` and ``sampled_surfaces`` has shape
    ``[K,T,V,3]``.  The earliest draw/face/candidate wins an exact distance tie.
    No ground truth or evaluation geometry is accepted.
    """

    query = _finite_array(points, "projection points", ndim=3)
    surfaces = _finite_array(sampled_surfaces, "sampled surfaces", ndim=4)
    raw_faces = np.asarray(faces)
    if (query.shape[2] != 3 or surfaces.shape[1] != query.shape[0] or
            surfaces.shape[3] != 3 or surfaces.shape[0] < 2):
        raise ValueError("Expected points [T,L,3] and surfaces [K,T,V,3], K >= 2")
    if (raw_faces.ndim != 2 or raw_faces.shape[1] != 3 or
            not np.issubdtype(raw_faces.dtype, np.integer) or not len(raw_faces)):
        raise ValueError("Integer triangle faces required")
    faces = raw_faces.astype(np.int64, copy=False)
    if faces.min() < 0 or faces.max() >= surfaces.shape[2]:
        raise ValueError("Surface face index outside the common topology")
    if type(face_chunk_size) is not int or face_chunk_size < 1:
        raise ValueError("face_chunk_size must be a positive integer")

    projected = np.empty_like(query)
    draw_indices = np.empty(query.shape[:2], dtype=np.int64)
    face_indices = np.empty(query.shape[:2], dtype=np.int64)
    barycentrics = np.empty(query.shape[:2] + (3,), dtype=_FLOAT)
    squared_distances = np.empty(query.shape[:2], dtype=_FLOAT)
    for frame in range(query.shape[0]):
        for landmark in range(query.shape[1]):
            best: tuple[np.ndarray, int, int, np.ndarray, float] | None = None
            point = query[frame, landmark]
            for draw in range(surfaces.shape[0]):
                for start in range(0, len(faces), face_chunk_size):
                    stop = min(start + face_chunk_size, len(faces))
                    triangles = surfaces[draw, frame, faces[start:stop]]
                    location, local_face, bary, distance = _closest_on_triangle_chunk(point, triangles)
                    face = start + local_face
                    # Strict update preserves earlier draw/face on an exact tie.
                    if best is None or distance < best[4]:
                        best = (location, draw, face, bary, distance)
            assert best is not None
            projected[frame, landmark], draw_indices[frame, landmark], face_indices[frame, landmark], \
                barycentrics[frame, landmark], squared_distances[frame, landmark] = best
    evidence = {
        "draw_indices": draw_indices,
        "face_indices": face_indices,
        "barycentrics": barycentrics,
        "squared_distances": squared_distances,
    }
    return projected, evidence


def union_surface_projected_mean(
    bank: SpatialModeBank,
    *,
    localized_radius: float,
    sampled_surfaces: np.ndarray,
    faces: np.ndarray,
    face_chunk_size: int = 4096,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Project the same-bank localized mean onto the sampled-surface union."""

    mean, _ = localized_mean(bank, localized_radius)
    return project_to_sampled_surface_union(
        mean, sampled_surfaces, faces, face_chunk_size=face_chunk_size)
