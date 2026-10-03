"""Robust stable-region Sim(3) decomposition without removing true pose.

This method requires a user-verified dimensionally stable region and metric
world coordinates (or already compensated camera motion). It does not infer
object/camera scale identifiability from monocular observations. Correcting
scale about the fitted stable-region center retains that region's world
translation, rotation, and scale-normalized articulation everywhere else.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class SimilarityFit:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    weights: np.ndarray
    residuals: np.ndarray


@dataclass
class ScaleDecomposition:
    corrected: np.ndarray
    scales: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray
    centers: np.ndarray
    local_residuals: np.ndarray
    fit_weights: np.ndarray
    camera_convention: str


def _points(value, name):
    x = np.asarray(value, float)
    if x.ndim != 2 or x.shape[1] != 3 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError(f"{name} must be a finite (N,3) array, N>=3")
    return x


def _weighted_fit(source, target, weights):
    total = weights.sum()
    if total <= 0 or np.count_nonzero(weights > 0) < 3:
        raise ValueError("at least three positive weights are required")
    w = weights / total
    c_source = w @ source
    c_target = w @ target
    x, y = source - c_source, target - c_target
    if np.linalg.matrix_rank(x * np.sqrt(w[:, None])) < 2 or np.linalg.matrix_rank(y * np.sqrt(w[:, None])) < 2:
        raise ValueError("similarity requires noncollinear source and target points")
    cov = (x * w[:, None]).T @ y
    u, singular, vh = np.linalg.svd(cov)
    signs = np.ones(3)
    signs[-1] = -1 if np.linalg.det(vh.T @ u.T) < 0 else 1
    rotation = (vh.T * signs) @ u.T
    denominator = float(np.sum(w[:, None] * x ** 2))
    scale = float(np.dot(singular, signs) / denominator)
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("estimated similarity scale is degenerate")
    translation = c_target - scale * (rotation @ c_source)
    return scale, rotation, translation


def _weighted_median(values, weights):
    order = np.argsort(values)
    return float(values[order[np.searchsorted(np.cumsum(weights[order]), .5 * weights.sum())]])


def fit_similarity(source, target, weights=None, *, robust: bool = True,
                   max_iterations: int = 30, hypotheses: int = 48,
                   seed: int = 0) -> SimilarityFit:
    """Fit target = scale * source @ rotation.T + translation.

    Weighted Umeyama fitting initializes a deterministic sampled-triple
    median-residual search, followed by Tukey IRLS when ``robust=True``.
    Complexity is O(N * (hypotheses + max_iterations)), with 3x3 SVDs.
    Robustness assumes a majority of correctly corresponding stable points;
    no finite-sample guarantee is made when that assumption is violated.
    """
    x, y = _points(source, "source"), _points(target, "target")
    if y.shape != x.shape:
        raise ValueError("source and target shapes must agree")
    base = np.ones(len(x)) if weights is None else np.asarray(weights, float)
    if base.shape != (len(x),) or not np.isfinite(base).all() or (base < 0).any():
        raise ValueError("weights must be finite nonnegative (N,)")
    if not isinstance(robust, (bool, np.bool_)):
        raise ValueError("robust must be boolean")
    if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1 or not isinstance(hypotheses, (int, np.integer)) or hypotheses < 0:
        raise ValueError("invalid robust iteration limits")
    scale, rotation, translation = _weighted_fit(x, y, base)

    def residual(s, r, t):
        return np.linalg.norm(s * x @ r.T + t - y, axis=1)

    used = base.copy()
    if robust:
        rng = np.random.default_rng(seed)
        best = _weighted_median(residual(scale, rotation, translation), base)
        positive = np.flatnonzero(base > 0)
        probabilities = base[positive] / base[positive].sum()
        for _ in range(hypotheses):
            ids = rng.choice(positive, 3, replace=False, p=probabilities)
            try:
                s, r, t = _weighted_fit(x[ids], y[ids], base[ids])
            except ValueError:
                continue
            score = _weighted_median(residual(s, r, t), base)
            if score < best:
                best, scale, rotation, translation = score, s, r, t
        geometric_floor = max(1., np.linalg.norm(np.ptp(y, axis=0))) * 1e-10
        for _ in range(max_iterations):
            errors = residual(scale, rotation, translation)
            sigma = max(_weighted_median(errors, base) / .67448975, geometric_floor)
            ratio = errors / (4.685 * sigma)
            used = base * np.maximum(0., 1. - ratio ** 2) ** 2
            used[ratio >= 1.] = 0.
            try:
                s, r, t = _weighted_fit(x, y, used)
            except ValueError as exc:
                raise ValueError("robust consensus lacks three noncollinear stable correspondences") from exc
            change = abs(scale - s) + np.linalg.norm(rotation - r) + np.linalg.norm(translation - t)
            scale, rotation, translation = s, r, t
            if change <= 1e-10:
                break
    return SimilarityFit(scale, rotation, translation, used, residual(scale, rotation, translation))


def _sequence(rest, trajectories, anchor_frame):
    p = _points(rest, "rest")
    q = np.asarray(trajectories, float)
    if q.ndim != 3 or q.shape[1:] != p.shape or len(q) < 1 or not np.isfinite(q).all():
        raise ValueError("trajectories must be finite (T,V,3)")
    if not isinstance(anchor_frame, (int, np.integer)) or not 0 <= anchor_frame < len(q):
        raise ValueError("anchor_frame outside trajectory")
    return p, q


def stabilize_scale(rest, trajectories, stable_mask, *, anchor_frame: int = 0,
                    confidence=None, camera_convention: str = "fixed_metric_world",
                    robust: bool = True) -> ScaleDecomposition:
    """Remove relative global scale drift estimated on explicit stable vertices.

    ``stable_mask`` must be a boolean (V,) array; it is never inferred by
    assuming an entire body is rigid. Confidence optionally has shape (T,V).
    The requested anchor is unchanged exactly. Its estimated size is the
    reference size. Returned ``local_residuals`` are shape (T,V,3) in rest
    axes after removing similarity; they retain scale-normalized articulation.
    Camera convention must be fixed_metric_world or camera_compensated_world.
    """
    p, q = _sequence(rest, trajectories, anchor_frame)
    if stable_mask is None:
        raise ValueError("a verified stable_mask is required; unknown stability must abstain")
    mask = np.asarray(stable_mask)
    if mask.dtype.kind != "b" or mask.shape != (len(p),) or np.count_nonzero(mask) < 3:
        raise ValueError("stable_mask must be boolean (V,) with at least three stable points")
    if camera_convention not in ("fixed_metric_world", "camera_compensated_world"):
        raise ValueError("unknown camera/scale convention: scale is not identifiable")
    conf = np.ones(q.shape[:2]) if confidence is None else np.asarray(confidence, float)
    if conf.shape != q.shape[:2] or not np.isfinite(conf).all() or ((conf < 0) | (conf > 1)).any():
        raise ValueError("confidence must be finite (T,V) in [0,1]")
    reference_center = p[mask].mean(0)
    fits = [fit_similarity(p[mask], frame[mask], conf[t, mask], robust=robust)
            for t, frame in enumerate(q)]
    scales = np.array([fit.scale for fit in fits])
    rotations = np.stack([fit.rotation for fit in fits])
    translations = np.stack([fit.translation for fit in fits])
    centers = np.stack([fit.scale * fit.rotation @ reference_center + fit.translation for fit in fits])
    corrected = centers[:, None] + (q - centers[:, None]) * (scales[anchor_frame] / scales)[:, None, None]
    corrected[anchor_frame] = q[anchor_frame]
    residuals = np.stack([(frame - centers[t]) @ rotations[t] / scales[t] - (p - reference_center)
                          for t, frame in enumerate(q)])
    used = np.zeros(q.shape[:2])
    used[:, mask] = np.stack([fit.weights for fit in fits])
    return ScaleDecomposition(corrected, scales, rotations, translations, centers,
                              residuals, used, camera_convention)


def scale_controls(rest, trajectories, *, anchor_frame: int = 0) -> dict[str, np.ndarray]:
    """Original, frame-bbox diagonal stabilization, and global least-squares Sim(3).

    These intentionally naive controls use all vertices, including articulated
    ones, and may confound size with articulation or orientation. Their frame
    centroids and exact anchor are retained. No ground truth chooses a control.
    """
    p, q = _sequence(rest, trajectories, anchor_frame)
    sizes = np.linalg.norm(np.ptp(q, axis=1), axis=1)
    if (sizes <= 1e-12).any():
        raise ValueError("bbox control cannot normalize zero-size geometry")
    centers = q.mean(axis=1, keepdims=True)
    bbox = centers + (q - centers) * (sizes[anchor_frame] / sizes)[:, None, None]
    bbox[anchor_frame] = q[anchor_frame]
    global_fit = stabilize_scale(p, q, np.ones(len(p), bool), anchor_frame=anchor_frame, robust=False)
    return {"original": q.copy(), "bbox": bbox, "global_sim3": global_fit.corrected}


def demo() -> dict:
    """Controlled scale drift with separate articulation and true pose."""
    import time
    start = time.perf_counter()
    rest = np.array([[-1., -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [3, 0, 0]])
    deformed = rest.copy()
    deformed[-1] += [0., 2., 1.]
    rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    truth = deformed @ rotation.T + [5, -2, 3]
    tracks = np.stack([rest, 1.4 * deformed @ rotation.T + [5, -2, 3]])
    stable = stabilize_scale(rest, tracks, np.array([True, True, True, True, False]))
    controls = scale_controls(rest, tracks)
    controls["stable_region"] = stable.corrected
    return {"method": 5, "validation": "constructed numerical controls only",
            "estimated_scales": stable.scales.tolist(),
            "scores": {name: {"constructed_error": float(np.linalg.norm(value[1] - truth)),
                               "anchor_max_error": float(np.max(np.abs(value[0] - rest)))}
                       for name, value in controls.items()},
            "elapsed_seconds": time.perf_counter() - start,
            "missing_for_natural_validation": ["verified size-stable region", "metric camera convention", "natural scale-drift census"]}
