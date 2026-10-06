"""C02: weighted projection that protects a declared action subspace.

This operator is a mathematical control. It does not estimate ground-truth
motion and it does not claim that the declared constraints represent all
motion in a generated 4D sequence.
"""
from __future__ import annotations

import numpy as np


def project_protected_step(
    d: np.ndarray,
    C: np.ndarray,
    W: np.ndarray | None = None,
    pins: list[int] | tuple[int, ...] | None = None,
) -> np.ndarray:
    """Project ``d`` into the null space of ``C`` under positive metric ``W``.

    Pinned coordinates are held at zero. The solve uses a pseudoinverse of the
    constraint Gram matrix, so redundant rows are valid and deterministic.
    Inputs are copied and the returned vector is always a fresh float array.
    """
    d = np.asarray(d, dtype=float)
    C = np.asarray(C, dtype=float)
    if d.ndim != 1:
        raise ValueError("d must be a one-dimensional vector")
    if C.ndim != 2 or C.shape[1] != d.size:
        raise ValueError("C must be a two-dimensional matrix with len(d) columns")
    if not np.all(np.isfinite(d)) or not np.all(np.isfinite(C)):
        raise ValueError("d and C must be finite")
    if W is None:
        W = np.eye(d.size)
    W = np.asarray(W, dtype=float)
    if W.shape != (d.size, d.size) or not np.all(np.isfinite(W)):
        raise ValueError("W must be a finite square matrix matching d")
    if not np.allclose(W, W.T, rtol=1e-10, atol=1e-12):
        raise ValueError("W must be symmetric")
    try:
        np.linalg.cholesky(W)
    except np.linalg.LinAlgError as exc:
        raise ValueError("W must be positive definite") from exc
    pinned = np.array(sorted(set(pins or ())), dtype=int)
    if np.any(pinned < 0) or np.any(pinned >= d.size):
        raise ValueError("pins must contain valid coordinate indices")
    free = np.ones(d.size, dtype=bool)
    free[pinned] = False
    result = np.zeros_like(d)
    if not np.any(free):
        return result
    # Eliminate pinned variables before forming the KKT Gram system. This
    # preserves the declared pins even when W couples pinned/free coordinates.
    free_indices = np.flatnonzero(free)
    W_ff = W[np.ix_(free_indices, free_indices)]
    C_f = C[:, free_indices]
    d_f = d[free_indices]
    if C_f.size:
        rhs = C_f @ d_f
        gram = C_f @ np.linalg.solve(W_ff, C_f.T)
        correction = np.linalg.solve(W_ff, C_f.T @ np.linalg.pinv(gram) @ rhs)
        d_f = d_f - correction
    result[free_indices] = d_f
    return result
