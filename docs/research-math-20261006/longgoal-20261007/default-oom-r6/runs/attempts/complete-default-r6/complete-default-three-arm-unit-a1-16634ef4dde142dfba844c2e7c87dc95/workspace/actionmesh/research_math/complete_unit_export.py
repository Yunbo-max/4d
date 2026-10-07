"""Lossless conversion of official ActionMesh deformation exports.

Official save_deformation writes (-z, x, y), whereas GLB and the census scorer
use the original mesh coordinates. This adapter restores (x, y, z). It grants
no model, benchmark or scientific qualification.
"""
import numpy as np


def restore_vertices(exported):
    array = np.asarray(exported)
    if (array.ndim != 3 or array.shape[0] != 16 or array.shape[-1] != 3
            or array.shape[1] < 3 or not np.issubdtype(array.dtype, np.floating)
            or not np.isfinite(array).all()):
        raise ValueError('Require 16 finite floating-point full mesh frames')
    restored = array[:, :, [1, 2, 0]].copy()
    restored[:, :, 2] *= -1
    return restored


def validate_sequence(vertices, faces):
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    if (vertices.ndim != 3 or vertices.shape[0] != 16 or vertices.shape[-1] != 3
            or vertices.shape[1] < 3 or not np.issubdtype(vertices.dtype, np.floating)
            or not np.isfinite(vertices).all()):
        raise ValueError('Require 16 finite floating-point full mesh frames')
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or not np.issubdtype(faces.dtype, np.integer)
            or faces.min() < 0 or faces.max() >= vertices.shape[1]):
        raise ValueError('Require nonempty in-range integer triangle topology')
    return dict(vertices=vertices.copy(), faces=faces.copy(),
                frame_indices=np.arange(16, dtype=np.int64),
                timesteps=np.arange(16, dtype=np.float32),
                query_vertex_ids=np.arange(vertices.shape[1], dtype=np.int64))
