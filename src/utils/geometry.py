"""Plain geometry on numpy arrays. Nothing here knows about scene graphs or datasets."""

from __future__ import annotations

import numpy as np


def triangle_areas(triangles: np.ndarray) -> np.ndarray:
    """[F, 3, 3] triangles -> [F] areas."""
    edge_1 = triangles[:, 1] - triangles[:, 0]
    edge_2 = triangles[:, 2] - triangles[:, 0]
    return 0.5 * np.linalg.norm(np.cross(edge_1, edge_2), axis=1)


def sample_surface(vertices: np.ndarray, faces: np.ndarray, spacing: float, rng: np.random.Generator) -> np.ndarray:
    """Points on a triangle mesh, about one per spacing x spacing of area, plus every vertex used.

    vertices [V, 3], faces [F, 3] -> points [P, 3]. Every triangle gets at least one point, so
    small parts of a mesh are never lost.
    """
    if not spacing > 0:  # also catches NaN
        raise ValueError(f"spacing must be positive, got {spacing}")
    if len(faces) == 0:
        return np.zeros((0, 3))
    triangles = vertices[faces]
    counts = np.ceil(triangle_areas(triangles) / spacing**2).astype(np.int64)
    counts = np.maximum(counts, 1)
    which = np.repeat(np.arange(len(triangles)), counts)
    # uniform points in a triangle: fold the unit square onto the triangle
    u, v = rng.random(len(which)), rng.random(len(which))
    outside = u + v > 1
    u[outside], v[outside] = 1 - u[outside], 1 - v[outside]
    t = triangles[which]
    points = t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])
    return np.concatenate([points, vertices[np.unique(faces)]])
