import numpy as np
import pytest

from src.utils.geometry import sample_surface, triangle_areas

# a 2 x 1 m rectangle on the floor, made of two triangles
RECT_VERTICES = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
RECT_FACES = np.array([[0, 1, 2], [0, 2, 3]])


def test_triangle_areas():
    assert triangle_areas(RECT_VERTICES[RECT_FACES]) == pytest.approx([1.0, 1.0])


def test_sampling_density_follows_the_area():
    points = sample_surface(RECT_VERTICES, RECT_FACES, spacing=0.1, rng=np.random.default_rng(0))
    assert len(points) == 2 * 100 + 4   # 100 points per square meter, plus the 4 vertices


def test_points_lie_on_the_surface():
    points = sample_surface(RECT_VERTICES, RECT_FACES, spacing=0.05, rng=np.random.default_rng(0))
    assert (points[:, 0] >= 0).all() and (points[:, 0] <= 2).all()
    assert (points[:, 1] >= 0).all() and (points[:, 1] <= 1).all()
    assert (points[:, 2] == 0).all()


def test_every_vertex_is_kept():
    points = sample_surface(RECT_VERTICES, RECT_FACES, spacing=0.5, rng=np.random.default_rng(0))
    for v in RECT_VERTICES:
        assert (points == v).all(axis=1).any()


def test_tiny_triangles_are_not_lost():
    tiny = np.array([[0.0, 0.0, 0.0], [1e-4, 0.0, 0.0], [0.0, 1e-4, 0.0]])
    points = sample_surface(tiny, np.array([[0, 1, 2]]), spacing=0.1, rng=np.random.default_rng(0))
    assert len(points) == 1 + 3


def test_same_seed_same_points():
    a = sample_surface(RECT_VERTICES, RECT_FACES, 0.1, np.random.default_rng(3))
    b = sample_surface(RECT_VERTICES, RECT_FACES, 0.1, np.random.default_rng(3))
    assert np.array_equal(a, b)


def test_no_faces_no_points():
    assert sample_surface(RECT_VERTICES, np.zeros((0, 3), dtype=int), 0.1, np.random.default_rng(0)).shape == (0, 3)


@pytest.mark.parametrize("spacing", [0.0, -0.1, float("nan")])
def test_spacing_must_be_positive(spacing):
    with pytest.raises(ValueError, match="spacing must be positive"):
        sample_surface(RECT_VERTICES, RECT_FACES, spacing, np.random.default_rng(0))
