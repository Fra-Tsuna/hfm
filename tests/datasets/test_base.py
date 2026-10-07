import numpy as np
import pytest

from src.datasets.base import AnnotatedScene, SceneDataset


def test_interface_cannot_be_used_directly():
    with pytest.raises(TypeError):
        SceneDataset()


def test_an_adapter_must_implement_every_method():
    class OnlyListsScenes(SceneDataset):
        def scene_ids(self, split):
            return []

    with pytest.raises(TypeError):
        OnlyListsScenes()


def test_extra_is_not_shared_between_scenes():
    a = AnnotatedScene("a", [], [], np.zeros((0, 3)), [])
    b = AnnotatedScene("b", [], [], np.zeros((0, 3)), [])
    a.extra["key"] = 1
    assert b.extra == {}


def test_toy_dataset_lists_its_scenes(toy_dataset):
    assert toy_dataset.scene_ids("train") == ["two_rooms"]
    assert toy_dataset.scene_ids("val") == []


def test_toy_scene_is_consistent(toy_dataset):
    """The hand-made scene used by the pipeline tests must itself be sound."""
    scene = toy_dataset.load_scene("two_rooms")
    region_ids = {r.region_id for r in scene.regions}
    assert {o.region_id for o in scene.objects} <= region_ids
    point_sets = [r.walkable_points for r in scene.regions] + [r.surface_points for r in scene.regions]
    point_sets += [o.points for o in scene.objects] + [d.points for d in scene.doors] + [scene.wall_points]
    for points in point_sets:
        assert points.ndim == 2 and points.shape[1] == 3 and len(points) > 0
        assert np.isfinite(points).all()


def test_toy_rooms_are_bounded_in_3d(toy_dataset):
    """surface_points bound the room: a real extent along x, y and z, enclosing the walkable floor."""
    for region in toy_dataset.load_scene("two_rooms").regions:
        lo, hi = region.surface_points.min(axis=0), region.surface_points.max(axis=0)
        assert (hi - lo > 0.5).all(), (region.region_id, hi - lo)
        assert (region.walkable_points >= lo).all() and (region.walkable_points <= hi).all()


def test_toy_dataset_rejects_unknown_scenes(toy_dataset):
    with pytest.raises(KeyError):
        toy_dataset.load_scene("not_a_scene")
