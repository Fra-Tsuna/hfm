import numpy as np
import pytest

from src.datasets.base import AnnotatedScene, SceneDataset, label_map_sha1
from src.datasets.hm3dsem.dataset import HM3DSemDataset
from src.scenegraph.graph import NodeType
from src.scenegraph.vocabulary import CATEGORIES
from tests.datasets.conftest import ToyDataset


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


def check_label_tables(dataset: SceneDataset) -> list[str]:
    """Problems with a dataset's label tables; every adapter's tables must pass."""
    problems = []
    for table, node_type in ((dataset.ROOM_CATEGORIES, NodeType.ROOM), (dataset.OBJECT_CATEGORIES, NodeType.OBJECT)):
        for raw, category in table.items():
            if category not in CATEGORIES[node_type]:
                problems.append(f"{raw!r} -> {category!r} is not a canonical {node_type.name.lower()} category")
    both = set(dataset.OBJECT_CATEGORIES) & set(dataset.NOT_OBJECTS)
    if both:
        problems.append(f"labels both mapped and dropped: {sorted(both)}")
    for raw, reason in dataset.NOT_OBJECTS.items():
        if not (isinstance(reason, str) and reason):
            problems.append(f"{raw!r} has no reason for not being an object")
    return problems


ADAPTERS = [ToyDataset(), HM3DSemDataset(root="unused")]


@pytest.mark.parametrize("dataset", ADAPTERS, ids=lambda d: type(d).__name__)
def test_every_adapter_has_valid_label_tables(dataset):
    for table in ("ROOM_CATEGORIES", "OBJECT_CATEGORIES", "NOT_OBJECTS"):
        assert isinstance(getattr(dataset, table, None), dict), f"{type(dataset).__name__} has no {table}"
    assert check_label_tables(dataset) == []
    assert len(label_map_sha1(dataset)) == 40


def test_label_table_problems_are_found(toy_dataset):
    class Broken(type(toy_dataset)):
        ROOM_CATEGORIES = {"bedroom": "sleeping_room"}
        OBJECT_CATEGORIES = {"bed": "bed", "tap": "lighting"}
        NOT_OBJECTS = {"tap": ""}

    problems = check_label_tables(Broken())
    assert any("sleeping_room" in p for p in problems)
    assert any("both mapped and dropped" in p for p in problems)
    assert any("no reason" in p for p in problems)


def test_label_map_fingerprint(toy_dataset):
    sha = label_map_sha1(toy_dataset)
    assert len(sha) == 40 and sha == label_map_sha1(type(toy_dataset)())

    class Reordered(type(toy_dataset)):
        OBJECT_CATEGORIES = {"toilet": "toilet", "bed": "bed"}   # same content, other order

    class Changed(type(toy_dataset)):
        OBJECT_CATEGORIES = {"bed": "bed", "toilet": "sink"}

    assert label_map_sha1(Reordered()) == sha
    assert label_map_sha1(Changed()) != sha
