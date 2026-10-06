import re

import pytest

from src.scenegraph import vocabulary
from src.scenegraph.graph import NodeType
from src.scenegraph.vocabulary import (
    CATEGORIES,
    OTHER,
    UNKNOWN,
    VOCAB_VERSION,
    category_id,
    category_name,
    num_categories,
    vocab_sha1,
)

# Every released vocabulary, oldest first. Changing the vocabulary means bumping VOCAB_VERSION
# and adding its snapshot here; ids are positions, so each release may only append to the previous.
V0 = {
    NodeType.BUILDING: ("building",),
    NodeType.ROOM: (
        "unknown", "other", "bathroom", "bedroom", "kitchen", "living_room", "dining_room", "office",
        "hallway", "stairs", "closet", "laundry_room", "garage", "outdoor",
    ),
    NodeType.OBJECT: (
        "unknown", "other", "chair", "sofa", "bed", "table", "cabinet", "chest_of_drawers", "shelving",
        "counter", "stool", "cushion", "picture", "mirror", "curtain", "blinds", "plant", "lighting",
        "sink", "toilet", "bathtub", "shower", "towel", "tv_monitor", "appliance", "fireplace", "clothes",
    ),
}
RELEASES = {"v0": V0}


def test_every_node_type_has_a_namespace():
    assert set(CATEGORIES) == set(NodeType)


@pytest.mark.parametrize("node_type", [NodeType.ROOM, NodeType.OBJECT])
def test_unknown_and_other_come_first(node_type):
    assert category_id(node_type, UNKNOWN) == 0
    assert category_id(node_type, OTHER) == 1


def test_building_has_a_single_category():
    assert CATEGORIES[NodeType.BUILDING] == ("building",)


@pytest.mark.parametrize("node_type", list(NodeType))
def test_names_are_unique_snake_case(node_type):
    names = CATEGORIES[node_type]
    assert len(set(names)) == len(names)
    assert all(re.fullmatch(r"[a-z]+(_[a-z]+)*", n) for n in names), names


@pytest.mark.parametrize("node_type", list(NodeType))
def test_id_and_name_are_inverse(node_type):
    assert num_categories(node_type) == len(CATEGORIES[node_type])
    for i, name in enumerate(CATEGORIES[node_type]):
        assert category_id(node_type, name) == i
        assert category_name(node_type, i) == name


def test_unknown_name_raises():
    with pytest.raises(ValueError):
        category_id(NodeType.OBJECT, "spaceship")


@pytest.mark.parametrize("idx", [-1, -27, 27, 100])
def test_out_of_range_id_raises(idx):
    with pytest.raises(ValueError, match="out of range"):
        category_name(NodeType.OBJECT, idx)


def test_current_vocabulary_is_a_released_snapshot():
    assert VOCAB_VERSION in RELEASES, f"{VOCAB_VERSION} has no snapshot in RELEASES"
    for node_type in NodeType:
        assert CATEGORIES[node_type] == RELEASES[VOCAB_VERSION][node_type], (
            f"{node_type.name} categories differ from the {VOCAB_VERSION} snapshot: bump VOCAB_VERSION")


def test_each_release_only_appends_to_the_previous():
    versions = list(RELEASES)
    for older, newer in zip(versions, versions[1:]):
        for node_type in NodeType:
            old, new = RELEASES[older][node_type], RELEASES[newer][node_type]
            assert new[:len(old)] == old, f"{newer} changes existing {node_type.name} ids of {older}"


def test_sha1_is_stable_and_tracks_content(monkeypatch):
    sha = vocab_sha1()
    assert re.fullmatch(r"[0-9a-f]{40}", sha)
    assert vocab_sha1() == sha
    monkeypatch.setitem(vocabulary.CATEGORIES, NodeType.ROOM, (*CATEGORIES[NodeType.ROOM], "attic"))
    assert vocab_sha1() != sha
