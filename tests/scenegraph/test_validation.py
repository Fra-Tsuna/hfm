import numpy as np
import pytest

from src.scenegraph.graph import GEOM_DIM, EdgeType, NodeType, SceneGraph, geometry_mask
from src.scenegraph.vocabulary import vocab_sha1
from tests.validation import check_containment, validate, validate_provenance

MAP_SHA1 = "0123456789abcdef0123456789abcdef01234567"   # stands in for a dataset map fingerprint


@pytest.fixture
def processed(toy):
    """The toy graph as a dataset builder would produce it: with its provenance stamped."""
    toy.raw_metadata["canonical_vocab_sha1"] = vocab_sha1()
    toy.raw_metadata["dataset_map_sha1"] = MAP_SHA1
    return toy


def test_graph_with_the_expected_provenance_passes(processed):
    assert validate_provenance(processed, vocab_sha1(), MAP_SHA1) == []


@pytest.mark.parametrize("key", ["canonical_vocab_sha1", "dataset_map_sha1"])
def test_missing_fingerprint_fails(processed, key):
    del processed.raw_metadata[key]
    problems = validate_provenance(processed, vocab_sha1(), MAP_SHA1)
    assert problems == [f"toy: raw_metadata has no {key}"]


def test_graph_from_another_vocabulary_fails(processed):
    problems = validate_provenance(processed, "f" * 40, MAP_SHA1)
    assert len(problems) == 1 and "vocabulary" in problems[0]


def test_graph_from_another_dataset_map_fails(processed):
    problems = validate_provenance(processed, vocab_sha1(), "f" * 40)
    assert len(problems) == 1 and "dataset map" in problems[0]


def test_validation_does_not_rewrite_provenance(processed):
    validate_provenance(processed, "f" * 40, "f" * 40)
    assert processed.raw_metadata["canonical_vocab_sha1"] == vocab_sha1()
    assert processed.raw_metadata["dataset_map_sha1"] == MAP_SHA1


def test_valid_graph_passes(toy):
    assert validate(toy) == []


def test_graph_with_only_a_building_passes():
    node_type = np.array([0])
    graph = SceneGraph(
        scene_id="empty",
        node_type=node_type,
        category_id=np.array([0]),
        geometry=np.zeros((1, GEOM_DIM)),
        geometry_norm=np.zeros((1, GEOM_DIM)),
        geometry_mask=geometry_mask(node_type),
        edge_index=np.zeros((2, 0), dtype=np.int64),
        edge_type=np.zeros(0, dtype=np.int64),
        room_parent=np.array([-1]),
        object_parent=np.array([-1]),
        raw_metadata={"nodes": [{"raw_label": None, "dataset_category_id": None}]},
    )
    assert validate(graph) == []


# Each corruption breaks the toy graph in one way: (description, how, expected problem).
# Toy graph: 0 building, 1-2 rooms, 3-4 objects; edges 0->1, 0->2, 1->3, 2->4, 1-2.

def wrong_dtype(g):
    g.category_id = g.category_id.astype(np.int32)


def wrong_shape(g):
    g.geometry = g.geometry[:, :6]


def not_an_array(g):
    g.edge_type = g.edge_type.tolist()


def unknown_node_type(g):
    g.node_type[3] = 9


def two_buildings(g):
    g.node_type[1] = NodeType.BUILDING


def object_category_outside_vocabulary(g):
    g.category_id[3] = 999


def negative_category(g):
    g.category_id[1] = -1


def mask_mismatch(g):
    g.geometry_mask[1, 6] = True


def nan_geometry(g):
    g.geometry[3, 0] = np.nan


def inf_normalized_geometry(g):
    g.geometry_norm[3, 0] = np.inf


def value_in_masked_slot(g):
    g.geometry[1, 6] = 0.5   # rooms have no theta


def zero_size_box(g):
    g.geometry[4, 3] = 0.0


def unknown_edge_type(g):
    g.edge_type[0] = 7


def edge_to_missing_node(g):
    g.edge_index[1, 2] = 42


def edge_between_wrong_types(g):
    g.edge_type[2] = EdgeType.ROOM_CONNECTS_ROOM   # 1->3 is room->object, not room-room


def self_loop(g):
    g.edge_index[:, 4] = [1, 1]


def undirected_edge_stored_backwards(g):
    g.edge_index[:, 4] = [2, 1]


def duplicate_edge(g):
    g.edge_index = np.concatenate([g.edge_index, g.edge_index[:, :1]], axis=1)
    g.edge_type = np.concatenate([g.edge_type, g.edge_type[:1]])


def object_with_two_rooms(g):
    g.edge_index = np.concatenate([g.edge_index, [[2], [3]]], axis=1)
    g.edge_type = np.concatenate([g.edge_type, [EdgeType.ROOM_CONTAINS_OBJECT]])


def object_without_room(g):
    g.edge_index = g.edge_index[:, [0, 1, 3, 4]]
    g.edge_type = g.edge_type[[0, 1, 3, 4]]
    g.object_parent[3] = -1


def building_inside_a_room(g):
    g.edge_index = np.concatenate([g.edge_index, [[1], [0]]], axis=1)
    g.edge_type = np.concatenate([g.edge_type, [EdgeType.BUILDING_CONTAINS_ROOM]])


def parent_array_disagrees(g):
    g.object_parent[3] = 2


def room_parent_array_disagrees(g):
    g.room_parent[1] = -1


def node_metadata_missing(g):
    g.raw_metadata["nodes"] = g.raw_metadata["nodes"][:-1]


def node_metadata_key_missing(g):
    del g.raw_metadata["nodes"][3]["dataset_category_id"]


def raw_label_not_a_string(g):
    g.raw_metadata["nodes"][3]["raw_label"] = 12


def dataset_category_id_a_bool(g):
    g.raw_metadata["nodes"][3]["dataset_category_id"] = True


CORRUPTIONS = [
    (wrong_dtype, "category_id has dtype int32"),
    (wrong_shape, "geometry has shape (5, 6)"),
    (not_an_array, "edge_type is a list"),
    (unknown_node_type, "unknown node types [9]"),
    (two_buildings, "2 building nodes"),
    (object_category_outside_vocabulary, "OBJECT category ids [999]"),
    (negative_category, "ROOM category ids [-1]"),
    (mask_mismatch, "geometry_mask does not match"),
    (nan_geometry, "geometry has NaN or inf"),
    (inf_normalized_geometry, "geometry_norm has NaN or inf"),
    (value_in_masked_slot, "non-zero values in masked slots"),
    (zero_size_box, "size <= 0"),
    (unknown_edge_type, "unknown edge types [7]"),
    (edge_to_missing_node, "nodes [42] that do not exist"),
    (edge_between_wrong_types, "connect the wrong node types"),
    (self_loop, "self loops"),
    (undirected_edge_stored_backwards, "source < target"),
    (duplicate_edge, "duplicate edges"),
    (object_with_two_rooms, "OBJECT nodes are not contained by exactly one room"),
    (object_without_room, "OBJECT nodes are not contained by exactly one room"),
    (building_inside_a_room, "the building is contained by another node"),
    (parent_array_disagrees, "object_parent disagrees"),
    (room_parent_array_disagrees, "room_parent disagrees"),
    (node_metadata_missing, "one entry per node"),
    (node_metadata_key_missing, "node 3 metadata has no dataset_category_id"),
    (raw_label_not_a_string, "node 3 raw_label must be"),
    (dataset_category_id_a_bool, "node 3 dataset_category_id must be"),
]


@pytest.mark.parametrize("corrupt, expected", CORRUPTIONS, ids=[c.__name__ for c, _ in CORRUPTIONS])
def test_corrupted_graph_is_reported(toy, corrupt, expected):
    corrupt(toy)
    problems = validate(toy)
    assert any(expected in p for p in problems), problems


def test_cycle_is_reported(toy):
    """With typed edges a cycle cannot come from valid edges, so build the parent loop directly."""
    toy.edge_index[:, 0] = [2, 1]   # room 2 contains room 1 ...
    toy.edge_index[:, 1] = [1, 2]   # ... and room 1 contains room 2
    problems = check_containment(toy)
    assert any("cycle" in p for p in problems), problems
