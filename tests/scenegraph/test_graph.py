import copy

import numpy as np
import pytest

from src.scenegraph.graph import (
    EDGE_ENDPOINTS,
    GEOM_DIM,
    GEOM_FIELDS,
    POS,
    SIZE,
    THETA,
    EdgeType,
    NodeType,
    SceneGraph,
    geometry_mask,
)


@pytest.fixture
def toy():
    """A building with two connected rooms, one object in each.

    node:  0 building   1 room   2 room   3 object (in room 1)   4 object (in room 2)
    """
    node_type = np.array([0, 1, 1, 2, 2])
    geometry = np.zeros((5, GEOM_DIM))
    geometry[1:, POS] = np.arange(12).reshape(4, 3)    # distinct centers, to see where nodes move
    geometry[1:, SIZE] = 1.0
    geometry[3:, THETA] = [0.1, 0.2]
    return SceneGraph(
        scene_id="toy",
        node_type=node_type,
        category_id=np.array([0, 3, 2, 7, 4]),
        geometry=geometry,
        geometry_norm=geometry / 10,
        geometry_mask=geometry_mask(node_type),
        edge_index=np.array([[0, 0, 1, 2, 1],
                             [1, 2, 3, 4, 2]]),
        edge_type=np.array([0, 0, 1, 1, 2]),
        room_parent=np.array([-1, 0, 0, -1, -1]),
        object_parent=np.array([-1, -1, -1, 1, 2]),
        raw_metadata={"nodes": [{"id": f"n{i}"} for i in range(5)]},
    )


def test_geometry_layout_covers_every_field():
    assert GEOM_FIELDS[POS] == ("px", "py", "pz")
    assert GEOM_FIELDS[SIZE] == ("sx", "sy", "sz")
    assert GEOM_FIELDS[THETA] == "theta"


def test_geometry_mask_rows_follow_node_type():
    mask = geometry_mask(np.array([0, 1, 2, 1]))
    assert mask.shape == (4, GEOM_DIM)
    assert mask[0].tolist() == [False] * 7                    # building: nothing
    assert mask[1].tolist() == [True] * 6 + [False]           # room: no theta
    assert mask[2].tolist() == [True] * 7                     # object: everything
    assert mask[3].tolist() == mask[1].tolist()


def test_geometry_mask_of_no_nodes():
    assert geometry_mask(np.array([], dtype=np.int64)).shape == (0, GEOM_DIM)


def test_geometry_mask_rejects_unknown_node_types():
    with pytest.raises(ValueError, match="unknown node types"):
        geometry_mask(np.array([0, 99]))


def test_edge_endpoints_cover_every_edge_type():
    assert set(EDGE_ENDPOINTS) == set(EdgeType)


def test_counts_and_selection(toy):
    assert toy.num_nodes == 5 and toy.num_edges == 5
    assert toy.nodes_of(NodeType.ROOM).tolist() == [1, 2]
    assert toy.nodes_of(NodeType.OBJECT).tolist() == [3, 4]
    assert toy.edges_of(EdgeType.ROOM_CONTAINS_OBJECT).tolist() == [[1, 2], [3, 4]]
    assert toy.edges_of(EdgeType.ROOM_CONNECTS_ROOM).tolist() == [[1], [2]]


def test_repr_is_a_short_summary(toy):
    assert repr(toy) == "SceneGraph(toy: 5 nodes (1 building, 2 room, 2 object), 5 edges)"


def test_reorder_moves_every_per_node_array(toy):
    order = [4, 3, 2, 1, 0]
    r = toy.reorder_nodes(order)
    for name in ("node_type", "category_id", "geometry", "geometry_norm", "geometry_mask"):
        assert np.array_equal(getattr(r, name), getattr(toy, name)[order]), name
    assert [n["id"] for n in r.raw_metadata["nodes"]] == ["n4", "n3", "n2", "n1", "n0"]


def test_reorder_relabels_parents_and_edges(toy):
    r = toy.reorder_nodes([4, 3, 2, 1, 0])          # old i -> new 4 - i
    assert r.object_parent.tolist() == [2, 3, -1, -1, -1]
    assert r.room_parent.tolist() == [-1, -1, 4, 4, -1]
    assert r.edge_index.tolist() == [[4, 4, 3, 2, 2],
                                     [3, 2, 1, 0, 3]]   # last edge: undirected, kept as source < target
    assert r.edge_type.tolist() == toy.edge_type.tolist()


@pytest.mark.parametrize("seed", range(5))
def test_reorder_keeps_the_same_graph(toy, seed):
    """Every edge and parent still connects the same pair of nodes, identified by their metadata id.

    Edges and parents are compared as sorted lists, not sets, so a lost or duplicated edge is caught.
    """
    order = np.random.default_rng(seed).permutation(toy.num_nodes)
    r = toy.reorder_nodes(order)
    ids_old = [n["id"] for n in toy.raw_metadata["nodes"]]
    ids_new = [n["id"] for n in r.raw_metadata["nodes"]]

    def edges(g, ids):
        assert g.edge_index.shape == (2, len(g.edge_type))
        out = []
        for t, a, b in zip(g.edge_type, *g.edge_index):
            ends = (ids[a], ids[b])
            out.append((t, *(sorted(ends) if t == EdgeType.ROOM_CONNECTS_ROOM else ends)))
        return sorted(out)

    def parents(g, ids):
        parent = np.maximum(g.object_parent, g.room_parent)
        return sorted((ids[i], ids[p]) for i, p in enumerate(parent) if p >= 0)

    assert edges(r, ids_new) == edges(toy, ids_old)
    assert parents(r, ids_new) == parents(toy, ids_old)


def test_reorder_then_inverse_gives_back_the_original(toy):
    order = np.array([2, 0, 4, 1, 3])
    back = toy.reorder_nodes(order).reorder_nodes(np.argsort(order))
    for name in ("node_type", "category_id", "geometry", "geometry_norm", "geometry_mask",
                 "edge_index", "edge_type", "room_parent", "object_parent"):
        assert np.array_equal(getattr(back, name), getattr(toy, name)), name
    assert back.raw_metadata == toy.raw_metadata


def test_reorder_does_not_modify_the_original(toy):
    before = copy.deepcopy(toy)
    toy.reorder_nodes([4, 3, 2, 1, 0])
    for name in ("node_type", "category_id", "geometry", "geometry_norm", "geometry_mask",
                 "edge_index", "edge_type", "room_parent", "object_parent"):
        assert np.array_equal(getattr(toy, name), getattr(before, name)), name
    assert toy.raw_metadata == before.raw_metadata


def test_reordered_graph_shares_no_metadata_with_the_original(toy):
    r = toy.reorder_nodes([4, 3, 2, 1, 0])
    r.raw_metadata["nodes"][0]["id"] = "changed"
    assert all(n["id"] != "changed" for n in toy.raw_metadata["nodes"])


@pytest.mark.parametrize("order", [
    [0, 1, 2, 3, 3],        # a node twice, another missing
    [0, 1, 2, 3],           # too short
    [0, 1, 2, 3, 4, 5],     # too long
    [0, 1, 2, 3, 5],        # index out of range
    [-1, 1, 2, 3, 4],       # negative index
])
def test_reorder_rejects_orders_that_are_not_permutations(toy, order):
    with pytest.raises(ValueError, match="permutation"):
        toy.reorder_nodes(order)
