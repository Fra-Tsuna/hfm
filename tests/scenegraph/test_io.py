import json

import numpy as np
import pytest

from src.scenegraph.graph import GEOM_DIM, SceneGraph, geometry_mask
from src.scenegraph.io import load_json, load_npz, save_json, save_npz

FORMATS = {
    "npz": (save_npz, load_npz),
    "json": (save_json, load_json),
}


@pytest.fixture(params=list(FORMATS))
def round_trip(request, tmp_path):
    """Save a graph and load it back, once per file format."""
    save, load = FORMATS[request.param]

    def run(graph):
        path = tmp_path / f"graph.{request.param}"
        save(graph, path)
        return load(path)

    return run


def assert_identical(a: SceneGraph, b: SceneGraph):
    """Same values, dtypes and shapes in every field."""
    def same(x, y):
        return x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y)

    assert a.scene_id == b.scene_id
    assert same(a.node_type, b.node_type)
    assert same(a.category_id, b.category_id)
    assert same(a.geometry, b.geometry)
    assert same(a.geometry_norm, b.geometry_norm)
    assert same(a.geometry_mask, b.geometry_mask)
    assert same(a.edge_index, b.edge_index)
    assert same(a.edge_type, b.edge_type)
    assert same(a.room_parent, b.room_parent)
    assert same(a.object_parent, b.object_parent)
    assert a.raw_metadata == b.raw_metadata


def only_building() -> SceneGraph:
    """The smallest graph: one building, no rooms, no objects, no edges."""
    node_type = np.array([0])
    return SceneGraph(
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
        raw_metadata={},
    )


def test_round_trip_is_exact(toy, round_trip):
    toy.raw_metadata = {
        "nodes": [{"raw_label": None if i == 0 else f"label {i}", "dataset_category_id": i} for i in range(5)],
        "nested": {"list": [1, 2.5, None, True], "text": "àèéìòù"},
    }
    assert_identical(round_trip(toy), toy)


def test_graph_without_edges_keeps_its_shapes(round_trip):
    loaded = round_trip(only_building())
    assert loaded.edge_index.shape == (2, 0)
    assert loaded.edge_type.shape == (0,)
    assert_identical(loaded, only_building())


def test_floats_keep_every_digit(toy, round_trip):
    toy.geometry[1, :3] = [0.1 + 0.2, 1 / 3, 1e-300]
    assert np.array_equal(round_trip(toy).geometry, toy.geometry)


def test_loading_gives_the_documented_dtypes(toy, round_trip):
    toy.node_type = toy.node_type.astype(np.int32)
    toy.edge_index = toy.edge_index.astype(np.int16)
    toy.geometry = toy.geometry.astype(np.float32)
    loaded = round_trip(toy)
    assert loaded.node_type.dtype == np.int64
    assert loaded.edge_index.dtype == np.int64
    assert loaded.geometry.dtype == np.float64
    assert loaded.geometry_mask.dtype == np.bool_


def test_json_file_is_readable_by_hand(toy, tmp_path):
    save_json(toy, tmp_path / "graph.json")
    content = json.loads((tmp_path / "graph.json").read_text())
    assert content["scene_id"] == "toy"
    assert content["node_type"] == [0, 1, 1, 2, 2]
    assert content["edge_index"] == [[0, 0, 1, 2, 1], [1, 2, 3, 4, 2]]
    assert content["geometry_mask"][0] == [False] * GEOM_DIM


@pytest.mark.parametrize("save", [save_npz, save_json])
def test_metadata_that_is_not_json_is_rejected(toy, tmp_path, save):
    toy.raw_metadata = {"array": np.zeros(3)}
    with pytest.raises(TypeError):
        save(toy, tmp_path / "graph")

