"""Saving and loading a SceneGraph.

.npz is the canonical file: compressed and exact. .json holds the same content in a form you
can read by hand; it is exact too, because Python writes floats with all their digits, and
loading gives every array back its dtype and shape. The shapes matter for JSON, which turns
an empty array into []: an empty edge_index must come back as (2, 0), not (0,).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.scenegraph.graph import GEOM_DIM, SceneGraph


def save_npz(graph: SceneGraph, path) -> None:
    """Write the graph to a compressed .npz file, the canonical format.

    Every field is stored under its own name; raw_metadata is stored as one JSON string, so it
    must be JSON-serializable and free of NaN and inf.
    """
    np.savez_compressed(
        path,
        scene_id=np.array(graph.scene_id),
        node_type=graph.node_type,
        category_id=graph.category_id,
        geometry=graph.geometry,
        geometry_norm=graph.geometry_norm,
        geometry_mask=graph.geometry_mask,
        edge_index=graph.edge_index,
        edge_type=graph.edge_type,
        room_parent=graph.room_parent,
        object_parent=graph.object_parent,
        # one JSON string; allow_nan=False rejects NaN and inf, which standard JSON cannot hold
        raw_metadata=np.array(json.dumps(graph.raw_metadata, allow_nan=False)),
    )


def load_npz(path) -> SceneGraph:
    """Read a graph written by save_npz, with every array in its documented dtype and shape."""
    with np.load(path, allow_pickle=False) as f:  # never execute code from a file
        return SceneGraph(
            scene_id=str(f["scene_id"]),
            node_type=np.asarray(f["node_type"], dtype=np.int64),
            category_id=np.asarray(f["category_id"], dtype=np.int64),
            geometry=np.asarray(f["geometry"], dtype=np.float64).reshape(-1, GEOM_DIM),
            geometry_norm=np.asarray(f["geometry_norm"], dtype=np.float64).reshape(-1, GEOM_DIM),
            geometry_mask=np.asarray(f["geometry_mask"], dtype=np.bool_).reshape(-1, GEOM_DIM),
            edge_index=np.asarray(f["edge_index"], dtype=np.int64).reshape(2, -1),
            edge_type=np.asarray(f["edge_type"], dtype=np.int64),
            room_parent=np.asarray(f["room_parent"], dtype=np.int64),
            object_parent=np.asarray(f["object_parent"], dtype=np.int64),
            raw_metadata=json.loads(str(f["raw_metadata"])),
        )


def save_json(graph: SceneGraph, path, indent: int = 1) -> None:
    """Write the graph to a .json file, the same content as save_npz in a form you can read by hand.

    Arrays become nested lists, one key per field. Raises ValueError on NaN or inf values.
    """
    content = {
        "scene_id": graph.scene_id,
        "node_type": graph.node_type.tolist(),
        "category_id": graph.category_id.tolist(),
        "geometry": graph.geometry.tolist(),
        "geometry_norm": graph.geometry_norm.tolist(),
        "geometry_mask": graph.geometry_mask.tolist(),
        "edge_index": graph.edge_index.tolist(),
        "edge_type": graph.edge_type.tolist(),
        "room_parent": graph.room_parent.tolist(),
        "object_parent": graph.object_parent.tolist(),
        "raw_metadata": graph.raw_metadata,
    }
    # allow_nan=False: standard JSON has no NaN or inf, so such values raise instead of
    # producing a file other JSON readers reject (a valid graph has none anyway)
    Path(path).write_text(json.dumps(content, indent=indent, allow_nan=False))


def load_json(path) -> SceneGraph:
    """Read a graph written by save_json, giving every array back its dtype and shape."""
    c = json.loads(Path(path).read_text())
    return SceneGraph(
        scene_id=c["scene_id"],
        node_type=np.asarray(c["node_type"], dtype=np.int64),
        category_id=np.asarray(c["category_id"], dtype=np.int64),
        geometry=np.asarray(c["geometry"], dtype=np.float64).reshape(-1, GEOM_DIM),
        geometry_norm=np.asarray(c["geometry_norm"], dtype=np.float64).reshape(-1, GEOM_DIM),
        geometry_mask=np.asarray(c["geometry_mask"], dtype=np.bool_).reshape(-1, GEOM_DIM),
        edge_index=np.asarray(c["edge_index"], dtype=np.int64).reshape(2, -1),
        edge_type=np.asarray(c["edge_type"], dtype=np.int64),
        room_parent=np.asarray(c["room_parent"], dtype=np.int64),
        object_parent=np.asarray(c["object_parent"], dtype=np.int64),
        raw_metadata=c["raw_metadata"],
    )
