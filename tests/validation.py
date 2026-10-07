"""Checks on processed scene graphs, used by the data tests.

Every check returns a list of problems; an empty list means the graph passes.

validate() runs every structural check:
  every object has exactly one room parent          check_containment
  every room has exactly one building parent        check_containment
  no containment cycles                             check_containment
  all geometric values are finite                   check_nodes
  all categories belong to the vocabulary           check_nodes
  every edge references existing nodes              check_edges
Provenance is checked separately by validate_provenance, which needs the expected fingerprints.
"""

from __future__ import annotations

import numpy as np

from src.scenegraph import vocabulary
from src.scenegraph.graph import (
    EDGE_ENDPOINTS,
    GEOM_DIM,
    GRAPH_METADATA_KEYS,
    NODE_METADATA_KEYS,
    SIZE,
    EdgeType,
    NodeType,
    SceneGraph,
    geometry_mask,
)


def validate(graph: SceneGraph) -> list[str]:
    """Every structural check, in order; stops early when later checks could not run safely."""
    problems = check_arrays(graph)
    if problems:  # wrong dtypes or shapes: the other checks would index the wrong things
        return problems
    problems += check_nodes(graph)
    problems += check_edges(graph)
    edges_point_to_nodes = ((graph.edge_index >= 0) & (graph.edge_index < graph.num_nodes)).all()
    if edges_point_to_nodes:  # containment follows the edges, so it needs valid indices
        problems += check_containment(graph)
    problems += check_node_metadata(graph)
    return problems


def _check_array(graph: SceneGraph, name: str, array, dtype, shape: tuple) -> list[str]:
    if not isinstance(array, np.ndarray):
        return [f"{graph.scene_id}: {name} is a {type(array).__name__}, not a numpy array"]
    problems = []
    if array.dtype != dtype:
        problems.append(f"{graph.scene_id}: {name} has dtype {array.dtype}, expected {np.dtype(dtype)}")
    if array.shape != shape:
        problems.append(f"{graph.scene_id}: {name} has shape {array.shape}, expected {shape}")
    return problems


def check_arrays(graph: SceneGraph) -> list[str]:
    """Every array has its documented dtype, and a shape consistent with N nodes and E edges."""
    n = len(graph.node_type)
    e = len(graph.edge_type)
    problems = []
    problems += _check_array(graph, "node_type", graph.node_type, np.int64, (n,))
    problems += _check_array(graph, "category_id", graph.category_id, np.int64, (n,))
    problems += _check_array(graph, "geometry", graph.geometry, np.float64, (n, GEOM_DIM))
    problems += _check_array(graph, "geometry_norm", graph.geometry_norm, np.float64, (n, GEOM_DIM))
    problems += _check_array(graph, "geometry_mask", graph.geometry_mask, np.bool_, (n, GEOM_DIM))
    problems += _check_array(graph, "edge_index", graph.edge_index, np.int64, (2, e))
    problems += _check_array(graph, "edge_type", graph.edge_type, np.int64, (e,))
    problems += _check_array(graph, "room_parent", graph.room_parent, np.int64, (n,))
    problems += _check_array(graph, "object_parent", graph.object_parent, np.int64, (n,))
    return problems


def check_nodes(graph: SceneGraph) -> list[str]:
    """Node types, categories and geometry."""
    sid = graph.scene_id
    problems = []

    unknown = np.setdiff1d(graph.node_type, list(NodeType))
    if len(unknown):
        return [f"{sid}: unknown node types {unknown.tolist()}"]
    n_buildings = int((graph.node_type == NodeType.BUILDING).sum())
    if n_buildings != 1:
        problems.append(f"{sid}: {n_buildings} building nodes, expected exactly 1")

    for t in NodeType:
        categories = graph.category_id[graph.node_type == t]
        outside = categories[(categories < 0) | (categories >= vocabulary.num_categories(t))]
        if len(outside):
            problems.append(f"{sid}: {t.name} category ids {sorted(set(outside.tolist()))} are not in the vocabulary")

    expected_mask = geometry_mask(graph.node_type)
    if not np.array_equal(graph.geometry_mask, expected_mask):
        problems.append(f"{sid}: geometry_mask does not match the node types")
    for name, geometry in (("geometry", graph.geometry), ("geometry_norm", graph.geometry_norm)):
        if not np.isfinite(geometry).all():
            problems.append(f"{sid}: {name} has NaN or inf values")
        if (geometry[~expected_mask] != 0).any():
            problems.append(f"{sid}: {name} has non-zero values in masked slots")

    boxes = graph.node_type != NodeType.BUILDING
    if (graph.geometry[boxes][:, SIZE] <= 0).any():
        problems.append(f"{sid}: some room or object box has a size <= 0")
    return problems


def check_edges(graph: SceneGraph) -> list[str]:
    """Edges reference existing nodes of the right types, with no loops or duplicates."""
    sid = graph.scene_id
    n = graph.num_nodes
    source, target = graph.edge_index

    unknown = np.setdiff1d(graph.edge_type, list(EdgeType))
    if len(unknown):
        return [f"{sid}: unknown edge types {unknown.tolist()}"]
    missing = graph.edge_index[(graph.edge_index < 0) | (graph.edge_index >= n)]
    if len(missing):
        return [f"{sid}: edges point to nodes {sorted(set(missing.tolist()))} that do not exist ({n} nodes)"]

    problems = []
    for t, (source_type, target_type) in EDGE_ENDPOINTS.items():
        of_type = graph.edge_type == t
        wrong = (graph.node_type[source[of_type]] != source_type) | (graph.node_type[target[of_type]] != target_type)
        if wrong.any():
            problems.append(f"{sid}: {int(wrong.sum())} {t.name} edges connect the wrong node types")
    if (source == target).any():
        problems.append(f"{sid}: {int((source == target).sum())} edges are self loops")
    undirected = graph.edge_type == EdgeType.ROOM_CONNECTS_ROOM
    if (source[undirected] > target[undirected]).any():
        problems.append(f"{sid}: ROOM_CONNECTS_ROOM edges must be stored once, with source < target")
    edges = list(zip(graph.edge_type.tolist(), source.tolist(), target.tolist()))
    if len(set(edges)) != len(edges):
        problems.append(f"{sid}: {len(edges) - len(set(edges))} duplicate edges")
    return problems


def check_containment(graph: SceneGraph) -> list[str]:
    """Exactly one parent per room and object, parent arrays agreeing with the edges, no cycles."""
    sid = graph.scene_id
    n = graph.num_nodes
    problems = []

    expected_room_parent = np.full(n, -1)
    expected_object_parent = np.full(n, -1)
    n_parents = np.zeros(n, dtype=int)
    for t, expected in ((EdgeType.BUILDING_CONTAINS_ROOM, expected_room_parent),
                        (EdgeType.ROOM_CONTAINS_OBJECT, expected_object_parent)):
        parent, child = graph.edges_of(t)
        np.add.at(n_parents, child, 1)
        expected[child] = parent

    for t, contained_by in ((NodeType.ROOM, "building"), (NodeType.OBJECT, "room")):
        wrong = (graph.node_type == t) & (n_parents != 1)
        if wrong.any():
            problems.append(f"{sid}: {int(wrong.sum())} {t.name} nodes are not contained by exactly one {contained_by}")
    if (n_parents[graph.node_type == NodeType.BUILDING] != 0).any():
        problems.append(f"{sid}: the building is contained by another node")
    if not np.array_equal(graph.room_parent, expected_room_parent):
        problems.append(f"{sid}: room_parent disagrees with the BUILDING_CONTAINS_ROOM edges")
    if not np.array_equal(graph.object_parent, expected_object_parent):
        problems.append(f"{sid}: object_parent disagrees with the ROOM_CONTAINS_OBJECT edges")

    # walk up from every node; a chain longer than the number of nodes must loop
    parent = np.maximum(expected_room_parent, expected_object_parent)
    for start in range(n):
        node, steps = start, 0
        while parent[node] >= 0 and steps <= n:
            node, steps = parent[node], steps + 1
        if steps > n:
            problems.append(f"{sid}: containment has a cycle through node {start}")
            break
    return problems


def check_node_metadata(graph: SceneGraph) -> list[str]:
    """raw_metadata["nodes"] has one entry per node, each with the NODE_METADATA_KEYS."""
    sid = graph.scene_id
    nodes = graph.raw_metadata.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != graph.num_nodes:
        return [f"{sid}: raw_metadata['nodes'] must be a list with one entry per node"]
    problems = []
    for i, entry in enumerate(nodes):
        missing = [key for key in NODE_METADATA_KEYS if key not in entry]
        if missing:
            problems.append(f"{sid}: node {i} metadata has no {', '.join(missing)}")
            continue
        if not (entry["raw_label"] is None or isinstance(entry["raw_label"], str)):
            problems.append(f"{sid}: node {i} raw_label must be a string or None")
        dataset_id = entry["dataset_category_id"]
        if not (dataset_id is None or isinstance(dataset_id, str)
                or (isinstance(dataset_id, int) and not isinstance(dataset_id, bool))):
            problems.append(f"{sid}: node {i} dataset_category_id must be an int, a string or None")
    return problems


def validate_provenance(graph: SceneGraph, vocab_sha1: str, dataset_map_sha1: str) -> list[str]:
    """Check that a processed dataset graph records the vocabulary and mapping we expect.

    The expected fingerprints are passed in by the pipeline that processed the graph, rather
    than read from the current vocabulary: a graph built with an older vocabulary then fails
    as an old graph, instead of having its provenance silently rewritten.
    """
    problems = []
    for key in GRAPH_METADATA_KEYS:
        if key not in graph.raw_metadata:
            problems.append(f"{graph.scene_id}: raw_metadata has no {key}")
    stored_vocab = graph.raw_metadata.get("canonical_vocab_sha1")
    if stored_vocab is not None and stored_vocab != vocab_sha1:
        problems.append(f"{graph.scene_id}: built with vocabulary {stored_vocab}, expected {vocab_sha1}")
    stored_map = graph.raw_metadata.get("dataset_map_sha1")
    if stored_map is not None and stored_map != dataset_map_sha1:
        problems.append(f"{graph.scene_id}: built with dataset map {stored_map}, expected {dataset_map_sha1}")
    return problems
