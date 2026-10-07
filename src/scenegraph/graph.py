"""The hierarchical scene graph: building -> rooms -> objects, plus room-room connectivity."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class NodeType(IntEnum):
    BUILDING = 0
    ROOM = 1
    OBJECT = 2


class EdgeType(IntEnum):
    BUILDING_CONTAINS_ROOM = 0
    ROOM_CONTAINS_OBJECT = 1
    ROOM_CONNECTS_ROOM = 2   # undirected, stored once with source < target


# (source node type, target node type) of each edge type
EDGE_ENDPOINTS = {
    EdgeType.BUILDING_CONTAINS_ROOM: (NodeType.BUILDING, NodeType.ROOM),
    EdgeType.ROOM_CONTAINS_OBJECT: (NodeType.ROOM, NodeType.OBJECT),
    EdgeType.ROOM_CONNECTS_ROOM: (NodeType.ROOM, NodeType.ROOM),
}

# Geometry of a node: a box, as 7 numbers
#   px, py, pz   box center (meters)
#   sx, sy, sz   box size along its own axes (meters)
#   theta        yaw: rotation of the box around the vertical axis (radians)
# All node types share these 7 slots so that geometry is one [N, 7] array, but not every slot
# means something for every type. The geometry mask marks the slots that do (True) and the
# ones that are only padding (False, and always 0 in the geometry arrays):
#   building  nothing        it has no box of its own
#   room      center + size  rooms are axis-aligned, so theta is not used
#   object    everything     objects are boxes rotated around the vertical axis
# Models use the mask to ignore padding, e.g. to leave padded slots out of a geometry loss.
GEOM_FIELDS = ("px", "py", "pz", "sx", "sy", "sz", "theta")
GEOM_DIM = len(GEOM_FIELDS)
POS = slice(0, 3)     # geometry[:, POS]   -> centers
SIZE = slice(3, 6)    # geometry[:, SIZE]  -> sizes
THETA = 6             # geometry[:, THETA] -> yaws

GEOM_MASK = {
    NodeType.BUILDING: np.zeros(GEOM_DIM, dtype=bool),               # [0 0 0 0 0 0 0]
    NodeType.ROOM: np.array([1, 1, 1, 1, 1, 1, 0], dtype=bool),      # [1 1 1 1 1 1 0]
    NodeType.OBJECT: np.ones(GEOM_DIM, dtype=bool),                  # [1 1 1 1 1 1 1]
}


def geometry_mask(node_type: np.ndarray) -> np.ndarray:
    """[N] node types -> [N, GEOM_DIM] mask: row i is GEOM_MASK of node i's type.

    Example: node_type [0, 1, 2] -> [[0 0 0 0 0 0 0],
                                     [1 1 1 1 1 1 0],
                                     [1 1 1 1 1 1 1]]
    """
    unknown = np.setdiff1d(node_type, list(NodeType))
    if len(unknown):
        raise ValueError(f"unknown node types {unknown.tolist()}")
    mask = np.zeros((len(node_type), GEOM_DIM), dtype=bool)
    for t, m in GEOM_MASK.items():
        mask[node_type == t] = m
    return mask


# Keys every entry of raw_metadata["nodes"] must have, one entry per node, in node order.
# They are provenance (where the canonical category came from), not model input:
#   raw_label            str or None         the dataset's own label, e.g. "kitchen cabinet door"
#   dataset_category_id  int, str or None    the dataset's own category id, if it has one
# None means the dataset provides no such value for that node (e.g. the building).
NODE_METADATA_KEYS = ("raw_label", "dataset_category_id")

# Keys every processed dataset graph must have at the top level of raw_metadata: the provenance
# of its canonical category_ids.
#   canonical_vocab_sha1  vocab_sha1() of the vocabulary the category_ids refer to
#   dataset_map_sha1      fingerprint of the dataset mapping (raw label -> canonical) that was used
# The dataset builder that assigns the category_ids writes them; saving and loading only preserve
# them. Transient graphs (e.g. synthetic ones built in tests) do not need them.
GRAPH_METADATA_KEYS = ("canonical_vocab_sha1", "dataset_map_sha1")


@dataclass(eq=False)
class SceneGraph:
    """One scene with N nodes and E edges. Node order carries no meaning.

    scene_id       str
    node_type      [N]     int64    NodeType
    category_id    [N]     int64    canonical category, index into the vocabulary of the node's type
    geometry       [N, 7]  float64  raw geometry in the scene frame; masked slots are 0
    geometry_norm  [N, 7]  float64  normalized geometry; masked slots are 0
    geometry_mask  [N, 7]  bool     meaningful slots, given by the node type
    edge_index     [2, E]  int64    (source, target)
    edge_type      [E]     int64    EdgeType
    room_parent    [N]     int64    for a room, its building node; -1 otherwise
    object_parent  [N]     int64    for an object, its room node; -1 otherwise
    raw_metadata   dict             anything else about the scene, JSON-serializable;
                                    raw_metadata["nodes"] holds one dict per node, with at least
                                    the NODE_METADATA_KEYS
    """

    scene_id: str
    node_type: np.ndarray
    category_id: np.ndarray
    geometry: np.ndarray
    geometry_norm: np.ndarray
    geometry_mask: np.ndarray
    edge_index: np.ndarray
    edge_type: np.ndarray
    room_parent: np.ndarray
    object_parent: np.ndarray
    raw_metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        counts = ", ".join(f"{(self.node_type == t).sum()} {t.name.lower()}" for t in NodeType)
        return f"SceneGraph({self.scene_id}: {self.num_nodes} nodes ({counts}), {self.num_edges} edges)"

    @property
    def num_nodes(self) -> int:
        return len(self.node_type)

    @property
    def num_edges(self) -> int:
        return len(self.edge_type)

    def nodes_of(self, t: NodeType) -> np.ndarray:
        """Indices of the nodes of one type."""
        return np.flatnonzero(self.node_type == t)

    def edges_of(self, t: EdgeType) -> np.ndarray:
        """[2, E_t] edges of one type."""
        return self.edge_index[:, self.edge_type == t]

    def reorder_nodes(self, new_order) -> SceneGraph:
        """Same graph with nodes in a new order: new node k is old node new_order[k].

        Why: node order carries no meaning, so permutation tests use this to shuffle a graph
        and check that models and losses do not notice.
        Edges, parents and per-node metadata are relabeled to point to the new indices.
        """
        perm = np.asarray(new_order, dtype=np.int64)
        if perm.shape != (self.num_nodes,) or not np.array_equal(np.sort(perm), np.arange(self.num_nodes)):
            raise ValueError(f"new_order must be a permutation of 0..{self.num_nodes - 1}")
        old_to_new = np.empty_like(perm)
        old_to_new[perm] = np.arange(len(perm))

        def relabel_parent(parent):
            parent = parent[perm]
            return np.where(parent >= 0, old_to_new[np.maximum(parent, 0)], -1)

        edge_index = old_to_new[self.edge_index]
        undirected = self.edge_type == EdgeType.ROOM_CONNECTS_ROOM
        edge_index[:, undirected] = np.sort(edge_index[:, undirected], axis=0)

        raw_metadata = copy.deepcopy(self.raw_metadata)   # the new graph shares nothing with this one
        if "nodes" in raw_metadata:   # per-node metadata follows its node
            raw_metadata["nodes"] = [raw_metadata["nodes"][i] for i in perm]

        return SceneGraph(
            scene_id=self.scene_id,
            node_type=self.node_type[perm],
            category_id=self.category_id[perm],
            geometry=self.geometry[perm],
            geometry_norm=self.geometry_norm[perm],
            geometry_mask=self.geometry_mask[perm],
            edge_index=edge_index,
            edge_type=self.edge_type.copy(),
            room_parent=relabel_parent(self.room_parent),
            object_parent=relabel_parent(self.object_parent),
            raw_metadata=raw_metadata,
        )
