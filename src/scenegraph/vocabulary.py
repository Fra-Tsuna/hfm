"""The canonical closed vocabulary shared by every dataset.

A graph's category_id always refers to these lists, never to a dataset's own category ids.
Each dataset maps its raw labels into this vocabulary with its own mapping; the raw label and
the dataset's category id are kept in the node metadata, so finer classes can be recovered.
Processed graphs record which vocabulary and mapping produced their category_ids (see
GRAPH_METADATA_KEYS in graph.py).

The room and object namespaces both start with two classes that must not be confused:
  unknown   we cannot tell what it is (no label, or a label too ambiguous to map safely)
  other     we can tell what it is, and it is none of the canonical classes
Ambiguous labels go to unknown rather than being forced into a wrong class. The building
namespace has a single category and needs neither: every graph has exactly one building.

The vocabulary is deliberately coarse for the first experiments. Rules for changing it:
append only, never reorder or remove (ids are positions), and bump VOCAB_VERSION.
"""

from __future__ import annotations

import hashlib
import json

from src.scenegraph.graph import NodeType

VOCAB_VERSION = "v1"   # v1: added the room category hall_or_stairwell

UNKNOWN = "unknown"
OTHER = "other"

BUILDING_CATEGORIES = (
    "building",
)

ROOM_CATEGORIES = (
    UNKNOWN,
    OTHER,
    "bathroom",       # also toilets and washrooms
    "bedroom",
    "kitchen",
    "living_room",    # also family room, lounge, tv room
    "dining_room",
    "office",         # also study, library
    "hallway",        # also corridor, entryway, foyer
    "stairs",         # stairwells and landings
    "closet",         # also walk-in wardrobe
    "laundry_room",   # also utility room
    "garage",
    "outdoor",        # balcony, porch, terrace, patio
    "hall_or_stairwell",  # a hall or a stairwell, for datasets that do not tell them apart (HM3D)
)

OBJECT_CATEGORIES = (
    UNKNOWN,
    OTHER,
    "chair",          # also armchair, office chair
    "sofa",
    "bed",
    "table",          # also desk
    "cabinet",        # also wardrobe, cupboard, sideboard
    "chest_of_drawers",
    "shelving",       # also bookshelf, rack
    "counter",        # also kitchen island
    "stool",          # also ottoman, pouffe
    "cushion",        # also pillow
    "picture",        # also painting, photo, poster
    "mirror",
    "curtain",
    "blinds",
    "plant",
    "lighting",       # lamps, ceiling lights, chandeliers
    "sink",
    "toilet",
    "bathtub",
    "shower",
    "towel",
    "tv_monitor",
    "appliance",      # fridge, oven, washing machine, microwave, ...
    "fireplace",
    "clothes",
)

CATEGORIES = {
    NodeType.BUILDING: BUILDING_CATEGORIES,
    NodeType.ROOM: ROOM_CATEGORIES,
    NodeType.OBJECT: OBJECT_CATEGORIES,
}


def num_categories(node_type: NodeType) -> int:
    return len(CATEGORIES[NodeType(node_type)])


def category_id(node_type: NodeType, name: str) -> int:
    """Canonical id of a category name, within the namespace of the node type."""
    return CATEGORIES[NodeType(node_type)].index(name)


def category_name(node_type: NodeType, idx: int) -> str:
    names = CATEGORIES[NodeType(node_type)]
    if not 0 <= idx < len(names):   # reject negative ids instead of letting Python wrap them around
        raise ValueError(f"category id {idx} out of range for {NodeType(node_type).name} (0..{len(names) - 1})")
    return names[idx]


def vocab_sha1() -> str:
    """Fingerprint of the vocabulary content (not of this file, so comments do not change it).

    The dataset builder that assigns canonical category_ids stores this value in the graph as
    raw_metadata["canonical_vocab_sha1"] (see GRAPH_METADATA_KEYS); saving and loading preserve
    it unchanged. A graph keeps the fingerprint of the vocabulary it was built with, so a graph
    from an older vocabulary stays recognizable as such: validation compares it with the
    vocabulary the pipeline expects, never re-stamps it.
    """
    content = {"version": VOCAB_VERSION, **{t.name: list(names) for t, names in CATEGORIES.items()}}
    return hashlib.sha1(json.dumps(content, sort_keys=True).encode()).hexdigest()
