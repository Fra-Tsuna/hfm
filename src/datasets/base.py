"""What every dataset provides, in a form that does not depend on the dataset.

A dataset adapter reads its own raw files and returns an AnnotatedScene: regions, objects,
walls and doors as 3D points, with the dataset's own labels left untouched. Everything after
that (mapping labels to the canonical vocabulary, boxes, connectivity, building the
SceneGraph) is the same for every dataset.

Coordinates: meters, right-handed, Z up. Each adapter converts its dataset to this frame.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np


@dataclass(eq=False)
class Region:
    """A room-like region of the scene.

    raw_label            the dataset's room label, e.g. "bedroom"; None if the dataset has none
                         (the room then becomes unknown; TODO: infer it from the objects inside)
    dataset_category_id  the dataset's own room category id, if it has one
    walkable_points      [P, 3] points on its floor and stairs
    surface_points       [S, 3] points on its floor, walls and ceiling, which bound the room
    """

    region_id: Union[int, str]
    raw_label: Optional[str]
    dataset_category_id: Optional[Union[int, str]]
    walkable_points: np.ndarray
    surface_points: np.ndarray


@dataclass(eq=False)
class ObjectInstance:
    """An object of the scene.

    raw_label            the dataset's object label, e.g. "kitchen cabinet door"
    dataset_category_id  the dataset's own object category id, if it has one
    region_id            the region it belongs to, if the dataset says so
    points               [Q, 3] points on its surface
    """

    instance_id: Union[int, str]
    raw_label: Optional[str]
    dataset_category_id: Optional[Union[int, str]]
    region_id: Optional[Union[int, str]]
    points: np.ndarray


@dataclass(eq=False)
class Door:
    """A door-like opening (door, door frame, archway) between regions: points [D, 3] on its surface."""

    instance_id: Union[int, str]
    points: np.ndarray


@dataclass(eq=False)
class AnnotatedScene:
    """One scene as a dataset adapter delivers it.

    regions      room-like regions
    objects      object instances
    wall_points  [W, 3] points on every wall of the scene, used to tell openings from walls
    doors        door-like openings
    extra        anything dataset-specific worth keeping; copied into the graph's raw_metadata
    """

    scene_id: str
    regions: list[Region]
    objects: list[ObjectInstance]
    wall_points: np.ndarray
    doors: list[Door]
    extra: dict = field(default_factory=dict)


class SceneDataset(ABC):
    """The interface every dataset implements.

    An adapter lists its scenes, reads one scene into an AnnotatedScene, and declares how its
    labels map to the canonical vocabulary (src/scenegraph/vocabulary.py) with three plain tables,
    keyed by the raw label exactly as the dataset writes it:

    ROOM_CATEGORIES    raw room label -> canonical room category
    OBJECT_CATEGORIES  raw object label -> canonical object category
    NOT_OBJECTS        raw object label -> why it is not an object node (e.g. "architecture")

    A label in none of the tables, or no label at all, becomes "unknown": a label is never
    forced into a category it was not explicitly mapped to.
    """

    ROOM_CATEGORIES: dict[str, str]
    OBJECT_CATEGORIES: dict[str, str]
    NOT_OBJECTS: dict[str, str]

    @abstractmethod
    def scene_ids(self, split: str) -> list[str]:
        """The scenes of a split, e.g. "train", "val" or "test"."""

    @abstractmethod
    def load_scene(self, scene_id: str) -> AnnotatedScene:
        """Read one scene from the dataset's raw files."""


def label_map_sha1(dataset: SceneDataset) -> str:
    """Fingerprint of a dataset's label tables, stored in every graph it builds as dataset_map_sha1."""
    content = {
        "room_categories": dataset.ROOM_CATEGORIES,
        "object_categories": dataset.OBJECT_CATEGORIES,
        "not_objects": dataset.NOT_OBJECTS,
    }
    return hashlib.sha1(json.dumps(content, sort_keys=True).encode()).hexdigest()
