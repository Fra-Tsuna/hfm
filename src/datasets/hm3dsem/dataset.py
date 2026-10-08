"""HM3D-Semantics v0.2 as an AnnotatedScene.

Layout on disk: <root>/<split>/<id>-<hash>/<hash>.semantic.txt and <hash>.semantic.glb.
  semantic.txt  one row per annotated instance: id, HEX color, "name", region id (-1: none)
  semantic.glb  the scene mesh; a texture paints every instance with its HEX color
Each mesh face gets the instance whose color is under it, so no simulator is needed.

The GLB is already in our frame (meters, Z up), so no conversion is applied.
The meshes carry no room labels. The HM3D authors published one proposal per region, computed by
voting over the objects it contains with a bed counting 10 votes (Per_Scene_Region_Weighted_Votes.csv,
from the statistics of github.com/matterport/habitat-matterport-3dresearch). When that file is given,
each region's raw_label is its proposal, e.g. "Bedroom" or "Tie: Bedroom & Office"; otherwise None.

Which instances play a structural role (walkable floor, room surfaces, walls, doors) is decided
here from their exact names, because names are HM3D-specific. Every other instance is an object with
its raw name untouched; the label map decides later what it becomes.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import trimesh

from src.datasets.base import AnnotatedScene, Door, ObjectInstance, Region, SceneDataset
from src.utils.geometry import sample_surface

NO_INSTANCE = -1  # face whose color matches no annotated instance
NO_REGION = -1    # instance that belongs to no region


# Structural roles, by exact HM3D name as found in the val annotations.
# Names not listed here are objects, e.g. "floor lamp", "wall clock", "door knob".

# Surfaces you can walk on: they make up a region's floor.
WALKABLE = {
    "floor", "shower floor",
    "stairs", "stair", "stair step", "step", "shower step", "doorstep",
}

# Surfaces that bound a room: floors, walls and ceilings.
ROOM_SURFACES = {
    "floor", "shower floor",
    "wall", "bath wall", "shower wall", "kitchen wall", "fireplace wall", "recessed wall",
    "closet mirror wall", "compound wall", "stair wall", "staircase wall", "ceiling wall",
    "ceiling", "bedroom ceiling", "shower ceiling", "ceiling lower", "ceiling dome", "ceiling arch",
    "ceiling under stairs", "ceiling under staircase",
}

# Walls that block passage between regions.
WALLS = {
    "wall", "bath wall", "shower wall", "kitchen wall", "fireplace wall", "recessed wall",
    "closet mirror wall", "compound wall", "stair wall", "staircase wall", "ceiling wall",
    "partition",
}

# Openings people walk through. Not doors: furniture doors ("cabinet door", "desk door"),
# door hardware ("door knob", "door hinge") and hatches to another level ("attic door", "ceiling door").
DOORS = {
    "door", "door frame", "doorpost", "sliding door", "sliding glass door",
    "closet door", "elevator door", "garage door", "shower door", "shower door frame",
    "arch", "entrance arch",
    "door window", "door/window", "window/door", "door/window frame",
}


def read_semantic_txt(path) -> dict[int, dict]:
    """instance id -> {"color": "5F517D", "name": "wall", "region": 0}; names exactly as annotated."""
    instances = {}
    with open(path, newline="") as f:
        rows = csv.reader(f)  # csv, because quoted names may contain commas
        header = next(rows)
        if not header[0].startswith("HM3D Semantic Annotations"):
            raise ValueError(f"{path} is not an HM3D semantic annotation file")
        for row in rows:
            if row:
                instances[int(row[0])] = {
                    "color": row[1].upper(),
                    "name": row[2],
                    "region": int(row[3]),
                }
    return instances


def read_room_labels(path) -> dict[tuple[str, int], str]:
    """(scene id, region id) -> the authors' weighted room proposal, as written apart from the CSV spacing."""
    labels = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            # the proposal column starts with a space: " Bedroom"
            labels[(row["Scene Name"], int(row["Region #"]))] = row["Weighted Room Proposal"].strip()
    return labels


def label_faces(mesh: trimesh.Trimesh, color_to_instance: dict[int, int]) -> np.ndarray:
    """[F] instance id of every face, NO_INSTANCE if its color is not an annotated instance.

    The color under the face center decides; the corners are only used when the center color
    is unknown, because on small instances they often land on a neighbor's color.
    """
    image = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB"))
    height, width = image.shape[:2]
    corners_uv = mesh.visual.uv[mesh.faces]                        # [F, 3, 2]
    uv = np.concatenate([corners_uv.mean(axis=1, keepdims=True), corners_uv], axis=1)  # [F, 4, 2]
    px = np.clip((uv[..., 0] * width).astype(np.int64), 0, width - 1)
    py = np.clip(((1.0 - uv[..., 1]) * height).astype(np.int64), 0, height - 1)  # image rows go down
    rgb = image[py, px].astype(np.int64)  # widen only the sampled pixels, not the whole texture
    colors = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    # look every distinct color up once, then spread the result back to all faces
    unique_colors, where = np.unique(colors, return_inverse=True)
    unique_ids = np.array([color_to_instance.get(int(c), NO_INSTANCE) for c in unique_colors], dtype=np.int64)
    ids = unique_ids[where].reshape(colors.shape)

    labels = ids[:, 0].copy()
    for corner in (1, 2, 3):
        unknown = labels == NO_INSTANCE
        labels[unknown] = ids[unknown, corner]
    return labels


def load_labelled_mesh(glb_path, color_to_instance: dict[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """vertices [V, 3], faces [F, 3] and the instance id of every face [F], for the whole scene."""
    scene = trimesh.load(glb_path, process=False)
    vertices, faces, labels = [], [], []
    offset = 0
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        mesh = scene.geometry[geometry_name]
        vertices.append(trimesh.transformations.transform_points(mesh.vertices, transform))
        faces.append(mesh.faces + offset)
        labels.append(label_faces(mesh, color_to_instance))
        offset += len(mesh.vertices)
    return np.concatenate(vertices), np.concatenate(faces), np.concatenate(labels)


class HM3DSemDataset(SceneDataset):
    """HM3D-Semantics scenes under root, e.g. <...>/scene_datasets/hm3d.

    room_labels_file  the authors' weighted room proposals per region; None leaves rooms unlabelled
    point_spacing     distance between points sampled on surfaces, meters
    seed              seed of the surface sampling, so a scene always gives the same points
    """

    # Room proposals of the HM3D authors -> canonical room categories. Ties ("Tie: Bedroom & Office")
    # are not listed, so they become unknown.
    ROOM_CATEGORIES = json.loads((Path(__file__).parent / "room_mapping.json").read_text())
    OBJECT_CATEGORIES = {}   # TODO: map the HM3D object names
    NOT_OBJECTS = {}         # TODO: list the HM3D names that are not objects

    def __init__(self, root, room_labels_file=None, point_spacing: float = 0.025, seed: int = 0):
        self.root = Path(root)
        self.room_labels = read_room_labels(room_labels_file) if room_labels_file else {}
        self.point_spacing = point_spacing
        self.seed = seed

    def scene_ids(self, split: str) -> list[str]:
        split_dir = self.root / split
        if not split_dir.is_dir():
            return []
        return sorted(d.name for d in split_dir.iterdir() if self._is_annotated(d))

    def _annotation_files(self, scene_dir: Path) -> tuple[Path, Path]:
        stem = scene_dir.name.split("-", 1)[-1]
        return scene_dir / f"{stem}.semantic.txt", scene_dir / f"{stem}.semantic.glb"

    def _is_annotated(self, scene_dir: Path) -> bool:
        """A scene can be loaded only if both its annotation table and its semantic mesh exist."""
        txt_path, glb_path = self._annotation_files(scene_dir)
        return scene_dir.is_dir() and txt_path.exists() and glb_path.exists()

    def _scene_dir(self, scene_id: str) -> Path:
        candidates = [d for d in self.root.glob(f"*/{scene_id}") if self._is_annotated(d)]
        if len(candidates) != 1:
            raise KeyError(f"no annotated HM3D scene {scene_id!r} under {self.root}")
        return candidates[0]

    def load_scene(self, scene_id: str) -> AnnotatedScene:
        scene_dir = self._scene_dir(scene_id)
        txt_path, glb_path = self._annotation_files(scene_dir)
        instances = read_semantic_txt(txt_path)
        color_to_instance = {int(inst["color"], 16): iid for iid, inst in instances.items()}
        vertices, faces, face_instance = load_labelled_mesh(glb_path, color_to_instance)
        rng = np.random.default_rng(self.seed)

        # faces of every instance, in instance id order so the sampling is reproducible
        order = np.argsort(face_instance, kind="stable")
        ids, starts = np.unique(face_instance[order], return_index=True)
        faces_of = {int(i): faces[chunk] for i, chunk in zip(ids, np.split(order, starts[1:])) if i != NO_INSTANCE}

        points_of = {}
        for iid in sorted(faces_of):
            points_of[iid] = sample_surface(vertices, faces_of[iid], self.point_spacing, rng)

        regions = []
        for region in sorted({inst["region"] for inst in instances.values()} - {NO_REGION}):
            members = [iid for iid, inst in instances.items() if inst["region"] == region and iid in points_of]
            walkable = [points_of[i] for i in members if instances[i]["name"] in WALKABLE]
            surfaces = [points_of[i] for i in members if instances[i]["name"] in ROOM_SURFACES]
            regions.append(Region(
                region_id=region,
                raw_label=self.room_labels.get((scene_id, region)),
                dataset_category_id=None,
                walkable_points=np.concatenate(walkable) if walkable else np.zeros((0, 3)),
                surface_points=np.concatenate(surfaces) if surfaces else np.zeros((0, 3)),
            ))

        objects, doors, walls = [], [], []
        for iid in sorted(points_of):
            name = instances[iid]["name"]
            if name in DOORS:
                doors.append(Door(instance_id=iid, points=points_of[iid]))
            elif name in WALLS:
                walls.append(points_of[iid])
            elif name in WALKABLE or name in ROOM_SURFACES:
                continue  # floors, stairs and ceilings are already part of their region
            else:
                region = instances[iid]["region"]
                objects.append(ObjectInstance(
                    instance_id=iid,
                    raw_label=name,
                    dataset_category_id=None,
                    region_id=None if region == NO_REGION else region,
                    points=points_of[iid],
                ))

        return AnnotatedScene(
            scene_id=scene_id,
            regions=regions,
            objects=objects,
            wall_points=np.concatenate(walls) if walls else np.zeros((0, 3)),
            doors=doors,
            extra={
                "dataset": "HM3D-Semantics v0.2",
                "split": scene_dir.parent.name,
                "unlabelled_face_fraction": float((face_instance == NO_INSTANCE).mean()),
                "instances_without_geometry": sorted(set(instances) - set(points_of)),
            },
        )
