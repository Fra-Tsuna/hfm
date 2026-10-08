from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from src.datasets.hm3dsem.dataset import (
    DOORS,
    NO_INSTANCE,
    ROOM_SURFACES,
    WALKABLE,
    WALLS,
    HM3DSemDataset,
    label_faces,
    name_key,
    read_semantic_txt,
)

HM3D_ROOT = Path.home() / "Desktop/Repos/habitat-MP3D/data/scene_datasets/hm3d"
SCENE = "00823-7MXmsvcQjpJ"
needs_hm3d = pytest.mark.skipif(not (HM3D_ROOT / "val" / SCENE).exists(), reason="HM3D val data not on disk")


def test_read_semantic_txt(tmp_path):
    path = tmp_path / "x.semantic.txt"
    path.write_text('HM3D Semantic Annotations\n1,5f517d,"wall",0\n2,A500B5,"Kitchen  Cabinet, Lower",3\n3,000001,"plant",-1\n')
    instances = read_semantic_txt(path)
    assert instances[1] == {"color": "5F517D", "name": "wall", "region": 0}
    assert instances[2]["name"] == "Kitchen  Cabinet, Lower"   # exactly as annotated, comma kept
    assert instances[3]["region"] == -1


def test_name_key_only_normalizes_for_lookup():
    assert name_key("Kitchen  Cabinet, Lower") == "kitchen cabinet, lower"
    assert name_key(" Shower   Floor ") in WALKABLE


def test_read_semantic_txt_rejects_other_files(tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("something else\n1,FFFFFF,\"wall\",0\n")
    with pytest.raises(ValueError):
        read_semantic_txt(path)


@pytest.mark.parametrize("name, walkable, surface, wall, door", [
    ("floor", True, True, False, False),
    ("shower floor", True, True, False, False),
    ("stairs", True, False, False, False),
    ("floor lamp", False, False, False, False),
    ("ceiling", False, True, False, False),
    ("ceiling under stairs", False, True, False, False),   # a ceiling, not stairs
    ("ceiling lamp", False, False, False, False),
    ("wall", False, True, True, False),
    ("wall clock", False, False, False, False),
    ("partition", False, False, True, False),
    ("door", False, False, False, True),
    ("door frame", False, False, False, True),
    ("closet door", False, False, False, True),
    ("door knob", False, False, False, False),
    ("cabinet door", False, False, False, False),
    ("desk door", False, False, False, False),
    ("garage door railing", False, False, False, False),
])
def test_structural_roles_from_names(name, walkable, surface, wall, door):
    roles = (name in WALKABLE, name in ROOM_SURFACES, name in WALLS, name in DOORS)
    assert roles == (walkable, surface, wall, door)


def test_role_names_are_written_as_lookup_keys():
    for name in WALKABLE | ROOM_SURFACES | WALLS | DOORS:
        assert name == name_key(name), name


def make_scene_folder(root, scene_id, with_txt=True, with_glb=True):
    folder = root / "val" / scene_id
    folder.mkdir(parents=True)
    stem = scene_id.split("-", 1)[1]
    if with_txt:
        (folder / f"{stem}.semantic.txt").write_text("HM3D Semantic Annotations\n")
    if with_glb:
        (folder / f"{stem}.semantic.glb").write_bytes(b"")


def test_a_scene_needs_both_annotation_files(tmp_path):
    make_scene_folder(tmp_path, "00001-both")
    make_scene_folder(tmp_path, "00002-onlyTxt", with_glb=False)
    make_scene_folder(tmp_path, "00003-onlyGlb", with_txt=False)
    dataset = HM3DSemDataset(tmp_path)
    assert dataset.scene_ids("val") == ["00001-both"]
    with pytest.raises(KeyError):
        dataset.load_scene("00002-onlyTxt")


def test_label_faces_reads_the_texture_color():
    """Three triangles on a 3-pixel texture: A | unknown | B."""
    a, unknown, b = (0x11, 0x22, 0x33), (0x99, 0x99, 0x99), (0x44, 0x55, 0x66)
    image = Image.new("RGB", (3, 1))
    image.putdata([a, unknown, b])
    vertices = np.zeros((9, 3))
    vertices[:, 0] = np.arange(9)
    faces = np.arange(9).reshape(3, 3)
    u = np.array([
        0.1, 0.2, 0.15,   # inside A
        0.8, 0.9, 0.85,   # inside B
        0.1, 0.5, 0.9,    # center on the unknown pixel, first corner on A
    ])
    uv = np.column_stack([u, np.full(9, 0.5)])
    mesh = trimesh.Trimesh(
        vertices, faces, process=False,
        visual=trimesh.visual.TextureVisuals(uv=uv, material=trimesh.visual.material.PBRMaterial(baseColorTexture=image)),
    )
    color_to_instance = {0x112233: 7, 0x445566: 8}
    assert label_faces(mesh, color_to_instance).tolist() == [7, 8, 7]
    assert label_faces(mesh, {}).tolist() == [NO_INSTANCE] * 3


def test_missing_root_has_no_scenes(tmp_path):
    assert HM3DSemDataset(tmp_path).scene_ids("val") == []


@pytest.fixture(scope="module")
def dataset():
    return HM3DSemDataset(HM3D_ROOT)


@pytest.fixture(scope="module")
def scene(dataset):
    return dataset.load_scene(SCENE)


@needs_hm3d
def test_val_split_has_the_annotated_scenes(dataset):
    assert len(dataset.scene_ids("val")) == 36
    assert SCENE in dataset.scene_ids("val")


@needs_hm3d
def test_unknown_scene_raises(dataset):
    with pytest.raises(KeyError):
        dataset.load_scene("00000-notAscene")


@needs_hm3d
def test_real_scene_is_consistent(scene):
    assert len(scene.regions) == 23
    assert all(r.raw_label is None for r in scene.regions)   # HM3D regions carry no room label
    region_ids = {r.region_id for r in scene.regions}
    assert {o.region_id for o in scene.objects} - {None} <= region_ids
    point_sets = [r.walkable_points for r in scene.regions] + [r.surface_points for r in scene.regions]
    point_sets += [o.points for o in scene.objects] + [d.points for d in scene.doors] + [scene.wall_points]
    for points in point_sets:
        assert points.ndim == 2 and points.shape[1] == 3
        assert np.isfinite(points).all()


@needs_hm3d
def test_structural_instances_are_not_objects(scene):
    structural = WALKABLE | ROOM_SURFACES | WALLS | DOORS
    assert all(name_key(o.raw_label) not in structural for o in scene.objects)
    assert len(scene.doors) > 0 and len(scene.wall_points) > 0


@needs_hm3d
def test_every_role_name_occurs_in_the_val_annotations(dataset):
    """The role lists are typed by hand: a name that never occurs is a typo."""
    names = set()
    for scene_id in dataset.scene_ids("val"):
        txt, _ = dataset._annotation_files(HM3D_ROOT / "val" / scene_id)
        names |= {inst["name"] for inst in read_semantic_txt(txt).values()}
    assert (WALKABLE | ROOM_SURFACES | WALLS | DOORS) - {name_key(n) for n in names} == set()


@needs_hm3d
def test_object_labels_are_the_raw_names(scene):
    txt, _ = HM3DSemDataset(HM3D_ROOT)._annotation_files(HM3D_ROOT / "val" / SCENE)
    raw = {iid: inst["name"] for iid, inst in read_semantic_txt(txt).items()}
    assert all(o.raw_label == raw[o.instance_id] for o in scene.objects)
