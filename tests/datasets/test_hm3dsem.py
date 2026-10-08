from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from src.datasets.hm3dsem.dataset import (
    NO_INSTANCE,
    STRUCTURE,
    HM3DSemDataset,
    label_faces,
    read_room_labels,
    read_semantic_txt,
)

HM3D_ROOT = Path.home() / "Desktop/Repos/habitat-MP3D/data/scene_datasets/hm3d"
SCENE = "00823-7MXmsvcQjpJ"
ROOM_LABELS = Path(__file__).resolve().parents[2] / "data/raw/hm3d/Per_Scene_Region_Weighted_Votes.csv"
ALL_NAMES = Path(__file__).resolve().parents[2] / "data/raw/hm3d/HM3D_CountsOfObjectTypes.csv"
needs_hm3d = pytest.mark.skipif(
    not (HM3D_ROOT / "val" / SCENE).exists() or not ROOM_LABELS.exists(), reason="HM3D val data not on disk"
)


def test_read_semantic_txt(tmp_path):
    path = tmp_path / "x.semantic.txt"
    path.write_text('HM3D Semantic Annotations\n1,5f517d,"wall",0\n2,A500B5,"Kitchen  Cabinet, Lower",3\n3,000001,"plant",-1\n')
    instances = read_semantic_txt(path)
    assert instances[1] == {"color": "5F517D", "name": "wall", "region": 0}
    assert instances[2]["name"] == "Kitchen  Cabinet, Lower"   # exactly as annotated, comma kept
    assert instances[3]["region"] == -1



def test_read_semantic_txt_rejects_other_files(tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("something else\n1,FFFFFF,\"wall\",0\n")
    with pytest.raises(ValueError):
        read_semantic_txt(path)


@pytest.mark.parametrize("name, structure", [
    ("floor", "floor"),
    ("shower floor", "floor"),
    ("patio floor", "floor"),
    ("stairs", "stairs"),
    ("landing", "stairs"),
    ("floor lamp", None),
    ("ceiling", "ceiling"),
    ("ceiling under stairs", "ceiling"),   # a ceiling, not stairs
    ("ceiling lamp", None),
    ("wall", "wall"),
    ("bathroom wall", "wall"),
    ("partition", "wall"),
    ("wall clock", None),
    ("door", "door"),
    ("doorway", "door"),
    ("closet door", "door"),
    ("door knob", None),
    ("cabinet door", None),
    ("desk door", None),
    ("garage door railing", None),
])
def test_structure_of_names(name, structure):
    assert STRUCTURE.get(name) == structure


def test_structure_classes():
    assert set(STRUCTURE.values()) == {"floor", "stairs", "wall", "ceiling", "door"}



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


def test_read_room_labels(tmp_path):
    path = tmp_path / "votes.csv"
    path.write_text(
        "Scene Name,Region #,Bedroom,Office,Weighted Room Proposal\n"
        "00001-abc,0,10,0, Bedroom\n"
        "00001-abc,3,1,1, Tie: Bedroom & Office\n"
    )
    assert read_room_labels(path) == {("00001-abc", 0): "Bedroom", ("00001-abc", 3): "Tie: Bedroom & Office"}


@pytest.fixture(scope="module")
def dataset():
    return HM3DSemDataset(HM3D_ROOT, room_labels_file=ROOM_LABELS)


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
    assert all(r.raw_label is not None for r in scene.regions)   # every region has a proposal
    region_ids = {r.region_id for r in scene.regions}
    assert {o.region_id for o in scene.objects} - {None} <= region_ids
    point_sets = [r.walkable_points for r in scene.regions] + [r.surface_points for r in scene.regions]
    point_sets += [o.points for o in scene.objects] + [d.points for d in scene.doors] + [scene.wall_points]
    for points in point_sets:
        assert points.ndim == 2 and points.shape[1] == 3
        assert np.isfinite(points).all()


@needs_hm3d
def test_structural_instances_are_not_objects(scene):
    assert all(o.raw_label not in STRUCTURE for o in scene.objects)
    assert len(scene.doors) > 0 and len(scene.wall_points) > 0


@pytest.mark.skipif(not ALL_NAMES.exists(), reason="official HM3D name list not on disk")
def test_every_structure_name_occurs_in_hm3d():
    """The mapping is typed by hand: a name missing from the official name list is a typo."""
    names = {line.rsplit(";", 1)[0] for line in ALL_NAMES.read_text().splitlines()[1:]}
    assert set(STRUCTURE) - names == set()


@needs_hm3d
def test_object_labels_are_the_raw_names(scene):
    txt, _ = HM3DSemDataset(HM3D_ROOT)._annotation_files(HM3D_ROOT / "val" / SCENE)
    raw = {iid: inst["name"] for iid, inst in read_semantic_txt(txt).items()}
    assert all(o.raw_label == raw[o.instance_id] for o in scene.objects)


@needs_hm3d
def test_every_room_proposal_is_mapped_or_a_tie():
    """Ties are left out of ROOM_CATEGORIES on purpose; every other proposal must be mapped."""
    proposals = set(read_room_labels(ROOM_LABELS).values())
    unmapped = {p for p in proposals if p not in HM3DSemDataset.ROOM_CATEGORIES and not p.startswith("Tie: ")}
    assert unmapped == set()


def test_without_a_labels_file_no_region_has_a_label(tmp_path):
    assert HM3DSemDataset(tmp_path).room_labels == {}


def official_name_counts() -> dict[str, int]:
    """Every HM3D name with its number of instances, from the authors' statistics."""
    counts = {}
    for line in ALL_NAMES.read_text().splitlines()[1:]:
        name, count = line.rsplit(";", 1)
        counts[name] = int(count)
    return counts


@pytest.mark.skipif(not ALL_NAMES.exists(), reason="official HM3D name list not on disk")
def test_every_frequent_object_name_is_decided():
    """Names with at least 10 instances are mapped or explicitly not objects, never left to unknown."""
    frequent = {n for n, c in official_name_counts().items() if c >= 10 and n not in STRUCTURE}
    decided = set(HM3DSemDataset.OBJECT_CATEGORIES) | set(HM3DSemDataset.NOT_OBJECTS)
    assert frequent - decided == set()


@pytest.mark.skipif(not ALL_NAMES.exists(), reason="official HM3D name list not on disk")
def test_every_object_table_name_occurs_in_hm3d():
    """The tables are written by hand: a name missing from the official name list is a typo."""
    names = set(official_name_counts())
    assert set(HM3DSemDataset.OBJECT_CATEGORIES) - names == set()
    assert set(HM3DSemDataset.NOT_OBJECTS) - names == set()


def test_object_tables_and_structure_do_not_overlap():
    tables = set(HM3DSemDataset.OBJECT_CATEGORIES) | set(HM3DSemDataset.NOT_OBJECTS)
    assert tables & set(STRUCTURE) == set()


def test_not_object_reasons():
    assert set(HM3DSemDataset.NOT_OBJECTS.values()) == {"architecture", "fixture", "part", "unlabeled", "ambiguous"}
    assert HM3DSemDataset.NOT_OBJECTS["unknown"] == "unlabeled"
