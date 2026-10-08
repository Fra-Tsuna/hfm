import numpy as np
import pytest

from src.datasets.base import AnnotatedScene, Door, ObjectInstance, Region, SceneDataset

STEP = 0.05  # spacing of the hand-made points, meters


def floor(x0, x1, y0, y1, z=0.0):
    """Points on a horizontal rectangle."""
    x, y = np.meshgrid(np.arange(x0, x1, STEP), np.arange(y0, y1, STEP))
    return np.column_stack([x.ravel(), y.ravel(), np.full(x.size, z)])


def wall(x, y0, y1, z0=0.0, z1=2.5):
    """Points on a vertical rectangle at constant x."""
    y, z = np.meshgrid(np.arange(y0, y1, STEP), np.arange(z0, z1, STEP))
    return np.column_stack([np.full(y.size, x), y.ravel(), z.ravel()])


def wall_along_x(y, x0, x1, z0=0.0, z1=2.5):
    """Points on a vertical rectangle at constant y."""
    x, z = np.meshgrid(np.arange(x0, x1, STEP), np.arange(z0, z1, STEP))
    return np.column_stack([x.ravel(), np.full(x.size, y), z.ravel()])


def room_surfaces(x0, x1, y0, y1, height=2.5):
    """Floor, ceiling and the four walls of a rectangular room."""
    return np.concatenate([
        floor(x0, x1, y0, y1),
        floor(x0, x1, y0, y1, z=height),
        wall(x0, y0, y1, z1=height),
        wall(x1, y0, y1, z1=height),
        wall_along_x(y0, x0, x1, z1=height),
        wall_along_x(y1, x0, x1, z1=height),
    ])


def box(center, size):
    """Points filling an axis-aligned box."""
    lo = np.asarray(center) - np.asarray(size) / 2
    axes = [np.arange(lo[i], lo[i] + size[i] + 1e-9, STEP) for i in range(3)]
    return np.stack(np.meshgrid(*axes), axis=-1).reshape(-1, 3)


class ToyDataset(SceneDataset):
    """One hand-made scene, two 3 x 3 m rooms side by side.

        y ^
        3 +-----------+-----------+
          | bedroom   | bathroom  |
          |   bed     :   toilet  |      ':' doorway in the wall x = 3, y in [1, 1.9]
          |           |   tap     |
        0 +-----------+-----------+--> x
          0           3           6
    """

    ROOM_CATEGORIES = {"bedroom": "bedroom", "bathroom": "bathroom"}
    OBJECT_CATEGORIES = {"bed": "bed", "toilet": "toilet"}
    NOT_OBJECTS = {"tap": "fixture"}

    def scene_ids(self, split):
        return {"train": ["two_rooms"], "val": [], "test": []}[split]

    def load_scene(self, scene_id):
        if scene_id != "two_rooms":
            raise KeyError(f"no scene {scene_id!r} in the toy dataset")
        wall_between = np.concatenate([wall(3.0, 0.0, 1.0), wall(3.0, 1.9, 3.0)])
        return AnnotatedScene(
            scene_id=scene_id,
            regions=[
                Region(0, "bedroom", None, floor(0.0, 2.95, 0.0, 3.0), room_surfaces(0.0, 2.95, 0.0, 3.0)),
                Region(1, "bathroom", None, floor(3.05, 6.0, 0.0, 3.0), room_surfaces(3.05, 6.0, 0.0, 3.0)),
            ],
            objects=[
                ObjectInstance(10, "bed", None, 0, box([1.0, 1.5, 0.25], [2.0, 1.6, 0.5])),
                ObjectInstance(11, "toilet", None, 1, box([5.5, 2.5, 0.4], [0.4, 0.6, 0.8])),
                ObjectInstance(12, "tap", None, 1, box([5.0, 0.2, 1.0], [0.1, 0.1, 0.1])),
                ObjectInstance(13, None, None, 1, box([4.0, 0.5, 0.2], [0.3, 0.3, 0.4])),
            ],
            wall_points=wall_between,
            doors=[Door(20, wall(3.0, 1.0, 1.9, 0.0, 2.1))],
            extra={"source": "hand-made"},
        )


@pytest.fixture
def toy_dataset():
    return ToyDataset()
