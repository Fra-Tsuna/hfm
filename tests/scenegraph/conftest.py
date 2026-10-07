import numpy as np
import pytest

from src.scenegraph.graph import GEOM_DIM, POS, SIZE, THETA, SceneGraph, geometry_mask


@pytest.fixture
def toy():
    """A building with two connected rooms, one object in each.

    node:  0 building   1 room   2 room   3 object (in room 1)   4 object (in room 2)
    """
    node_type = np.array([0, 1, 1, 2, 2])
    geometry = np.zeros((5, GEOM_DIM))
    geometry[1:, POS] = np.arange(12).reshape(4, 3)    # distinct centers, to see where nodes move
    geometry[1:, SIZE] = 1.0
    geometry[3:, THETA] = [0.1, 0.2]
    return SceneGraph(
        scene_id="toy",
        node_type=node_type,
        category_id=np.array([0, 3, 2, 7, 4]),
        geometry=geometry,
        geometry_norm=geometry / 10,
        geometry_mask=geometry_mask(node_type),
        edge_index=np.array([[0, 0, 1, 2, 1],
                             [1, 2, 3, 4, 2]]),
        edge_type=np.array([0, 0, 1, 1, 2]),
        room_parent=np.array([-1, 0, 0, -1, -1]),
        object_parent=np.array([-1, -1, -1, 1, 2]),
        raw_metadata={"nodes": [{"id": f"n{i}"} for i in range(5)]},
    )
