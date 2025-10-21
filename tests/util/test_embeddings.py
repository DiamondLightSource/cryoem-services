from __future__ import annotations

import numpy as np

from cryoemservices.util.embeddings import distance_matrix


def test_distance_matrix_2d():
    coords = np.array([[1, 2, 3], [3, 2, 1]])
    dm = distance_matrix(coords)
    assert dm.shape == (3, 3)
    expected_result = np.array(
        [
            [0, np.sqrt(2), 2 * np.sqrt(2)],
            [np.sqrt(2), 0, np.sqrt(2)],
            [2 * np.sqrt(2), np.sqrt(2), 0],
        ]
    )
    assert dm == expected_result


def test_distance_matrix_3d():
    coords = np.array([[1, 2, 3], [3, 2, 1], [1, 2, 3]])
    dm = distance_matrix(coords)
    assert dm.shape == (3, 3)
    assert dm[0][0] == 0
    assert dm[0][2] == dm[2][0]
    coords = np.array([[1, 2, 3, 4], [3, 2, 1, 4], [1, 2, 3, 4]])
    dm = distance_matrix(coords)
    assert dm.shape == (4, 4)
    assert dm[0][0] == 0
    assert dm[0][2] == dm[2][0]
