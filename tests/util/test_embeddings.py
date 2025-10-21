from __future__ import annotations

import numpy as np
import pytest

from cryoemservices.util.embeddings import distance_exp_score, distance_matrix


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
    assert (dm == expected_result).all()


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


def test_distance_exp_score_wrong_dimensions():
    with pytest.raises(ValueError):
        selections = np.array([0, 0, 0, 0])
        distances = np.array([1, 2, 3, 4, 5])
        distance_exp_score(selections, distances)


def test_distance_exp_score_no_selections():
    selections = np.array([0, 0, 0, 0, 0])
    distances = np.array([1, 2, 3, 4, 5])
    score = distance_exp_score(selections, distances)
    assert score == 1


def test_distance_exp_score_with_selections():
    selections = np.array([0, 1, 0, 4, 0])
    distances = np.array([1, 2, 3, 4, 5])
    score = distance_exp_score(selections, distances)
    assert score == (np.exp(-2 / 5) + 4 * np.exp(-1 / 5)) / 5
