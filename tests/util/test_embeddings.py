from __future__ import annotations

import numpy as np
import pytest

from cryoemservices.util.embeddings import (
    distance_augmented_sort,
    distance_exp_score,
    distance_matrix,
)


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
        distances = np.array([[0, 1], [1, 0]])
        distance_exp_score(selections, distances)


def test_distance_exp_score_no_selections():
    selections = np.array([0, 0, 0])
    distances = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    score = distance_exp_score(selections, distances)
    assert (score == np.array([1, 1, 1])).all()


def test_distance_exp_score_with_selections():
    selections = np.array([0, 1, 0])
    distances = np.array([[0, 2, 3], [2, 0, 4], [3, 4, 0]])
    score = distance_exp_score(selections, distances)
    assert (score == np.array([np.exp(-1), np.exp(-10), np.exp(-0.5)])).all()


def test_augmented_sort_with_no_distances():
    scores = np.array([5, 4, 3, 2, 1])
    classes = np.array([0, 0, 1, 0, 1])
    distance_matrix = np.array([[0, 0], [0, 0]])
    sorted_indices = distance_augmented_sort(scores, classes, distance_matrix)
    assert (sorted_indices == np.array([0, 1, 2, 3, 4])).all()


def test_augmented_sort_with_small_distance():
    scores = np.array([5, 4, 3, 2, 1])
    classes = np.array([0, 0, 1, 0, 1])
    distance_matrix = np.array([[0, 1], [1, 0]])
    sorted_indices = distance_augmented_sort(scores, classes, distance_matrix)
    assert (sorted_indices == np.array([0, 2, 1, 3, 4])).all()


def test_augmented_sort_with_3_classes():
    scores = np.array([5, 4, 3, 2, 1])
    classes = np.array([0, 2, 1, 0, 1])
    distance_matrix = np.array([[0, 1, 10], [1, 0, 9], [10, 9, 0]])
    sorted_indices = distance_augmented_sort(scores, classes, distance_matrix)
    assert (sorted_indices == np.array([0, 1, 2, 3, 4])).all()
