import numpy as np


def distance_matrix(coords: np.array) -> np.array:
    diffs = np.zeros((len(coords), len(coords[0]), len(coords[0])))
    for i, xs in enumerate(coords):
        diffs[i] = np.subtract.outer(xs, xs)
    return np.sqrt(np.sum(diffs**2, axis=0))


def distance_exp_score(selection_hist: np.array, distances: np.array) -> float:
    if len(selection_hist) != len(distances):
        raise ValueError(
            f"Size of selection histogram must match the number of distances: {len(selection_hist)} != {len(distances)}"
        )
    if not (num_selections := np.sum(selection_hist)):
        return 1
    normalised_distances = distances / np.min(distances)
    return (
        np.sum(
            selection_hist
            * np.exp(-(selection_hist / num_selections) / normalised_distances)
        )
        / num_selections
    )
