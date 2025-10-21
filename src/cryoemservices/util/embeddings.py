import numpy as np


def distance_matrix(coords: np.array) -> np.array:
    diffs = np.zeros((len(coords), len(coords[0]), len(coords[0])))
    for i, xs in enumerate(coords):
        diffs[i] = np.subtract.outer(xs, xs)
    return np.sqrt(np.sum(diffs**2, axis=0))
