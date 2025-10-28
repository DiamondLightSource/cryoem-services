import numpy as np


def distance_matrix(coords: np.array) -> np.array:
    diffs = np.zeros((len(coords), len(coords[0]), len(coords[0])))
    for i, xs in enumerate(coords):
        diffs[i] = np.subtract.outer(xs, xs)
    return np.sqrt(np.sum(diffs**2, axis=0))


def distance_exp_score(selection_hist: np.array, distance_matrix: np.array) -> np.array:
    if len(selection_hist) != len(distance_matrix[0]):
        raise ValueError(
            f"Size of selection histogram must match the number of distances: {len(selection_hist)} != {len(distance_matrix[0])}"
        )
    if not (num_selections := np.sum(selection_hist)) or not np.sum(distance_matrix):
        return np.full(selection_hist.shape, 1)

    nonzero_min = np.min(distance_matrix[distance_matrix != 0])
    normalised_distances = distance_matrix / nonzero_min
    normalised_distances[normalised_distances == 0] = 0.1
    return (
        np.sum(
            selection_hist
            * np.exp(-(selection_hist / num_selections) / normalised_distances),
            axis=1,
        )
        / num_selections
    )


def distance_augmented_sort(
    scores: np.array, classes: np.array, distance_matrix: np.array
) -> np.array:
    selected_indices = np.zeros(scores.shape, dtype=np.uint32)
    combined_scores = np.zeros(scores.shape)
    selection_hist = np.zeros(distance_matrix.shape[0], dtype=np.uint32)
    for n in range(scores.shape[0]):
        class_exp_scores = distance_exp_score(selection_hist, distance_matrix)
        exp_scores_per_particle = np.array([class_exp_scores[ci] for ci in classes])
        combined_scores = scores * exp_scores_per_particle
        if not np.sum(combined_scores):
            missing_indices = set(range(selected_indices.shape[0])) - set(
                selected_indices
            )
            for i, mi in enumerate(missing_indices):
                selected_indices[
                    selected_indices.shape[0] - len(missing_indices) + i
                ] = mi
            break
        else:
            selected_index = np.argmax(combined_scores)
            selected_indices[n] = selected_index
            scores[selected_index] = 0
            selection_hist[classes[selected_index]] += 1
    return selected_indices
