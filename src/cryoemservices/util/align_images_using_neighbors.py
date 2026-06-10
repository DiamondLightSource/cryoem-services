import logging
from itertools import combinations
from pathlib import Path
from typing import cast

import cv2
import numpy as np

from cryoemservices.util.image_processing import (
    apply_sobel_edge_filter,
    create_hanning_window,
    threshold_image,
    write_image,
)

logger = logging.getLogger(__name__)


def _filter_components(
    binary: np.ndarray,
    min_component_area: int | None = None,
):
    """
    Removes components in a binary image with an area smaller than the threshold.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        image=binary,
        labels=8,
    )
    # Populate blank image with only the components that meet the criteria
    filtered = np.zeros_like(binary)
    for i in range(1, n):
        if (
            min_component_area is not None
            and stats[i, cv2.CC_STAT_AREA] >= min_component_area
        ):
            filtered[labels == i] = 255
    return filtered


def _fill_holes(
    binary: np.ndarray,
    morph_close_kernel: int | None = None,
    morph_open_kernel: int | None = None,
):
    """
    Performs morphological closing to connect disjointed components in the binary
    image using the kernel size provided, before performing morphological opening
    to separate components that have been joined together by thin bridges using a
    different, usually smaller, kernel size.
    """
    if morph_close_kernel is not None:
        kernel_close = cv2.getStructuringElement(
            shape=cv2.MORPH_ELLIPSE,
            ksize=(morph_close_kernel, morph_close_kernel),
        )
        binary = cv2.morphologyEx(
            binary,
            op=cv2.MORPH_CLOSE,
            kernel=kernel_close,
        )
    if morph_open_kernel is not None:
        kernel_open = cv2.getStructuringElement(
            shape=cv2.MORPH_ELLIPSE,
            ksize=(morph_open_kernel, morph_open_kernel),
        )
        binary = cv2.morphologyEx(
            binary,
            op=cv2.MORPH_OPEN,
            kernel=kernel_open,
        )
    # Attempt to fill in holes after connecting components
    filled = np.zeros_like(binary)
    contours, _ = cv2.findContours(
        binary,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(
        filled,
        contours=contours,
        contourIdx=-1,
        color=255,
        thickness=cv2.FILLED,
    )
    return cast(np.ndarray, filled)


def _preprocess(
    img: np.ndarray,
    median_blur: int | None = None,
    gaussian_blur: float | None = None,
    use_hanning: bool = False,
    sobel_kernel: int | None = None,
    threshold_percentile: float | None = None,
    min_component_area: int | None = None,
    morph_close_kernel: int | None = None,
    morph_open_kernel: int | None = None,
    # Image saving parameters
    save_images: bool = False,
    save_dir: Path | None = None,
    name: str = "",
):
    height, width = img.shape[:2]
    processed = img

    # Apply median blur
    if median_blur is not None:
        processed = cv2.medianBlur(processed, median_blur)
        if save_images and save_dir and name:
            write_image(
                processed,
                save_dir,
                f"{name}_median.png",
            )
    # Apply Gaussian blur
    if gaussian_blur is not None:
        processed = cv2.GaussianBlur(processed, ksize=(0, 0), sigmaX=gaussian_blur)
        if save_images and save_dir and name:
            write_image(
                processed,
                save_dir,
                f"{name}_gaussian.png",
            )
    # Apply Hanning window
    if use_hanning:
        window = create_hanning_window(width=width, height=height).astype(np.float32)
        processed = processed.astype(np.float32) * window
        cv2.normalize(
            src=processed,
            dst=processed,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
        )
        processed = processed.astype(np.uint8)
        if save_images and save_dir and name:
            write_image(
                processed,
                save_dir,
                f"{name}_hanning.png",
            )
    # Apply Sobel filter
    if sobel_kernel is not None:
        processed = apply_sobel_edge_filter(processed, kernel_size=sobel_kernel)
        if save_images and save_dir and name:
            write_image(
                processed,
                save_dir,
                f"{name}_sobel.png",
            )
    # Threshold image
    binary = threshold_image(
        processed,
        percentile=threshold_percentile,
    )
    if save_images and save_dir and name:
        write_image(
            binary,
            save_dir,
            f"{name}_threshold.png",
        )
    # Filter out small features after thresholding
    if min_component_area is not None:
        binary = _filter_components(binary, min_component_area=min_component_area)
        if save_images and save_dir and name:
            write_image(
                binary,
                save_dir,
                f"{name}_filtered.png",
            )
    # Connect disjointed components in the binary image
    binary = _fill_holes(
        binary,
        morph_close_kernel=morph_close_kernel,
        morph_open_kernel=morph_open_kernel,
    )
    if save_images and save_dir and name:
        write_image(
            binary,
            save_dir,
            f"{name}_filled.png",
        )

    return binary


def _detect_features(
    binary: np.ndarray,
    min_feature_area: int | None = None,
    max_feature_area: int | None = None,
    min_solidity: float | None = None,
    min_ellipse_fit: float | None = None,
    max_aspect_ratio: float | None = None,
    # Image/file saving parameters
    save_images: bool = False,
    save_tables: bool = False,
    save_dir: Path | None = None,
    name: str = "",
    marker_size: int = 3,
    line_thickness: int = 2,
    font_scale: float = 1.0,
    text_offset: int = 20,
) -> np.ndarray:
    """
    Identifies features in a thresholded image that fulfil the criteria specified.
    Returns a list of descriptors for each feature, along with either None if
    'save_images' is False or an annotated version of the image if True. The
    returned features array will contain the following columns:
    - x- and y- coordinates of the feature's centroid
    - The areas of the feature contour and it convex hull
    - The minor and major axes of the fitted ellipse
    - The angulr orientation of the fitted ellipse
    """
    contours, _ = cv2.findContours(
        binary,
        mode=cv2.RETR_LIST,
        method=cv2.CHAIN_APPROX_SIMPLE,
    )

    # Create RGB version of binary image for annotation or a None placeholder
    annotated = (
        cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        if save_images and save_dir and name
        else None
    )

    # Keep only features that match criteria, and extract descriptors
    features_list = []
    index = 0
    for contour in contours:
        # Remove features that cannot be used for ellipse fitting
        if len(contour) < 5:
            continue
        # Check contour area feature
        area = cv2.contourArea(contour)
        if min_feature_area is not None and area < min_feature_area:
            continue
        if max_feature_area is not None and area > max_feature_area:
            continue

        # Check convex hull area
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        # Filter out bad fits
        if hull_area <= 0:
            continue

        # Estimate solidity (how much of the hull the feature takes up)
        solidity = area / hull_area
        if solidity < min_solidity:
            continue

        # Elliptical fit for the feature
        ellipse = cv2.fitEllipse(hull)
        (ex, ey), (w, h), angle = ellipse
        # w = short axis
        # h = long axis

        # Exclude bad elliptical fits
        # Much larger than actual area
        ellipse_area = np.pi * (w / 2) * (h / 2)
        if min_ellipse_fit is not None and hull_area / ellipse_area < min_ellipse_fit:
            continue

        # Too circular
        aspect = min(w, h) / max(w, h)
        if max_aspect_ratio is not None and aspect > max_aspect_ratio:
            continue

        # Adjust the returned angle so that:
        # up = 0,
        # clockwise = positive
        # anticlockwise = negative
        if angle > 90:
            angle -= 180

        # Find the centroids of the features using the convex hull
        hull_moments = cv2.moments(hull)
        cx = hull_moments["m10"] / hull_moments["m00"]
        cy = hull_moments["m01"] / hull_moments["m00"]

        # Append results
        features_list.append(
            (
                cx,
                cy,
                area,
                hull_area,
                w,
                h,
                angle,
            )
        )

        # Annotate image
        if annotated is not None:
            # Draw the outline of the fitted contour and convex hull
            cv2.drawContours(
                annotated,
                contours=[contour, hull],
                contourIdx=-1,
                color=(0, 0, 255),
                thickness=line_thickness,
            )
            # Draw the fitted ellipse
            cv2.ellipse(
                annotated,
                box=ellipse,
                color=(0, 255, 255),
                thickness=line_thickness,
            )
            # Add a marker
            cv2.circle(
                annotated,
                center=(int(cx), int(cy)),
                radius=marker_size,
                color=(0, 255, 0),
                thickness=line_thickness,
            )
            # Add the index number
            cv2.putText(
                annotated,
                text=f"{index}",
                org=(int(cx) + text_offset, int(cy)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale,
                color=(255, 255, 255),
                thickness=line_thickness,
                lineType=cv2.LINE_AA,
            )

        # Increment the index for next loop once successful
        index += 1

    # Convert to array or return empty array
    if features_list:
        features = np.array(features_list, dtype=np.float32)
    else:
        features = np.empty((0, 7), dtype=np.float32)

    # Optionally save results
    if annotated is not None and name and save_dir:
        write_image(
            annotated,
            save_dir,
            f"{name}_features.png",
        )
    if save_tables and save_dir and name:
        np.savetxt(
            save_dir / f"{name}_features.tsv",
            np.column_stack((np.arange(len(features)), features)),
            fmt=[
                "%-5d",
                "%-10.3f",
                "%-10.3f",
                "%-10.3f",
                "%-10.3f",
                "%-10.3f",
                "%-10.3f",
                "%-10.3f",
            ],
            header=(
                f"{'':<3} "
                f"{'x':<10} "
                f"{'y':<10} "
                f"{'cont_area':<10} "
                f"{'hull_area':<10} "
                f"{'w':<10} "
                f"{'h':<10} "
                f"{'angle':<10} "
            ),
        )
    return features


def _build_descriptor(
    features: np.ndarray,
    max_neighbor_distance: float | None = None,
    eps: float = 1e-9,
    # Save parameters
    save_tables: bool = False,
    save_dir: Path | None = None,
    name: str = "",
):
    """
    Build geometric descriptors describing the arrangement of neighbouring
    features around each parent feature.

    Descriptors captured for a given feature O include:
    - The ratio of the distances between two neighbours N (near) and F (far)
    - The signed angle formed by the vector NOF
    - The natural logs of the contour and convex hull areas of N and F
    - The natural logs of the minor and major axes of the ellipses fitted to N and F
    """
    # Safe early exit if no features were provided
    if len(features) == 0:
        descriptors = np.empty((0, 13), dtype=np.float32)
    else:
        # Precompute distances between points via matrix operations
        coords = features[:, :2]  # x, y
        distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

        desc_list = []
        num_features = len(features)
        for o in range(num_features):
            # Only look at nearest neighbours
            if max_neighbor_distance is not None:
                neighbors = np.where(
                    (distances[o] <= max_neighbor_distance) & (distances[o] > 0)
                )[0]
            else:
                neighbors = np.array([i for i in range(num_features) if i != o])

            # Skip if there aren't at least 2 neighboirs
            if len(neighbors) < 2:
                continue

            O = coords[o]

            # Iterate through all unique neighbour combinations
            for a, b in combinations(neighbors, 2):
                dA = distances[o, a]
                dB = distances[o, b]

                # Order neighbours from nearest to furthest
                if dA < dB:
                    n, f = a, b
                    d_near, d_far = dA, dB
                else:
                    n, f = b, a
                    d_near, d_far = dB, dA

                # Calculate the signed angle of vector NOF
                N = coords[n]
                F = coords[f]

                NO = O - N
                OF = F - O
                cross = (NO[0] * OF[1]) - (NO[1] * OF[0])
                dot = NO @ OF
                angle = np.arctan2(cross, dot)

                # Extract contour and ellipse descriptors
                feat_near = features[n][2:6]
                feat_far = features[f][2:6]

                # Precompute logs of the features
                (
                    area_n,
                    hull_n,
                    wn,
                    hn,
                ) = np.log(np.where(feat_near <= 0, eps, feat_near))
                (
                    area_f,
                    hull_f,
                    wf,
                    hf,
                ) = np.log(np.where(feat_far <= 0, eps, feat_far))

                # Append
                desc_list.append(
                    [
                        o,
                        n,
                        f,
                        d_near / (d_far + eps),
                        angle,
                        area_n,
                        hull_n,
                        wn,
                        hn,
                        area_f,
                        hull_f,
                        wf,
                        hf,
                    ]
                )
        if len(desc_list) == 0:
            descriptors = np.empty((0, 13), dtype=np.float32)
        else:
            descriptors = np.asarray(desc_list, dtype=np.float32)

    # Save tables if set
    if save_tables and save_dir and name:
        desc = np.column_stack((np.arange(len(descriptors)), descriptors))
        np.savetxt(
            save_dir / f"{name}_desc.tsv",
            desc,
            fmt=[
                "%-5d",
                "%-3d",
                "%-3d",
                "%-3d",
                "%-10.3f",
                "%-10.3f",
                "%-10.3f",
                "%-10.3f",
                "%-10.3f",
                "%-10.3f",
                "%-10.3f",
                "%-10.3f",
                "%-10.3f",
                "%-10.3f",
            ],
            header=(
                f"{'':<3} "
                f"{'o':<3} "
                f"{'n':<3} "
                f"{'f':<3} "
                f"{'d_ratio':<10} "
                f"{'angle':<10} "
                f"{'area_near':<10} "
                f"{'hull_near':<10} "
                f"{'w_near':<10} "
                f"{'h_near':<10} "
                f"{'area_far':<10} "
                f"{'hull_far':<10} "
                f"{'w_far':<10} "
                f"{'h_far':<10} "
            ),
        )
    return descriptors


def _calculate_similarity(
    ref_desc: np.ndarray, mov_desc: np.ndarray, eps: float = 1e-9
):
    """
    Compare descriptor sets using vectorised NumPy operations.

    This replaces extremely slow Python nested loops.
    """

    # Expand dimensions so that every descriptor is compared against every other
    ref = ref_desc[:, None, :]  # 'Ref' will be the outermost axis
    mov = mov_desc[None, :, :]  # 'Mov' will be the next axis in

    # Compare neighbour distance ratios
    ratio_ref = ref[..., 0]
    ratio_mov = mov[..., 0]
    d_ratio = np.abs(ratio_mov - ratio_ref) / (ratio_ref + ratio_mov + eps)

    # Compare signed angles
    angle_ref = ref[..., 1]
    angle_mov = mov[..., 1]
    d_theta = (
        np.abs(
            np.arctan2(
                np.sin(angle_mov - angle_ref),
                np.cos(angle_mov - angle_ref),
            )
        )
        / np.pi
    )

    # Compare ellipse dimensions
    d_area_near = np.linalg.norm(mov[..., 2:4] - ref[..., 2:4], axis=-1)
    d_area_far = np.linalg.norm(mov[..., 6:8] - ref[..., 6:8], axis=-1)
    d_area = 0.5 * (d_area_near + d_area_far)

    # Compare combined contour area-hull area differences
    d_elps_near = np.linalg.norm(mov[..., 4:6] - ref[..., 4:6], axis=-1)
    d_elps_far = np.linalg.norm(mov[..., 8:10] - ref[..., 8:10], axis=-1)
    d_elps = 0.5 * (d_elps_near + d_elps_far)

    # Calculate the combined feature-space distance
    # Smaller distances indicate larger similarity scores
    D = 4.0 * d_theta + 2.0 * d_ratio + 0.5 * d_elps + 1.0 * d_area
    return cast(np.ndarray, np.exp(-D))


def _match_features(
    ref_features: np.ndarray,
    mov_features: np.ndarray,
    max_neighbor_distance: float | None = None,
    min_score: float | None = None,
    # Saving parameters
    save_tables: bool = False,
    save_dir: Path | None = None,
):
    """
    Constructs descriptors for each feature in the arrays provided, which are then
    used to compute a similarity score between the two features. Only the features
    for which they are each other's best match are retained.

    For instance, for Point A in the reference image and Point B in the moving one,
    Point B must be the highest scoring candidate for Point A, and Point A must be
    Point B's highest scoring candidate for it to be considered.
    """

    ref_desc = _build_descriptor(
        ref_features,
        max_neighbor_distance=max_neighbor_distance,
        save_tables=save_tables,
        save_dir=save_dir,
        name="ref",
    )
    mov_desc = _build_descriptor(
        mov_features,
        max_neighbor_distance=max_neighbor_distance,
        save_tables=save_tables,
        save_dir=save_dir,
        name="mov",
    )

    if len(ref_desc) == 0 or len(mov_desc) == 0:
        logger.warning("Could not compile descriptors from the detected features")
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
        )

    # Calculate the similarity scorees between all descriptors
    len_ref, len_mov = len(ref_features), len(mov_features)
    scores = np.zeros((len_ref, len_mov), dtype=np.float32)
    for i in range(len_ref):
        ref_desc_curr = ref_desc[ref_desc[:, 0] == i][:, 3:]
        # Early skip if there are no descriptors for this feature
        if len(ref_desc_curr) == 0:
            continue
        for j in range(len_mov):
            mov_desc_curr = mov_desc[mov_desc[:, 0] == j][:, 3:]
            # Early skip if there are no descriptors for this feature
            if len(mov_desc_curr) == 0:
                continue

            # Calculate the similarity score between each feature's descriptors
            score_curr = _calculate_similarity(
                ref_desc_curr,
                mov_desc_curr,
            )

            # Pick the highest score in mov for each descriptor in ref, then average
            # Do the reverse for mov
            score_ref = np.mean(np.max(score_curr, axis=1))
            score_mov = np.mean(np.max(score_curr, axis=0))
            # Add the average to the score matrix
            scores[i, j] = 0.5 * (score_ref + score_mov)

    if save_tables and save_dir is not None:
        np.savetxt(
            save_dir / "scores.tsv",
            np.column_stack((np.arange(len(scores)), scores)),
            fmt=(["%-5d"] + ["%-10.3f"] * len(mov_features)),
            header=(
                f"{'':<3} " + "".join(f"{i:<10} " for i in range(len(mov_features)))
            ),
        )

    # Keep only mutual best matches and those above min threshold
    ref_best = np.argmax(scores, axis=1)
    mov_best = np.argmax(scores, axis=0)
    ref_list: list[int] = []
    mov_list: list[int] = []
    score_list: list[float] = []
    for ref_idx, mov_idx in enumerate(ref_best):
        if mov_best[mov_idx] == ref_idx:
            score = float(scores[ref_idx, mov_idx])
            if min_score is not None and score < min_score:
                continue
            ref_list.append(ref_idx)
            mov_list.append(mov_idx)
            score_list.append(score)
    ref_matches = ref_features[ref_list][:, :2]
    mov_matches = mov_features[mov_list][:, :2]
    # Display logs as appropriate
    if ref_list:
        logger.info(
            (
                "Found matches:\n"
                + "".join(
                    f"{i} -> {j} ({score})\n"
                    for i, j, score in zip(ref_list, mov_list, score_list)
                )
            )
        )
    else:
        logger.warning("No matches found for current criteria")
    return ref_matches, mov_matches


def _draw_matches(
    ref: np.ndarray,
    mov: np.ndarray,
    # List of features
    ref_features: np.ndarray,
    mov_features: np.ndarray,
    # List of coordinates of matches
    ref_match: np.ndarray,
    mov_match: np.ndarray,
    # Image saving parameters
    save_dir: Path,
    marker_size: int = 3,
    line_thickness: int = 2,
    font_scale: float = 1.0,
    text_offset: int = 20,
):
    """
    Overlay the original reference and moving images on top of one another and add
    annotations showing how the features from the reference image map onto those in
    the moving image.
    """

    annotated = cv2.addWeighted(
        src1=ref,
        alpha=0.5,
        src2=mov,
        beta=0.5,
        gamma=0,
    )
    annotated = cv2.cvtColor(annotated, code=cv2.COLOR_GRAY2BGR)

    # Mark the points
    for (x0, y0), (x1, y1) in zip(ref_match, mov_match):
        x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
        cv2.line(
            annotated,
            pt1=(x0, y0),
            pt2=(x1, y1),
            color=(0, 255, 255),
            thickness=line_thickness,
        )
    for features, color in (
        (ref_features, (0, 255, 0)),
        (mov_features, (0, 0, 255)),
    ):
        for i, (x, y) in enumerate(features[:, :2]):
            x, y = int(x), int(y)
            cv2.circle(
                annotated,
                center=(x, y),
                radius=marker_size,
                color=color,
                thickness=line_thickness,
            )
            cv2.putText(
                annotated,
                text=f"{i}",
                org=(x + text_offset, y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale,
                color=color,
                thickness=line_thickness,
                lineType=cv2.LINE_AA,
            )
    write_image(
        annotated,
        save_dir,
        "matched_holes.png",
    )


def align_images_using_neighbors(
    reference_array: np.ndarray,
    moving_array: np.ndarray,
    # Preprocessing
    median_blur: int | None = None,
    gaussian_blur: float | None = 3,
    sobel_kernel: int | None = 3,
    use_hanning: bool = False,
    threshold_percentile: float | None = 99,
    min_component_area: int | None = 200,
    morph_close_kernel: int | None = 19,
    morph_open_kernel: int | None = 3,
    # Feature detection
    min_feature_area: int | None = 400,
    max_feature_area: int | None = 5000,
    min_solidity: float | None = 0.6,
    min_ellipse_fit: float | None = 0.4,
    max_aspect_ratio: float | None = 0.9,
    # Similarity calculation and registration
    max_neighbor_distance: float | None = 400,
    min_score: float | None = 0.2,
    ransac_threshold: float = 10,
    # Debug options
    save_images: bool = False,
    save_tables: bool = False,
    save_dir: Path | None = None,
):
    """
    Image registration function that identifies features in the reference and moving
    images and aligns them by using the geometry of the features' neighbours to work
    out the transformation matrix.

    Parameters
    ----------
    reference_array: np.ndarray
        The reference image. Currently, only 2D grayscale images are supported.
    moving_array: np.ndarray
        The image to be aligned.
    median_blur: int | None
        The kernel size of the median blur to be applied to the image.
        The default is None.
    gaussian_blur: float | None
        The sigma value to use to Gaussian blur the image.
        The default is 3.
    sobel_kernel: int | None
        The kernel size to use for the Sobel edge filter. Larger kernel sizes
        mean that the pixel intensity is computed over a larger radius.
        The default is 3.
    use_hanning: bool
        Decide whether to apply a Hanning window to the image. This dampens the
        intensity of the image smoothly towards the edges, emphasising features
        towards the centre of the image.
        The default is False.
    threshold_percentile: float | None
        The nth percentile of the pixel values to threshold the image to. If
        set to None, The function will use the Otsu method to attempt to find
        a suitable threshold value.
        The default is 99.0.
    min_component_area: int | None
        The minimum pixel area a feature in the thresholded image must have to
        be considered for feature detection. This helps reduce noisy artifacts
        left over after the initial thresholding operation.
        The default is 200.
    morph_close_kernel: int | None
        The kernel size to use when joining fragmented features and filling in
        holes formed by the features after thresholding.
        The default is 19.
    morph_open_kernel: int | None
        The kernel size to use when separating features in the thresholded image.
        This is used to separate features that might have been joined together
        by the previous step due to their close proximity to one another.
        The default is 3
    min_feature_area: int | None
        The minimimum pixel area of the feature for it to be used for computing
        the transformation matrix.
        The default is 400.
    max_feature_area: int | None
        The maximum pixel area of the feature to be used.
        The default is 5000.
    min_solidity: float | None
        The minimum ratio between the feature's actual area and the area of the
        smallest convex hull it is bound by. This is used to reject features
        that are highly irregular in shape.
        The default is 0.6.
    min_ellipse_fit: float | None
        The min ratio of the fitted ellipse's area to the convex hull area of the
        feature. A ratio of 1.0 means that their areas are identical, while 0.0
        means that the ellipse area is inifinitely larger than that of the hull's.
        The default value is 0.4.
    max_aspect_ratio: float | None
        The maximum aspect ratio of the ellipse fitted around a feature, beyond
        which the feature will not be used to compute the transformation matrix
        with. A ratio of 1 indicates that the ellipse is a perfect sphere, and
        a value of 0.5 indicates that its major axis is twice as long as its
        minor one.
        The default is 0.9.
    max_neighbor_distance: float | None
        The maximum pixel distance between two features for them to be included
        in the similarity score computation.
        The default is 400.0.
    min_score: float | None
        The minimum similarity score between features in the reference and moving
        images for their coordinates to be considered when computing the final
        transformation matrix.
        The default is 0.2
    ransac_threshold: float
        The maximum pixel distance the points between the reference and moving
        images are allowed to differ from one another before the computed
        transform is considered bad.
        The default is 5.
    save_images: bool
        Toggle whether to save intermediate images.
        The default is False.
    save_tables: bool
        Toggle whether to save intermediate tables.
        The default is False.
    save_dir: Path | None
        If either 'save_tables' or 'save_images' is set tot True, this determines
        the folder to save the output files to. If this is not set, a warning will
        be logged and the files will not be generated.
        The default is None.

    Returns
    -------
    result: dict[str, np.ndarray]
        Contains the aligned image and the transformation matrix, stored under the
        keys "aligned" and "transform" respectively.

    """

    # Let Python control multithreading
    cv2.setNumThreads(1)

    # Check if save parameters have been set correctly
    if (save_tables or save_images) and save_dir is None:
        logger.warning(
            "No save directory provided even though saving of intermediate results "
            "was specified. Intermediate results will not be saved."
        )

    # Use the height and width to determine suitable values for figures
    height, width = reference_array.shape[:2]

    if (num_pixels := height * width) > 4096**2:
        line_thickness = 4
        marker_size = 5
        font_scale = 2.0
        text_offset = 32
    elif num_pixels > 2048**2:
        line_thickness = 3
        marker_size = 4
        font_scale = 1.5
        text_offset = 24
    elif num_pixels > 1024**2:
        line_thickness = 2
        marker_size = 3
        font_scale = 1.0
        text_offset = 16
    elif num_pixels > 512**2:
        line_thickness = 1
        marker_size = 3
        font_scale = 0.75
        text_offset = 12
    elif num_pixels > 256**2:
        line_thickness = 1
        marker_size = 3
        font_scale = 0.5
        text_offset = 10
    else:
        line_thickness = 1
        marker_size = 2
        font_scale = 0.25
        text_offset = 8

    # Preprocess images to get binaries
    ref_bin = _preprocess(
        reference_array,
        median_blur=median_blur,
        gaussian_blur=gaussian_blur,
        use_hanning=use_hanning,
        sobel_kernel=sobel_kernel,
        threshold_percentile=threshold_percentile,
        min_component_area=min_component_area,
        morph_close_kernel=morph_close_kernel,
        morph_open_kernel=morph_open_kernel,
        # Image saving
        save_images=save_images,
        save_dir=save_dir,
        name="ref",
    )
    mov_bin = _preprocess(
        moving_array,
        median_blur=median_blur,
        gaussian_blur=gaussian_blur,
        use_hanning=use_hanning,
        sobel_kernel=sobel_kernel,
        threshold_percentile=threshold_percentile,
        min_component_area=min_component_area,
        morph_close_kernel=morph_close_kernel,
        morph_open_kernel=morph_open_kernel,
        # Image saving
        save_images=save_images,
        save_dir=save_dir,
        name="mov",
    )

    # Identify features in binarised images
    ref_features = _detect_features(
        ref_bin,
        min_feature_area=min_feature_area,
        max_feature_area=max_feature_area,
        min_solidity=min_solidity,
        min_ellipse_fit=min_ellipse_fit,
        max_aspect_ratio=max_aspect_ratio,
        # Image saving
        save_images=save_images,
        save_tables=save_tables,
        save_dir=save_dir,
        name="ref",
        marker_size=marker_size,
        line_thickness=line_thickness,
        font_scale=font_scale,
        text_offset=text_offset,
    )
    mov_features = _detect_features(
        mov_bin,
        min_feature_area=min_feature_area,
        max_feature_area=max_feature_area,
        min_solidity=min_solidity,
        min_ellipse_fit=min_ellipse_fit,
        max_aspect_ratio=max_aspect_ratio,
        # Image saving
        save_images=save_images,
        save_tables=save_tables,
        save_dir=save_dir,
        name="mov",
        marker_size=marker_size,
        line_thickness=line_thickness,
        font_scale=font_scale,
        text_offset=text_offset,
    )
    if len(ref_features) == 0 or len(mov_features) == 0:
        logger.warning("Could not identify any features in the images")
        return {
            "aligned": moving_array,
            "transform": None,
        }

    # Run the feature matching algorithm
    ref_match, mov_match = _match_features(
        ref_features,
        mov_features,
        max_neighbor_distance=max_neighbor_distance,
        min_score=min_score,
        # Saving parameters
        save_tables=save_tables,
        save_dir=save_dir,
    )
    if len(ref_match) == 0 or len(mov_match) == 0:
        logger.warning("Could not identify matching features between the images")
        return {
            "aligned": moving_array,
            "transform": None,
        }
    if save_images and save_dir:
        _draw_matches(
            reference_array,
            moving_array,
            ref_features,
            mov_features,
            ref_match,
            mov_match,
            save_dir=save_dir,
            marker_size=marker_size,
            line_thickness=line_thickness,
            font_scale=font_scale,
            text_offset=text_offset,
        )

    # Use the matched points to estimate the similarity transform
    transform, _ = cv2.estimateAffinePartial2D(
        from_=np.ascontiguousarray(mov_match),
        to=np.ascontiguousarray(ref_match),
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
    )
    if transform is None:
        logger.warning("Affine transform estimation failed")
        return {
            "aligned": moving_array,
            "transform": transform,
        }
    aligned = cv2.warpAffine(
        moving_array,
        M=transform,
        dsize=(
            moving_array.shape[1],
            moving_array.shape[0],
        ),
    )
    if save_images and save_dir:
        overlay = cv2.addWeighted(
            src1=reference_array,
            alpha=0.5,
            src2=aligned,
            beta=0.5,
            gamma=0,
        )
        write_image(
            overlay,
            save_dir,
            "overlay.png",
        )

    return {
        "aligned": aligned,
        "transform": transform,
    }
