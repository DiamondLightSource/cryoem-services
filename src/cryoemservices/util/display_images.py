from pathlib import Path

import mrcfile
import numpy as np


def generate_binned_mrc(input_path: Path, binning: int) -> Path:
    """Produce binned mrc files for display purposes"""
    with mrcfile.open(input_path) as mrc:
        input_header = mrc.header
        input_data = mrc.data

    # Bin the data and set values to a range of 0-127
    input_size = np.array(input_data.shape)
    output_size = (input_size / binning).astype("int")
    if not np.sum(input_size / output_size) == binning * 3:
        reduction = abs(output_size * binning - input_size)
        input_data = input_data[reduction[0] :, reduction[1] :, reduction[2] :]
    reshaped_data = input_data.reshape(
        output_size[0], binning, output_size[1], binning, output_size[2], binning
    )
    binned_data = reshaped_data.mean(5).mean(3).mean(1)
    binned_data -= binned_data.min()
    binned_data *= 127 / binned_data.max()

    # Edge clip all directions as segmentations often have edge artifacts
    binned_data[-5:] = 0
    binned_data[:5] = 0
    binned_data[:, :5] = 0
    binned_data[:, -5:] = 0
    binned_data[:, :, :5] = 0
    binned_data[:, :, -5:] = 0

    # Save output binned mrc
    mini_mrc_name = str(input_path.with_suffix("")) + f"_bin{binning}.mrc"
    with mrcfile.new(mini_mrc_name, overwrite=True) as mrc:
        mrc.set_data(binned_data.astype("int8"))
        mrc.header.cella = input_header.cella
    return Path(mini_mrc_name)


def spherical_clipping(
    input_tomogram: Path,
    output_tomogram: Path | None = None,
    tilt_axis: float | None = None,
    edge_cut: int = 10,
):
    """Clip a tomogram with zeros beyond a circle defined by the tilt size"""
    with mrcfile.open(input_tomogram) as mrc:
        data = np.copy(mrc.data)
        pixel_size = mrc.voxel_size

    # Data is ZYX, need to project around either Y or X depending on tilt axis
    side_projection = 2 if tilt_axis is not None and -45 < tilt_axis < 45 else 1

    dist_mat_bool = np.zeros((data.shape[0], data.shape[side_projection]), dtype=bool)
    radius = data.shape[side_projection] / 2 - edge_cut

    for i in range(edge_cut, data.shape[0] - edge_cut):
        for j in range(edge_cut, data.shape[side_projection] - edge_cut):
            # Record where distance from centre exceeds central slice size
            dist_ij = int(np.sqrt((i - radius) ** 2 + (j - radius) ** 2))
            dist_mat_bool[i + edge_cut, j + edge_cut] = dist_ij < radius

    if side_projection == 2:
        data[:, : edge_cut + 1, :] = 0
        data[:, data.shape[1] - edge_cut :, :] = 0
        for i in range(edge_cut, data.shape[1] - edge_cut):
            data[:, i, :][not dist_mat_bool] = 0
    else:
        data[:, :, : edge_cut + 1] = 0
        data[:, :, data.shape[2] - edge_cut :] = 0
        for i in range(edge_cut, data.shape[2] - edge_cut):
            data[:, :, i][not dist_mat_bool] = 0

    with mrcfile.new(
        output_tomogram if output_tomogram is not None else input_tomogram,
        overwrite=True,
    ) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = pixel_size
    return output_tomogram
