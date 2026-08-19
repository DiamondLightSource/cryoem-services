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
