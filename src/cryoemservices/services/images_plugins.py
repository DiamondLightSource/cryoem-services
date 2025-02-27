from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

import mrcfile
import numpy as np
import pandas as pd
import PIL.Image
import starfile
from PIL import ImageDraw, ImageEnhance, ImageFilter, ImageFont

logger = logging.getLogger("cryoemservices.services.images_plugins")
logger.setLevel(logging.INFO)


def required_parameters(parameters: Callable, required_keys: list[str]):
    """Make sure all parameters which aren't nullable are there"""
    for param_key in required_keys:
        if parameters(param_key) in [False, "False", None, "None"]:
            logger.error(f"Required key {param_key} not valid: {parameters(param_key)}")
            return False
    return True


def mrc_to_jpeg(plugin_params: Callable):
    if not required_parameters(plugin_params, ["file"]):
        return False
    filepath = Path(plugin_params("file"))
    allframes = plugin_params("all_frames")
    if not filepath.is_file():
        logger.error(f"File {filepath} not found")
        return False
    start = time.perf_counter()
    try:
        with mrcfile.open(filepath) as mrc:
            data = mrc.data
    except ValueError:
        logger.error(
            f"File {filepath} could not be opened. It may be corrupted or not in mrc format"
        )
        return False
    outfile = filepath.with_suffix(".jpeg")
    outfiles = []
    if len(data.shape) == 2:
        mean = np.mean(data)
        sdev = np.std(data)
        sigma_min = mean - 3 * sdev
        sigma_max = mean + 3 * sdev
        data = np.ndarray.copy(data)
        data[data < sigma_min] = sigma_min
        data[data > sigma_max] = sigma_max
        data = data - data.min()
        data = data * 255 / data.max()
        data = data.astype("uint8")
        im = PIL.Image.fromarray(data, mode="L")

        if plugin_params("pixel_spacing"):
            scalebar_nm = float(plugin_params("pixel_spacing")) / 10 * data.shape[0] / 3
            colour_im = im.convert("RGB")
            dim = ImageDraw.Draw(colour_im)
            dim.line(
                ((20, data.shape[0] / 3), (20, data.shape[0] * 2 / 3)),
                fill="yellow",
                width=5,
            )
            font_to_use = ImageFont.load_default(size=26)
            dim.text(
                (25, data.shape[0] / 2),
                f"{scalebar_nm:.0f} nm",
                anchor="lm",
                font=font_to_use,
                fill="yellow",
            )
            colour_im.save(outfile)
        else:
            im.save(outfile)
    elif len(data.shape) == 3:
        if allframes:
            for i, frame in enumerate(data):
                frame = frame - frame.min()
                frame = frame * 255 / frame.max()
                frame = frame.astype("uint8")
                im = PIL.Image.fromarray(frame, mode="L")
                frame_outfile = outfile.parent / f"{outfile.stem}_{i+1}.jpeg"
                im.save(frame_outfile)
                outfiles.append(frame_outfile)
        else:
            data = data - data[0].min()
            data = data * 255 / data[0].max()
            data = data.astype("uint8")
            im = PIL.Image.fromarray(data[0], mode="L")
            im.save(outfile)
    timing = time.perf_counter() - start

    logger.info(
        f"Converted mrc to jpeg {filepath} -> {outfile} in {timing:.1f} seconds",
        extra={"image-processing-time": timing},
    )
    if outfiles:
        return outfiles
    return outfile


def picked_particles(plugin_params: Callable):
    if not required_parameters(plugin_params, ["file", "diameter", "outfile"]):
        return False
    basefilename = Path(plugin_params("file"))
    if basefilename.suffix == ".jpeg":
        logger.info(f"Replacing jpeg extension with mrc extension for {basefilename}")
        basefilename = basefilename.with_suffix(".mrc")
    coords = plugin_params("coordinates")
    selected_coords = plugin_params("selected_coordinates")
    pixel_size = plugin_params("pixel_size")
    if not pixel_size:
        # Legacy case of zocalo-relion
        pixel_size = plugin_params("angpix") or 1
    diam = plugin_params("diameter")
    contrast_factor = plugin_params("contrast_factor") or 6
    outfile = plugin_params("outfile")
    if not basefilename.is_file():
        logger.error(f"File {basefilename} not found")
        return False
    radius = (diam / pixel_size) // 2
    start = time.perf_counter()
    try:
        with mrcfile.open(basefilename) as mrc:
            data = mrc.data
    except ValueError:
        logger.error(
            f"File {basefilename} could not be opened. It may be corrupted or not in mrc format"
        )
        return False
    mean = np.mean(data)
    sdev = np.std(data)
    sigma_min = mean - 3 * sdev
    sigma_max = mean + 3 * sdev
    data = np.ndarray.copy(data)
    data[data < sigma_min] = sigma_min
    data[data > sigma_max] = sigma_max
    data = data - data.min()
    data = data * 255 / data.max()
    data = data.astype("uint8")
    with PIL.Image.fromarray(data).convert(mode="RGB") as bim:
        enhancer = ImageEnhance.Contrast(bim)
        enhanced = enhancer.enhance(contrast_factor)
        fim = enhanced.filter(ImageFilter.BLUR)
        dim = ImageDraw.Draw(fim)
        if coords and coords[0]:
            # Orange circles for all coordinates
            for x, y in coords:
                dim.ellipse(
                    [
                        (float(x) - radius, float(y) - radius),
                        (float(x) + radius, float(y) + radius),
                    ],
                    width=8,
                    outline="#f58a07",
                )
        else:
            logger.warning(f"No coordinates provided for {basefilename}")
        if selected_coords and selected_coords[0]:
            # Green circles if selected coordinates are provided
            for x, y in selected_coords:
                dim.ellipse(
                    [
                        (float(x) - radius, float(y) - radius),
                        (float(x) + radius, float(y) + radius),
                    ],
                    width=12,
                    outline="#98df8a",
                )
        try:
            fim.save(outfile)
        except FileNotFoundError:
            logger.error(
                f"Trying to save to file {outfile} but directory does not exist"
            )
            return False
    timing = time.perf_counter() - start
    logger.info(
        f"Particle picker image {outfile} saved in {timing:.1f} seconds",
        extra={"image-processing-time": timing},
    )
    return outfile


def mrc_central_slice(plugin_params: Callable):
    if not required_parameters(plugin_params, ["file"]):
        return False
    filepath = Path(plugin_params("file"))
    skip_rescaling = plugin_params("skip_rescaling")
    if not filepath.is_file():
        logger.error(f"File {filepath} not found")
        return False
    start = time.perf_counter()
    try:
        with mrcfile.open(filepath) as mrc:
            data = mrc.data
    except ValueError:
        logger.error(
            f"File {filepath} could not be opened. It may be corrupted or not in mrc format"
        )
        return False
    outfile = str(filepath.with_suffix("")) + "_thumbnail.jpeg"
    if len(data.shape) != 3:
        logger.error(
            f"File {filepath} is not 3-dimensional. Cannot extract central slice"
        )
        return False

    # Extract central slice
    total_slices = data.shape[0]
    central_slice_index = int(total_slices / 2)
    central_slice_data = data[central_slice_index, :, :]

    # Write as jpeg
    central_slice_data = np.ndarray.copy(central_slice_data)
    if not skip_rescaling:
        mean = np.mean(central_slice_data)
        sdev = np.std(central_slice_data)
        sigma_min = mean - 3 * sdev
        sigma_max = mean + 3 * sdev
        central_slice_data[central_slice_data < sigma_min] = sigma_min
        central_slice_data[central_slice_data > sigma_max] = sigma_max
    central_slice_data = central_slice_data - central_slice_data.min()
    central_slice_data = central_slice_data * 255 / central_slice_data.max()
    central_slice_data = central_slice_data.astype("uint8")
    im = PIL.Image.fromarray(central_slice_data, mode="L")
    im.thumbnail((512, 512))
    im.save(outfile)
    timing = time.perf_counter() - start

    logger.info(
        f"Converted mrc to jpeg {filepath} -> {outfile} in {timing:.1f} seconds",
        extra={"image-processing-time": timing},
    )
    return outfile


def mrc_to_apng(plugin_params: Callable):
    if not required_parameters(plugin_params, ["file"]):
        return False
    filepath = Path(plugin_params("file"))
    skip_rescaling = plugin_params("skip_rescaling")
    if not filepath.is_file():
        logger.error(f"File {filepath} not found")
        return False
    start = time.perf_counter()
    try:
        with mrcfile.open(filepath) as mrc:
            data = mrc.data
    except ValueError:
        logger.error(
            f"File {filepath} could not be opened. It may be corrupted or not in mrc format"
        )
        return False
    outfile = str(filepath.with_suffix("")) + "_movie.png"

    if len(data.shape) == 3:
        images_to_append = []
        for frame in data:
            frame = np.ndarray.copy(frame)
            if not skip_rescaling:
                mean = np.mean(frame)
                sdev = np.std(frame)
                sigma_min = mean - 3 * sdev
                sigma_max = mean + 3 * sdev
                frame[frame < sigma_min] = sigma_min
                frame[frame > sigma_max] = sigma_max
            frame = frame - frame.min()
            frame = frame * 255 / frame.max()
            clipped_frame = np.random.choice([0, 1], np.shape(frame))
            clipped_frame[2:-2, 2:-2] = frame[2:-2, 2:-2]
            frame = clipped_frame.astype("uint8")
            im = PIL.Image.fromarray(frame, mode="L")
            im.thumbnail((512, 512))
            images_to_append.append(im)
        try:
            im_frame0 = images_to_append[0]
            im_frame0.save(outfile, save_all=True, append_images=images_to_append[1:])
        except IndexError:
            logger.error(f"Unable to save movie to file {outfile}")
            return False
    else:
        logger.error(f"File {filepath} is not a 3D volume")
        return False
    timing = time.perf_counter() - start
    logger.info(
        f"Converted mrc to apng {filepath} -> {outfile} in {timing:.1f} seconds"
    )
    return outfile


def particles_3d_in_frame(
    framedata: np.ndarray,
    framenum: int,
    radius_box: int,
    coords_file: Path,
):
    """Function to plot the pick locations of particles in a frame of a volume"""
    # Rescale the frame image
    frame = np.ndarray.copy(framedata)
    mean = np.mean(frame)
    sdev = np.std(frame)
    sigma_min = mean - 3 * sdev
    sigma_max = mean + 3 * sdev
    frame[frame < sigma_min] = sigma_min
    frame[frame > sigma_max] = sigma_max
    frame = frame - frame.min()
    frame = frame * 255 / frame.max()
    frame = frame.astype("uint8")

    im = PIL.Image.fromarray(frame, mode="L")
    colour_im = im.convert("RGB")
    dim = ImageDraw.Draw(colour_im)

    # Find the coordinates of the picks
    try:
        all_coords: pd.DataFrame = starfile.read(coords_file)["cryolo"]
        pick_x_coords = all_coords["CoordinateX"].to_numpy(dtype=float)
        pick_y_coords = all_coords["CoordinateY"].to_numpy(dtype=float)
        pick_z_coords = all_coords["CoordinateZ"].to_numpy(dtype=float)
        pick_width = all_coords["EstWidth"].to_numpy(dtype=float)
        pick_height = all_coords["EstHeight"].to_numpy(dtype=float)
        frame_count = all_coords["NumBoxes"].to_numpy(dtype=int)
    except KeyError:
        logger.warning(f"Cannot find picks in {coords_file}")
        colour_im.thumbnail((512, 512))
        return colour_im

    # Find the locations of picks which cross into this frame
    picks_appearing_in_frame = np.abs(pick_z_coords - framenum) - frame_count // 2 <= 0
    frame_x_coords = pick_x_coords[picks_appearing_in_frame]
    frame_y_coords = pick_y_coords[picks_appearing_in_frame]
    if len(frame_x_coords) and len(frame_y_coords):
        # Red circles for all coordinates
        for cid in range(len(frame_x_coords)):
            # Scale the pick radius based on the distance of frame from centre
            scaled_width = np.sqrt(
                pick_width[cid] ** 2
                - (pick_z_coords[picks_appearing_in_frame][cid] - framenum) ** 2
            )
            scaled_height = np.sqrt(
                pick_height[cid] ** 2
                - (pick_z_coords[picks_appearing_in_frame][cid] - framenum) ** 2
            )
            dim.ellipse(
                [
                    (
                        frame_x_coords[cid] + radius_box - scaled_width / 2,
                        frame_y_coords[cid] + radius_box - scaled_height / 2,
                    ),
                    (
                        frame_x_coords[cid] + radius_box + scaled_width / 2,
                        frame_y_coords[cid] + radius_box + scaled_height / 2,
                    ),
                ],
                width=4,
                outline="#f52407",
            )

    colour_im.thumbnail((512, 512))
    return colour_im


def picked_particles_3d_central_slice(plugin_params: Callable):
    if not required_parameters(plugin_params, ["file", "coordinates_file", "box_size"]):
        return False
    filename = plugin_params("file")
    coords_file = plugin_params("coordinates_file")
    box_size = plugin_params("box_size")
    radius_box = box_size // 2
    if not Path(filename).is_file() or not Path(coords_file).is_file():
        logger.error(f"File {filename} or {coords_file} not found")
        return False
    try:
        with mrcfile.open(filename) as mrc:
            data = mrc.data
    except ValueError:
        logger.error(f"MRC file {filename} could not be opened")
        return False
    if not len(data.shape) == 3:
        logger.error(f"File {filename} is not a 3D volume")
        return False

    # Extract central slice
    total_slices = data.shape[0]
    central_slice_index = int(total_slices / 2)
    central_slice_data = data[central_slice_index, :, :]

    colour_im = particles_3d_in_frame(
        framedata=central_slice_data,
        framenum=central_slice_index,
        radius_box=radius_box,
        coords_file=Path(coords_file),
    )

    outfile = str(Path(coords_file).with_suffix("")) + "_thumbnail.jpeg"
    colour_im.save(outfile)

    logger.info(f"3D particle picker central slice {outfile} saved")
    return outfile


def picked_particles_3d_apng(plugin_params: Callable):
    if not required_parameters(plugin_params, ["file", "coordinates_file", "box_size"]):
        return False
    filename = plugin_params("file")
    coords_file = plugin_params("coordinates_file")
    box_size = plugin_params("box_size")
    radius_box = box_size // 2
    if not Path(filename).is_file() or not Path(coords_file).is_file():
        logger.error(f"File {filename} or {coords_file} not found")
        return False
    try:
        with mrcfile.open(filename) as mrc:
            data = mrc.data
    except ValueError:
        logger.error(f"MRC file {filename} could not be opened")
        return False
    if not len(data.shape) == 3:
        logger.error(f"File {filename} is not a 3D volume")
        return False

    images_to_append = []
    for framenum, frame in enumerate(data):
        colour_im = particles_3d_in_frame(
            framedata=frame,
            framenum=framenum,
            radius_box=radius_box,
            coords_file=Path(coords_file),
        )
        images_to_append.append(colour_im)

    outfile = str(Path(coords_file).with_suffix("")) + "_movie.png"
    try:
        im_frame0 = images_to_append[0]
        im_frame0.save(outfile, save_all=True, append_images=images_to_append[1:])
    except IndexError:
        logger.error(f"Unable to save movie to file {outfile}")
        return False

    logger.info(f"3D particle picker movie {outfile} saved")
    return outfile
