from __future__ import annotations

import logging
import time
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
import PIL.Image
import starfile
from PIL import ImageDraw, ImageEnhance, ImageFilter

logger = logging.getLogger("cryoemservices.services.images_plugins")
logger.setLevel(logging.INFO)


def mrc_to_jpeg(plugin_params):
    filename = plugin_params.parameters("file")
    allframes = plugin_params.parameters("all_frames")
    if not filename or filename == "None":
        logger.error("Skipping mrc to jpeg conversion: filename not specified")
        return False
    filepath = Path(filename)
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
        try:
            im.save(outfile)
        except FileNotFoundError:
            logger.error(
                f"Trying to save to file {outfile} but directory does not exist"
            )
            return False
    elif len(data.shape) == 3:
        if allframes:
            for i, frame in enumerate(data):
                frame = frame - frame.min()
                frame = frame * 255 / frame.max()
                frame = frame.astype("uint8")
                im = PIL.Image.fromarray(frame, mode="L")
                frame_outfile = str(outfile).replace(".jpeg", f"_{i+1}.jpeg")
                try:
                    im.save(frame_outfile)
                except FileNotFoundError:
                    logger.error(
                        f"Trying to save to file {frame_outfile} but directory does not exist"
                    )
                    return False
                outfiles.append(frame_outfile)
        else:
            data = data - data[0].min()
            data = data * 255 / data[0].max()
            data = data.astype("uint8")
            im = PIL.Image.fromarray(data[0], mode="L")
            try:
                im.save(outfile)
            except FileNotFoundError:
                logger.error(
                    f"Trying to save to file {outfile} but directory does not exist"
                )
                return False
    timing = time.perf_counter() - start

    logger.info(
        f"Converted mrc to jpeg {filename} -> {outfile} in {timing:.1f} seconds",
        extra={"image-processing-time": timing},
    )
    if outfiles:
        return outfiles
    return outfile


def picked_particles(plugin_params):
    basefilename = Path(plugin_params.parameters("file"))
    if basefilename.suffix == ".jpeg":
        logger.info(f"Replacing jpeg extension with mrc extension for {basefilename}")
        basefilename = basefilename.with_suffix(".mrc")
    coords = plugin_params.parameters("coordinates")
    selected_coords = plugin_params.parameters("selected_coordinates")
    pixel_size = plugin_params.parameters("pixel_size")
    if not pixel_size:
        # Legacy case of zocalo-relion
        pixel_size = plugin_params.parameters("angpix")
    diam = plugin_params.parameters("diameter")
    contrast_factor = plugin_params.parameters("contrast_factor") or 6
    outfile = plugin_params.parameters("outfile")
    if not outfile:
        logger.error(f"Outfile incorrectly specified: {outfile}")
        return False
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
    except FileNotFoundError:
        logger.error(f"File {basefilename} could not be opened")
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


def mrc_central_slice(plugin_params):
    filename = plugin_params.parameters("file")
    skip_rescaling = plugin_params.parameters("skip_rescaling")

    if not filename or filename == "None":
        logger.error("Skipping mrc to jpeg conversion: filename not specified")
        return False
    filepath = Path(filename)
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
    try:
        im.save(outfile)
    except FileNotFoundError:
        logger.error(f"Trying to save to file {outfile} but directory does not exist")
        return False
    timing = time.perf_counter() - start

    logger.info(
        f"Converted mrc to jpeg {filename} -> {outfile} in {timing:.1f} seconds",
        extra={"image-processing-time": timing},
    )

    Path(outfile).chmod(0o740)
    return outfile


def mrc_to_apng(plugin_params):
    filename = plugin_params.parameters("file")
    skip_rescaling = plugin_params.parameters("skip_rescaling")

    if not filename or filename == "None":
        logger.error("Skipping mrc to jpeg conversion: filename not specified")
        return False
    filepath = Path(filename)

    filepath.chmod(0o740)

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
        except (IndexError, FileNotFoundError):
            logger.error(f"Unable to save movie to file {outfile}")
            return False
    else:
        logger.error(f"File {filepath} is not a 3D volume")
        return False
    timing = time.perf_counter() - start
    logger.info(
        f"Converted mrc to apng {filename} -> {outfile} in {timing:.1f} seconds"
    )

    Path(outfile).chmod(0o740)
    return outfile


def particles_3d_in_frame(
    framedata: np.ndarray,
    framenum: int,
    radius: int,
    radius_box: int,
    pick_x_coords: np.ndarray,
    pick_y_coords: np.ndarray,
    pick_z_coords: np.ndarray,
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

    # Find the locations of picks which cross into this frame
    picks_appearing_in_frame = np.abs(pick_z_coords - framenum) < radius
    frame_x_coords = pick_x_coords[picks_appearing_in_frame]
    frame_y_coords = pick_y_coords[picks_appearing_in_frame]
    if len(frame_x_coords) and len(frame_y_coords):
        # Red circles for all coordinates
        for cid in range(len(frame_x_coords)):
            # Scale the pick radius based on the distance of frame from centre
            cid_radius = np.sqrt(
                radius**2
                - (pick_z_coords[picks_appearing_in_frame][cid] - framenum) ** 2
            )
            dim.ellipse(
                [
                    (
                        frame_x_coords[cid] + radius_box - cid_radius,
                        frame_y_coords[cid] + radius_box - cid_radius,
                    ),
                    (
                        frame_x_coords[cid] + radius_box + cid_radius,
                        frame_y_coords[cid] + radius_box + cid_radius,
                    ),
                ],
                width=4,
                outline="#f52407",
            )

    colour_im.thumbnail((512, 512))
    return colour_im


def picked_particles_3d_central_slice(plugin_params):
    filename = plugin_params.parameters("file")
    coords_file = plugin_params.parameters("coordinates_file")
    diameter_pixels = plugin_params.parameters("diameter_pixels")
    box_size = plugin_params.parameters("box_size")
    if not Path(filename).is_file() or not Path(coords_file).is_file():
        logger.error(f"File {filename} or {coords_file} not found")
        return False
    radius = diameter_pixels // 2
    radius_box = box_size // 2

    outfile = str(Path(coords_file).with_suffix("")) + "_thumbnail.jpeg"

    try:
        with mrcfile.open(filename) as mrc:
            data = mrc.data
    except ValueError:
        logger.error(f"MRC file {filename} could not be opened")
        return False
    if not len(data.shape) == 3:
        logger.error(f"File {filename} is not a 3D volume")
        return False
    try:
        all_coords: pd.DataFrame = starfile.read(coords_file)["cryolo"]
        pick_x_coords = all_coords["CoordinateX"].to_numpy(dtype=float)
        pick_y_coords = all_coords["CoordinateY"].to_numpy(dtype=float)
        pick_z_coords = all_coords["CoordinateZ"].to_numpy(dtype=float)
    except KeyError:
        logger.warning(f"Cannot find picks in {coords_file}")
        pick_x_coords = np.array([])
        pick_y_coords = np.array([])
        pick_z_coords = np.array([])

    # Extract central slice
    total_slices = data.shape[0]
    central_slice_index = int(total_slices / 2)
    central_slice_data = data[central_slice_index, :, :]

    colour_im = particles_3d_in_frame(
        framedata=central_slice_data,
        framenum=central_slice_index,
        radius=radius,
        radius_box=radius_box,
        pick_x_coords=pick_x_coords,
        pick_y_coords=pick_y_coords,
        pick_z_coords=pick_z_coords,
    )
    try:
        colour_im.save(outfile)
    except FileNotFoundError:
        logger.error(f"Trying to save to file {outfile} but directory does not exist")
        return False

    logger.info(f"3D particle picker central slice {outfile} saved")
    return outfile


def picked_particles_3d_apng(plugin_params):
    filename = plugin_params.parameters("file")
    coords_file = plugin_params.parameters("coordinates_file")
    diameter_pixels = plugin_params.parameters("diameter_pixels")
    box_size = plugin_params.parameters("box_size")
    if not Path(filename).is_file() or not Path(coords_file).is_file():
        logger.error(f"File {filename} or {coords_file} not found")
        return False
    radius = diameter_pixels // 2
    radius_box = box_size // 2

    outfile = str(Path(coords_file).with_suffix("")) + "_movie.png"

    try:
        with mrcfile.open(filename) as mrc:
            data = mrc.data
    except ValueError:
        logger.error(f"MRC file {filename} could not be opened")
        return False
    if not len(data.shape) == 3:
        logger.error(f"File {filename} is not a 3D volume")
        return False
    try:
        all_coords: pd.DataFrame = starfile.read(coords_file)["cryolo"]
        pick_x_coords = all_coords["CoordinateX"].to_numpy(dtype=float)
        pick_y_coords = all_coords["CoordinateY"].to_numpy(dtype=float)
        pick_z_coords = all_coords["CoordinateZ"].to_numpy(dtype=float)
    except KeyError:
        logger.warning(f"Cannot find picks in {coords_file}")
        pick_x_coords = np.array([])
        pick_y_coords = np.array([])
        pick_z_coords = np.array([])

    images_to_append = []
    for framenum, frame in enumerate(data):
        colour_im = particles_3d_in_frame(
            framedata=frame,
            framenum=framenum,
            radius=radius,
            radius_box=radius_box,
            pick_x_coords=pick_x_coords,
            pick_y_coords=pick_y_coords,
            pick_z_coords=pick_z_coords,
        )
        images_to_append.append(colour_im)
    try:
        im_frame0 = images_to_append[0]
        im_frame0.save(outfile, save_all=True, append_images=images_to_append[1:])
    except (IndexError, FileNotFoundError):
        logger.error(f"Unable to save movie to file {outfile}")
        return False

    logger.info(f"3D particle picker movie {outfile} saved")
    return outfile
