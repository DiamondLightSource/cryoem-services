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

from cryoemservices.services.cryolo import grid_bar_histogram
from cryoemservices.util.clem_array_functions import convert_to_rgb

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
        data = data / data.max() * 255
        data = data.astype("uint8")
        im = PIL.Image.fromarray(data)

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
            # Apply thumbnailing if 2D image without text overlay
            im.thumbnail((1024, 1024))
            im.save(outfile)
    elif len(data.shape) == 3:
        if allframes:
            for i, frame in enumerate(data):
                frame = frame - frame.min()
                frame = frame / frame.max() * 255
                frame = frame.astype("uint8")
                im = PIL.Image.fromarray(frame)
                frame_outfile = outfile.parent / f"{outfile.stem}_{i + 1}.jpeg"
                im.save(frame_outfile)
                outfiles.append(frame_outfile)
        else:
            data = data - data[0].min()
            data = data / data[0].max() * 255
            data = data.astype("uint8")
            im = PIL.Image.fromarray(data[0])
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

    if plugin_params("flatten_image"):
        flat_data = grid_bar_histogram(data, peak_width=1)
        if flat_data is not None:
            contrast_factor = 1
            data = flat_data

    mean = np.mean(data)
    sdev = np.std(data)
    sigma_min = mean - 3 * sdev
    sigma_max = mean + 3 * sdev
    data = np.ndarray.copy(data)
    data[data < sigma_min] = sigma_min
    data[data > sigma_max] = sigma_max
    data = data - data.min()
    data = data / data.max() * 255
    data = data.astype("uint8")
    with PIL.Image.fromarray(data).convert("RGB") as bim:
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
            fim.thumbnail((1024, 1024))
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
    if plugin_params("remove_input"):
        basefilename.unlink()
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
    central_slice_data = central_slice_data / central_slice_data.max() * 255
    central_slice_data = central_slice_data.astype("uint8")
    im = PIL.Image.fromarray(central_slice_data)
    im.thumbnail((512, 512))
    im.save(outfile)
    timing = time.perf_counter() - start

    logger.info(
        f"Converted mrc to jpeg {filepath} -> {outfile} in {timing:.1f} seconds",
        extra={"image-processing-time": timing},
    )
    return outfile


def mrc_projection(plugin_params: Callable):
    if not required_parameters(plugin_params, ["file", "projection"]):
        return False
    filepath = Path(plugin_params("file"))
    projection_type: str = plugin_params("projection")
    thickness_ang = plugin_params("thickness_ang")
    pixel_spacing = plugin_params("pixel_spacing")
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
    outfile = str(filepath.with_suffix("")) + f"_proj{projection_type}.jpeg"
    if len(data.shape) != 3:
        logger.error(f"File {filepath} is not 3-dimensional. Cannot make projection")
        return False

    # Do projection
    if projection_type.lower() in ("xy", "yx"):
        projected_data = np.mean(data, axis=0)
    elif projection_type.lower() in ("yz", "zy"):
        projected_data = np.mean(data, axis=1)
    elif projection_type.lower() in ("zx", "xz"):
        projected_data = np.mean(data, axis=2)
    else:
        logger.error(f"Unknown projection type {projection_type}")
        return False

    # Write as jpeg
    mean = np.mean(projected_data)
    sdev = np.std(projected_data)
    sigma_min = mean - 3 * sdev
    sigma_max = mean + 3 * sdev
    projected_data[projected_data < sigma_min] = sigma_min
    projected_data[projected_data > sigma_max] = sigma_max
    projected_data = projected_data - projected_data.min()
    projected_data = projected_data / projected_data.max() * 255
    projected_data = projected_data.astype("uint8")
    im = PIL.Image.fromarray(projected_data)

    if pixel_spacing:
        if thickness_ang:
            # Show estimated thickness if given
            scalebar_pixels = thickness_ang / float(pixel_spacing)
            scalebar_description = f"Thickness: {thickness_ang / 10:.0f} nm"
        else:
            # Add scale bar if no thickness provided
            scalebar_pixels = projected_data.shape[0] / 3
            scalebar_nm = float(pixel_spacing) / 10 * projected_data.shape[0] / 3
            scalebar_description = f"{scalebar_nm:.0f} nm"
        colour_im = im.convert("RGB")
        dim = ImageDraw.Draw(colour_im)
        dim.line(
            (
                (20, projected_data.shape[0] / 2 - scalebar_pixels / 2),
                (20, projected_data.shape[0] / 2 + scalebar_pixels / 2),
            ),
            fill="yellow",
            width=5,
        )
        font_to_use = ImageFont.load_default(size=26)
        dim.text(
            (25, projected_data.shape[0] / 2),
            scalebar_description,
            anchor="lm",
            font=font_to_use,
            fill="yellow",
        )
        colour_im.save(outfile)
    else:
        im.save(outfile)
    timing = time.perf_counter() - start

    logger.info(
        f"Made {projection_type} projection of mrc to jpeg {filepath} -> {outfile} in {timing:.1f} seconds",
        extra={"image-processing-time": timing},
    )
    return outfile


def mrc_to_apng(plugin_params: Callable):
    if not required_parameters(plugin_params, ["file"]):
        return False
    filepath = Path(plugin_params("file"))
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
            if not plugin_params("skip_rescaling"):
                mean = np.mean(frame)
                sdev = np.std(frame)
                sigma_min = mean - 3 * sdev
                sigma_max = mean + 3 * sdev
                frame[frame < sigma_min] = sigma_min
                frame[frame > sigma_max] = sigma_max
            frame = frame - frame.min()
            frame = frame / frame.max() * 255
            if plugin_params("jitter_edge"):
                clipped_frame = np.random.choice([0, 1], np.shape(frame))
                clipped_frame[2:-2, 2:-2] = frame[2:-2, 2:-2]
                frame = clipped_frame.astype("uint8")
            else:
                frame = frame.astype("uint8")
            im = PIL.Image.fromarray(frame)
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
    frame = frame / frame.max() * 255
    frame = frame.astype("uint8")

    colour_im = PIL.Image.fromarray(frame).convert("RGB")
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


def tiff_to_apng(plugin_params: Callable):
    """
    Converts TIFF images/image stacks into PNGs.

    This function only works with unsigned 8-bit (grayscale or RGB) TIFF images, as
    Pillow cannot correctly parse 16-bit or higher channels, nor can it save float-
    based images as PNGs.
    """
    # Check that the essential parameters are provided
    if not required_parameters(plugin_params, ["input_file", "output_file"]):
        return False

    # Load parameters
    input_file = Path(plugin_params("input_file"))
    output_file = Path(plugin_params("output_file"))
    target_size: tuple[int | None, int | None] = (
        tuple(plugin_params("target_size"))
        if plugin_params("target_size") is not None
        else (None, None)
    )
    target_height, target_width = target_size
    color: str | None = (
        str(plugin_params("color")) if plugin_params("color") is not None else None
    )

    # Verify that the input file exists
    if not input_file.is_file():
        logger.error(f"File {input_file} not found")
        return False

    # Start of function
    start = time.perf_counter()
    img = PIL.Image.open(input_file)

    # Collect image frames
    frames: list[PIL.Image.ImageFile.ImageFile] = []
    try:
        while True:
            frame = img.copy()
            # Resize image if target size provided
            if target_height and target_width:
                frame.thumbnail((target_width, target_height))
            # Convert only grayscale 8-bit images if a color LUT is provided
            if color is not None and frame.mode == "L":
                frame = PIL.Image.fromarray(
                    convert_to_rgb(np.asarray(frame, dtype="uint8"), color)
                )
            # Skip colorisation step and notify why
            elif color is not None and frame.mode != "L" and img.tell() == 0:
                logger.debug(
                    f"Image format {frame.mode} not valid for color conversion"
                )

            # Append frame and load next frame in sequence
            frames.append(frame)
            img.seek(img.tell() + 1)
    except EOFError:
        pass  # All frames in the image will have been loaded by this point
    img.seek(0)  # Reset the image after being done

    # Save as PNG
    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True)
    try:
        frames[0].save(
            output_file,
            save_all=True,
            append_images=frames[1:],
        )
    except Exception:
        logger.error(f"Unable to create PNG from TIFF file {input_file}", exc_info=True)
        return False

    # Report on successful processing result
    timing = time.perf_counter() - start
    logger.info(
        f"Converted TIFF to PNG {input_file} -> {output_file} in {timing:.1f} seconds",
        extra={"image-processing-time": timing},
    )
    return str(output_file)


def tilt_series_alignment(plugin_params: Callable):
    if not required_parameters(plugin_params, ["file", "aln_file", "pixel_size"]):
        return False
    filepath = Path(plugin_params("file"))
    aln_file = Path(plugin_params("aln_file"))
    pixel_size = float(plugin_params("pixel_size"))
    if not filepath.is_file() or not aln_file.is_file():
        logger.error(f"File {filepath} or {aln_file} not found")
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
    outfile = str(filepath.with_suffix("")) + "_alignment.jpeg"
    if len(data.shape) == 3:
        # Extract central slice
        total_slices = data.shape[0]
        central_slice_index = total_slices // 2
        central_slice_data = data[central_slice_index, :, :]
    elif len(data.shape) == 2:
        central_slice_data = data
    else:
        logger.error(
            f"File {filepath} is not 2- or 3-dimensional. Cannot extract central slice"
        )
        return False

    # Read in the tilt alignment file
    tilts = []
    angles = []
    x_shifts = []
    y_shifts = []
    with open(aln_file) as alns:
        while True:
            line = alns.readline()
            if not line:
                break
            if not line.startswith("#"):
                tilts.append(float(line.split()[-1]))
                angles.append(float(line.split()[1]))
                x_shifts.append(float(line.split()[3]))
                y_shifts.append(float(line.split()[4]))

    # Pad the image to make the shifts more visible - x and y end up flipped here
    flat_size = central_slice_data.shape
    pad_x = flat_size[0] // 8
    pad_y = flat_size[1] // 8
    cen = [
        central_slice_data.shape[1] / 2 + pad_y,
        central_slice_data.shape[0] / 2 + pad_x,
    ]
    mrc_data = np.pad(
        central_slice_data,
        ((pad_x, pad_x), (pad_y, pad_y)),
        mode="constant",
        constant_values=np.mean(central_slice_data),
    )

    # Show image of the central slice
    mean = np.mean(mrc_data)
    sdev = np.std(mrc_data)
    sigma_min = mean - 3 * sdev
    sigma_max = mean + 3 * sdev
    data = np.ndarray.copy(mrc_data)
    data[data < sigma_min] = sigma_min
    data[data > sigma_max] = sigma_max
    data = data - data.min()
    data = data / data.max() * 255
    data = data.astype("uint8")
    im = PIL.Image.fromarray(data)
    colour_im = im.convert("RGB")
    dim = ImageDraw.Draw(colour_im)

    # Add a box for each tilt
    outliers = []
    for tid, tlt in enumerate(tilts):
        # Apply the shifts to the centre position
        shifted_cen_x = cen[0] + x_shifts[tid] / pixel_size
        shifted_cen_y = cen[1] + y_shifts[tid] / pixel_size

        # Find distance to edge of image, accounting for y tilt
        x_half_width = flat_size[1] / 2
        y_half_width = (flat_size[0] * np.cos(tlt * np.pi / 180)) / 2

        # Flip the angles if greater than 45 degrees
        if angles[tid] > 45:
            angles[tid] -= 90
        elif angles[tid] < -45:
            angles[tid] += 90

        # Assign colours based on the size of the shift and mark outliers
        if np.sqrt(x_shifts[tid] ** 2 + y_shifts[tid] ** 2) > 1000:
            outline_colour = "#f52407"  # red
            outliers.append(str(tlt))
        elif np.sqrt(x_shifts[tid] ** 2 + y_shifts[tid] ** 2) > 100:
            outline_colour = "#f5a927"  # orange
        elif np.sqrt(x_shifts[tid] ** 2 + y_shifts[tid] ** 2) > 10:
            outline_colour = "#f5e90a"  # yellow
        else:
            outline_colour = "#0af549"  # green

        # Construct a rectangle, rotated by the tilt axis angle
        rotation_matrix = [
            [np.cos(angles[tid] * np.pi / 180), -np.sin(angles[tid] * np.pi / 180)],
            [np.sin(angles[tid] * np.pi / 180), np.cos(angles[tid] * np.pi / 180)],
        ]
        x_corners = []
        y_corners = []
        for rid in range(4):
            corner_rotations = np.matmul(
                rotation_matrix,
                [
                    [x_half_width, y_half_width],
                    [-x_half_width, y_half_width],
                    [-x_half_width, -y_half_width],
                    [x_half_width, -y_half_width],
                ][rid],
            )
            x_corners.append(corner_rotations[0])
            y_corners.append(corner_rotations[1])

        dim.polygon(
            [
                (
                    shifted_cen_x + x_corners[0],
                    shifted_cen_y + y_corners[0],
                ),
                (
                    shifted_cen_x + x_corners[1],
                    shifted_cen_y + y_corners[1],
                ),
                (
                    shifted_cen_x + x_corners[2],
                    shifted_cen_y + y_corners[2],
                ),
                (
                    shifted_cen_x + x_corners[3],
                    shifted_cen_y + y_corners[3],
                ),
            ],
            width=int(20 - np.abs(len(tilts) / 2 - tid)),
            outline=outline_colour,
        )

    # Text to record outliers and colour key
    if outliers:
        dim.text(
            (100, cen[1] * 2 - 200),
            "Outliers (deg): " + " ".join(outliers),
            fill="#f52407",
            font_size=150,
        )
    dim.text((100, 10), "Shift over 100 nm", fill="#f52407", font_size=150)
    dim.text((100, 160), "Shift 10 to 100 nm", fill="#f5a927", font_size=150)
    dim.text((100, 320), "Shift 1 to 10 nm", fill="#f5e90a", font_size=150)
    dim.text((100, 460), "Shift under 1 nm", fill="#0af549", font_size=150)
    colour_im.thumbnail((1024, 1024))
    colour_im.save(outfile)
    timing = time.perf_counter() - start
    logger.info(
        f"Done tilt series alignment {filepath} -> {outfile} in {timing:.1f} seconds",
        extra={"image-processing-time": timing},
    )
    return outfile
