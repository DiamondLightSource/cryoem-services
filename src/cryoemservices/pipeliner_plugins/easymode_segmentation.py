import os

import mrcfile
import numpy as np
import psutil
from scipy.ndimage import zoom


def segment_loop(xi, yi, zi, volume, model, tile_size, overlap, logger):
    (pz, py, px) = tile_size
    (oz, oy, ox) = overlap
    d, h, w = volume.shape  # 672, 1440, 2016
    sz, sy, sx = pz - 2 * oz, py - 2 * oy, px - 2 * ox  # non-overlapped patch size, 160

    z_start = zi * sz - oz
    y_start = yi * sy - oy
    x_start = xi * sx - ox
    vz0 = max(0, z_start)
    vy0 = max(0, y_start)
    vx0 = max(0, x_start)
    vz1 = min(d, z_start + pz)
    vy1 = min(h, y_start + py)
    vx1 = min(w, x_start + px)
    tile = np.zeros((pz, py, px), dtype=volume.dtype)
    tile[
        vz0 - z_start : vz1 - z_start,
        vy0 - y_start : vy1 - y_start,
        vx0 - x_start : vx1 - x_start,
    ] = volume[vz0:vz1, vy0:vy1, vx0:vx1]
    tile = np.expand_dims(np.array([tile]), axis=-1)
    segmented_tile = model.predict(tile, verbose=0, batch_size=1).squeeze(-1)
    logger.info(
        f"Segmented {xi} {yi} {zi} at {psutil.Process(os.getpid()).memory_info().rss / 1024**2}"
    )
    z_end = min(zi * sz + sz, d)
    y_end = min(yi * sy + sy, h)
    x_end = min(xi * sx + sx, w)
    az = z_end - zi * sz
    ay = y_end - yi * sy
    ax = x_end - xi * sx
    return (
        segmented_tile[0, oz : oz + sz, oy : oy + sy, ox : ox + sx][:az, :ay, :ax] * 127
    ).astype(np.int8)


def _segment_tomogram_instance(volume, model, tile_size, overlap, logger):
    (pz, py, px) = tile_size
    (oz, oy, ox) = overlap
    d, h, w = volume.shape  # 672, 1440, 2016
    sz, sy, sx = pz - 2 * oz, py - 2 * oy, px - 2 * ox  # non-overlapped patch size, 160
    z_boxes = max(1, (d + sz - 1) // sz)  # 5
    y_boxes = max(1, (h + sy - 1) // sy)  # 9
    x_boxes = max(1, (w + sx - 1) // sx)  # 13
    logger.info(f"{volume.shape}, {z_boxes}, {y_boxes}, {x_boxes}")
    out = np.ones((d, h, w), dtype=np.int8) * 0
    for zi in range(z_boxes):
        for yi in range(y_boxes):
            for xi in range(x_boxes):
                z_pos, y_pos, x_pos = zi * sz, yi * sy, xi * sx
                z_end = min(z_pos + sz, d)
                y_end = min(y_pos + sy, h)
                x_end = min(x_pos + sx, w)
                out[z_pos:z_end, y_pos:y_end, x_pos:x_end] = segment_loop(
                    xi, yi, zi, volume, model, tile_size, overlap, logger
                )
    return out


def _pad_volume(volume, min_pad=16, div=32):
    j, k, l = volume.shape
    pads = []
    for n in (j, k, l):
        total_pad = max(2 * min_pad, ((n + 2 * min_pad + div - 1) // div) * div - n)
        before = total_pad // 2
        after = total_pad - before
        pads.append((before, after))
    padded = np.pad(volume, pads, mode="reflect")
    return padded, tuple(pads)


def segment_tomogram(
    model, tomogram_path, tta=1, batch_size=1, model_apix=10, input_apix=10, logger=None
):
    logger.info("Reading tomogram")
    with mrcfile.open(tomogram_path) as m:
        volume = m.data.astype(np.float32)
        oj, ok, ol = volume.shape
    logger.info("Read tomogram")
    scale = float(input_apix) / float(model_apix)
    volume = zoom(volume, scale, order=1)
    _k_margin = min(int(0.2 * ok), 64)
    _l_margin = min(int(0.2 * ol), 64)
    volume -= np.mean(volume[:, _k_margin:-_k_margin, _l_margin:-_l_margin])
    volume /= np.std(volume[:, _k_margin:-_k_margin, _l_margin:-_l_margin]) + 1e-7
    volume, padding = _pad_volume(volume)
    tile_size = (min(256, oj), min(256, ok), min(256, ol))
    overlap = (
        0 if tile_size[0] == oj else 48,
        0 if tile_size[1] == ok else 48,
        0 if tile_size[2] == ol else 48,
    )
    logger.info(
        f"Doing instance {psutil.Process(os.getpid()).memory_info().rss / 1024**2}"
    )
    segmented_volume = _segment_tomogram_instance(
        volume, model, tile_size, overlap, logger
    )
    logger.info(
        f"Done instance {psutil.Process(os.getpid()).memory_info().rss / 1024**2}"
    )
    (j0, j1), (k0, k1), (l0, l1) = padding
    segmented_volume = segmented_volume[
        j0 : segmented_volume.shape[0] - j1,
        k0 : segmented_volume.shape[1] - k1,
        l0 : segmented_volume.shape[2] - l1,
    ]
    sj, sk, sl = segmented_volume.shape
    segmented_volume = zoom(segmented_volume, (oj / sj, ok / sk, ol / sl), order=1)
    return segmented_volume, input_apix
