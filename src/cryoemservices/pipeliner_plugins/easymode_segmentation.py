import os

import mrcfile
import numpy as np
import psutil
from scipy.ndimage import zoom


def _segment_tomogram_instance(volume, model, tile_size, overlap, logger):
    (pz, py, px) = tile_size
    (oz, oy, ox) = overlap
    d, h, w = volume.shape
    sz, sy, sx = pz - 2 * oz, py - 2 * oy, px - 2 * ox
    z_boxes = max(1, (d + sz - 1) // sz)
    y_boxes = max(1, (h + sy - 1) // sy)
    x_boxes = max(1, (w + sx - 1) // sx)
    out = np.zeros((d, h, w), dtype=np.float32)
    wgt = np.zeros((d, h, w), dtype=np.float32)
    for zi in range(z_boxes):
        for yi in range(y_boxes):
            for xi in range(x_boxes):
                logger.info(f"Starting {xi} {yi} {zi}")
                z_start = zi * sz - oz
                y_start = yi * sy - oy
                x_start = xi * sx - ox
                vz0 = max(0, z_start)
                vy0 = max(0, y_start)
                vx0 = max(0, x_start)
                vz1 = min(d, z_start + pz)
                vy1 = min(h, y_start + py)
                vx1 = min(w, x_start + px)
                extracted = volume[vz0:vz1, vy0:vy1, vx0:vx1]
                tile = np.zeros((pz, py, px), dtype=volume.dtype)
                tz0 = vz0 - z_start
                ty0 = vy0 - y_start
                tx0 = vx0 - x_start
                tile[
                    tz0 : tz0 + extracted.shape[0],
                    ty0 : ty0 + extracted.shape[1],
                    tx0 : tx0 + extracted.shape[2],
                ] = extracted
                logger.info(f"{extracted.shape} {tile.shape}")
                z_pos, y_pos, x_pos = zi * sz, yi * sy, xi * sx
                tile = np.expand_dims(np.array([tile]), axis=-1)
                segmented_tile = model.predict(tile, verbose=0, batch_size=1)
                logger.info(
                    f"Segmented at {psutil.Process(os.getpid()).memory_info().rss / 1024**2}"
                )
                segmented_tile = segmented_tile.squeeze(-1)
                center = segmented_tile[0, oz : oz + sz, oy : oy + sy, ox : ox + sx]
                z_end = min(z_pos + sz, d)
                y_end = min(y_pos + sy, h)
                x_end = min(x_pos + sx, w)
                az, ay, ax = z_end - z_pos, y_end - y_pos, x_end - x_pos
                out[z_pos:z_end, y_pos:y_end, x_pos:x_end] += center[:az, :ay, :ax]
                wgt[z_pos:z_end, y_pos:y_end, x_pos:x_end] += 1
    wgt[wgt == 0] = 1
    return out / wgt


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
    logger.info("Doing instance")
    segmented_volume = _segment_tomogram_instance(
        volume, model, tile_size, overlap, logger
    )
    logger.info("Done instance")
    (j0, j1), (k0, k1), (l0, l1) = padding
    segmented_volume = segmented_volume[
        j0 : segmented_volume.shape[0] - j1,
        k0 : segmented_volume.shape[1] - k1,
        l0 : segmented_volume.shape[2] - l1,
    ]
    sj, sk, sl = segmented_volume.shape
    segmented_volume = zoom(segmented_volume, (oj / sj, ok / sk, ol / sl), order=1)
    return segmented_volume, input_apix
