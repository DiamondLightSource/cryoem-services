import mrcfile
import numpy as np
from scipy.ndimage import zoom


def segment_tile(xi, yi, zi, volume, model, tile_size, overlap):
    (oz, oy, ox) = overlap
    # non-overlapped patch sizes
    sz = tile_size - 2 * oz
    sy = tile_size - 2 * oy
    sx = tile_size - 2 * ox

    z_start = zi * sz - oz
    y_start = yi * sy - oy
    x_start = xi * sx - ox
    vz0 = max(0, z_start)
    vy0 = max(0, y_start)
    vx0 = max(0, x_start)
    vz1 = min(volume.shape[0], z_start + tile_size)
    vy1 = min(volume.shape[1], y_start + tile_size)
    vx1 = min(volume.shape[2], x_start + tile_size)

    tile = np.zeros((tile_size, tile_size, tile_size), dtype=volume.dtype)
    tile[
        vz0 - z_start : vz1 - z_start,
        vy0 - y_start : vy1 - y_start,
        vx0 - x_start : vx1 - x_start,
    ] = volume[vz0:vz1, vy0:vy1, vx0:vx1]
    tile = np.expand_dims(np.array([tile]), axis=-1)
    segmented_tile = model.predict(tile, verbose=0, batch_size=1).squeeze(-1)
    return segmented_tile[0, oz : oz + sz, oy : oy + sy, ox : ox + sx]


def _segment_tomogram_instance(volume, model, tile_size, overlap):
    (oz, oy, ox) = overlap
    d, h, w = volume.shape
    # non-overlapped patch sizes
    sz = tile_size - 2 * oz
    sy = tile_size - 2 * oy
    sx = tile_size - 2 * ox

    z_boxes = max(1, (d + sz - 1) // sz)
    y_boxes = max(1, (h + sy - 1) // sy)
    x_boxes = max(1, (w + sx - 1) // sx)
    out = np.ones(volume.shape, dtype=np.int8) * 0
    for zi in range(z_boxes):
        for yi in range(y_boxes):
            for xi in range(x_boxes):
                z_pos, y_pos, x_pos = zi * sz, yi * sy, xi * sx
                z_end = min(z_pos + sz, d)
                y_end = min(y_pos + sy, h)
                x_end = min(x_pos + sx, w)
                az, ay, ax = z_end - z_pos, y_end - y_pos, x_end - x_pos
                seg_vol_float = segment_tile(
                    xi, yi, zi, volume, model, tile_size, overlap
                )[:az, :ay, :ax]
                out[z_pos:z_end, y_pos:y_end, x_pos:x_end] = (
                    seg_vol_float * 127
                ).astype(np.int8)
    return out


def _pad_volume(volume, min_pad=16, div=32, min_size=None):
    j, k, l = volume.shape
    pads = []
    for n in (j, k, l):
        total_pad = max(2 * min_pad, ((n + 2 * min_pad + div - 1) // div) * div - n)
        if min_size and total_pad + n < min_size:
            total_pad = min_size - n
        before = total_pad // 2
        after = total_pad - before
        pads.append((before, after))
    padded = np.pad(volume, pads, mode="reflect")
    return padded, tuple(pads)


def segment_tomogram(
    model,
    tomogram_path,
    tta=1,
    batch_size=1,
    model_apix=10,
    input_apix=10,
    tile_size=256,
):
    with mrcfile.open(tomogram_path) as m:
        volume = m.data.astype(np.float32)
        oj, ok, ol = volume.shape
    scale = float(input_apix) / float(model_apix)
    volume = zoom(volume, scale, order=1)
    _k_margin = min(int(0.2 * ok), 64)
    _l_margin = min(int(0.2 * ol), 64)
    volume -= np.mean(volume[:, _k_margin:-_k_margin, _l_margin:-_l_margin])
    volume /= np.std(volume[:, _k_margin:-_k_margin, _l_margin:-_l_margin]) + 1e-7
    volume, padding = _pad_volume(volume, min_size=tile_size)
    overlap = (
        0 if tile_size >= oj else 48,
        0 if tile_size >= ok else 48,
        0 if tile_size >= ol else 48,
    )
    segmented_volume = _segment_tomogram_instance(volume, model, tile_size, overlap)
    (j0, j1), (k0, k1), (l0, l1) = padding
    segmented_volume = segmented_volume[
        j0 : segmented_volume.shape[0] - j1,
        k0 : segmented_volume.shape[1] - k1,
        l0 : segmented_volume.shape[2] - l1,
    ]
    segmented_volume = zoom(segmented_volume, 1 / scale, order=1)
    return segmented_volume, input_apix
