from __future__ import annotations

from pydantic import BaseModel


class TomographyOptions(BaseModel):
    """The parameters used by the tomography services"""

    """Parameters that Murfey will set"""
    # Pixel size in Angstroms in the input movies
    pixel_size: float = 0.885
    # Dose in electrons per squared Angstrom per frame
    dose_per_frame: float = 1.277
    # Gain-reference image in MRC format
    gain_ref: str = "Movies/gain.mrc"
    # Acceleration voltage (in kV)
    voltage: int = 300
    # Use binning=2 for super-resolution K2 movies
    motion_corr_binning: int = 1
    # eer format grouping
    eer_grouping: int = 20

    frame_count: int = 10
    tilt_axis: float = 85.0
    defocus: float = -4
    invert_hand: int = -1

    """Parameters we set differently from pipeliner defaults"""
    # Spherical aberration
    spher_aber: float = 2.7
    # Amplitude contrast (Q0)
    ampl_contrast: float = 0.1
