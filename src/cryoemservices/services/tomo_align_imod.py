import subprocess
from pathlib import Path
from typing import Any, List, Optional

import mrcfile
from pydantic import BaseModel, Field, ValidationError
from txrm2tiff.main import convert_and_save
from workflows.recipe import wrap_subscribe

from cryoemservices.services.common_service import CommonService
from cryoemservices.util.models import MockRW
from cryoemservices.util.relion_service_options import (
    RelionServiceOptions,
    update_relion_options,
)


class ImodTomoParameters(BaseModel):
    stack_file: str = Field(..., min_length=1)
    txrm_file: str = Field(..., min_length=1)
    pixel_size: float
    vol_z: int = 700
    out_bin: int = 1
    tilt_axis: float = 0
    bead_size: int = 250
    bead_count: int = 4
    wbp: int = 1
    sirt: int = 1
    sirt_leave_iterations: int = 5
    patch: int = 0
    patch_size: int = 200
    patch_overlap: float = 0.5
    flip_vol: int = 0
    flip_vol_post_reconstruction: bool = True
    manual_tilt_offset: Optional[float] = None
    cpus: int = 1
    relion_options: RelionServiceOptions


class ImodTomoAlign(CommonService):
    """
    A service for grouping and aligning tomography tilt-series with Imod
    """

    # Logger name
    _logger_name = "cryoemservices.services.tomo_align_imod"

    # Job name
    job_type = "relion.reconstructtomograms"

    # Values to extract for ISPyB
    input_file_list_of_lists: List[Any]
    refined_tilts: List[float]
    x_shift: List[float]
    y_shift: List[float]
    rot_centre_z_list: List[str]
    tilt_offset: Optional[float] = None
    thickness_pixels: int | None = None
    rot: float | None = None
    mag: float | None = None
    alignment_quality: Optional[float] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.refined_tilts = []
        self.rot_centre_z_list = []

    def initializing(self):
        """Subscribe to a queue. Received messages must be acknowledged."""
        self.log.info("TomoAlign service starting")
        wrap_subscribe(
            self._transport,
            self._environment["queue"] or "tomo_align",
            self.tomo_align,
            acknowledgement=True,
            allow_non_recipe_messages=True,
        )

    def tomo_align(self, rw, header: dict, message: dict):
        """Main function which interprets and processes received messages"""
        if not rw:
            self.log.info("Received a simple message")
            if not isinstance(message, dict):
                self.log.error("Rejected invalid simple message")
                self._transport.nack(header)
                return

            # Create a wrapper-like object that can be passed to functions
            # as if a recipe wrapper was present.
            rw = MockRW(self._transport)
            rw.recipe_step = {"parameters": message}

        try:
            if isinstance(message, dict):
                tomo_params = ImodTomoParameters(
                    **{**rw.recipe_step.get("parameters", {}), **message}
                )
            else:
                tomo_params = ImodTomoParameters(
                    **{**rw.recipe_step.get("parameters", {})}
                )
        except (ValidationError, TypeError) as e:
            self.log.warning(
                f"TomoAlign parameter validation failed for message: {message} "
                f"and recipe parameters: {rw.recipe_step.get('parameters', {})} "
                f"with exception: {e}"
            )
            rw.transport.nack(header)
            return

        # TODO
        aln_file = "dummy"
        rot_centre_z = 0
        shift_plot_suffix = "_xy_shift_plot.json"

        # Update the relion options
        tomo_params.relion_options = update_relion_options(
            tomo_params.relion_options, dict(tomo_params)
        )

        # Do txrm conversion
        self.log.info(f"Input file {tomo_params.txrm_file}")
        tifftomo = Path(tomo_params.stack_file).with_suffix(".tiff")
        convert_and_save(tomo_params.txrm_file, str(tifftomo), custom_reference=None)
        subprocess.run(["tif2mrc", str(tifftomo), tomo_params.stack_file])
        tifftomo.unlink(missing_ok=True)
        if not Path(tomo_params.stack_file).is_file():
            self.log.error(
                f"Converting {tomo_params.txrm_file} to {tomo_params.stack_file} failed"
            )
            rw.transport.nack(header)
            return

        # Find the input image dimensions
        with mrcfile.open(self.input_file_list_of_lists[0][0]) as mrc:
            mrc_header = mrc.header

        # Run batchruntomo
        adoc_file = write_batch_directive_file(tomo_params)
        imod_output_path = Path(tomo_params.stack_file).with_suffix("_Vol.rec")
        imod_result = subprocess.run(
            [
                "batchruntomo",
                "-directive",
                str(adoc_file),
                "-cpus",
                str(tomo_params.cpus),
            ]
        )
        if imod_result.returncode or not imod_output_path.is_file():
            self.log.error(
                f"batchruntomo failed with exitcode {imod_result.returncode}:\n"
                + imod_result.stderr.decode("utf8", "replace")
            )
            # Update failure processing status
            rw.send_to("failure", {})
            rw.transport.nack(header)
            return

        # Insert tomogram into ispyb
        ispyb_command_list: list[dict] = [
            {
                "ispyb_command": "insert_tomogram",
                "volume_file": str(imod_output_path),
                "stack_file": tomo_params.stack_file,
                "size_x": int(mrc_header.nx / tomo_params.out_bin),
                "size_y": int(mrc_header.ny / tomo_params.out_bin),
                "size_z": int(tomo_params.vol_z / tomo_params.out_bin),
                "pixel_spacing": tomo_params.pixel_size,
                "tilt_angle_offset": str(
                    self.tilt_offset or tomo_params.manual_tilt_offset
                ),
                "z_shift": rot_centre_z,
                "file_directory": str(imod_output_path.parent),
                "central_slice_image": imod_output_path.name + "_Vol_thumbnail.jpeg",
                "tomogram_movie": imod_output_path.name + "_Vol_movie.png",
                "xy_shift_plot": imod_output_path.name + shift_plot_suffix,
                "proj_xy": imod_output_path.name + "_Vol_projXY.jpeg",
                "proj_xz": imod_output_path.name + "_Vol_projXZ.jpeg",
                "alignment_quality": str(self.alignment_quality),
            },
            {
                "ispyb_command": "insert_processed_tomogram",
                "file_path": tomo_params.stack_file,
                "processing_type": "Stack",
            },
            {
                "ispyb_command": "insert_processed_tomogram",
                "file_path": f"{imod_output_path.parent}/{imod_output_path.name}_alignment.jpeg",
                "processing_type": "Alignment",
            },
        ]

        # Forward results to images service
        self.log.info(f"Sending to images service {tomo_params.stack_file}")
        rw.send_to(
            "images",
            {
                "image_command": "tilt_series_alignment",
                "file": tomo_params.stack_file,
                "aln_file": str(aln_file),
                "pixel_size": tomo_params.pixel_size,
            },
        )
        rw.send_to(
            "images",
            {
                "image_command": "mrc_central_slice",
                "file": tomo_params.stack_file,
            },
        )
        rw.send_to(
            "images",
            {
                "image_command": "mrc_to_apng",
                "file": tomo_params.stack_file,
            },
        )
        self.log.info(f"Sending to images service {imod_output_path}")
        rw.send_to(
            "images",
            {
                "image_command": "mrc_central_slice",
                "file": str(imod_output_path),
            },
        )
        rw.send_to(
            "images",
            {
                "image_command": "mrc_to_apng",
                "file": str(imod_output_path),
            },
        )

        self.log.info("Sending to images service for XY and XZ projections")
        side_projection = (
            "YZ"
            if tomo_params.tilt_axis is not None and -45 < tomo_params.tilt_axis < 45
            else "XZ"
        )
        for projection_type in ["XY", side_projection]:
            images_call_params: dict[str, str | float] = {
                "image_command": "mrc_projection",
                "file": str(imod_output_path),
                "projection": projection_type,
                "pixel_spacing": tomo_params.pixel_size,
            }
            if projection_type in ["XZ", "YZ"] and self.thickness_pixels:
                images_call_params["thickness_ang"] = (
                    self.thickness_pixels * tomo_params.pixel_size
                )
            rw.send_to("images", images_call_params)

        # Forward results to denoise service
        self.log.info(f"Sending to denoise service {imod_output_path}")
        rw.send_to(
            "denoise",
            {
                "volume": str(imod_output_path),
                "output_dir": str(imod_output_path.parent.parent.parent / "Denoise"),
                "relion_options": dict(tomo_params.relion_options),
            },
        )

        # Insert tomogram into ispyb
        ispyb_parameters = {
            "ispyb_command": "multipart_message",
            "ispyb_command_list": ispyb_command_list,
        }
        self.log.info(f"Sending to ispyb {ispyb_parameters}")
        rw.send_to("ispyb_connector", ispyb_parameters)

        # Remove any temporary files
        for tmp_file in imod_output_path.parent.glob(
            f"{Path(tomo_params.stack_file).stem}*~"
        ):
            tmp_file.unlink()

        # Update success processing status
        rw.send_to("success", {})
        self.log.info(f"Done tomogram alignment for {tomo_params.stack_file}")
        rw.transport.ack(header)


def write_batch_directive_file(tomo_params: ImodTomoParameters):
    adoc_file = Path(tomo_params.stack_file).with_suffix(".adoc")
    with open(adoc_file, "w") as adoc:
        # Commands for copytomocoms
        adoc.write(f"setupset.datasetDirectory={adoc_file.parent}\n")
        adoc.write(f"setupset.copyarg.name={Path(tomo_params.stack_file).stem}\n")
        "setupset.copyarg.userawtlt=1"
        # "setupset.copyarg.dual=0"
        adoc.write(f"setupset.copyarg.pixel={tomo_params.pixel_size / 10}\n")
        adoc.write(f"setupset.copyarg.gold={tomo_params.beam_size}\n")
        adoc.write(f"setupset.copyarg.rotation={tomo_params.tilt_axis}\n")
        # "setupset.copyarg.extract=0"

        # Preprocessing
        adoc.write("runtime.Preprocessing.any.removeXrays=1\n")

        # Coarse Alignment
        # if tomo_params.patch:
        #    "comparam.prenewst.newstack.BinByFactor=2"

        # Tracking Choices
        adoc.write(f"runtime.Fiducials.any.trackingMethod={tomo_params.patch}\n")
        adoc.write("runtime.Fiducials.any.seedingMethod=1\n")

        # Beadtracking
        if not tomo_params.patch:
            adoc.write("comparam.track.beadtrack.LightBeads=0\n")
            adoc.write("comparam.track.beadtrack.LocalAreaTracking=1\n")
            adoc.write("comparam.track.beadtrack.SobelFilterCentering=1\n")
            adoc.write("comparam.track.beadtrack.KernelSigmaForSobel=1.5\n")
            adoc.write("runtime.BeadTracking.any.numberOfRuns=2\n")

        # Auto Seed Finding
        if not tomo_params.patch:
            adoc.write(
                f"comparam.autofidseed.autofidseed.TargetNumberOfBeads={tomo_params.bead_count}\n"
            )
            adoc.write("comparam.autofidseed.autofidseed.AdjustSizes=1\n")
            adoc.write("comparam.autofidseed.autofidseed.TwoSurfaces=0\n")

        # RAPTOR Parameters

        # Patch Tracking
        if tomo_params.patch:
            adoc.write(
                f"comparam.xcorr_pt.tiltxcorr.SizeOfPatchesXandY={tomo_params.patch_size} {tomo_params.patch_size}\n"
            )
            adoc.write(
                f"comparam.xcorr_pt.tiltxcorr.OverlapOfPatchesXandY={tomo_params.patch_overlap} {tomo_params.patch_overlap}\n"
            )
            adoc.write("comparam.xcorr_pt.tiltxcorr.IterateCorrelations=1\n")
            adoc.write("runtime.PatchTracking.any.adjustTiltAngles=1\n")

        # Alignment
        adoc.write("comparam.align.tiltalign.SurfacesToAnalyze=1\n")
        adoc.write("comparam.align.tiltalign.LocalAlignments=1\n")
        adoc.write("comparam.align.tiltalign.MagOption=3\n")
        adoc.write("comparam.align.tiltalign.TiltOption=5\n")
        adoc.write("comparam.align.tiltalign.RotOption=3\n")
        adoc.write("comparam.align.tiltalign.RobustFitting=1\n")

        # Aligned Stack Parameters
        adoc.write("runtime.AlignedStack.any.linearInterpolation=1\n")
        adoc.write(f"runtime.AlignedStack.any.binByFactor={tomo_params.out_bin}\n")

        # Reconstruction + SIRT Parameters
        adoc.write(f"comparam.tilt.tilt.THICKNESS={tomo_params.vol_z}\n")
        "comparam.tilt.tilt.LOG="
        adoc.write(f"runtime.Reconstruction.any.useSirt={tomo_params.sirt}\n")
        adoc.write(f"runtime.Reconstruction.any.doBackprojAlso={tomo_params.wbp}\n")
        if tomo_params.sirt:
            adoc.write(
                f"comparam.sirtsetup.sirtsetup.LeaveIterations={tomo_params.sirt_leave_iterations}\n"
            )

        # Postprocessing
        adoc.write(f"runtime.Trimvol.any.reorient={tomo_params.flip_vol}\n")
    return adoc_file
