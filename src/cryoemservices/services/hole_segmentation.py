from pathlib import Path

from pydantic import BaseModel, ValidationError
from workflows.recipe import wrap_subscribe

from cryoemservices.pipeliner_plugins.segment_holes import segment_single_image
from cryoemservices.services.common_service import CommonService
from cryoemservices.util.models import MockRW
from cryoemservices.util.slurm_submission import slurm_submission_for_services


class HoleSegmentationParameters(BaseModel):
    micrograph: Path
    output_dir: Path
    checkpoint: Path
    submit_to_slurm: bool = False
    binning: int = 4
    min_area_fraction: float = 0.001
    beam_threshold: float = 10
    hole_threshold: float = 100
    save_summary: bool = False
    slurm_memory: int = 10000


class HoleSegmentation(CommonService):
    """
    A service for segmenting out micrographs with segment anything
    """

    # Logger name
    _logger_name = "cryoemservices.services.hole_segmentation"

    def initializing(self):
        """Subscribe to a queue. Received messages must be acknowledged."""
        self.log.info("HoleSegmentation service starting")
        wrap_subscribe(
            self._transport,
            self._environment["queue"] or "hole_segmentation",
            self.segment_hole,
            acknowledgement=True,
            allow_non_recipe_messages=True,
        )

    def segment_using_slurm(self, command: list[str], output_image: Path, memory: int):
        """Submit jobs to a slurm cluster via the RestAPI"""
        image_results = {
            "hole_percentage": 0.0,
            "hole_detected": False,
            "beam_edge_detected": False,
        }

        slurm_outcome = slurm_submission_for_services(
            log=self.log,
            service_config_file=self._environment["config"],
            slurm_cluster=self._environment["slurm_cluster"],
            job_name="SegmentHole",
            command=command,
            project_dir=output_image.parent,
            output_file=output_image,
            cpus=1,
            use_gpu=True,
            use_singularity=False,
            memory_request=memory,
            script_extras="module load EM/cryoem-services",
        )

        if not slurm_outcome.returncode:
            # Read in the output logs
            slurm_output_file = output_image.with_suffix(".out")
            if Path(slurm_output_file).is_file():
                with open(slurm_output_file) as f:
                    for line in f.readlines():
                        if "Beam edge detected" in line:
                            image_results["beam_edge_detected"] = True
                        if "Hole area" in line:
                            image_results["hole_percent"] = float(line.split()[-1][:-1])
                            image_results["hole_detected"] = (
                                image_results["hole_percent"] > 0
                            )
        output_image.with_suffix(".out").unlink(missing_ok=True)
        output_image.with_suffix(".err").unlink(missing_ok=True)
        return slurm_outcome, image_results

    def segment_hole(self, rw, header: dict, message: dict):
        """Main function which interprets and processes received messages"""
        if not rw:
            self.log.info("Received a simple message")
            if not isinstance(message, dict):
                self.log.error("Rejected invalid simple message")
                self._reject_message(header, requeue=False)
                return

            # Create a wrapper-like object that can be passed to functions
            # as if a recipe wrapper was present.
            rw = MockRW(self._transport)
            rw.recipe_step = {"parameters": message}

        try:
            if isinstance(message, dict):
                segment_params = HoleSegmentationParameters(
                    **{**rw.recipe_step.get("parameters", {}), **message}
                )
            else:
                segment_params = HoleSegmentationParameters(
                    **{**rw.recipe_step.get("parameters", {})}
                )
        except (ValidationError, TypeError) as e:
            self.log.warning(
                f"Hole segmentation parameter validation failed for message: {message} "
                f"and recipe parameters: {rw.recipe_step.get('parameters', {})} "
                f"with exception: {e}"
            )
            self._reject_message(header, transport=rw.transport, requeue=False)
            return

        output_mask = (
            Path(segment_params.output_dir)
            / f"{segment_params.micrograph.stem}_mask.png"
        )
        if segment_params.submit_to_slurm:
            segment_command = [
                "cryoemservices.segment_holes",
                "--micrograph_dir",
                str(Path(segment_params.micrograph).parent),
                "--input_glob",
                Path(segment_params.micrograph).name,
                "--output_dir",
                str(segment_params.output_dir),
                "--checkpoint",
                str(segment_params.checkpoint),
                "--binning",
                str(segment_params.binning),
                "--min_area_fraction",
                str(segment_params.min_area_fraction),
                "--beam_threshold",
                str(segment_params.beam_threshold),
                "--hole_threshold",
                str(segment_params.hole_threshold),
                "--save_summary" if segment_params.save_summary else "",
            ]
            slurm_result, image_results = self.segment_using_slurm(
                segment_command,
                output_mask,
                segment_params.slurm_memory,
            )
            if slurm_result.returncode:
                self.log.error(f"Segmentation of {segment_params.micrograph} failed")
                self._reject_message(header, transport=rw.transport)
                return
        else:
            image_results, mask_results = segment_single_image(
                impath=Path(segment_params.micrograph),
                checkpoint=Path(segment_params.checkpoint),
                output_dir=Path(segment_params.output_dir),
                binning=segment_params.binning,
                min_area_fraction=segment_params.min_area_fraction,
                beam_threshold=segment_params.beam_threshold,
                hole_threshold=segment_params.hole_threshold,
                save_summary=segment_params.save_summary,
            )

        self.log.info("Sending to Murfey")
        rw.send_to(
            "murfey_feedback",
            {
                "register": "segmented_hole",
                "micrograph": str(segment_params.micrograph),
                "hole_percent": image_results.get("hole_percent", 0),
                "hole_detected": image_results.get("hole_detected", False),
                "beam_edge_detected": image_results.get("beam_edge_detected", False),
            },
        )

        self.log.info(f"Done hole segmentation for {segment_params.micrograph}.")
        rw.transport.ack(header)
