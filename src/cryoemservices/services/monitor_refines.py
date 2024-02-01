from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import workflows.recipe
from pydantic import BaseModel, Field, ValidationError, root_validator, validator
from workflows.services.common_service import CommonService


class MonitorParams(BaseModel):
    monitor_command: str = Field(default="setup", alias="register")
    microscope: Optional[str]
    visit: Optional[str]
    year: Optional[int]
    grid: Optional[str]
    class_number: Optional[int]
    batch_size: Optional[int]
    project_dir: Optional[str]
    class_reference: Optional[str]
    mask_file: Optional[str]
    pixel_size: Optional[float]
    mask_diameter: Optional[float]
    resolution: Optional[float]
    number_of_particles: Optional[int]

    @validator("monitor_command")
    def is_spa_or_tomo(cls, command):
        if command not in ["setup", "done_refinement", "done_bfactor"]:
            raise ValueError("Unknown command.")
        return command

    @root_validator(skip_on_failure=True)
    def check_command_values(cls, values):
        command = values.get("monitor_command")
        if command == "setup":
            if (
                not values.get("microscope")
                or not values.get("visit")
                or not values.get("year")
                or not values.get("grid")
                or not values.get("class_number")
            ):
                raise KeyError(
                    "The following keys must be provided for setup:"
                    "microscope, visit, year, grid, class_number"
                )
        elif command == "done_refinement":
            if (
                not values.get("batch_size")
                or not values.get("project_dir")
                or not values.get("class_reference")
                or not values.get("class_number")
                or not values.get("mask_file")
                or not values.get("mask_diameter")
                or not values.get("pixel_size")
                or not values.get("resolution")
            ):
                raise KeyError(
                    "The following keys must be provided for done_refinement:"
                    "batch_size, project_dir, class_reference, class_number, mask_file"
                    "mask_diameter, pxiel_size, resolution"
                )
        elif command == "done_bfactor":
            if (
                not values.get("project_dir")
                or not values.get("number_of_particles")
                or not values.get("resolution")
            ):
                raise KeyError(
                    "The following keys must be provided for done_bfactor:"
                    "project_dir, number_of_particles, resolution"
                )
        return values


class MonitorRefine(CommonService):
    """
    A service for monitoring the progress of refinements
    """

    # Human readable service name
    _service_name = "MonitorRefine"

    # Logger name
    _logger_name = "cryoemservices.services.monitor_refine"

    def initializing(self):
        """Subscribe to a queue. Received messages must be acknowledged."""
        self.log.info("Refinement monitoring service starting")
        workflows.recipe.wrap_subscribe(
            self._transport,
            "murfey_feedback_m12",
            self.monitor_refine,
            acknowledgement=True,
            log_extender=self.extend_log,
            allow_non_recipe_messages=True,
        )

    def monitor_refine(self, rw, header: dict, message: dict):
        class MockRW:
            def dummy(self, *args, **kwargs):
                pass

        if not rw:
            print(
                "Incoming message is not a recipe message. Simple messages can be valid"
            )
            if (
                not isinstance(message, dict)
                or not message.get("parameters")
                or not message.get("content")
            ):
                self.log.error("Rejected invalid simple message")
                self._transport.nack(header)
                return
            self.log.debug("Received a simple message")

            # Create a wrapper-like object that can be passed to functions
            # as if a recipe wrapper was present.
            rw = MockRW()
            rw.transport = self._transport
            rw.recipe_step = {"parameters": message["parameters"]}
            rw.environment = {"has_recipe_wrapper": False}
            rw.set_default_channel = rw.dummy
            rw.send = rw.dummy
            message = message["content"]

        try:
            if isinstance(message, dict):
                monitor_params = MonitorParams(
                    **{**rw.recipe_step.get("parameters", {}), **message}
                )
            else:
                monitor_params = MonitorParams(
                    **{**rw.recipe_step.get("parameters", {})}
                )
        except (ValidationError, TypeError) as e:
            self.log.warning(
                f"Monitor parameter validation failed for message: {message} "
                f"and recipe parameters: {rw.recipe_step.get('parameters', {})} "
                f"with exception: {e}"
            )
            rw.transport.nack(header)
            return

        if monitor_params.monitor_command == "setup":
            visit_tmp_dir = Path(
                f"/dls/{monitor_params.microscope}/data/{monitor_params.year}/{monitor_params.visit}/tmp/Refinement"
            )
            (visit_tmp_dir / "MotionCorr").unlink(missing_ok=True)
            (visit_tmp_dir / "MotionCorr").symlink_to(
                f"/dls/{monitor_params.microscope}/data/{monitor_params.year}/{monitor_params.visit}/"
                f"processed/{monitor_params.grid}/relion_murfey/MotionCorr"
            )
            (visit_tmp_dir / "CtfFind").unlink(missing_ok=True)
            (visit_tmp_dir / "CtfFind").symlink_to(
                f"/dls/{monitor_params.microscope}/data/{monitor_params.year}/{monitor_params.visit}/"
                f"processed/{monitor_params.grid}/relion_murfey/CtfFind"
            )

            with open(
                visit_tmp_dir / "MotionCorr/job002/corrected_micrographs.star", "r"
            ) as mc_file:
                for i in range(20):
                    line = mc_file.readline()
                    if "opticsGroup1" in line:
                        pixel_size = float(line.split()[-1])
                        break
            if not pixel_size:
                self.log.warning("No pixel size found")
                rw.transport.nack(header)
                return

            # Get the information about the 3d run
            class3d_all = Path(
                f"/dls/{monitor_params.microscope}/data/{monitor_params.year}/{monitor_params.visit}/"
                f"processed/{monitor_params.grid}/relion_murfey/Class3D"
            ).glob("job*")
            class3d_dir = list(class3d_all)[0]

            with open(class3d_dir / "run_it025_model.star") as class3d_output:
                postprocess_lines = class3d_output.readlines()
                for line in postprocess_lines:
                    if "_rlnPixelSize" in line:
                        downscaled_pixel_size = float(line.split()[-1])
                        break
            if not downscaled_pixel_size:
                self.log.warning("No class3d pixel size found")
                rw.transport.nack(header)
                return

            with open(class3d_dir / "run_it025_optimiser.star") as class3d_output:
                postprocess_lines = class3d_output.readlines()
                for line in postprocess_lines:
                    if "_rlnParticleDiameter" in line:
                        mask_diameter = float(line.split()[-1])
                        break
            if not mask_diameter:
                self.log.warning("No mask diameter found")
                rw.transport.nack(header)
                return

            # Boxsize conversion as in particle extraction, enlarged by 25%
            boxsize = mask_diameter / pixel_size / 1.1 * 1.25

            # Set up and run a first refinement job
            refine_message = {
                "recipes": ["em-spa-refine-test"],
                "parameters": {
                    "refine_job_dir": f"{visit_tmp_dir}/Refine3D/job004",
                    "class3d_dir": f"{class3d_dir}",
                    "micrographs_file": f"{visit_tmp_dir}/CtfFind/job006/micrographs_ctf.star",
                    "class_number": monitor_params.class_number,
                    "boxsize": boxsize,
                    "pixel_size": pixel_size,
                    "downscaled_pixel_size": downscaled_pixel_size,
                    "mask_diameter": mask_diameter,
                    "nr_iter": "25",
                    "picker_id": "0",
                    "refined_grp_uuid": "0",
                    "refined_class_uuid": "0",
                },
            }
            self.log.info("Running refinement")
            if isinstance(rw, MockRW):
                rw.transport.send(
                    destination="processing_recipe", message=refine_message
                )
            else:
                rw.send_to("processing_recipe", refine_message)

        elif monitor_params.monitor_command == "done_refinement":
            # Run bfactor jobs once the first one is done
            bfactor_particle_counts = [
                1000 * 2**n
                for n in range(int(np.log(monitor_params.batch_size) / np.log(2) + 1))
            ]
            for particle_count in bfactor_particle_counts:
                bfactor_message = {
                    "recipes": ["em-spa-bfactor-test"],
                    "parameters": {
                        "bfactor_directory": f"{monitor_params.project_dir}/bfactor_run/bfactor_{particle_count}",
                        "class_reference": monitor_params.class_reference,
                        "class_number": monitor_params.class_number,
                        "number_of_particles": particle_count,
                        "batch_size": monitor_params.batch_size,
                        "pixel_size": monitor_params.pixel_size,
                        "mask_file": monitor_params.mask_file,
                        "mask_diameter": monitor_params.mask_diameter,
                        "picker_id": "0",
                        "refined_class_uuid": "0",
                        "refined_grp_uuid": "0",
                        "project_dir": monitor_params.project_dir,
                    },
                }
                self.log.info(f"Running bfactor {particle_count} particles")
                if isinstance(rw, MockRW):
                    rw.transport.send(
                        destination="processing_recipe", message=bfactor_message
                    )
                else:
                    rw.send_to("processing_recipe", bfactor_message)

                with open(
                    f"{monitor_params.project_dir}/bfactor_resolutions.txt", "w"
                ) as bfile:
                    bfile.write(
                        f"{monitor_params.batch_size} {monitor_params.resolution}"
                    )

        elif monitor_params.monitor_command == "done_bfactor":
            with open(
                f"{monitor_params.project_dir}/bfactor_resolutions.txt", "a"
            ) as bfile:
                bfile.write(
                    f"{monitor_params.number_of_particles} {monitor_params.resolution}"
                )

            bfactor_results = np.genfromtxt(
                f"{monitor_params.project_dir}/bfactor_resolutions.txt"
            )

            # Fit and plot
            bfactor_fitting = np.polyfit(
                1 / np.array(bfactor_results[:, 1]) ** 2,
                np.log(np.array(bfactor_results[:, 0])),
                2,
            )
            plot_resolutions = np.arange(
                0.004, max(1 / bfactor_results[:, 1] ** 2) + 0.001, 0.002
            )
            plot_particles = (
                bfactor_fitting[2]
                + bfactor_fitting[1] * plot_resolutions
                + bfactor_fitting[0] * plot_resolutions**2
            )

            bfactor_linear = np.polyfit(
                1 / np.array(bfactor_results[:, 1]) ** 2,
                np.log(np.array(bfactor_results[:, 0])),
                1,
            )
            linear_particles = bfactor_linear[1] + bfactor_linear[0] * plot_resolutions

            plt.figure(figsize=(8, 8))
            plt.scatter(
                np.log(np.array(bfactor_results[:, 0])),
                1 / np.array(bfactor_results[:, 1]) ** 2,
                color="black",
                label="B-factor data",
            )
            plt.plot(
                plot_particles,
                plot_resolutions,
                color="black",
                linestyle="--",
                label=f"Quadratic fit {bfactor_fitting[2]} + {bfactor_fitting[1]}z + {bfactor_fitting[0]}z^2",
            )
            plt.plot(
                linear_particles,
                plot_resolutions,
                color="black",
                linestyle="-",
                label=f"Linear fit {bfactor_linear[1]} + {bfactor_linear[0]}z",
            )
            plt.xlabel("log particle count")
            plt.ylabel("1 / Resolution^2")
            plt.legend()
            plt.savefig(f"{monitor_params.project_dir}/bfactors.png")

            self.log.info(
                f"B-factor from {len(bfactor_results)} runs is {bfactor_linear[0]/2}"
            )
        self.log.info(f"Done monitoring {monitor_params.monitor_command}.")
        rw.transport.ack(header)
