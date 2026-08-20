from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ValidationError, field_validator
from workflows.recipe import RecipeWrapper, wrap_subscribe

from cryoemservices.services.common_service import CommonService
from cryoemservices.util.models import MockRW


class SIMOTFParameters(BaseModel):
    """
    These are the shared values used by PySIMRecon to run the 'sim-otf' function. These
    are read in from a config file under the section '[otf config]'.

    The values listed are the default values used by PySIMRecon. 'cudasirecon', which
    PySIMRecon runs under the hood, has a different set of preset values, which are
    overridden by these ones.

    Source:
    https://github.com/DiamondLightSource/PySIMRecon/commit/c039b09cbe3b510c032462c6817a517d2d7b2f99
    """

    # Number of phases
    nphases: int = 5
    # The diameter of the bead in microns
    beaddiam: float = 0.17
    # The k0 vector angle with which the PSF is taken
    angle: float = -0.264228
    # Do not perform bead size compensation, default False (do perform).
    nocompen: bool = False
    # The starting and end pixel for interpolation along kr axis
    fixorigin: tuple[int, int] = (2, 9)
    # The (effective) NA of the objective
    na: float = 0.9
    # The index of refraction of the immersion liquid.
    nimm: float = 1
    # User-supplied number as the background to subtract. If `None`, background will be estimated from image.
    background: int = 500


class SIMReconParameters(BaseModel):
    """
    These are the shared values used by PySIMRecon to run the 'sim-recon' function.
    These are read in from a config file under the section '[recon config]'.

    Source:
    https://github.com/DiamondLightSource/PySIMRecon/commit/c039b09cbe3b510c032462c6817a517d2d7b2f99
    """

    # Refractive index of air (or gaseous N2) is ~1.
    nimm: float = 1
    # Equivalent of 'bias offset' in softworx. Often left at 0 as small comapred to signal - a residual from when EMCCD had biases of 2000 or more.
    background: int = 200
    # Small means less smoothing, more chance of hammer stroke noise but higher resolution. Good estimate is the heuristic from SIMCheck.
    wiener: float = 0.0010
    # Use these pattern vector k0 angles for all directions (instead of inferring the rest of the angles from angle0).
    k0angles: tuple[float, float, float] = (-0.264228, 1.829976, -2.353254)
    # Number of SIM  directions.
    ndirs: int = 3
    # Number of pattern phases per SIM direction.
    nphases: int = 5
    # Detection objective's numerical aperture.
    na: float = 0.9
    # Using rotationally averaged OTF; otherwise using 3/2D OTF for 3/2D raw data.
    otfRA: int = 1
    # Dampen order-0 in final assembly; do not use for 2D SIM; good choice for high-background images.
    dampenOrder0: float = 0
    # Lateral zoom factor in the output over the input images.
    zoomfact: float = 2
    # Axial zoom factor.
    zzoom: float = 1
    # Output apodization gamma; 1.0 means triangular apo; lower value means less dampening of high-resolution info at the tradeoff of higher noise.
    gammaApo: float = 1
    # 1 = do not perform any bleach correction
    norescale: int = 1
    # z pixel size of PSF
    zresPSF: float = 0.125


class WavelengthParameters(BaseModel):
    """
    Override values for the individual wavelengths used in the SIM, along with the
    path to its OTF file.
    """

    # General
    wavelength: int

    # OTF values
    ls: float | None = None  # Line spacing
    beaddiam: float | None = None  # Bead diameter

    # OTF path
    otf_path: Path | None = None  # Path to OTF file


class PySIMReconParameters(BaseModel):
    visit_name: str
    file: Path
    output_dir: Path
    blue_params: WavelengthParameters = WavelengthParameters(wavelength=452)
    green_params: WavelengthParameters = WavelengthParameters(wavelength=525)
    red_params: WavelengthParameters = WavelengthParameters(wavelength=605)
    far_red_params: WavelengthParameters = WavelengthParameters(wavelength=655)
    sim_otf_params: SIMOTFParameters = SIMOTFParameters()
    sim_recon_params: SIMReconParameters = SIMReconParameters()

    @field_validator(
        "blue_params", "green_params", "red_params", "far_red_params", mode="before"
    )
    @classmethod
    def parse_stringified_dict(cls, value):
        # Parse strings as dicts
        if isinstance(value, str):
            try:
                # Evaluate as a JSON string first
                return json.loads(value)
            except Exception:
                try:
                    # Evaluate as a Python literal
                    return ast.literal_eval(value)
                except Exception as e:
                    raise ValueError(f"Could not evaluate {value} as a dict") from e
        # Return as-is
        return value


class SIMReconService(CommonService):
    """
    A service that will run PySIMRecon with the desired parameters on the incoming
    SIM data files.
    """

    _log_name = "cryoemservices.services.sim_recon"

    def initializing(self):
        """
        Subscribe to a queue. Received messages must be acknowledged.
        """
        self.log.info("SIM reconstruction service starting")
        # Subscribe service to RMQ queue
        wrap_subscribe(
            self._transport,
            self._environment["queue"] or "sim.reconstruction",
            self.call_pysimrecon,
            acknowledgement=True,
            allow_non_recipe_messages=True,
        )

    def call_pysimrecon(
        self,
        rw: RecipeWrapper | None,
        header: dict,
        message: dict | None,
    ):
        """
        Pass incoming message to the relevant plugin function
        """
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
                params = PySIMReconParameters(
                    **{**rw.recipe_step.get("parameters", {}), **message}
                )
            else:
                params = PySIMReconParameters(**rw.recipe_step.get("parameters", {}))
        except (ValidationError, TypeError) as e:
            self.log.error(
                f"PySIMReconParameters validation failed for message: {message} "
                f"and recipe parameters: {rw.recipe_step.get('parameters', {})} "
                f"with exception: {e}"
            )
            self._reject_message(header, transport=rw.transport, requeue=False)
            return

        self.log.info(
            "Running PySIMRecon with the following parameters:\n"
            f"{json.dumps(params.model_dump(), indent=2, default=str)}"
        )

        try:
            # ------------------------------------------------
            # Create the config files needed to run PySIMRecon
            # ------------------------------------------------

            # Find the visit directory and create a setup directory
            visit_idx = params.file.parts.index(params.visit_name)
            visit_dir = Path(*params.file.parts[: visit_idx + 1])
            setup_dir = visit_dir / "setup"
            setup_dir.mkdir(parents=True, exist_ok=True)

            # 1. 'defaults.cfg'
            # -----------------
            defaults_config_lines = []
            # Construct the OTF and reconstruction sections of the config
            for section_name, params_name in (
                ("[otf config]", "sim_otf_params"),
                ("[recon config]", "sim_recon_params"),
            ):
                defaults_config_lines.append(section_name)
                # Load relevant model from PySIMReconParameters
                params_section = cast(
                    SIMOTFParameters | SIMReconParameters,
                    getattr(params, params_name),
                )
                # Load and append values for each field
                for name in type(params_section).model_fields.keys():
                    # Serialise any tuple values encountered
                    if isinstance((value := getattr(params_section, name)), tuple):
                        value = ",".join([str(i) for i in value])
                    defaults_config_lines.append(f"{name}={value}")
                # Add a newline between sections
                defaults_config_lines.append("")
            # Save the output to the setup directory
            defaults_config = setup_dir / "defaults.cfg"
            with open(defaults_config, "w") as f:
                f.write("\n".join(defaults_config_lines))
            self.log.info(f"Created config file {defaults_config}")

            # 2. Configs for each wavelength
            # ------------------------------
            wavelength_configs: list[Path] = []
            otf_files: dict[int, Path] = {}
            for params_name in (
                "blue_params",
                "green_params",
                "red_params",
                "far_red_params",
            ):
                # Load the relevant model from PySIMReconParameters
                wavelength_params = cast(
                    WavelengthParameters, getattr(params, params_name)
                )
                wavlength_config_lines = []

                # 'ls' must be provided in order to continue
                if not wavelength_params.ls:
                    raise ValueError("No value for 'ls' was provided")
                # OTF file must also be present
                if not wavelength_params.otf_path:
                    raise ValueError("No OTF file path was provided")

                # Add sections and their corresponding values
                wavlength_config_lines.append("[otf config]")
                wavlength_config_lines.append(f"ls={wavelength_params.ls}")
                if wavelength_params.beaddiam:
                    wavlength_config_lines.append(
                        f"beaddiam={wavelength_params.beaddiam}"
                    )
                wavlength_config_lines.append("")
                wavlength_config_lines.append("[recon config]")
                wavlength_config_lines.append(f"ls={wavelength_params.ls}")
                wavlength_config_lines.append("")

                # Save the output to the setup directory and append file to list
                wavelength_config = setup_dir / f"{wavelength_params.wavelength}.cfg"
                with open(wavelength_config, "w") as f:
                    f.write("\n".join(wavlength_config_lines))
                wavelength_configs.append(wavelength_config)
                self.log.info(f"Created config file {wavelength_config}")

                # Extract and add OTF file to dict
                otf_files[wavelength_params.wavelength] = wavelength_params.otf_path

            # 3. 'config.ini'
            # ---------------
            master_config_lines = []

            # Populate the 'configs' section
            master_config_lines.append("[configs]")
            master_config_lines.append(f"directory={str(setup_dir)}")
            master_config_lines.append(f"defaults={str(defaults_config.name)}")
            # Iteratively add files for wavelengths
            for file in wavelength_configs:
                master_config_lines.append(f"{file.stem}={file.name}")

            # Populate the 'otfs' section
            master_config_lines.append("[otfs]")
            otf_parent: Path | None = None
            # Iteratively add files for wavelengths
            for wavelength, otf_file in otf_files.items():
                if not otf_parent:
                    otf_parent = otf_file.parent
                    master_config_lines.append(f"directory={str(otf_parent)}")
                master_config_lines.append(f"{wavelength}={str(otf_file.name)}")

            # Save the output to the setup directory
            master_config = setup_dir / "config.ini"
            with open(master_config, "w") as f:
                f.write("\n".join(master_config_lines))
            self.log.info(f"Created config file {master_config}")

        except Exception:
            self.log.error("Error creating PySIMRecon config files", exc_info=True)
            self._reject_message(header, transport=rw.transport, requeue=False)
            return

        # Ack message after completion
        rw.transport.ack(header)
        return
