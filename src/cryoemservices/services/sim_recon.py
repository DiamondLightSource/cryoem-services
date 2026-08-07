from __future__ import annotations

import ast
import json
from pathlib import Path

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


class PySIMReconService(CommonService):
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
            self.log.warning(
                f"PySIMReconParameters validation failed for message: {message} "
                f"and recipe parameters: {rw.recipe_step.get('parameters', {})} "
                f"with exception: {e}"
            )
            self._reject_message(header, transport=rw.transport, requeue=False)
            return

        self.log.info(
            "Running PySIMRecon with the following parameters:\n"
            f"{params.model_dump(mode='json')}"
        )
