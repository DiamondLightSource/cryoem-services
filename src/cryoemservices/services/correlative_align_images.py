from pathlib import Path
from typing import Any, cast

import cv2
import ispyb.sqlalchemy
import sqlalchemy.orm
from ispyb.sqlalchemy import _auto_db_schema as ISPyBDB
from pydantic import BaseModel, ValidationError
from sqlalchemy import select
from workflows.recipe import RecipeWrapper, wrap_subscribe

from cryoemservices.services.common_service import CommonService
from cryoemservices.util.config import config_from_file
from cryoemservices.util.image_processing.align_images_using_neighbors import (
    align_images_using_neighbors,
)
from cryoemservices.util.image_processing.shared import crop_image
from cryoemservices.util.models import MockRW


class AlignImagesParameters(BaseModel):
    # ISPyB Atlas atlasId values
    id_ref: int
    id_mov: int
    # Optional keys for manual testing
    image_ref: Path | None = None
    pixel_size_ref: float | None = None
    image_mov: Path | None = None
    pixel_size_mov: float | None = None
    save_dir: Path | None = None


def _get_atlas_proposal_session_experiment_type(
    session: sqlalchemy.orm.Session, atlas_id: int
):
    """
    Runs an ISPyB query to get the Atlas and its corresponding Proposal, BLSession,
    and ExperimentType rows.
    """

    statement = (
        select(
            ISPyBDB.Atlas, ISPyBDB.Proposal, ISPyBDB.BLSession, ISPyBDB.ExperimentType
        )
        .join(
            ISPyBDB.DataCollectionGroup,
            ISPyBDB.Atlas.dataCollectionGroupId
            == ISPyBDB.DataCollectionGroup.dataCollectionGroupId,
        )
        .join(
            ISPyBDB.ExperimentType,
            ISPyBDB.DataCollectionGroup.experimentTypeId
            == ISPyBDB.ExperimentType.experimentTypeId,
        )
        .join(
            ISPyBDB.BLSession,
            ISPyBDB.DataCollectionGroup.sessionId == ISPyBDB.BLSession.sessionId,
        )
        .join(
            ISPyBDB.Proposal,
            ISPyBDB.BLSession.proposalId == ISPyBDB.Proposal.proposalId,
        )
        .where(ISPyBDB.Atlas.atlasId == atlas_id)
    )
    result = session.execute(statement).one()  # Will error if no match is found
    return (
        cast(ISPyBDB.Atlas, result.Atlas),
        cast(ISPyBDB.Proposal, result.Proposal),
        cast(ISPyBDB.BLSession, result.BLSession),
        cast(ISPyBDB.ExperimentType, result.ExperimentType),
    )


class AlignImagesService(CommonService):
    """
    A CryoEM service to align to images to one another
    """

    _logger_name = __name__

    def initializing(self):
        """Subscribe to a queue. Received messages must be acknowledged."""
        # Set up ISPyB session maker
        service_config = config_from_file(self._environment["config"])
        self._database_session_maker = sqlalchemy.orm.sessionmaker(
            bind=sqlalchemy.create_engine(
                ispyb.sqlalchemy.url(credentials=service_config.ispyb_credentials),
                connect_args={"use_pure": True},
            )
        )
        # Subscribe service to RMQ queue
        wrap_subscribe(
            self._transport,
            self._environment["queue"] or "correlative.align_images",
            self.call_align_images,
            acknowledgement=True,
            allow_non_recipe_messages=True,
        )
        self.log.info("CorrelativeAlignImages service ready")

    def call_align_images(
        self,
        rw: RecipeWrapper | None,
        header: dict[str, Any],
        message: dict[str, Any] | None,
    ):
        """Pass incoming message to the relevant plugin function."""
        # Encase message in ReceipeWrapper if none was provided
        if not rw:
            self.log.info("Received a simple message")
            if not isinstance(message, dict):
                self.log.error("Rejected invalid simple message")
                self._reject_message(header, requeue=False)
                return
            # Create a wrapper-like object to be passed to functions
            rw = MockRW(self._transport)
            rw.recipe_step = {"paramters": message}

        try:
            if isinstance(message, dict):
                params = AlignImagesParameters(
                    **{**rw.recipe_step.get("parameters", {}), **message}
                )
            else:
                params = AlignImagesParameters(
                    **{**rw.recipe_step.get("parameters", {})}
                )
        except (ValidationError, TypeError) as e:
            self.log.error(
                f"AlignImagesParameters validation failed for message: {message} "
                f"and recipe parameters: {rw.recipe_step.get('parameters', {})} "
                f"with exception: {e}"
            )
            self._reject_message(header, transport=rw.transport, requeue=False)
            return

        # Acknowledge receipt of parameters
        self.log.info(
            "Running image alignment with the following parameters:\n"
            f"{params.model_dump(mode='json')}"
        )

        ###############################################################################
        # Image alignment logic goes here
        ###############################################################################

        # Load the ISPyB Atlas entries using the provided IDs
        try:
            with self._database_session_maker() as session:
                atlas_ref, proposal_ref, bl_session_ref, experiment_type_ref = (
                    _get_atlas_proposal_session_experiment_type(
                        session=session, atlas_id=params.id_ref
                    )
                )
                atlas_mov, proposal_mov, bl_session_mov, experiment_type_mov = (
                    _get_atlas_proposal_session_experiment_type(
                        session=session, atlas_id=params.id_mov
                    )
                )
                # Construct visit names
                visit_ref = f"{proposal_ref.proposalCode}{proposal_ref.proposalNumber}-{bl_session_ref.visit_number}"
                visit_mov = f"{proposal_mov.proposalCode}{proposal_mov.proposalNumber}-{bl_session_mov.visit_number}"
        except Exception:
            self.log.error(
                "Uncaught exception {e!r} while querying ISPyB, "
                "quarantining message and shutting down instance.",
                exc_info=True,
            )
            self._reject_message(header, transport=rw.transport)
            return

        # Align images differently depending on which data types are being compared
        try:
            match sorted((experiment_type_ref.name, experiment_type_mov.name)):
                case (
                    ["FIB", "Tomography"]
                    | ["FIB", "Lamella Tomography"]
                    | ["FIB", "Single Particle"]
                ):
                    self.log.info("Aligning FIB atlas to TEM one")
                    result = self._handle_fib_tem_case(
                        params,
                        atlas_ref,
                        visit_ref,
                        atlas_mov,
                        visit_mov,
                    )
                    if result["transform"] is not None:
                        self.log.info("Successfully aligned FIB atlas to TEM atlas")
                case _:
                    self.log.info(
                        "No image alignment algorithm implemented for this case yet"
                    )
        except Exception:
            self.log.error(
                "Error while attempting to perform image alignment", exc_info=True
            )
            self._reject_message(header, transport=rw.transport)

        # Ack message after completion
        rw.transport.ack(header)
        return

    def _handle_fib_tem_case(
        self,
        params: AlignImagesParameters,
        atlas_ref: ISPyBDB.Atlas,
        visit_ref: str,
        atlas_mov: ISPyBDB.Atlas,
        visit_mov: str,
    ):
        # Ensure image path and pixel size are present as params or in the atlas tables
        if not (
            (
                (atlas_ref.atlasImage and atlas_ref.pixelSize)
                or (params.image_ref and params.pixel_size_ref)
            )
            and (
                (atlas_mov.atlasImage and atlas_mov.pixelSize)
                or (params.image_mov and params.pixel_size_mov)
            )
        ):
            raise ValueError(
                "Could not determine the file path or pixel size for "
                "either the reference or moving image"
            )
        # Load image files and pixel sizes
        # Load from params, then fall back to table (useful for testing)
        file_ref = (
            params.image_ref
            if params.image_ref is not None
            else Path(atlas_ref.atlasImage)
        )
        img_ref = cv2.imread(file_ref, flags=cv2.IMREAD_GRAYSCALE)
        pixel_size_ref = (
            params.pixel_size_ref
            if params.pixel_size_ref is not None
            else atlas_ref.pixelSize
        )
        file_mov = (
            params.image_mov
            if params.image_mov is not None
            else Path(atlas_mov.atlasImage)
        )
        img_mov = cv2.imread(file_mov, flags=cv2.IMREAD_GRAYSCALE)
        pixel_size_mov = (
            params.pixel_size_mov
            if params.pixel_size_mov is not None
            else atlas_mov.pixelSize
        )

        # Get save directory from message
        save_dir = params.save_dir
        # Fall back to constructing a save directory using ISPyB-derived data
        if save_dir is None:
            visit_dir = file_ref.parents[-(file_ref.parts.index(visit_ref) + 1)]
            save_dir = (
                visit_dir / "processed" / "correlation" / visit_mov / file_mov.stem
            )

        # Store images under the visit name of the moving image
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
            self.log.info(f"Created save directory at {save_dir}")

        # Rescale images to same pixel size and crop to the same dimensions
        pixel_size_target = 4.0e-6
        resized_ref = cv2.resize(
            img_ref,
            dsize=None,
            dst=None,
            fx=pixel_size_ref / pixel_size_target,
            fy=pixel_size_ref / pixel_size_target,
            interpolation=cv2.INTER_AREA,
        )
        resized_mov = cv2.resize(
            img_mov,
            dsize=None,
            dst=None,
            fx=pixel_size_mov / pixel_size_target,
            fy=pixel_size_mov / pixel_size_target,
            interpolation=cv2.INTER_AREA,
        )
        height = min(resized_ref.shape[0], resized_mov.shape[0])
        width = min(resized_ref.shape[1], resized_mov.shape[1])
        cropped_ref = crop_image(resized_ref, width=width, height=height)
        cropped_mov = crop_image(resized_mov, width=width, height=height)

        # Perform image alignment
        # Parameters empirically determined to work for a pixel size of ~4.0e-6
        return align_images_using_neighbors(
            cropped_ref,
            cropped_mov,
            median_blur=None,
            gaussian_blur=0.5,
            sobel_kernel=3,
            use_hanning=False,
            min_component_area=20,
            threshold_percentile=98.5,
            morph_close_kernel=10,
            morph_open_kernel=2,
            min_feature_area=20,
            max_feature_area=1000,
            min_solidity=0.6,
            min_ellipse_fit=0.4,
            max_aspect_ratio=0.95,
            max_neighbor_distance=200,
            min_score=0.3,
            ransac_threshold=5,
            save_images=True,
            save_tables=True,
            save_dir=save_dir,
        )
