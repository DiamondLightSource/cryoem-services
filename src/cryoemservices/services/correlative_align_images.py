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
    id_ref: int | None = None
    id_mov: int | None = None

    # Optional keys for manual testing
    # --------------------------------
    # Reference image
    visit_ref: str | None = None
    experiment_type_ref: str | None = None
    image_ref: Path | None = None
    pixel_size_ref: float | None = None
    # Moving image
    visit_mov: str | None = None
    experiment_type_mov: str | None = None
    image_mov: Path | None = None
    pixel_size_mov: float | None = None
    # Save directory
    save_dir: Path | None = None
    # Image alignment params
    # These parameters have been empirically determined to work with a target pixel
    # size of 4e-6
    target_pixel_size: float = 4e-6
    median_blur: int | None = None
    gaussian_blur: float | None = 0.5
    sobel_kernel: int | None = 3
    use_hanning: bool = False
    min_component_area: int | None = 20
    threshold_percentile: float | None = 98.5
    morph_close_kernel: int | None = 10
    morph_open_kernel: int | None = 2
    min_feature_area: int | None = 20
    max_feature_area: int | None = 1000
    min_solidity: float | None = 0.6
    min_ellipse_fit: float | None = 0.4
    max_aspect_ratio: float | None = 0.95
    max_neighbor_distance: int | None = 200
    min_score: float | None = 0.3
    ransac_threshold: float = 5
    save_images: bool = True
    save_tables: bool = True


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
        if params.id_ref and params.id_mov:
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
                    # Extract and construct needed values
                    visit_ref = f"{proposal_ref.proposalCode}{proposal_ref.proposalNumber}-{bl_session_ref.visit_number}"
                    experiment_name_ref = experiment_type_ref.name
                    image_ref = Path(atlas_ref.atlasImage)
                    pixel_size_ref = atlas_ref.pixelSize

                    visit_mov = f"{proposal_mov.proposalCode}{proposal_mov.proposalNumber}-{bl_session_mov.visit_number}"
                    experiment_name_mov = experiment_type_mov.name
                    image_mov = Path(atlas_mov.atlasImage)
                    pixel_size_mov = atlas_mov.pixelSize

            except Exception:
                self.log.error(
                    "Uncaught exception {e!r} while querying ISPyB, "
                    "quarantining message and shutting down instance.",
                    exc_info=True,
                )
                self._reject_message(header, transport=rw.transport)
                return
        # Use the directly provided values in the message
        else:
            if (
                params.visit_ref is not None
                and params.experiment_type_ref is not None
                and params.image_ref is not None
                and params.pixel_size_ref is not None
                and params.visit_mov is not None
                and params.experiment_type_mov is not None
                and params.image_mov is not None
                and params.pixel_size_mov is not None
            ):
                # Extract and construct needed values
                visit_ref = params.visit_ref
                experiment_name_ref = params.experiment_type_ref
                image_ref = params.image_ref
                pixel_size_ref = params.pixel_size_ref

                visit_mov = params.visit_mov
                experiment_name_mov = params.experiment_type_mov
                image_mov = params.image_mov
                pixel_size_mov = params.pixel_size_mov
            else:
                self.log.error(
                    "Missing values needed to perform image correlation:\n"
                    f"visit_ref: {params.visit_ref}\n"
                    f"experiment_type_ref: {params.experiment_type_ref}\n"
                    f"image_ref: {params.image_ref}\n"
                    f"pixel_size_ref: {params.pixel_size_ref}\n"
                    f"visit_mov: {params.visit_mov}\n"
                    f"experiment_type_mov: {params.experiment_type_mov}\n"
                    f"image_mov: {params.image_mov}\n"
                    f"pixel_size_mov: {params.pixel_size_mov}\n"
                )
                self._reject_message(header, transport=rw.transport)
                return

        try:
            # Construct the save directory for the outputs
            visit_dir = image_ref.parents[-(image_ref.parts.index(visit_ref) + 1)]
            save_dir = (
                visit_dir / "processed" / "correlation" / visit_mov / image_mov.stem
            )
            save_dir.mkdir(parents=True, exist_ok=True)
            self.log.info(f"Created save directory at {save_dir}")

            # Align images differently depending on which data types are being compared
            match sorted((experiment_name_ref, experiment_name_mov)):
                case ["FIB", "Tomography"] | ["FIB", "Lamella Tomography"]:
                    self.log.info("Aligning FIB atlas to tomography atlas")
                    result = self._handle_fib_tomo_case(
                        image_ref,
                        pixel_size_ref,
                        image_mov,
                        pixel_size_mov,
                        save_dir,
                        params,
                    )
                    if result["transform"] is not None:
                        self.log.info(
                            "Successfully aligned FIB atlas to tomography atlas"
                        )
                    else:
                        self.log.error("Could not align FIB atlas to tomography atlas")
                        self._reject_message(header, transport=rw.transport)
                        return
                case _:
                    self.log.info(
                        "No image alignment algorithm implemented for this case yet"
                    )
        except Exception:
            self.log.error(
                "Error while attempting to perform image alignment", exc_info=True
            )
            self._reject_message(header, transport=rw.transport)
            return

        # Ack message after completion
        rw.transport.ack(header)
        return

    def _handle_fib_tomo_case(
        self,
        image_ref: Path,
        pixel_size_ref: float,
        image_mov: Path,
        pixel_size_mov: float,
        save_dir: Path,
        params: AlignImagesParameters,
    ):
        # Load image files and pixel sizes
        # Load from params, then fall back to table (useful for testing)
        img_ref = cv2.imread(image_ref, flags=cv2.IMREAD_GRAYSCALE)
        img_mov = cv2.imread(image_mov, flags=cv2.IMREAD_GRAYSCALE)

        # Rescale images to same pixel size and crop to the same dimensions
        resized_ref = cv2.resize(
            img_ref,
            dsize=None,
            dst=None,
            fx=pixel_size_ref / params.target_pixel_size,
            fy=pixel_size_ref / params.target_pixel_size,
            interpolation=cv2.INTER_AREA,
        )
        resized_mov = cv2.resize(
            img_mov,
            dsize=None,
            dst=None,
            fx=pixel_size_mov / params.target_pixel_size,
            fy=pixel_size_mov / params.target_pixel_size,
            interpolation=cv2.INTER_AREA,
        )
        height = min(resized_ref.shape[0], resized_mov.shape[0])
        width = min(resized_ref.shape[1], resized_mov.shape[1])
        cropped_ref = crop_image(resized_ref, width=width, height=height)
        cropped_mov = crop_image(resized_mov, width=width, height=height)

        # Perform image alignment
        return align_images_using_neighbors(
            cropped_ref,
            cropped_mov,
            median_blur=params.median_blur,
            gaussian_blur=params.gaussian_blur,
            sobel_kernel=params.sobel_kernel,
            use_hanning=params.use_hanning,
            min_component_area=params.min_component_area,
            threshold_percentile=params.threshold_percentile,
            morph_close_kernel=params.morph_close_kernel,
            morph_open_kernel=params.morph_open_kernel,
            min_feature_area=params.min_feature_area,
            max_feature_area=params.max_feature_area,
            min_solidity=params.min_solidity,
            min_ellipse_fit=params.min_ellipse_fit,
            max_aspect_ratio=params.max_aspect_ratio,
            max_neighbor_distance=params.max_neighbor_distance,
            min_score=params.min_score,
            ransac_threshold=params.ransac_threshold,
            save_images=params.save_images,
            save_tables=params.save_tables,
            save_dir=save_dir,
        )
