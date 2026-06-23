from pathlib import Path
from typing import Any, cast

import ispyb.sqlalchemy
import sqlalchemy.orm
from ispyb.sqlalchemy import _auto_db_schema as ISPyBDB
from pydantic import BaseModel, ValidationError
from sqlalchemy import select
from workflows.recipe import RecipeWrapper, wrap_subscribe

from cryoemservices.services.common_service import CommonService
from cryoemservices.util.config import config_from_file
from cryoemservices.util.models import MockRW


class AlignImagesParameters(BaseModel):
    id_ref: int  # ISPyB Atlas atlasId
    image_ref: Path
    pixel_size_ref: float
    id_mov: int  # ISPyB Atlas atlasId
    image_mov: Path
    pixel_size_mov: float


def _get_atlas_dcg_experiment_type(session: sqlalchemy.orm.Session, atlas_id: int):
    """
    Runs an ISPyB query to get the Atlas and its corresponding DataCollectionGroup
    and ExperimentType rows.
    """

    statement = (
        select(ISPyBDB.Atlas, ISPyBDB.DataCollectionGroup, ISPyBDB.ExperimentType)
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
        .where(ISPyBDB.Atlas.atlasId == atlas_id)
    )
    result = session.execute(statement).one()  # Will error if no match is found
    return (
        cast(ISPyBDB.Atlas, result.Atlas),
        cast(ISPyBDB.DataCollectionGroup, result.DataCollectionGroup),
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
                atlas_ref, dcg_ref, experiment_type_ref = (
                    _get_atlas_dcg_experiment_type(
                        session=session, atlas_id=params.id_ref
                    )
                )
                atlas_mov, dcg_mov, experiment_type_mov = (
                    _get_atlas_dcg_experiment_type(
                        session=session, atlas_id=params.id_mov
                    )
                )
        except Exception:
            self.log.error(
                "Uncaught exception {e!r} while querying ISPyB, "
                "quarantining message and shutting down instance.",
                exc_info=True,
            )
            self._reject_message(header, transport=rw.transport)
            return

        # Align images differently depending on which data types are being compared
        match (experiment_type_ref.name, experiment_type_mov.name):
            case ("Tomography", "FIB"):
                self.log.info("Aligning FIB atlas to tomography one")
            case _:
                self.log.info(
                    "No image alignment algorithm implemented for this case yet"
                )

        # Ack message after completion
        rw.transport.ack(header)
        return
