from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Tuple

import ispyb.sqlalchemy as models
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from cryoemservices.util.config import ServiceConfig

logger = logging.getLogger("cryoemservices.services.process_recipe_tools")


def get_processing_info(processing_id: int, session: Session) -> dict:
    """Find the ispyb recipe name, dcid, and any inserted processing parameters"""

    processing_job = (
        session.execute(
            select(models.ProcessingJob).where(
                models.ProcessingJob.processingJobId == processing_id
            )
        )
        .scalars()
        .first()
    )

    if not processing_job:
        logger.error(f"Reprocessing ID {processing_id} not found")
        return {}

    processing_info = {
        "recipe": processing_job.recipe,
        "ispyb_dcid": processing_job.dataCollectionId,
    }

    job_parameters = (
        session.execute(
            select(models.ProcessingJobParameter).where(
                models.ProcessingJobParameter.processingJobId == processing_id
            )
        )
        .scalars()
        .all()
    )

    processing_info["ispyb_reprocessing_parameters"] = {
        p.parameterKey: p.parameterValue for p in job_parameters
    }
    return processing_info


def get_image_directory(dcid: int, session: Session) -> str:
    """Find a data collection in ispyb"""
    dc_query = (
        session.execute(
            select(models.DataCollection).where(
                models.DataCollection.dataCollectionId == dcid
            )
        )
        .scalars()
        .first()
    )
    if dc_query is None or not dc_query.imageDirectory:
        logger.error(f"Data collection {dcid} not found or no image directory present")
        return ""
    return dc_query.imageDirectory


def get_visit_directory_from_image_directory(data_directory: Path) -> Path:
    """Return the visit directory. Assumes a structure similar to
    /${facility}/${beamline}/data/${year}/${visit}/..."""
    return Path("/".join(str(data_directory).split("/")[:6]))


def get_working_directory(data_directory: Path, recipe_id: str) -> Path:
    """Make a directory name for storing the run information"""
    visit = get_visit_directory_from_image_directory(data_directory)
    return visit / "tmp" / "wrapper" / data_directory.relative_to(visit) / recipe_id


def get_processed_directory(data_directory: Path, recipe_id: str) -> Path:
    """Make a directory name for the processed output data"""
    visit = get_visit_directory_from_image_directory(data_directory)
    return visit / "processed" / data_directory.relative_to(visit) / recipe_id


def ispyb_filter(
    message: dict,
    parameters: dict,
    config: ServiceConfig,
) -> Tuple[dict, dict]:
    """Filter recipes where a process has already been inserted into ispyb"""
    ispyb_process = parameters.get("ispyb_process")
    if not ispyb_process:
        # Pass all messages where there is not an existing process
        return message, parameters

    ispyb_sessionmaker = sessionmaker(
        bind=create_engine(
            models.url(credentials=config.ispyb_credentials),
            connect_args={"use_pure": True},
        )
    )
    with ispyb_sessionmaker() as session:
        processing_info = get_processing_info(ispyb_process, session)
        if not processing_info:
            raise ValueError(f"No ispyb entry found for ispyb_process={ispyb_process}")

        parameters.update(processing_info)
        dc_id = parameters["ispyb_dcid"]

        image_directory = get_image_directory(dc_id, session)
        if not image_directory:
            raise ValueError(f"No ispyb entry found for dcid={dc_id}")

    recipe_uuid = parameters.get("guid") or str(uuid.uuid4())

    parameters["ispyb_beamline"] = "microscope"
    parameters["ispyb_image_directory"] = image_directory
    parameters["ispyb_visit_directory"] = str(
        get_visit_directory_from_image_directory(Path(image_directory))
    )
    parameters["ispyb_working_directory"] = str(
        get_working_directory(Path(image_directory), recipe_uuid)
    )
    parameters["ispyb_results_directory"] = str(
        get_processed_directory(Path(image_directory), recipe_uuid)
    )

    # Prefix recipe name coming from ispyb with 'ispyb-'
    message["recipes"] = ["ispyb-" + parameters["recipe"]]

    return message, parameters
