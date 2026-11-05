from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import murfey.util.db as models
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from cryoemservices.util import ispyb_commands

logger = logging.getLogger("cryoemservices.util.murfey_db_commands")
logger.setLevel(logging.INFO)


def multipart_message(message: dict, parameters: Callable, session: Session):
    """
    Override of the multipart message command,
    which uses commands from this file rather than ispyb commands
    """
    commands = parameters("ispyb_command_list")
    if not commands or not isinstance(commands, list):
        logger.error("Received multipart message containing no command list")
        return False

    current_command = commands[0]
    ispyb_command = current_command.get("ispyb_command")
    command: Callable | None = globals().get(ispyb_command) or getattr(
        ispyb_commands, ispyb_command, None
    )
    if not command:
        logger.error(
            f"Multipart command {current_command} does not have a valid ispyb_command"
        )
        return False
    return ispyb_commands.run_multipart_command(message, parameters, session, command)


# Against ISPyB a ``buffer_store`` UUID is written to a buffer table which maps it
# to the auto generated primary key of the inserted row, so a later
# ``buffer_lookup`` of the same UUID resolves to that key. 

# The murfey database has no buffer table, and this file's ``buffer`` copies
# a ``buffer_lookup`` value straight into the command. 
# The stored row must be keyed on the UUID.


# List of commands that accept their primary key as a parameter.
BUFFER_STORE_PRIMARY_KEYS = {
    "insert_particle_picker": "particle_picker_id",
    "insert_particle_classification_group": "particle_classification_group_id",
    "insert_particle_classification": "particle_classification_id",
}


def buffer(message: dict, parameters: Callable, session: Session):
    """
    Override of the buffer command,
    which just uses the given value rather than doing a lookup
    """
    if isinstance(message.get("buffer_command"), dict):
        command = message.get("buffer_command", {}).get("ispyb_command", "")
    else:
        command = None
    if not command:
        logger.error(f"Invalid buffer call: no buffer command in {message}")
        return False

    # Look up the command in this file or in the ispyb_commands file
    command_function = globals().get(command) or getattr(ispyb_commands, command, None)
    if not command_function:
        logger.error(f"Invalid buffer call: unknown command {command} in {message}")
        return False

    program_id = parameters("program_id")
    if not program_id:
        logger.error("Invalid buffer call: program_id is undefined")
        return False

    # Prepare command: Resolve all references
    if message.get("buffer_lookup") and not isinstance(message["buffer_lookup"], dict):
        logger.error("Invalid buffer call: buffer_lookup is not a dictionary")
        return False
    for entry in list(message.get("buffer_lookup", [])):
        # Copy value into command variables
        message["buffer_command"][entry] = message["buffer_lookup"][entry]
        del message["buffer_lookup"][entry]

    # Store the buffered value by keying the new row on the UUID itself.
    if message.get("buffer_store"):
        primary_key = BUFFER_STORE_PRIMARY_KEYS.get(command)
        if primary_key:
            message["buffer_command"][primary_key] = message["buffer_store"]
        else:
            logger.warning(f"No primary key known for buffer store of {command}")
        del message["buffer_store"]

    # Run the actual command
    result = command_function(
        message=message["buffer_command"],
        parameters=parameters,
        session=session,
    )

    # If the command did not succeed then propagate failure
    if not result or not result.get("success"):
        logger.warning("Buffered command failed")
        return result

    # Finally, propagate result
    result["store_result"] = message.get("store_result")
    return result


def insert_movie(message: dict, parameters: Callable, session: Session):
    """Do nothing as this will already be present in the murfey database"""
    return {"success": True, "return_value": 0}


def insert_cryoem_initial_model(message: dict, parameters: Callable, session: Session):
    """Override of initial model search for if the link table does not exist"""

    def full_parameters(param):
        return ispyb_commands.parameters_with_replacement(param, message, parameters)

    try:
        values_im = models.CryoemInitialModel(
            resolution=full_parameters("resolution"),
            numberOfParticles=full_parameters("number_of_particles"),
            particleClassificationId=full_parameters("particle_classification_id"),
        )
        session.add(values_im)
        session.commit()
        logger.info(
            f"Created CryoEM Initial Model record {values_im.cryoemInitialModelId}"
        )
        return {"success": True, "return_value": values_im.cryoemInitialModelId}
    except SQLAlchemyError as e:
        logger.error(
            f"Inserting CryoEM Initial Model entry caused exception {e}",
            exc_info=True,
        )
        return False


def insert_tilt_image_alignment(message: dict, parameters: Callable, session: Session):
    """Override of tilt image alignment for murfey movie table"""

    def full_parameters(param):
        return ispyb_commands.parameters_with_replacement(param, message, parameters)

    if full_parameters("movie_id"):
        mvid = full_parameters("movie_id")
    else:
        movie_name = Path(full_parameters("path")).stem.replace("_motion_corrected", "")
        logger.info(
            f"Looking for Movie ID. Movie name: {movie_name} DCID: {full_parameters('dcid')}"
        )
        results = (
            session.query(models.Movie)
            .filter(
                models.Movie.data_collection_id == full_parameters("dcid"),
            )
            .all()
        )
        mvid = None
        for result in results:
            if movie_name in result.path:
                logger.info(f"Found Movie ID: {mvid}")
                mvid = result.murfey_id
        if not mvid:
            logger.error(f"No movie ID for {movie_name} in tilt image alignment")
            return False

    try:
        values = models.TiltImageAlignment(
            movieId=mvid,
            tomogramId=full_parameters("tomogram_id"),
            defocusU=full_parameters("defocus_u"),
            defocusV=full_parameters("defocus_v"),
            psdFile=full_parameters("psd_file"),
            resolution=full_parameters("resolution"),
            fitQuality=full_parameters("fit_quality"),
            refinedMagnification=full_parameters("refined_magnification"),
            refinedTiltAngle=full_parameters("refined_tilt_angle"),
            refinedTiltAxis=full_parameters("refined_tilt_axis"),
            residualError=full_parameters("residual_error"),
        )
        session.add(values)
        session.commit()
        logger.info(f"Created tilt image alignment record for {values.tomogramId}")
        return {"success": True, "return_value": values.tomogramId}
    except SQLAlchemyError as e:
        logger.error(
            f"Inserting Tilt Image Alignment entry caused exception {e}",
            exc_info=True,
        )
        return False


def update_processing_status(message: dict, parameters: Callable, session: Session):
    """Do nothing if AutoProcProgram status table does not exist"""
    return {"success": True, "return_value": 0}


def register_processing(message: dict, parameters: Callable, session: Session):
    """Do nothing if AutoProcProgram status table does not exist"""
    return {"success": True, "return_value": 0}
