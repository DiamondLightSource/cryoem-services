from __future__ import annotations

import logging
from typing import Callable

import murfey.util.processing_db as models
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from cryoemservices.util import ispyb_commands

logger = logging.getLogger("cryoemservices.util.murfey_db_commands")
logger.setLevel(logging.INFO)


def buffer(message: dict, parameters: Callable, session: Session):
    """
    Override of the buffer command,
    which just uses the given value rather than doing a lookup
    """
    command = message.get("buffer_command", {}).get("ispyb_command", "")
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
    for entry in list(message.get("buffer_lookup", [])):
        # Copy value into command variables
        message["buffer_command"][entry] = message["buffer_lookup"][entry]
        del message["buffer_lookup"][entry]

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


def update_processing_status(message: dict, parameters: Callable, session: Session):
    """Do nothing if AutoProcProgram status table does not exist"""
    return {"success": True, "return_value": 0}


def register_processing(message: dict, parameters: Callable, session: Session):
    """Do nothing if AutoProcProgram status table does not exist"""
    return {"success": True, "return_value": 0}
