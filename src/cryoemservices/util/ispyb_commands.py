from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

import ispyb.sqlalchemy as models
import sqlalchemy.exc

from cryoemservices.services import ispyb_buffer

logger = logging.getLogger("cryoemservices.util.ispyb_commands")
logger.setLevel(logging.INFO)


def multipart_message(rw, message, parameters, session):
    """The multipart_message command allows the recipe or client to specify a
    multi-stage operation. With this you can process a list of API calls.
    Each API call may have a return value that can be stored.
    Multipart_message takes care of chaining and checkpointing to make the
    overall call near-ACID compliant."""

    if not rw.environment.get("has_recipe_wrapper", True):
        logger.error("Multipart message call can not be used with simple messages")
        return False

    step = 1
    commands = rw.recipe_step["parameters"].get("ispyb_command_list")
    if isinstance(message, dict) and isinstance(
        message.get("ispyb_command_list"), list
    ):
        commands = message["ispyb_command_list"]
        step = message.get("checkpoint", 0) + 1
    if not commands:
        logger.error("Received multipart message containing no commands")
        return False

    current_command = commands[0]
    command = current_command.get("ispyb_command")
    if not command:
        logger.error(
            "Multipart command %s is not a valid ISPyB command", current_command
        )
        return False
    logger.debug(
        "Processing step %d of multipart message (%s) with %d further steps",
        step,
        command,
        len(commands) - 1,
        extra={"ispyb-message-parts": len(commands)} if step == 1 else {},
    )

    # Create a parameter lookup function specific to this step of the
    # multipart message
    def step_parameters(parameter, replace_variables=True):
        """Slight change in behaviour compared to 'parameters' in a direct call:
        If the value is defined in the command list item then this takes
        precedence. Otherwise we check the original message content. Finally,
        we look in the parameters dictionary of the recipe step for the
        multipart_message command.
        String replacement rules apply as usual."""
        if parameter in current_command:
            base_value = current_command[parameter]
        elif isinstance(message, dict) and parameter in message:
            base_value = message[parameter]
        else:
            base_value = rw.recipe_step["parameters"].get(parameter)
        if (
            not replace_variables
            or not base_value
            or not isinstance(base_value, str)
            or "$" not in base_value
        ):
            return base_value
        for key in sorted(rw.environment, key=len, reverse=True):
            if "${" + str(key) + "}" in base_value:
                base_value = base_value.replace(
                    "${" + str(key) + "}", str(rw.environment[key])
                )
            # Replace longest keys first, as the following replacement is
            # not well-defined when one key is a prefix of another:
            if f"${key}" in base_value:
                base_value = base_value.replace(f"${key}", str(rw.environment[key]))
        return base_value

    # If this step previously checkpointed then override the message passed
    # to the step.
    step_message = current_command
    if isinstance(message, dict):
        step_message = message.get("step_message", step_message)

    # Run the multipart step
    result = command(
        rw=rw, message=step_message, parameters=step_parameters, session=session
    )

    # Store step result if appropriate
    store_result = current_command.get("store_result")
    if store_result and result and "return_value" in result:
        rw.environment[store_result] = result["return_value"]
        logger.debug(
            "Storing result '%s' in environment variable '%s'",
            result["return_value"],
            store_result,
        )

    # If the current step has checkpointed then need to manage this
    if result and result.get("checkpoint"):
        logger.debug("Checkpointing for sub-command %s", command)

        if isinstance(message, dict):
            checkpoint_dictionary = message
        else:
            checkpoint_dictionary = {}
        checkpoint_dictionary["checkpoint"] = step - 1
        checkpoint_dictionary["ispyb_command_list"] = commands
        checkpoint_dictionary["step_message"] = result.get("return_value")
        return {
            "checkpoint": True,
            "return_value": checkpoint_dictionary,
        }

    # If the step did not succeed then propagate failure
    if not result or not result.get("success"):
        logger.debug("Multipart command failed")
        return result

    # Step has completed, so remove from queue
    commands.pop(0)

    # If the multipart command is finished then propagate success
    if not commands:
        logger.debug("and done.")
        return result

    # If there are more steps then checkpoint the current state and re-queue it
    logger.debug("Checkpointing remaining %d steps", len(commands))
    if isinstance(message, dict):
        checkpoint_dictionary = message
    else:
        checkpoint_dictionary = {}
    checkpoint_dictionary["checkpoint"] = step
    checkpoint_dictionary["ispyb_command_list"] = commands
    if "step_message" in checkpoint_dictionary:
        del checkpoint_dictionary["step_message"]
    return {"checkpoint": True, "return_value": checkpoint_dictionary}


def buffer(rw, message, parameters, session):
    """The buffer command supports running buffer lookups before running
    a command, and optionally storing the result in a buffer after running
    the command. It also takes care of checkpointing in case a required
    buffer value is not yet available.

    As an example, if you want to send this message to the ISPyB service:

    {
        "ispyb_command": "insert_thing",
        "parent_id": "$ispyb_thing_parent_id",
        "store_result": "ispyb_thing_id",
        "parameter_a": ...,
        "parameter_b": ...,
    }

    and want to look up the parent_id using the buffer with your unique
    reference UUID1 you could write:

    {
        "ispyb_command": "buffer",
        "program_id": "$ispyb_autoprocprogram_id",
        "buffer_lookup": {
            "parent_id": UUID1,
        },
        "buffer_command": {
            "ispyb_command": "insert_thing",
            "parameter_a": ...,
            "parameter_b": ...,
        },
        "buffer_store": UUID2,
        "store_result": "ispyb_thing_id",
    }

    which would also store the result under buffer reference UUID2.
    """

    if not isinstance(message, dict):
        logger.error(f"Invalid buffer call: {message} is not a dictionary")
        return False

    if not isinstance(message.get("buffer_command"), dict) or not message[
        "buffer_command"
    ].get("ispyb_command"):
        logger.error(f"Invalid buffer call: no buffer command in {message}")
        return False

    command_function = message["buffer_command"]["ispyb_command"]
    if not command_function:
        logger.error(f"Invalid buffer call: unknown command in {message}")
        return False

    # Prepare command: Resolve all references
    program_id = parameters("program_id")
    if message.get("buffer_lookup"):
        if not isinstance(message["buffer_lookup"], dict):
            logger.error(
                "Invalid buffer call: buffer_lookup dictionary is not a dictionary"
            )
            return False
        if not program_id:
            logger.error("Invalid buffer call: program_id is undefined")
            return False
        for entry in list(message["buffer_lookup"]):
            buffer_result = ispyb_buffer.load(
                session=session,
                program=program_id,
                uuid=message["buffer_lookup"][entry],
            )
            if buffer_result.success:
                # resolve value and continue
                message["buffer_command"][entry] = buffer_result.value
                del message["buffer_lookup"][entry]
                logger.debug(
                    f"Successfully resolved buffer reference {entry!r} to {buffer_result.value!r}"
                )
                continue

            logger.warning(
                "Buffer call could not be resolved: "
                f"entry {entry} not found for program {program_id}. "
                "Will wait 20 seconds before trying again"
            )
            time.sleep(20)
            return False

    # Run the actual command
    result = command_function(
        rw=rw,
        message=message["buffer_command"],
        session=session,
        parameters=parameters,
    )

    # Store result if appropriate
    store_result = message.get("store_result")
    if store_result and result and "return_value" in result:
        rw.environment[store_result] = result["return_value"]
        logger.debug(
            "Storing result '%s' in environment variable '%s'",
            result["return_value"],
            store_result,
        )

    # If the actual command has checkpointed then need to manage this
    if result and result.get("checkpoint"):
        logger.debug("Checkpointing for buffered function")
        message["buffer_command"] = result["return_value"]
        return {
            "checkpoint": True,
            "return_value": message,
        }

    # If the command did not succeed then propagate failure
    if not result or not result.get("success"):
        logger.warning("Buffered command failed")
        # to become debug level eventually, the actual function will do the warning
        return result

    # Optionally store a reference to the result in the buffer table
    if message.get("buffer_store"):
        logger.debug("Storing buffer result for UUID %r", message["buffer_store"])
        ispyb_buffer.store(
            session=session,
            program=program_id,
            uuid=message["buffer_store"],
            reference=result["return_value"],
        )

    # Finally, propagate result
    return result


def _get_movie_id(
    full_path,
    data_collection_id,
    db_session,
):
    logger.info(
        f"Looking for Movie ID. Movie name: {full_path} DCID: {data_collection_id}"
    )
    movie_name = str(Path(full_path).stem).replace("_motion_corrected", "")
    mv_query = db_session.query(models.Movie).filter(
        models.Movie.dataCollectionId == data_collection_id,
    )
    results = mv_query.all()
    correct_result = None
    if results:
        for result in results:
            if movie_name in result.movieFullPath:
                correct_result = result
    if correct_result:
        mvid = correct_result.movieId
        logger.info(f"Found Movie ID: {mvid}")
        return mvid
    else:
        logger.error(f"Unable to find movie ID for {movie_name}")
        return None


def insert_movie(rw, message, parameters, session):
    logger.info("Inserting Movie parameters.")

    try:
        if parameters.get("timestamp"):
            values = models.Movie(
                dataCollectionId=parameters("dcid"),
                movieNumber=parameters("movie_number"),
                movieFullPath=parameters("movie_path"),
                createdTimeStamp=datetime.fromtimestamp(
                    parameters("timestamp")
                ).strftime("%Y-%m-%d %H:%M:%S"),
            )
        else:
            values = models.Movie(
                dataCollectionId=parameters("dcid"),
                movieNumber=parameters("movie_number"),
                movieFullPath=parameters("movie_path"),
            )
        session.add(values)
        session.commit()
        logger.info(f"Created Movie record {values.movieId}")
        return {"success": True, "return_value": values.movieId}
    except sqlalchemy.exc.SQLAlchemyError as e:
        logger.error(
            "Inserting movie entry caused exception '%s'.",
            e,
            exc_info=True,
        )
        return False


def insert_motion_correction(rw, message, parameters, session):
    if message is None:
        message = {}
    logger.info("Inserting Motion Correction parameters.")

    def full_parameters(param):
        return message.get(param) or parameters(param)

    try:
        movie_id = None
        if full_parameters("movie_id") is None:
            movie_values = insert_movie(
                rw=rw,
                message=message,
                parameters={
                    "dcid": full_parameters("dcid"),
                    "movie_number": full_parameters("image_number"),
                    "movie_path": full_parameters("micrograph_full_path"),
                    "timestamp": full_parameters("created_time_stamp"),
                },
                session=session,
            )
            movie_id = movie_values["return_value"]

        values = models.MotionCorrection(
            dataCollectionId=full_parameters("dcid"),
            movieId=full_parameters("movie_id") or movie_id,
            autoProcProgramId=full_parameters("program_id"),
            imageNumber=full_parameters("image_number"),
            firstFrame=full_parameters("first_frame"),
            lastFrame=full_parameters("last_frame"),
            dosePerFrame=full_parameters("dose_per_frame"),
            doseWeight=full_parameters("dose_weight"),
            totalMotion=full_parameters("total_motion"),
            averageMotionPerFrame=full_parameters("average_motion_per_frame"),
            driftPlotFullPath=full_parameters("drift_plot_full_path"),
            micrographFullPath=full_parameters("micrograph_full_path"),
            micrographSnapshotFullPath=full_parameters("micrograph_snapshot_full_path"),
            patchesUsedX=full_parameters("patches_used_x"),
            patchesUsedY=full_parameters("patches_used_y"),
            fftFullPath=full_parameters("fft_full_path"),
            fftCorrectedFullPath=full_parameters("fft_corrected_full_path"),
            comments=full_parameters("comments"),
        )
        session.add(values)
        session.commit()
        logger.info(f"Created MotionCorrection record {values.motionCorrectionId}")
        return {"success": True, "return_value": values.motionCorrectionId}
    except sqlalchemy.exc.SQLAlchemyError as e:
        logger.error(
            "Inserting motion correction entry caused exception '%s'.",
            e,
            exc_info=True,
        )
        return False


def insert_relative_ice_thickness(rw, message, parameters, session):
    if message is None:
        message = {}
    dcid = parameters("dcid")
    logger.info(f"Inserting Relative Ice Thickness parameters. DCID: {dcid}")

    def full_parameters(param):
        return message.get(param) or parameters(param)

    try:
        values = models.RelativeIceThickness(
            motionCorrectionId=full_parameters("motion_correction_id"),
            autoProcProgramId=full_parameters("program_id"),
            minimum=full_parameters("minimum"),
            q1=full_parameters("q1"),
            median=full_parameters("median"),
            q3=full_parameters("q3"),
            maximum=full_parameters("maximum"),
        )
        session.add(values)
        session.commit()
        return {"success": True, "return_value": values.relativeIceThicknessId}
    except sqlalchemy.exc.SQLAlchemyError as e:
        logger.error(
            "Inserting relative ice thickness entry caused exception '%s'.",
            e,
            exc_info=True,
        )
        return False


def insert_ctf(rw, message, parameters, session):
    if message is None:
        message = {}
    dcid = parameters("dcid")
    logger.info(f"Inserting CTF parameters. DCID: {dcid}")

    def full_parameters(param):
        return message.get(param) or parameters(param)

    try:
        values = models.CTF(
            ctfId=full_parameters("ctf_id"),
            motionCorrectionId=full_parameters("motion_correction_id"),
            autoProcProgramId=full_parameters("program_id"),
            boxSizeX=full_parameters("box_size_x"),
            boxSizeY=full_parameters("box_size_y"),
            minResolution=full_parameters("min_resolution"),
            maxResolution=full_parameters("max_resolution"),
            minDefocus=full_parameters("min_defocus"),
            maxDefocus=full_parameters("max_defocus"),
            defocusStepSize=full_parameters("defocus_step_size"),
            astigmatism=full_parameters("astigmatism"),
            astigmatismAngle=full_parameters("astigmatism_angle"),
            estimatedResolution=full_parameters("estimated_resolution"),
            estimatedDefocus=full_parameters("estimated_defocus"),
            amplitudeContrast=full_parameters("amplitude_contrast"),
            ccValue=full_parameters("cc_value"),
            fftTheoreticalFullPath=full_parameters("fft_theoretical_full_path"),
            comments=full_parameters("comments"),
        )
        session.add(values)
        session.commit()
        logger.info(f"Created CTF record {values.ctfId} for DCID {dcid}")
        return {"success": True, "return_value": values.ctfId}
    except sqlalchemy.exc.SQLAlchemyError as e:
        logger.error(
            "Inserting CTF entry caused exception '%s'.",
            e,
            exc_info=True,
        )
        return False


def insert_particle_picker(rw, message, parameters, session):
    if message is None:
        message = {}
    dcid = parameters("dcid")
    logger.info(f"Inserting Particle Picker parameters. DCID: {dcid}")

    def full_parameters(param):
        return message.get(param) or parameters(param)

    try:
        values = models.ParticlePicker(
            particlePickerId=full_parameters("particle_picker_id"),
            programId=full_parameters("program_id"),
            firstMotionCorrectionId=full_parameters("motion_correction_id"),
            particlePickingTemplate=full_parameters("particle_picking_template"),
            particleDiameter=full_parameters("particle_diameter"),
            numberOfParticles=full_parameters("number_of_particles"),
            summaryImageFullPath=full_parameters("summary_image_full_path"),
        )
        session.add(values)
        session.commit()
        logger.info(
            f"Created ParticlePicker record {values.particlePickerId} "
            f"for DCID {dcid}"
        )
        return {"success": True, "return_value": values.particlePickerId}
    except sqlalchemy.exc.SQLAlchemyError as e:
        logger.error(
            "Inserting Particle Picker entry caused exception '%s'.",
            e,
            exc_info=True,
        )
        return False


def insert_particle_classification(rw, message, parameters, session):
    if message is None:
        message = {}
    dcid = parameters("dcid")

    def full_parameters(param):
        return message.get(param) or parameters(param)

    try:
        values = models.ParticleClassification(
            particleClassificationId=full_parameters("particle_classification_id"),
            particleClassificationGroupId=full_parameters(
                "particle_classification_group_id"
            ),
            classNumber=full_parameters("class_number"),
            classImageFullPath=full_parameters("class_image_full_path"),
            particlesPerClass=full_parameters("particles_per_class"),
            rotationAccuracy=full_parameters("rotation_accuracy"),
            translationAccuracy=full_parameters("translation_accuracy"),
            estimatedResolution=full_parameters("estimated_resolution"),
            overallFourierCompleteness=full_parameters("overall_fourier_completeness"),
            classDistribution=full_parameters("class_distribution"),
            selected=full_parameters("selected"),
            bFactorFitIntercept=full_parameters("bfactor_fit_intercept"),
            bFactorFitLinear=full_parameters("bfactor_fit_linear"),
            bFactorFitQuadratic=full_parameters("bfactor_fit_quadratic"),
        )
        particle_classification = (
            session.query(models.ParticleClassification)
            .filter(
                models.ParticleClassification.particleClassificationId
                == values.particleClassificationId,
            )
            .first()
        )
        if particle_classification:
            session.query(models.ParticleClassification).filter(
                models.ParticleClassification.particleClassificationId
                == values.particleClassificationId,
            ).update(
                {
                    k: v
                    for k, v in values.__dict__.items()
                    if k not in ["_sa_instance_state", "particleClassificationId"]
                    and v is not None
                }
            )
        else:
            session.add(values)
        session.commit()
        logger.info(
            "Created ParticleClassification record "
            f"{values.particleClassificationId} for DCID {dcid}"
        )
        return {"success": True, "return_value": values.particleClassificationId}
    except sqlalchemy.exc.SQLAlchemyError as e:
        logger.error(
            "Inserting particle classification entry caused exception '%s'.",
            e,
            exc_info=True,
        )
        return False


def insert_particle_classification_group(rw, message, parameters, session):
    if message is None:
        message = {}
    dcid = parameters("dcid")

    def full_parameters(param):
        return message.get(param) or parameters(param)

    logger.info(f"Inserting particle classification parameters. DCID: {dcid}")
    try:
        values = models.ParticleClassificationGroup(
            particleClassificationGroupId=full_parameters(
                "particle_classification_group_id"
            ),
            particlePickerId=full_parameters("particle_picker_id"),
            programId=full_parameters("program_id"),
            type=full_parameters("type"),
            batchNumber=full_parameters("batch_number"),
            numberOfParticlesPerBatch=full_parameters("number_of_particles_per_batch"),
            numberOfClassesPerBatch=full_parameters("number_of_classes_per_batch"),
            symmetry=full_parameters("symmetry"),
        )
        particle_classification_group = (
            session.query(models.ParticleClassificationGroup)
            .filter(
                models.ParticleClassificationGroup.particleClassificationGroupId
                == values.particleClassificationGroupId,
            )
            .first()
        )
        if particle_classification_group:
            session.query(models.ParticleClassificationGroup).filter(
                models.ParticleClassificationGroup.particleClassificationGroupId
                == values.particleClassificationGroupId,
            ).update(
                {
                    k: v
                    for k, v in values.__dict__.items()
                    if k not in ["_sa_instance_state", "particleClassificationGroupId"]
                    and v is not None
                }
            )
        else:
            session.add(values)
        session.commit()
        logger.info(
            "Created particle classification group record "
            f"{values.particleClassificationGroupId} for DCID {dcid}"
        )
        return {
            "success": True,
            "return_value": values.particleClassificationGroupId,
        }
    except sqlalchemy.exc.SQLAlchemyError as e:
        logger.error(
            "Inserting particle classification group entry caused exception '%s'.",
            e,
            exc_info=True,
        )
        return False


def insert_cryoem_initial_model(rw, message, parameters, session):
    if message is None:
        message = {}
    dcid = parameters("dcid")

    def full_parameters(param):
        return message.get(param) or parameters(param)

    logger.info(f"Inserting CryoEM Initial Model parameters. DCID: {dcid}")
    try:
        if not full_parameters("cryoem_initial_model_id"):
            values_im = models.CryoemInitialModel(
                resolution=full_parameters("resolution"),
                numberOfParticles=full_parameters("number_of_particles"),
            )
            session.add(values_im)
            session.commit()
            logger.info(
                "Created CryoEM Initial Model record "
                f"{values_im.cryoemInitialModelId} for DCID {dcid}"
            )
            initial_model_id = values_im.cryoemInitialModelId
        else:
            initial_model_id = full_parameters("cryoem_initial_model_id")
        session.execute(
            models.t_ParticleClassification_has_CryoemInitialModel.insert().values(
                cryoemInitialModelId=initial_model_id,
                particleClassificationId=full_parameters("particle_classification_id"),
            )
        )
        session.commit()
        return {"success": True, "return_value": initial_model_id}
    except sqlalchemy.exc.SQLAlchemyError as e:
        logger.error(
            "Inserting CryoEM Initial Model entry caused exception '%s'.",
            e,
            exc_info=True,
        )
        return False


def insert_bfactor_fit(rw, message, parameters, session):
    if message is None:
        message = {}
    dcid = parameters("dcid")

    def full_parameters(param):
        return message.get(param) or parameters(param)

    logger.info(f"Inserting bfactor calculation parameters. DCID: {dcid}")
    try:
        values = models.BFactorFit(
            bFactorFitId=full_parameters("bfactor_id"),
            particleClassificationId=full_parameters("particle_classification_id"),
            resolution=full_parameters("resolution"),
            numberOfParticles=full_parameters("number_of_particles"),
            particleBatchSize=full_parameters("particle_batch_size"),
        )
        bfactor = (
            session.query(models.BFactorFit)
            .filter(models.BFactorFit.bFactorFitId == values.bFactorFitId)
            .first()
        )
        if bfactor:
            session.query(models.BFactorFit).filter(
                models.BFactorFit.bFactorFitId == values.bFactorFitId,
            ).update(
                {
                    k: v
                    for k, v in values.__dict__.items()
                    if k not in ["_sa_instance_state", "bFactorFitId"] and v is not None
                }
            )
        else:
            session.add(values)
        session.commit()
        logger.info(f"Created bfactor record {values.bFactorFitId} for DCID {dcid}")
        return {
            "success": True,
            "return_value": values.bFactorFitId,
        }
    except sqlalchemy.exc.SQLAlchemyError as e:
        logger.error(
            "Inserting bfactor entry caused exception '%s'.",
            e,
            exc_info=True,
        )
        return False


def insert_tomogram(rw, message, parameters, session):
    if message is None:
        message = {}
    dcid = parameters("dcid")
    logger.info(f"Inserting Tomogram parameters. DCID: {dcid}")
    if not message:
        message = {}

    def full_parameters(param):
        return message.get(param) or parameters(param)

    try:
        values = models.Tomogram(
            tomogramId=full_parameters("tomogram_id"),
            dataCollectionId=full_parameters("dcid"),
            autoProcProgramId=full_parameters("program_id"),
            volumeFile=full_parameters("volume_file"),
            stackFile=full_parameters("stack_file"),
            sizeX=full_parameters("size_x"),
            sizeY=full_parameters("size_y"),
            sizeZ=full_parameters("size_z"),
            pixelSpacing=full_parameters("pixel_spacing"),
            residualErrorMean=full_parameters("residual_error_mean"),
            residualErrorSD=full_parameters("residual_error_sd"),
            xAxisCorrection=full_parameters("x_axis_correction"),
            tiltAngleOffset=full_parameters("tilt_angle_offset"),
            zShift=full_parameters("z_shift"),
            fileDirectory=full_parameters("file_directory"),
            centralSliceImage=full_parameters("central_slice_image"),
            tomogramMovie=full_parameters("tomogram_movie"),
            xyShiftPlot=full_parameters("xy_shift_plot"),
            projXY=full_parameters("proj_xy"),
            projXZ=full_parameters("proj_xz"),
            globalAlignmentQuality=full_parameters("alignment_quality"),
        )
        tomogram_row = (
            session.query(models.Tomogram)
            .filter(
                models.Tomogram.tomogramId == values.tomogramId,
            )
            .first()
        )
        if tomogram_row:
            session.query(models.Tomogram).filter(
                models.Tomogram.tomogramId == values.tomogramId,
            ).update(
                {
                    k: v
                    for k, v in values.__dict__.items()
                    if k not in ["_sa_instance_state", "tomogramId"] and v is not None
                }
            )
        else:
            session.add(values)
        session.commit()
        return {"success": True, "return_value": values.tomogramId}
    except sqlalchemy.exc.SQLAlchemyError as e:
        logger.error(
            "Inserting Tomogram entry caused exception '%s'.",
            e,
            exc_info=True,
        )
        return False


def insert_processed_tomogram(rw, message, parameters, session):
    if message is None:
        message = {}
    dcid = parameters("dcid")
    logger.info(f"Inserting Processed Tomogram parameters. DCID: {dcid}")

    def full_parameters(param):
        return message.get(param) or parameters(param)

    try:
        values = models.ProcessedTomogram(
            tomogramId=full_parameters("tomogram_id"),
            filePath=full_parameters("file_path"),
            processingType=full_parameters("processing_type"),
        )
        session.add(values)
        session.commit()
        return {"success": True, "return_value": values.processedTomogramId}
    except sqlalchemy.exc.SQLAlchemyError as e:
        logger.error(
            "Inserting Processed Tomogram entry caused exception '%s'.",
            e,
            exc_info=True,
        )
        return False


def insert_tilt_image_alignment(rw, message, parameters, session):
    if message is None:
        message = {}
    dcid = parameters("dcid")
    logger.info(f"Inserting Tilt Image Alignment parameters. DCID: {dcid}")

    def full_parameters(param):
        return message.get(param) or parameters(param)

    if full_parameters("movie_id"):
        mvid = full_parameters("movie_id")
    else:
        mvid = _get_movie_id(full_parameters("path"), dcid, session)

    if not mvid:
        logger.error("No movie ID for tilt image alignment")
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
        return {"success": True, "return_value": values.tomogramId}
    except sqlalchemy.exc.SQLAlchemyError as e:
        logger.error(
            "Inserting Tilt Image Alignment entry caused exception '%s'.",
            e,
            exc_info=True,
        )
        return False


def update_processing_status(rw, message, parameters, session):
    if message is None:
        message = {}

    def full_parameters(param):
        return message.get(param) or parameters(param)

    ppid = full_parameters("program_id")
    message = full_parameters("message")
    try:
        values = models.AutoProcProgram(
            autoProcProgramId=ppid,
            processingStatus={"success": 1, "failure": 0}.get(
                full_parameters("status")
            ),
            processingMessage=message,
            processingStartTime=full_parameters("start_time"),
            processingEndTime=full_parameters("update_time"),
        )
        # This is an update call, want it to throw an error if the row isn't present
        session.query(models.AutoProcProgram).filter(
            models.AutoProcProgram.autoProcProgramId == values.autoProcProgramId,
        ).update(
            {
                k: v
                for k, v in values.__dict__.items()
                if k not in ["_sa_instance_state", "autoProcProgramId"]
                and v is not None
            }
        )
        session.commit()
        logger.info(
            f"Updating program {ppid} with status {message}",
        )
        return {"success": True, "return_value": values.autoProcProgramId}
    except sqlalchemy.exc.SQLAlchemyError as e:
        logger.error(
            f"Updating program {ppid} status: {message} caused exception {e}.",
            exc_info=True,
        )
        return False


# These are needed for the old relion-zocalo wrapper
def do_add_program_attachment(rw, message, parameters, session):
    file_name = parameters.get("file_name")
    file_path = parameters.get("file_path")
    logger.error(
        f"Adding program attachments is no longer supported. "
        f"Skipping file {file_name} in {file_path}."
    )
    return {"success": True, "return_value": 0}


def do_register_processing(rw, message, parameters, session):
    program = parameters("program")
    cmdline = parameters("cmdline")
    environment = parameters("environment") or ""
    if isinstance(environment, dict):
        environment = ", ".join(f"{key}={value}" for key, value in environment.items())
    environment = environment[: min(255, len(environment))]
    rpid = parameters("rpid")
    if rpid and not rpid.isdigit():
        logger.error(f"Invalid processing id {rpid}")
        return False
    try:

        values = models.AutoProcProgram(
            processingJobId=rpid,
            processingPrograms=program,
            processingCommandLine=cmdline,
            processingEnvironment=environment,
        )

        session.add(values)
        session.commit()
        logger.info(
            f"Registered new program {program} for processing id {rpid} "
            f"with command line {cmdline} and environment {environment}."
        )
        return {"success": True, "return_value": values.autoProcProgramId}
    except sqlalchemy.exc.SQLAlchemyError as e:
        logger.error(
            f"Registering new program {program} for processing id {rpid} "
            f"with command line {cmdline} and environment {environment} "
            f"caused exception {e}.",
            exc_info=True,
        )
        return False
