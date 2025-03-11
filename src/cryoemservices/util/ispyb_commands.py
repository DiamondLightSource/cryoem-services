from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Callable

import ispyb.sqlalchemy as models
import sqlalchemy.exc
import sqlalchemy.orm

from cryoemservices.util import ispyb_buffer

logger = logging.getLogger("cryoemservices.util.ispyb_commands")
logger.setLevel(logging.INFO)


def parameters_with_replacement(param: str, message: dict, all_parameters: Callable):
    """
    Create a parameter lookup function specific to this call.
    Slight change in behaviour compared to 'parameters' in a direct call:
    If the value is defined in the command list item then this takes
    precedence.
    """
    if message.get(param) and "$" not in str(message[param]):
        # Precedence for command list items
        value_to_return = message[param]
    elif message.get(param):
        # Run lookup on dollar parameters
        value_to_return = all_parameters(message[param])
    else:
        # Lookup anything else
        value_to_return = all_parameters(param)
    if value_to_return == "None":
        value_to_return = None
    return value_to_return


def multipart_message(
    message: dict, parameters: Callable, session: sqlalchemy.orm.Session
):
    """
    The multipart_message command allows the recipe or client to specify a
    list of API calls to run.
    Each API call may have a return value that can be stored and passed on.
    """
    commands = parameters("ispyb_command_list")
    step = message.get("checkpoint", 0) + 1
    if not commands or not isinstance(commands, list):
        logger.error("Received multipart message containing no command list")
        return False

    current_command = commands[0]
    command = globals().get(current_command.get("ispyb_command"))
    if not command:
        logger.error(
            f"Multipart command {current_command} does not have a valid ispyb_command"
        )
        return False
    logger.info(
        f"Processing step {step} of multipart message ({current_command}) "
        f"with {len(commands)-1} further steps",
    )

    # Create a parameter lookup function specific to this step
    def step_parameters(parameter):
        return parameters_with_replacement(parameter, current_command, parameters)

    # If this step previously checkpointed then override the message passed
    # to the step.
    step_message = message.get("step_message", current_command)

    # Run the multipart step
    result = command(message=step_message, parameters=step_parameters, session=session)

    # If the step did not succeed then propagate failure
    if not result or not result.get("success"):
        logger.info("Multipart command failed")
        return result

    # Step has completed, so remove from queue
    commands.pop(0)

    # If the multipart command is finished then propagate success
    if not commands:
        logger.info("Multipart message done")
        result["store_result"] = current_command.get("store_result")
        return result

    # If there are more steps then checkpoint the current state and re-queue it
    logger.info(f"Checkpointing remaining {len(commands)} steps")
    checkpoint_dictionary = message
    checkpoint_dictionary["checkpoint"] = step
    checkpoint_dictionary["ispyb_command_list"] = commands
    if "step_message" in checkpoint_dictionary:
        del checkpoint_dictionary["step_message"]
    return {
        "checkpoint": True,
        "checkpoint_dict": checkpoint_dictionary,
        "store_result": current_command.get("store_result"),
        "return_value": result.get("return_value"),
    }


def buffer(message: dict, parameters: Callable, session: sqlalchemy.orm.Session):
    """
    The buffer command supports running buffer lookups before running
    a command, and storing the result in a buffer after running the command.
    """
    if not isinstance(message.get("buffer_command"), dict) or not message[
        "buffer_command"
    ].get("ispyb_command"):
        logger.error(f"Invalid buffer call: no buffer command in {message}")
        return False

    command_function = globals().get(message["buffer_command"]["ispyb_command"])
    if not command_function:
        logger.error(f"Invalid buffer call: unknown command in {message}")
        return False

    program_id = parameters("program_id")
    if not program_id:
        logger.error("Invalid buffer call: program_id is undefined")
        return False

    # Prepare command: Resolve all references
    if message.get("buffer_lookup"):
        if not isinstance(message["buffer_lookup"], dict):
            logger.error("Invalid buffer call: buffer_lookup is not a dictionary")
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
                logger.info(f"Buffer entry {entry!r} found: {buffer_result.value!r}")
                continue

            logger.warning(f"Buffer entry {entry} not found for program {program_id}.")
            return False

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

    # Optionally store a reference to the result in the buffer table
    if message.get("buffer_store"):
        logger.info("Storing buffer result for UUID %r", message["buffer_store"])
        ispyb_buffer.store(
            session=session,
            program=program_id,
            uuid=message["buffer_store"],
            reference=result["return_value"],
        )

    # Finally, propagate result
    result["store_result"] = message.get("store_result")
    return result


def _get_movie_id(
    full_path,
    data_collection_id,
    db_session,
):
    logger.info(
        f"Looking for Movie ID. Movie name: {full_path} DCID: {data_collection_id}"
    )
    movie_name = Path(full_path).stem.replace("_motion_corrected", "")
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


def insert_movie(message: dict, parameters: Callable, session: sqlalchemy.orm.Session):
    try:
        foil_hole_id = (
            parameters("foil_hole_id") if parameters("foil_hole_id") != "None" else None
        )
        if parameters("timestamp"):
            values = models.Movie(
                dataCollectionId=parameters("dcid"),
                foilHoleId=foil_hole_id,
                movieNumber=parameters("movie_number"),
                movieFullPath=parameters("movie_path"),
                createdTimeStamp=datetime.fromtimestamp(
                    parameters("timestamp")
                ).strftime("%Y-%m-%d %H:%M:%S"),
            )
        else:
            values = models.Movie(
                dataCollectionId=parameters("dcid"),
                foilHoleId=foil_hole_id,
                movieNumber=parameters("movie_number"),
                movieFullPath=parameters("movie_path"),
            )
        session.add(values)
        session.commit()
        logger.info(f"Created Movie record {values.movieId}")
        return {"success": True, "return_value": values.movieId}
    except sqlalchemy.exc.SQLAlchemyError as e:
        logger.error(
            f"Inserting movie entry caused exception {e}",
            exc_info=True,
        )
        return False


def insert_motion_correction(
    message: dict, parameters: Callable, session: sqlalchemy.orm.Session
):
    def full_parameters(param):
        return parameters_with_replacement(param, message, parameters)

    try:
        movie_id = None
        if full_parameters("movie_id") is None:

            def movie_parameters(p):
                mv_param = {
                    "dcid": full_parameters("dcid"),
                    "foil_hole_id": full_parameters("foil_hole_id"),
                    "movie_number": full_parameters("image_number"),
                    "movie_path": full_parameters("micrograph_full_path"),
                    "timestamp": full_parameters("created_time_stamp"),
                }
                return mv_param[p]

            movie_values = insert_movie(
                message=message,
                parameters=movie_parameters,
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
            f"Inserting motion correction entry caused exception {e}",
            exc_info=True,
        )
        return False


def insert_relative_ice_thickness(
    message: dict, parameters: Callable, session: sqlalchemy.orm.Session
):
    def full_parameters(param):
        return parameters_with_replacement(param, message, parameters)

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
        logger.info(f"Created Ice Thickness record {values.relativeIceThicknessId}")
        return {"success": True, "return_value": values.relativeIceThicknessId}
    except sqlalchemy.exc.SQLAlchemyError as e:
        logger.error(
            f"Inserting relative ice thickness entry caused exception {e}",
            exc_info=True,
        )
        return False


def insert_ctf(message: dict, parameters: Callable, session: sqlalchemy.orm.Session):
    def full_parameters(param):
        return parameters_with_replacement(param, message, parameters)

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
        logger.info(f"Created CTF record {values.ctfId}")
        return {"success": True, "return_value": values.ctfId}
    except sqlalchemy.exc.SQLAlchemyError as e:
        logger.error(
            f"Inserting CTF entry caused exception {e}",
            exc_info=True,
        )
        return False


def insert_particle_picker(
    message: dict, parameters: Callable, session: sqlalchemy.orm.Session
):
    def full_parameters(param):
        return parameters_with_replacement(param, message, parameters)

    try:
        values = models.ParticlePicker(
            particlePickerId=full_parameters("particle_picker_id"),
            programId=full_parameters("program_id"),
            firstMotionCorrectionId=full_parameters("motion_correction_id"),
            particlePickingTemplate=full_parameters("particle_picking_template"),
            particleDiameter=full_parameters("particle_diameter"),
            numberOfParticles=full_parameters("number_of_particles") or 0,
            summaryImageFullPath=full_parameters("summary_image_full_path"),
        )
        session.add(values)
        session.commit()
        logger.info(f"Created ParticlePicker record {values.particlePickerId}")
        return {"success": True, "return_value": values.particlePickerId}
    except sqlalchemy.exc.SQLAlchemyError as e:
        logger.error(
            f"Inserting Particle Picker entry caused exception {e}",
            exc_info=True,
        )
        return False


def insert_particle_classification(
    message: dict, parameters: Callable, session: sqlalchemy.orm.Session
):
    def full_parameters(param):
        return parameters_with_replacement(param, message, parameters)

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
            f"Created ParticleClassification record {values.particleClassificationId}"
        )
        return {"success": True, "return_value": values.particleClassificationId}
    except sqlalchemy.exc.SQLAlchemyError as e:
        logger.error(
            f"Inserting particle classification entry caused exception {e}",
            exc_info=True,
        )
        return False


def insert_particle_classification_group(
    message: dict, parameters: Callable, session: sqlalchemy.orm.Session
):
    def full_parameters(param):
        return parameters_with_replacement(param, message, parameters)

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
            binnedPixelSize=full_parameters("binned_pixel_size"),
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
            f"{values.particleClassificationGroupId}"
        )
        return {
            "success": True,
            "return_value": values.particleClassificationGroupId,
        }
    except sqlalchemy.exc.SQLAlchemyError as e:
        logger.error(
            f"Inserting particle classification group entry caused exception {e}",
            exc_info=True,
        )
        return False


def insert_cryoem_initial_model(
    message: dict, parameters: Callable, session: sqlalchemy.orm.Session
):
    def full_parameters(param):
        return parameters_with_replacement(param, message, parameters)

    try:
        if not full_parameters("cryoem_initial_model_id"):
            values_im = models.CryoemInitialModel(
                resolution=full_parameters("resolution"),
                numberOfParticles=full_parameters("number_of_particles"),
            )
            session.add(values_im)
            session.commit()
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
        logger.info(f"Created CryoEM Initial Model record {initial_model_id}")
        return {"success": True, "return_value": initial_model_id}
    except sqlalchemy.exc.SQLAlchemyError as e:
        logger.error(
            f"Inserting CryoEM Initial Model entry caused exception {e}",
            exc_info=True,
        )
        return False


def insert_bfactor_fit(
    message: dict, parameters: Callable, session: sqlalchemy.orm.Session
):
    def full_parameters(param):
        return parameters_with_replacement(param, message, parameters)

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
        logger.info(f"Created bfactor record {values.bFactorFitId}")
        return {
            "success": True,
            "return_value": values.bFactorFitId,
        }
    except sqlalchemy.exc.SQLAlchemyError as e:
        logger.error(
            f"Inserting bfactor entry caused exception {e}",
            exc_info=True,
        )
        return False


def insert_tomogram(
    message: dict, parameters: Callable, session: sqlalchemy.orm.Session
):
    if not message:
        message = {}

    def full_parameters(param):
        return parameters_with_replacement(param, message, parameters)

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
        logger.info(f"Created tomogram record {values.tomogramId}")
        return {"success": True, "return_value": values.tomogramId}
    except sqlalchemy.exc.SQLAlchemyError as e:
        logger.error(
            f"Inserting Tomogram entry caused exception {e}",
            exc_info=True,
        )
        return False


def insert_processed_tomogram(
    message: dict, parameters: Callable, session: sqlalchemy.orm.Session
):
    def full_parameters(param):
        return parameters_with_replacement(param, message, parameters)

    try:
        values = models.ProcessedTomogram(
            tomogramId=full_parameters("tomogram_id"),
            filePath=full_parameters("file_path"),
            processingType=full_parameters("processing_type"),
        )
        session.add(values)
        session.commit()
        logger.info(f"Created processed tomogram record {values.processedTomogramId}")
        return {"success": True, "return_value": values.processedTomogramId}
    except sqlalchemy.exc.SQLAlchemyError as e:
        logger.error(
            f"Inserting Processed Tomogram entry caused exception {e}",
            exc_info=True,
        )
        return False


def insert_tilt_image_alignment(
    message: dict, parameters: Callable, session: sqlalchemy.orm.Session
):
    def full_parameters(param):
        return parameters_with_replacement(param, message, parameters)

    if full_parameters("movie_id"):
        mvid = full_parameters("movie_id")
    else:
        mvid = _get_movie_id(full_parameters("path"), full_parameters("dcid"), session)

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
        logger.info(f"Created tilt image alignment record for {values.tomogramId}")
        return {"success": True, "return_value": values.tomogramId}
    except sqlalchemy.exc.SQLAlchemyError as e:
        logger.error(
            f"Inserting Tilt Image Alignment entry caused exception {e}",
            exc_info=True,
        )
        return False


def update_processing_status(
    message: dict, parameters: Callable, session: sqlalchemy.orm.Session
):
    def full_parameters(param):
        return parameters_with_replacement(param, message, parameters)

    ppid = full_parameters("program_id")
    status_message = full_parameters("status_message")

    completion_status = {"success": 1, "failure": 0}.get(full_parameters("status"))
    try:
        if completion_status is None:
            # Messages without a completion status update the processing start time
            values = models.AutoProcProgram(
                autoProcProgramId=ppid,
                processingMessage=status_message,
                processingStartTime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
        else:
            # For "success" and "failure" messages update the processing end time
            values = models.AutoProcProgram(
                autoProcProgramId=ppid,
                processingStatus=completion_status,
                processingMessage=status_message,
                processingEndTime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
        logger.info(f"Updating program {ppid} with status {status_message}")
        return {"success": True, "return_value": values.autoProcProgramId}
    except sqlalchemy.exc.SQLAlchemyError as e:
        logger.error(
            f"Updating program {ppid} status: {status_message} caused exception {e}.",
            exc_info=True,
        )
        return False


# These are needed for the old relion-zocalo wrapper
def add_program_attachment(
    message: dict, parameters: Callable, session: sqlalchemy.orm.Session
):
    file_name = parameters("file_name")
    file_path = parameters("file_path")
    logger.error(
        f"Adding program attachments is no longer supported. "
        f"Skipping file {file_name} in {file_path}."
    )
    return {"success": True, "return_value": 0}


def register_processing(
    message: dict, parameters: Callable, session: sqlalchemy.orm.Session
):
    program = parameters("program")
    cmdline = parameters("cmdline")
    environment = parameters("environment") or ""
    if isinstance(environment, dict):
        environment = ", ".join(f"{key}={value}" for key, value in environment.items())
    environment = environment[: min(255, len(environment))]
    rpid = parameters("rpid")
    if rpid and not str(rpid).isdigit():
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
