from __future__ import annotations

import string
import time
from collections import ChainMap
from datetime import datetime
from pathlib import Path
from typing import Optional

import ispyb
import ispyb.sqlalchemy as models
import sqlalchemy.exc
import sqlalchemy.orm
import workflows.recipe
from pydantic import BaseModel, validate_arguments
from workflows.services.common_service import CommonService

import cryoemservices.services.ispyb_buffer as buffer
from cryoemservices.util.models import MockRW


class ChainMapWithReplacement(ChainMap):
    def __init__(self, *maps, substitutions=None) -> None:
        super().__init__(*maps)
        self._substitutions = substitutions

    def __getitem__(self, k):
        v = super().__getitem__(k)
        if self._substitutions and isinstance(v, str) and "$" in v:
            template = string.Template(v)
            return template.substitute(**self._substitutions)
        return v


class MovieParams(BaseModel):
    dcid: int
    movie_number: Optional[int] = None  # image number
    movie_path: Optional[str] = None  # micrograph full path
    timestamp: Optional[float] = None


def lookup_command(command, refclass):
    return getattr(refclass, "do_" + command, None)


class EMISPyB(CommonService):
    """A service that receives information to be written to ISPyB."""

    # Human readable service name
    _service_name = "EMISPyB"

    # Logger name
    _logger_name = "cryoemservices.services.ispyb"

    # ispyb connection details
    ispyb = None
    _ispyb_sessionmaker = None

    def initializing(self):
        """Subscribe the ISPyB connector queue. Received messages must be
        acknowledged. Prepare ISPyB database connection."""
        self.log.info(f"ISPyB connector using ispyb v{ispyb.__version__}")
        self.ispyb = ispyb.open()
        self._ispyb_sessionmaker = sqlalchemy.orm.sessionmaker(
            bind=sqlalchemy.create_engine(
                ispyb.sqlalchemy.url(), connect_args={"use_pure": True}
            )
        )
        try:
            self.log.info("Cleaning up ISPyB buffer table...")
            with self._ispyb_sessionmaker() as session:
                buffer.evict(session=session)
        except Exception as e:
            self.log.warning(
                f"Encountered exception {e!r} while cleaning up ISPyB buffer table",
                exc_info=True,
            )
        self.log.info("ISPyB service ready")
        workflows.recipe.wrap_subscribe(
            self._transport,
            "ispyb_connector",
            self.receive_msg,
            acknowledgement=True,
            log_extender=self.extend_log,
            allow_non_recipe_messages=True,
        )

    def receive_msg(self, rw, header, message):
        """Do something with ISPyB."""

        if header.get("redelivered") == "true":
            # A redelivered message may just have been processed in a parallel instance,
            # which was connected to a different database server in the DB cluster. If
            # we were to process it immediately we may run into a DB synchronization
            # fault. Avoid this by giving the DB cluster a bit of time to settle.
            self.log.debug("Received redelivered message, holding for a moment.")
            time.sleep(0.5)

        if not rw:
            # Incoming message is not a recipe message. Simple messages can be valid
            self.log.info("Received a simple message")
            if (
                not isinstance(message, dict)
                or not message.get("parameters")
                or not message.get("content")
            ):
                self.log.error("Rejected invalid simple message")
                self._transport.nack(header)
                return

            # Create a wrapper-like object that can be passed to functions
            # as if a recipe wrapper was present.
            rw = MockRW(self._transport)
            rw.recipe_step = {"parameters": message["parameters"]}
            if isinstance(message["content"], dict) and isinstance(
                message["parameters"], dict
            ):
                message["content"].update(message["parameters"])
            message = message["content"]

        command = rw.recipe_step["parameters"].get("ispyb_command")
        if not command:
            self.log.error("Received message is not a valid ISPyB command")
            rw.transport.nack(header)
            return
        command_function = lookup_command(command, self)
        if not command_function:
            self.log.error("Received unknown ISPyB command (%s)", command)
            rw.transport.nack(header)
            return

        self.log.debug("Running ISPyB call %s", command)
        txn = rw.transport.transaction_begin(subscription_id=header["subscription"])
        rw.set_default_channel("output")

        parameter_map = ChainMapWithReplacement(
            message if isinstance(message, dict) else {},
            rw.recipe_step["parameters"],
            substitutions=rw.environment,
        )

        def parameters(parameter, replace_variables=True):
            if isinstance(message, dict):
                base_value = message.get(
                    parameter, rw.recipe_step["parameters"].get(parameter)
                )
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
                if "${" + key + "}" in base_value:
                    base_value = base_value.replace(
                        "${" + key + "}", str(rw.environment[key])
                    )
                # Replace longest keys first, as the following replacement is
                # not well-defined when one key is a prefix of another:
                if "$" + key in base_value:
                    base_value = base_value.replace("$" + key, str(rw.environment[key]))
            return base_value

        try:
            with self._ispyb_sessionmaker() as session:
                result = command_function(
                    rw=rw,
                    message=message,
                    parameters=parameters,
                    parameter_map=parameter_map,
                    session=session,
                    transaction=txn,
                    header=header,
                )
        except Exception as e:
            self.log.error(
                f"Uncaught exception {e!r} in ISPyB function {command!r}, "
                "quarantining message and shutting down instance.",
                exc_info=True,
            )
            rw.transport.transaction_abort(txn)
            rw.transport.nack(header)
            self._request_termination()
            return

        store_result = rw.recipe_step["parameters"].get("store_result")
        if store_result and result and "return_value" in result:
            rw.environment[store_result] = result["return_value"]
            self.log.debug(
                "Storing result '%s' in environment variable '%s'",
                result["return_value"],
                store_result,
            )
        if result and result.get("success"):
            rw.send({"result": result.get("return_value")}, transaction=txn)
            rw.transport.ack(header, transaction=txn)
        elif result and result.get("checkpoint") and not result.get("delay"):
            rw.checkpoint(
                result.get("return_value"),
                delay=rw.recipe_step["parameters"].get("delay", result.get("delay")),
                transaction=txn,
            )
            rw.transport.ack(header, transaction=txn)
        else:
            rw.transport.transaction_abort(txn)
            rw.transport.nack(header)
            return
        rw.transport.transaction_commit(txn)

    def do_multipart_message(self, rw, message, **kwargs):
        """The multipart_message command allows the recipe or client to specify a
        multi-stage operation. With this you can process a list of API calls.
        Each API call may have a return value that can be stored.
        Multipart_message takes care of chaining and checkpointing to make the
        overall call near-ACID compliant."""

        if not rw.environment.get("has_recipe_wrapper", True):
            self.log.error(
                "Multipart message call can not be used with simple messages"
            )
            return False

        step = 1
        commands = rw.recipe_step["parameters"].get("ispyb_command_list")
        if isinstance(message, dict) and isinstance(
            message.get("ispyb_command_list"), list
        ):
            commands = message["ispyb_command_list"]
            step = message.get("checkpoint", 0) + 1
        if not commands:
            self.log.error("Received multipart message containing no commands")
            return False

        current_command = commands[0]
        command = current_command.get("ispyb_command")
        if not command:
            self.log.error(
                "Multipart command %s is not a valid ISPyB command", current_command
            )
            return False
        command_function = lookup_command(command, self)
        if not command_function:
            self.log.error("Received unknown ISPyB command (%s)", command)
            return False
        self.log.debug(
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
                if "${" + key + "}" in base_value:
                    base_value = base_value.replace(
                        "${" + key + "}", str(rw.environment[key])
                    )
                # Replace longest keys first, as the following replacement is
                # not well-defined when one key is a prefix of another:
                if "$" + key in base_value:
                    base_value = base_value.replace("$" + key, str(rw.environment[key]))
            return base_value

        kwargs["parameters"] = step_parameters

        # If this step previously checkpointed then override the message passed
        # to the step.
        step_message = current_command
        if isinstance(message, dict):
            step_message = message.get("step_message", step_message)

        # Run the multipart step
        result = command_function(rw=rw, message=step_message, **kwargs)

        # Store step result if appropriate
        store_result = current_command.get("store_result")
        if store_result and result and "return_value" in result:
            rw.environment[store_result] = result["return_value"]
            self.log.debug(
                "Storing result '%s' in environment variable '%s'",
                result["return_value"],
                store_result,
            )

        # If the current step has checkpointed then need to manage this
        if result and result.get("checkpoint"):
            self.log.debug("Checkpointing for sub-command %s", command)

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
                "delay": result.get("delay"),
            }

        # If the step did not succeed then propagate failure
        if not result or not result.get("success"):
            self.log.debug("Multipart command failed")
            return result

        # Step has completed, so remove from queue
        commands.pop(0)

        # If the multipart command is finished then propagate success
        if not commands:
            self.log.debug("and done.")
            return result

        # If there are more steps then checkpoint the current state
        # and put it back on the queue (with no delay)
        self.log.debug("Checkpointing remaining %d steps", len(commands))
        if isinstance(message, dict):
            checkpoint_dictionary = message
        else:
            checkpoint_dictionary = {}
        checkpoint_dictionary["checkpoint"] = step
        checkpoint_dictionary["ispyb_command_list"] = commands
        if "step_message" in checkpoint_dictionary:
            del checkpoint_dictionary["step_message"]
        return {"checkpoint": True, "return_value": checkpoint_dictionary}

    def do_buffer(self, rw, message, session, parameters, header, **kwargs):
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
            self.log.error(f"Invalid buffer call: {message} is not a dictionary")
            return False

        if not isinstance(message.get("buffer_command"), dict) or not message[
            "buffer_command"
        ].get("ispyb_command"):
            self.log.error(f"Invalid buffer call: no buffer command in {message}")
            return False

        command_function = lookup_command(
            message["buffer_command"]["ispyb_command"], self
        )
        if not command_function:
            self.log.error(f"Invalid buffer call: unknown command in {message}")
            return False

        if ("buffer_expiry_time" not in message) or (
            header.get("dlq-reinjected") in {True, "True", "true", 1}
        ):
            message["buffer_expiry_time"] = time.time() + 600

        # Prepare command: Resolve all references
        program_id = parameters("program_id")
        if message.get("buffer_lookup"):
            if not isinstance(message["buffer_lookup"], dict):
                self.log.error(
                    "Invalid buffer call: buffer_lookup dictionary is not a dictionary"
                )
                return False
            if not program_id:
                self.log.error("Invalid buffer call: program_id is undefined")
                return False
            for entry in list(message["buffer_lookup"]):
                buffer_result = buffer.load(
                    session=session,
                    program=program_id,
                    uuid=message["buffer_lookup"][entry],
                )
                if buffer_result.success:
                    # resolve value and continue
                    message["buffer_command"][entry] = buffer_result.value
                    del message["buffer_lookup"][entry]
                    self.log.debug(
                        f"Successfully resolved buffer reference {entry!r} to {buffer_result.value!r}"
                    )
                    continue

                if message["buffer_expiry_time"] < time.time():
                    self.log.warning(
                        f"Buffer call could not be resolved: entry {entry} not found for program {program_id}"
                    )
                    return False

                # value can not yet be resolved, put request back in the queue
                return {"checkpoint": True, "return_value": message, "delay": 20}

        # Run the actual command
        result = command_function(
            rw=rw,
            message=message["buffer_command"],
            session=session,
            parameters=parameters,
            **kwargs,
        )

        # Store result if appropriate
        store_result = message.get("store_result")
        if store_result and result and "return_value" in result:
            rw.environment[store_result] = result["return_value"]
            self.log.debug(
                "Storing result '%s' in environment variable '%s'",
                result["return_value"],
                store_result,
            )

        # If the actual command has checkpointed then need to manage this
        if result and result.get("checkpoint"):
            self.log.debug("Checkpointing for buffered function")
            message["buffer_command"] = result["return_value"]
            return {
                "checkpoint": True,
                "return_value": message,
                "delay": result.get("delay"),
            }

        # If the command did not succeed then propagate failure
        if not result or not result.get("success"):
            self.log.warning("Buffered command failed")
            # to become debug level eventually, the actual function will do the warning
            return result

        # Optionally store a reference to the result in the buffer table
        if message.get("buffer_store"):
            self.log.debug("Storing buffer result for UUID %r", message["buffer_store"])
            buffer.store(
                session=session,
                program=program_id,
                uuid=message["buffer_store"],
                reference=result["return_value"],
            )

        # Finally, propagate result
        return result

    # EM-specific parts from here
    def _get_movie_id(
        self,
        full_path,
        data_collection_id,
        db_session,
    ):
        self.log.info(
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
            self.log.info(f"Found Movie ID: {mvid}")
            return mvid
        else:
            self.log.error(f"Unable to find movie ID for {movie_name}")
            return None

    @validate_arguments(config={"arbitrary_types_allowed": True})
    def do_insert_movie(self, parameter_map: MovieParams, session, **kwargs):
        self.log.info("Inserting Movie parameters.")

        try:
            if parameter_map.timestamp:
                values = models.Movie(
                    dataCollectionId=parameter_map.dcid,
                    movieNumber=parameter_map.movie_number,
                    movieFullPath=parameter_map.movie_path,
                    createdTimeStamp=datetime.fromtimestamp(
                        parameter_map.timestamp
                    ).strftime("%Y-%m-%d %H:%M:%S"),
                )
            else:
                values = models.Movie(
                    dataCollectionId=parameter_map.dcid,
                    movieNumber=parameter_map.movie_number,
                    movieFullPath=parameter_map.movie_path,
                )
            session.add(values)
            session.commit()
            self.log.info(f"Created Movie record {values.movieId}")
            return {"success": True, "return_value": values.movieId}
        except sqlalchemy.exc.SQLAlchemyError as e:
            self.log.error(
                "Inserting movie entry caused exception '%s'.",
                e,
                exc_info=True,
            )
            return False

    def do_insert_motion_correction(self, parameters, session, message=None, **kwargs):
        if message is None:
            message = {}
        self.log.info("Inserting Motion Correction parameters.")

        def full_parameters(param):
            return message.get(param) or parameters(param)

        try:
            movie_id = None
            if full_parameters("movie_id") is None:
                movie_values = self.do_insert_movie(
                    parameter_map=MovieParams(
                        dcid=full_parameters("dcid"),
                        movie_number=full_parameters("image_number"),
                        movie_path=full_parameters("micrograph_full_path"),
                        timestamp=full_parameters("created_time_stamp"),
                    ),
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
                micrographSnapshotFullPath=full_parameters(
                    "micrograph_snapshot_full_path"
                ),
                patchesUsedX=full_parameters("patches_used_x"),
                patchesUsedY=full_parameters("patches_used_y"),
                fftFullPath=full_parameters("fft_full_path"),
                fftCorrectedFullPath=full_parameters("fft_corrected_full_path"),
                comments=full_parameters("comments"),
            )
            session.add(values)
            session.commit()
            self.log.info(
                f"Created MotionCorrection record {values.motionCorrectionId}"
            )
            return {"success": True, "return_value": values.motionCorrectionId}
        except sqlalchemy.exc.SQLAlchemyError as e:
            self.log.error(
                "Inserting motion correction entry caused exception '%s'.",
                e,
                exc_info=True,
            )
            return False

    def do_insert_relative_ice_thickness(
        self, parameters, session, message=None, **kwargs
    ):
        if message is None:
            message = {}
        dcid = parameters("dcid")
        self.log.info(f"Inserting Relative Ice Thickness parameters. DCID: {dcid}")

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
            self.log.error(
                "Inserting relative ice thickness entry caused exception '%s'.",
                e,
                exc_info=True,
            )
            return False

    def do_insert_ctf(self, parameters, session, message=None, **kwargs):
        if message is None:
            message = {}
        dcid = parameters("dcid")
        self.log.info(f"Inserting CTF parameters. DCID: {dcid}")

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
            self.log.info(f"Created CTF record {values.ctfId} for DCID {dcid}")
            return {"success": True, "return_value": values.ctfId}
        except sqlalchemy.exc.SQLAlchemyError as e:
            self.log.error(
                "Inserting CTF entry caused exception '%s'.",
                e,
                exc_info=True,
            )
            return False

    def do_insert_particle_picker(self, parameters, session, message=None, **kwargs):
        if message is None:
            message = {}
        dcid = parameters("dcid")
        self.log.info(f"Inserting Particle Picker parameters. DCID: {dcid}")

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
            self.log.info(
                f"Created ParticlePicker record {values.particlePickerId} "
                f"for DCID {dcid}"
            )
            return {"success": True, "return_value": values.particlePickerId}
        except sqlalchemy.exc.SQLAlchemyError as e:
            self.log.error(
                "Inserting Particle Picker entry caused exception '%s'.",
                e,
                exc_info=True,
            )
            return False

    def do_insert_particle_classification(
        self, parameters, session, message=None, **kwargs
    ):
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
                overallFourierCompleteness=full_parameters(
                    "overall_fourier_completeness"
                ),
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
            self.log.info(
                "Created ParticleClassification record "
                f"{values.particleClassificationId} for DCID {dcid}"
            )
            return {"success": True, "return_value": values.particleClassificationId}
        except sqlalchemy.exc.SQLAlchemyError as e:
            self.log.error(
                "Inserting particle classification entry caused exception '%s'.",
                e,
                exc_info=True,
            )
            return False

    def do_insert_particle_classification_group(
        self, parameters, session, message=None, **kwargs
    ):
        if message is None:
            message = {}
        dcid = parameters("dcid")

        def full_parameters(param):
            return message.get(param) or parameters(param)

        self.log.info(f"Inserting particle classification parameters. DCID: {dcid}")
        try:
            values = models.ParticleClassificationGroup(
                particleClassificationGroupId=full_parameters(
                    "particle_classification_group_id"
                ),
                particlePickerId=full_parameters("particle_picker_id"),
                programId=full_parameters("program_id"),
                type=full_parameters("type"),
                batchNumber=full_parameters("batch_number"),
                numberOfParticlesPerBatch=full_parameters(
                    "number_of_particles_per_batch"
                ),
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
                        if k
                        not in ["_sa_instance_state", "particleClassificationGroupId"]
                        and v is not None
                    }
                )
            else:
                session.add(values)
            session.commit()
            self.log.info(
                "Created particle classification group record "
                f"{values.particleClassificationGroupId} for DCID {dcid}"
            )
            return {
                "success": True,
                "return_value": values.particleClassificationGroupId,
            }
        except sqlalchemy.exc.SQLAlchemyError as e:
            self.log.error(
                "Inserting particle classification group entry caused exception '%s'.",
                e,
                exc_info=True,
            )
            return False

    def do_insert_cryoem_initial_model(
        self, parameters, session, message=None, **kwargs
    ):
        if message is None:
            message = {}
        dcid = parameters("dcid")

        def full_parameters(param):
            return message.get(param) or parameters(param)

        self.log.info(f"Inserting CryoEM Initial Model parameters. DCID: {dcid}")
        try:
            if not full_parameters("cryoem_initial_model_id"):
                values_im = models.CryoemInitialModel(
                    resolution=full_parameters("resolution"),
                    numberOfParticles=full_parameters("number_of_particles"),
                )
                session.add(values_im)
                session.commit()
                self.log.info(
                    "Created CryoEM Initial Model record "
                    f"{values_im.cryoemInitialModelId} for DCID {dcid}"
                )
                initial_model_id = values_im.cryoemInitialModelId
            else:
                initial_model_id = full_parameters("cryoem_initial_model_id")
            session.execute(
                models.t_ParticleClassification_has_CryoemInitialModel.insert().values(
                    cryoemInitialModelId=initial_model_id,
                    particleClassificationId=full_parameters(
                        "particle_classification_id"
                    ),
                )
            )
            session.commit()
            return {"success": True, "return_value": initial_model_id}
        except ispyb.ISPyBException as e:
            self.log.error(
                "Inserting CryoEM Initial Model entry caused exception '%s'.",
                e,
                exc_info=True,
            )
            return False

    def do_insert_bfactor_fit(self, parameters, session, message=None, **kwargs):
        if message is None:
            message = {}
        dcid = parameters("dcid")

        def full_parameters(param):
            return message.get(param) or parameters(param)

        self.log.info(f"Inserting bfactor calculation parameters. DCID: {dcid}")
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
                        if k not in ["_sa_instance_state", "bFactorFitId"]
                        and v is not None
                    }
                )
            else:
                session.add(values)
            session.commit()
            self.log.info(
                f"Created bfactor record {values.bFactorFitId} for DCID {dcid}"
            )
            return {
                "success": True,
                "return_value": values.bFactorFitId,
            }
        except sqlalchemy.exc.SQLAlchemyError as e:
            self.log.error(
                "Inserting bfactor entry caused exception '%s'.",
                e,
                exc_info=True,
            )
            return False

    def do_insert_tomogram(self, parameters, session, message=None, **kwargs):
        if message is None:
            message = {}
        dcid = parameters("dcid")
        self.log.info(f"Inserting Tomogram parameters. DCID: {dcid}")
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
                        if k not in ["_sa_instance_state", "tomogramId"]
                        and v is not None
                    }
                )
            else:
                session.add(values)
            session.commit()
            return {"success": True, "return_value": values.tomogramId}
        except sqlalchemy.exc.SQLAlchemyError as e:
            self.log.error(
                "Inserting Tomogram entry caused exception '%s'.",
                e,
                exc_info=True,
            )
            return False

    def do_insert_processed_tomogram(self, parameters, session, message=None, **kwargs):
        if message is None:
            message = {}
        dcid = parameters("dcid")
        self.log.info(f"Inserting Processed Tomogram parameters. DCID: {dcid}")

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
            self.log.error(
                "Inserting Processed Tomogram entry caused exception '%s'.",
                e,
                exc_info=True,
            )
            return False

    def do_insert_tilt_image_alignment(
        self, parameters, session, message=None, **kwargs
    ):
        if message is None:
            message = {}
        dcid = parameters("dcid")
        self.log.info(f"Inserting Tilt Image Alignment parameters. DCID: {dcid}")

        def full_parameters(param):
            return message.get(param) or parameters(param)

        if full_parameters("movie_id"):
            mvid = full_parameters("movie_id")
        else:
            mvid = self._get_movie_id(full_parameters("path"), dcid, session)

        if not mvid:
            self.log.error("No movie ID for tilt image alignment")
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
            self.log.error(
                "Inserting Tilt Image Alignment entry caused exception '%s'.",
                e,
                exc_info=True,
            )
            return False

    # These are needed for the old relion-zocalo wrapper
    def do_add_program_attachment(self, parameters, **kwargs):
        params = self.ispyb.mx_processing.get_program_attachment_params()
        params["parentid"] = parameters("program_id")
        try:
            programid = int(params["parentid"])
        except ValueError:
            programid = None
        if not programid:
            self.log.warning("Encountered invalid program ID '%s'", params["parentid"])
            return False
        params["file_name"] = parameters("file_name", replace_variables=False)
        params["file_path"] = parameters("file_path", replace_variables=False)
        params["importance_rank"] = parameters(
            "importance_rank", replace_variables=False
        )
        fqpn = Path(params["file_path"]) / params["file_name"]

        if not fqpn.is_file():
            self.log.error(
                "Not adding attachment '%s' to data processing: File does not exist",
                str(fqpn),
            )
            return False

        params["file_type"] = str(parameters("file_type")).lower()
        if params["file_type"] not in ("log", "result", "graph"):
            self.log.warning(
                "Attachment type '%s' unknown, defaulting to 'log'", params["file_type"]
            )
            params["file_type"] = "log"

        self.log.debug("Writing program attachment to database: %s", params)

        result = self.ispyb.mx_processing.upsert_program_attachment(
            list(params.values())
        )
        return {"success": True, "return_value": result}

    def do_register_processing(self, parameters, **kwargs):
        program = parameters("program")
        cmdline = parameters("cmdline")
        environment = parameters("environment") or ""
        if isinstance(environment, dict):
            environment = ", ".join(
                f"{key}={value}" for key, value in environment.items()
            )
        environment = environment[: min(255, len(environment))]
        rpid = parameters("rpid")
        if rpid and not rpid.isdigit():
            self.log.error(f"Invalid processing id {rpid}")
            return False
        try:
            result = self.ispyb.mx_processing.upsert_program_ex(
                job_id=rpid,
                name=program,
                command=cmdline,
                environment=environment,
            )
            self.log.info(
                f"Registered new program {program} for processing id {rpid} "
                f"with command line {cmdline} and environment {environment} "
                f"with result {result}."
            )
            return {"success": True, "return_value": result}
        except ispyb.ISPyBException as e:
            self.log.error(
                f"Registering new program {program} for processing id {rpid} "
                f"with command line {cmdline} and environment {environment} "
                f"caused exception {e}.",
                exc_info=True,
            )
            return False

    def do_update_processing_status(self, parameters, **kwargs):
        ppid = parameters("program_id")
        message = parameters("message")
        status = parameters("status")
        try:
            result = self.ispyb.mx_processing.upsert_program_ex(
                program_id=ppid,
                status={"success": 1, "failure": 0}.get(status),
                time_start=parameters("start_time"),
                time_update=parameters("update_time"),
                message=message,
            )
            self.log.info(
                f"Updating program {ppid} with status {message}",
            )
            # result is just ppid
            return {"success": True, "return_value": result}
        except ispyb.ISPyBException as e:
            self.log.error(
                f"Updating program {ppid} status: {message} caused exception {e}.",
                exc_info=True,
            )
            return False
