from __future__ import annotations

import copy
import os
import uuid
from pathlib import Path

import workflows.recipe
from backports.entry_points_selectable import entry_points
from workflows.services.common_service import CommonService

from cryoemservices.util.config import config_from_file


class ProcessRecipe(CommonService):
    """
    Service that takes in a data collection ID or a processing recipe,
    and mangles these into something that can be processed by downstream services.
    """

    # Human readable service name
    _service_name = "ProcessRecipe"

    # Logger name
    _logger_name = "cryoemservices.services.process_recipe"

    recipe_basepath: Path = Path.cwd()
    message_filters: dict = {}

    def filter_load_recipes_from_files(self, message: dict, parameters: dict):
        """Load named recipes from configured location"""
        for recipefile in message.get("recipes", []):
            recipe_location = self.recipe_basepath / f"{recipefile}.json"
            if not recipe_location.is_file():
                raise ValueError(f"Cannot find recipe in location {recipe_location}")
            with open(recipe_location, "r") as rcp:
                named_recipe = workflows.recipe.Recipe(recipe=rcp.read())
            try:
                named_recipe.validate()
            except workflows.Error as e:
                raise ValueError(f"Named recipe {recipefile} failed validation. {e}")
            message["recipe"] = message["recipe"].merge(named_recipe)
        return message, parameters

    def filter_apply_parameters(self, message: dict, parameters: dict):
        """Fill in any placeholders in the recipe of the form {name} using the
        parameters data structure"""
        message["recipe"].apply_parameters(parameters)
        return message, parameters

    def initializing(self):
        """Subscribe to the processing_recipe queue."""
        self.log.info("ProcessRecipe service starting")
        service_config = config_from_file(Path(os.environ["CRYOEMSERVICES_CONFIG"]))
        self.recipe_basepath = service_config.recipe_directory

        self.message_filters = {
            **{
                f.name: f.load()
                for f in entry_points(
                    group="cryoemservices.services.process_recipe.filters"
                )
            },
            "load_recipes_from_files": self.filter_load_recipes_from_files,
            "apply_parameters": self.filter_apply_parameters,
        }

        workflows.recipe.wrap_subscribe(
            self._transport,
            "processing_recipe",
            self.process,
            acknowledgement=True,
            log_extender=self.extend_log,
            allow_non_recipe_messages=True,
        )

    def process(self, rw, header, message):
        """Process an incoming processing request."""
        # Load processing parameters
        parameters = message.get("parameters", {})
        if not isinstance(parameters, dict):
            self.log.error("Rejected parameters not given as dictionary")
            self._transport.nack(header)
            return

        # Unless 'guid' is already defined then generate a unique recipe IDs for
        # this request, which is attached to all downstream log records and can
        # be used to determine unique file paths.
        recipe_id = parameters.get("guid") or str(uuid.uuid4())
        parameters["guid"] = recipe_id

        # From here on add the global ID to all log messages
        with self.extend_log("recipe_ID", recipe_id):
            self.log.info("Received processing request:\n" + str(message))
            self.log.info("Received processing parameters:\n" + str(parameters))

            filtered_message: dict = copy.deepcopy(message)
            filtered_parameters: dict = copy.deepcopy(parameters)

            # Create empty recipe
            filtered_message["recipe"] = workflows.recipe.Recipe()

            # Apply all specified filters in order to message and parameters
            for name, f in self.message_filters.items():
                try:
                    filtered_message, filtered_parameters = f(
                        message=filtered_message, parameters=filtered_parameters
                    )
                except Exception as e:
                    self.log.error(f"Rejected message due to filter {name} error: {e}")
                    self._transport.nack(header)
                    return

            self.log.info("Mangled processing request:\n" + str(filtered_message))
            self.log.info("Mangled processing parameters:\n" + str(filtered_parameters))

            # Conditionally acknowledge receipt of the message
            txn = self._transport.transaction_begin(
                subscription_id=header["subscription"]
            )
            self._transport.ack(header, transaction=txn)

            rw = workflows.recipe.RecipeWrapper(
                recipe=filtered_message["recipe"], transport=self._transport
            )
            rw.environment = {"ID": recipe_id}
            rw.start(transaction=txn)

            # Commit transaction
            self._transport.transaction_commit(txn)
            self.log.info("Processed incoming message")
