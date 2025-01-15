from __future__ import annotations

import uuid
from importlib.metadata import entry_points
from pathlib import Path

import workflows.recipe
from workflows.services.common_service import CommonService

from cryoemservices.util.config import ServiceConfig, config_from_file


def filter_load_recipes_from_files(
    message: dict, parameters: dict, config: ServiceConfig
):
    """Load named recipes from configured location"""
    for recipefile in message.get("recipes", []):
        recipe_location = config.recipe_directory / f"{recipefile}.json"
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


def filter_apply_parameters(message: dict, parameters: dict, config: ServiceConfig):
    """Fill in any placeholders in the recipe of the form {name} using the
    parameters data structure"""
    message["recipe"].apply_parameters(parameters)
    return message, parameters


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

    def initializing(self):
        """Subscribe to the processing_recipe queue."""
        self.log.info("ProcessRecipe service starting")
        self.message_filters = {
            **{
                f.name: f.load()
                for f in entry_points(
                    group="cryoemservices.services.process_recipe.filters"
                )
            },
            "load_recipes_from_files": filter_load_recipes_from_files,
            "apply_parameters": filter_apply_parameters,
        }

        workflows.recipe.wrap_subscribe(
            self._transport,
            self._environment["queue"] or "processing_recipe",
            self.process,
            acknowledgement=True,
            log_extender=self.extend_log,
            allow_non_recipe_messages=True,
        )

    def process(self, rw, header, message):
        """Process an incoming processing request."""
        # Find config
        service_config = config_from_file(self._environment["config"])

        # Load processing parameters
        self.log.info(f"Received processing request: {str(message)}")
        parameters = message.get("parameters", {})
        if not isinstance(parameters, dict):
            self.log.error("Rejected parameters not given as dictionary")
            self._transport.nack(header)
            return

        # Unless 'guid' is already defined then generate a unique recipe ID
        recipe_id = parameters.get("guid") or str(uuid.uuid4())
        parameters["guid"] = recipe_id

        # Add an empty recipe to the message
        message["recipe"] = workflows.recipe.Recipe()

        # Apply all specified filters in order to message and parameters
        for name, f in self.message_filters.items():
            try:
                message, parameters = f(
                    message=message,
                    parameters=parameters,
                    config=service_config,
                )
            except Exception as e:
                self.log.info(f"Rejected message due to filter {name} error: {e}")
                self._transport.nack(header)
                return

        self.log.info(f"Filtered processing request: {str(message)}")
        self.log.info(f"Filtered parameters: {str(parameters)}")

        # Conditionally acknowledge receipt of the message
        txn = self._transport.transaction_begin(subscription_id=header["subscription"])
        self._transport.ack(header, transaction=txn)

        rw = workflows.recipe.RecipeWrapper(
            recipe=message["recipe"], transport=self._transport
        )
        rw.environment = {"ID": recipe_id}
        rw.start(transaction=txn)

        # Commit transaction
        self._transport.transaction_commit(txn)
        self.log.info("Processed incoming message")
