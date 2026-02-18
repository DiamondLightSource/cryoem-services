from __future__ import annotations

import functools
import json
import logging
import string
from typing import Any

logger = logging.getLogger("cryoemservices.util.recipe")


def wrap_subscribe(
    transport_layer,
    channel,
    callback,
    acknowledgement=True,
):
    """Internal method to create an intercepting function for incoming messages
    to interpret recipes. This function is then used to subscribe to a channel
    on the transport layer.
    """

    @functools.wraps(callback)
    def unwrap_recipe(header, message):
        """This is a helper function unpacking incoming messages when they are
        in a recipe format. Other messages are passed through unmodified.
        :param header:  A dictionary of message headers. If the header contains
                        an entry 'workflows-recipe' then the message is parsed
                        and the embedded recipe information is passed on in a
                        RecipeWrapper object to the target function.
        :param message: Incoming deserialized message object.
        """
        if header.get("workflows-recipe") is True:
            rw = RecipeWrapper(message=message, transport=transport_layer)
            return callback(rw, header, message.get("payload"))
        return callback(None, header, message)

    return transport_layer.subscribe(
        channel, unwrap_recipe, acknowledgement=acknowledgement
    )


class Recipe:
    """Object containing a processing recipe that can be passed to services.
    A recipe describes how all involved services are connected together, how
    data should be passed and how errors should be handled."""

    recipe: dict[Any, Any] = {}
    """The processing recipe is encoded in this dictionary."""

    def __init__(self, recipe=None):
        """Constructor allows passing in a recipe dictionary."""
        if isinstance(recipe, str):
            self.recipe = json.loads(recipe)
        elif recipe:
            self.recipe = recipe
        if self.recipe:
            self.validate()

    def __getitem__(self, item):
        """Allow direct dictionary access to recipe elements."""
        return self.recipe.__getitem__(item)

    def validate(self):
        """Check whether the encoded recipe is valid"""
        if not self.recipe:
            raise Exception("Invalid recipe: No recipe defined")

        # Make all keys strings
        for key in self.recipe.keys():
            if isinstance(key, int):
                self.recipe[str(key)] = self.recipe[key]
                del self.recipe[key]

        # Without a 'start' node nothing would happen
        if not self.recipe.get("start"):
            raise Exception('Invalid recipe: "start" node empty or missing')
        if not type(self.recipe["start"]) in [int, str]:
            raise Exception('Invalid recipe: "start" must be an integer or string')

        touched_nodes = ["start"]

        def follow_recipe(node: str):
            touched_nodes.append(node)
            if self.recipe[node]["output"]:
                if not isinstance(self.recipe[node]["output"], dict):
                    raise ValueError(f"Invalid output for recipe node {node}")
                for k, v in self.recipe[node]["output"].items():
                    if str(v) not in touched_nodes:
                        follow_recipe(str(v))

        # Test recipe for unreferenced nodes
        follow_recipe(str(self.recipe["start"]))
        for key in self.recipe.keys():
            if key not in touched_nodes:
                raise KeyError(f"Recipe node {key} is not accessed")

    def apply_parameters(self, parameters):
        """Recursively apply dictionary entries in 'parameters' to {item}s in recipe
        structure, leaving undefined {item}s as they are.
        """
        # The python formatter class is used to resolve {item} references
        formatter = string.Formatter()

        def _recursive_apply(item):
            """Helper function to recursively apply replacements."""
            if isinstance(item, str):
                return formatter.vformat(item, (), parameters)
            elif isinstance(item, dict):
                return {
                    _recursive_apply(key): _recursive_apply(value)
                    for key, value in item.items()
                }
            elif isinstance(item, list):
                return [_recursive_apply(x) for x in item]
            else:
                raise TypeError(
                    f"Cannot format recipe item {item} of type {type(item)}"
                )

        self.recipe = _recursive_apply(self.recipe)


class RecipeWrapper:
    """A wrapper object which contains a recipe and a number of functions to make
    life easier for recipe users.
    """

    def __init__(self, message=None, transport=None, recipe=None, environment=None):
        """Create a RecipeWrapper object from a wrapped message.
        References to the transport layer are required to send directly to
        connected downstream processes.
        """
        if not transport:
            raise ValueError("Transport object is required")
        if message:
            self.recipe = Recipe(message["recipe"])
            self.recipe_pointer = int(message["recipe-pointer"])
            self.recipe_step = self.recipe[self.recipe_pointer]
            self.recipe_path = message.get("recipe-path", [])
            if environment is None:
                self.environment = message.get("environment", {})
            else:
                self.environment = environment
            self.payload = message.get("payload")
        elif recipe:
            if isinstance(recipe, Recipe):
                self.recipe = recipe
            else:
                self.recipe = Recipe(recipe)
            self.recipe_pointer = None
            self.recipe_step = None
            self.recipe_path = []
            self.environment = environment or {}
            self.payload = None
        else:
            raise ValueError("A message or recipe is required for a RecipeWrapper")
        self.transport = transport

    def send_to(self, channel, *args, **kwargs):
        """Send messages to another service that is connected to the currently
        running service via the recipe. Discard messages if the recipe does
        not have anything connected to the specified output channel.
        """
        if not self.recipe_step:
            raise ValueError("This RecipeWrapper object does not contain a recipe step")
        if channel not in self.recipe_step.get("output", []):
            raise ValueError(
                "The current recipe step does not have an output channel with this name"
            )

        self._send_to_destination(self.recipe_step["output"][channel], *args, **kwargs)

    def start(self, header=None, **kwargs):
        """Trigger the start of a recipe, sending the defined payloads to the
        recipients set in the recipe. Any parameters to this function are
        passed to the transport send/broadcast methods.
        If the wrapped recipe has already been started then a ValueError will
        be raised.
        """
        if self.recipe_step:
            raise ValueError("This recipe has already been started.")
        for destination, payload in self.recipe["start"]:
            self._send_to_destination(destination, header, payload, kwargs)

    def checkpoint(self, message, header=None, **kwargs):
        """Send a message to the current recipe destination. This can be used to
        keep a state for longer processing tasks.
        """
        if not self.recipe_step:
            raise ValueError("This RecipeWrapper object does not contain a recipe step")
        self._send_to_destination(
            self.recipe_pointer, header, message, kwargs, add_path_step=False
        )

    def _generate_full_recipe_message(self, destination, message, add_path_step):
        """Factory function to generate independent message objects for
        downstream recipients with different destinations."""
        if add_path_step and self.recipe_pointer:
            recipe_path = self.recipe_path + [self.recipe_pointer]
        else:
            recipe_path = self.recipe_path

        return {
            "environment": self.environment,
            "payload": message,
            "recipe": self.recipe.recipe,
            "recipe-path": recipe_path,
            "recipe-pointer": destination,
        }

    def _send_to_destination(
        self,
        destination,
        header,
        payload,
        transport_kwargs,
        add_path_step=True,
    ):
        """Helper function to send a message to a specific recipe destination."""
        if header:
            header = header.copy()
            header["workflows-recipe"] = True
        else:
            header = {"workflows-recipe": True}

        dest_kwargs = transport_kwargs.copy()
        if "exchange" in self.recipe[destination]:
            dest_kwargs.setdefault("exchange", self.recipe[destination]["exchange"])

        if self.recipe[destination].get("queue"):
            message = self._generate_full_recipe_message(
                destination, payload, add_path_step
            )
            self.transport.send(
                self.recipe[destination]["queue"],
                message,
                headers=header,
                **dest_kwargs,
            )
