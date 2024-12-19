from __future__ import annotations


class MockRW:
    def __init__(self, transport):
        self.transport = transport
        self.recipe_step = {}
        self.environment = {"has_recipe_wrapper": False}

    def set_default_channel(self, *args, **kwargs):
        pass

    def send(self, *args, **kwargs):
        pass

    def send_to(self, destination, parameters):
        self.transport.send(destination, parameters)
