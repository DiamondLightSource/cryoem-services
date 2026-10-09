import signal


class MockRW:
    def __init__(self, transport):
        self._transport = transport
        self.recipe_step = {}
        self.environment = {"has_recipe_wrapper": False}

    @property
    def transport(self):
        return self._transport

    def set_default_channel(self, *args, **kwargs):
        pass

    def send(self, *args, **kwargs):
        pass

    def send_to(self, destination, parameters):
        self.transport.send(destination, parameters)


class InterruptHandler:
    def __init__(self):
        self.interrupted = False
        signal.signal(signal.SIGINT, self.set_interrupted)
        signal.signal(signal.SIGTERM, self.set_interrupted)

    def __enter__(self):
        self.interrupted = False
        return self

    def __exit__(self, tp, val, tb):
        pass

    def set_interrupted(self, sig, frame):
        self.interrupted = True
