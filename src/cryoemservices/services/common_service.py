from __future__ import annotations

import enum
import itertools
import logging
import queue
from typing import Any, Dict

import workflows
import workflows.logging


class Priority(enum.IntEnum):
    """
    Priorities for the service-internal priority queue. This ensures that eg.
    frontend commands are always processed before timer events.
    """

    COMMAND = 1
    TIMER = 2
    TRANSPORT = 3
    IDLE = 4


class CommonService:
    """
    Base class for workflow services. A service is a piece of software that runs
    in an isolated environment, communicating only via pipes with the outside
    world. Units of work are injected via a pipe. Results, status and log
    messages, etc. are written out via a pipe. Any task can be encapsulated
    as a service, for example a service that counts spots on an image passed
    as a filename, and returns the number of counts.

    To instantiate a service two Pipe-like objects should be passed to the
    constructors, one to communicate from the service to the frontend, one to
    communicate from the frontend to the service.
    """

    # Human readable service name -----------------------------------------------

    _service_name = "unnamed service"

    # Logger name ---------------------------------------------------------------

    _logger_name = "workflows.service"  # The logger can be accessed via self.log

    # Overrideable functions ----------------------------------------------------

    def initializing(self):
        """Service initialization. This function is run before any commands are
        received from the frontend. This is the place to request channel
        subscriptions with the messaging layer, and register callbacks.
        This function can be overridden by specific service implementations."""
        pass

    # Any keyword arguments set on service invocation

    start_kwargs: Dict[Any, Any] = {}

    # Not so overrideable functions ---------------------------------------------

    def __init__(self, *args, **kwargs):
        """Service constructor. Parameters include optional references to two
        pipes: frontend= for messages from the service to the frontend,
        and commands= for messages from the frontend to the service.
        A dictionary can optionally be passed with environment=, which is then
        available to the service during runtime."""
        self.__pipe_frontend = None
        self._environment = kwargs.get("environment", {})
        self._transport = None
        self.__callback_register = {}
        self.__service_status = None
        self.__shutdown = False
        self.__queue = queue.PriorityQueue()
        self._idle_callback = None
        self._idle_time = None

        # Logger will be overwritten in start() function
        self.log = logging.getLogger(self._logger_name)

    def __send_to_frontend(self, data_structure):
        """Put a message in the pipe for the frontend."""
        if self.__pipe_frontend:
            self.__pipe_frontend.send(data_structure)

    @property
    def config(self):
        return self._environment.get("config")

    @property
    def transport(self):
        return self._transport

    @transport.setter
    def transport(self, value):
        if self._transport:
            raise AttributeError("Transport already defined")
        self._transport = value

    def start_transport(self):
        """If a transport object has been defined then connect it now."""
        if self.transport:
            if self.transport.connect():
                self.log.debug("Service successfully connected to transport layer")
            else:
                raise RuntimeError("Service could not connect to transport layer")
            # direct all transport callbacks into the main queue
            self._transport_interceptor_counter = itertools.count()
            self.transport.subscription_callback_set_intercept(
                self._transport_interceptor
            )
            metrics = self._environment.get("metrics")
            if metrics:
                import prometheus_client
                from workflows.transport.middleware.prometheus import (
                    PrometheusMiddleware,
                )

                self.log.debug("Instrumenting transport")
                source = f"{self.__module__}:{self.__class__.__name__}"
                instrument = PrometheusMiddleware(source=source)
                self._transport.add_middleware(instrument)
                port = metrics["port"]
                self.log.debug(f"Starting metrics endpoint on port {port}")
                prometheus_client.start_http_server(port=port)
        else:
            self.log.debug("No transport layer defined for service. Skipping.")

    def stop_transport(self):
        """If a transport object has been defined then tear it down."""
        if self.transport:
            self.log.debug("Stopping transport object")
            self.transport.disconnect()

    def _transport_interceptor(self, callback):
        """Takes a callback function and returns a function that takes headers and
        messages and places them on the main service queue."""

        def add_item_to_queue(header, message):
            queue_item = (
                Priority.TRANSPORT,
                next(
                    self._transport_interceptor_counter
                ),  # insertion sequence to keep messages in order
                (callback, header, message),
            )
            self.__queue.put(
                queue_item
            )  # Block incoming transport until insertion completes

        return add_item_to_queue

    def connect(self, frontend):
        """Inject pipes connecting the service to the frontend. Two arguments are
        supported: frontend= for messages from the service to the frontend,
        and commands= for messages from the frontend to the service.
        The injection should happen before the service is started, otherwise the
        underlying file descriptor references may not be handled correctly."""
        self.__pipe_frontend = frontend

    def _log_send(self, logrecord):
        """Forward log records to the frontend."""
        self.__send_to_frontend({"band": "log", "payload": logrecord})

    def _register(self, message_band, callback):
        """Register a callback function for a specific message band."""
        self.__callback_register[message_band] = callback

    def _register_idle(self, idle_time, callback):
        """Register a callback function that is run when idling for a given
        time span (in seconds)."""
        self._idle_callback = callback
        self._idle_time = idle_time

    def _shutdown(self):
        """Terminate the service from the service side."""
        self.__shutdown = True

    def initialize_logging(self):
        """Reset the logging for the service process. All logged messages are
        forwarded to the frontend. If any filtering is desired, then this must
        take place on the service side."""
        # Reset logging to pass logrecords into the queue to the frontend only.
        # Existing handlers may be broken as they were copied into a new process,
        # so should be discarded.
        for loggername in [None] + list(logging.Logger.manager.loggerDict.keys()):
            logger = logging.getLogger(loggername)
            while logger.handlers:
                logger.removeHandler(logger.handlers[0])

        # Re-enable logging to console
        root_logger = logging.getLogger()

        # By default pass all warning (and higher) level messages to the frontend
        root_logger.setLevel(logging.WARN)
        root_logger.addHandler(workflows.logging.CallbackHandler(self._log_send))

        # Set up the service logger and pass all info (and higher) level messages
        # (or other level if set differently)
        self.log = logging.getLogger(self._logger_name)

        self.log.setLevel(logging.INFO)

        # Additionally, write all critical messages directly to console
        console = logging.StreamHandler()
        console.setLevel(logging.CRITICAL)
        root_logger.addHandler(console)

    def start(self, **kwargs):
        """Start listening to command queue, process commands in main loop,
        set status, etc...
        This function is most likely called by the frontend in a separate
        process."""

        # Keep a copy of keyword arguments for use in subclasses
        self.start_kwargs.update(kwargs)
        try:
            self.initialize_logging()
            self.start_transport()
            self.initializing()

            while not self.__shutdown:  # main loop
                try:
                    task = self.__queue.get(True, self._idle_time or 2)
                    run_idle_task = False
                except queue.Empty:
                    run_idle_task = True

                if self.transport and not self.transport.is_connected():
                    raise workflows.Disconnected("Connection lost")

                if run_idle_task:
                    if self._idle_time:
                        # run this outside the 'except' to avoid exception chaining
                        if self._idle_callback:
                            self._idle_callback()
                    continue

                if task[0] == Priority.COMMAND:
                    message = task[2]
                    if message and "band" in message:
                        processor = self.__callback_register.get(message["band"])
                        if processor is None:
                            self.log.warning(
                                "received message on unregistered band\n%s", message
                            )
                        else:
                            processor(message.get("payload"))
                    else:
                        self.log.warning(
                            "received message without band information\n%s", message
                        )
                elif task[0] == Priority.TRANSPORT:
                    callback, header, message = task[2]
                    callback(header, message)
                else:
                    self.log.warning("Unknown item on main service queue\n%r", task)

        except KeyboardInterrupt:
            self.log.warning("Ctrl+C detected. Shutting down.")

        except Exception as e:
            self.process_uncaught_exception(e)
            self.stop_transport()
            return

        try:
            self.stop_transport()
        except Exception as e:
            self.process_uncaught_exception(e)

    def process_uncaught_exception(self, e):
        """This is called to handle otherwise uncaught exceptions from the service.
        The service will terminate either way, but here we can do things such as
        gathering useful environment information and logging for posterity."""
        # Add information about the actual exception to the log message
        # This includes the file, line and piece of code causing the exception.
        # exc_info=True adds the full stack trace to the log message.
        (
            exc_file_fullpath,
            exc_file,
            exc_lineno,
            exc_func,
            exc_line,
        ) = workflows.logging.get_exception_source()
        added_information = {
            "workflows_exc_lineno": exc_lineno,
            "workflows_exc_funcName": exc_func,
            "workflows_exc_line": exc_line,
            "workflows_exc_pathname": exc_file_fullpath,
            "workflows_exc_filename": exc_file,
        }
        for field in filter(lambda x: x.startswith("workflows_log_"), dir(e)):
            added_information[field[14:]] = getattr(e, field, None)
        self.log.critical(
            "Unhandled service exception: %s", e, exc_info=True, extra=added_information
        )
