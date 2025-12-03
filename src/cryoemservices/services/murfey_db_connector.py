from __future__ import annotations

from murfey.server.murfey_db import get_murfey_db_session
from workflows.recipe import wrap_subscribe

from cryoemservices.services.ispyb_connector import EMISPyB
from cryoemservices.util import ispyb_commands, murfey_db_commands


class MurfeyDBConnector(EMISPyB):
    """
    A service that receives information to be written to the Murfey database.
    Designed to override the ispyb connector in cases where ispyb is not available
    """

    # Logger name
    _logger_name = "cryoemservices.services.murfey_db_connector"

    def initializing(self):
        """Subscribe the ISPyB connector queue. Received messages must be
        acknowledged. Prepare Murfey database connection."""
        self._database_session_maker = get_murfey_db_session
        self.log.info("ISPyB service ready")
        wrap_subscribe(
            self._transport,
            self._environment["queue"] or "ispyb_connector",
            self.insert_into_ispyb,
            acknowledgement=True,
            allow_non_recipe_messages=True,
        )

    @staticmethod
    def get_command(command):
        if getattr(murfey_db_commands, command, None):
            return getattr(murfey_db_commands, command, None)
        return getattr(ispyb_commands, command, None)
