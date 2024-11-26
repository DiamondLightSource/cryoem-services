from __future__ import annotations

import logging
from typing import NamedTuple, Optional

import ispyb.sqlalchemy
import sqlalchemy.exc

logger = logging.getLogger("cryoemservices.util.ispyb_buffer")
logger.setLevel(logging.INFO)


class BufferResult(NamedTuple):
    success: bool
    value: Optional[int]


def load(*, session, program: int, uuid: int) -> BufferResult:
    """Load an entry from the zc_ZocaloBuffer table.

    Given an AutoProcProgramID and a client-defined unique reference (uuid)
    retrieve a reference value from the database if possible.
    """
    query = (
        session.query(ispyb.sqlalchemy.ZcZocaloBuffer)
        .filter(ispyb.sqlalchemy.ZcZocaloBuffer.AutoProcProgramID == program)
        .filter(ispyb.sqlalchemy.ZcZocaloBuffer.UUID == uuid)
    )
    try:
        result = query.one()
        logger.info(
            f"buffer lookup for {program}.{uuid} succeeded (={result.Reference})"
        )
        return BufferResult(success=True, value=result.Reference)
    except sqlalchemy.exc.NoResultFound:
        logger.info(f"buffer lookup for {program}.{uuid} failed")
        return BufferResult(success=False, value=None)


def store(*, session, program: int, uuid: int, reference: int):
    """Write an entry into the zc_ZocaloBuffer table.

    The buffer table allows decoupling of the message-sending client
    and the database server-side assigned primary keys. The client defines
    a unique reference (uuid) that it will use to refer to a real primary
    key value (reference). All uuids are relative to an AutoProcProgramID
    and will be stored for a limited time based on the underlying
    AutoProcProgram record.
    """
    entry = ispyb.sqlalchemy.ZcZocaloBuffer(
        AutoProcProgramID=program,
        UUID=uuid,
        Reference=reference,
    )
    session.merge(entry)
    logger.info(f"buffering value {reference} for {program}.{uuid}")
    session.commit()
