from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, model_validator


class ServiceConfig(BaseModel):
    rabbitmq_credentials: Path
    recipe_directory: Path
    ispyb_credentials: Optional[Path] = None
    slurm_credentials: dict[str, Path] = {}
    graylog_host: str = ""
    graylog_port: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def check_port_present_if_host_is(cls, values: dict) -> Optional[dict]:
        if values.get("graylog_host") and not values.get("graylog_port"):
            raise ValueError("The Graylog port must be set if the Graylog host is")
        return values

    model_config = ConfigDict(extra="allow")


def config_from_file(config_file_path: Path) -> ServiceConfig:
    with open(config_file_path, "r") as config_stream:
        config = yaml.safe_load(config_stream)
    return ServiceConfig(**config)
