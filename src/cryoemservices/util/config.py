from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict


class ServiceConfig(BaseModel):
    rabbitmq_credentials: Path
    recipe_directory: Path
    transport_type: str = "PikaTransport"
    slurm_credentials: Optional[Path] = None
    slurm_cluster: Optional[str] = ""

    model_config = ConfigDict(extra="allow")


def config_from_file(config_file_path: Path) -> ServiceConfig:
    with open(config_file_path, "r") as config_stream:
        config = yaml.safe_load(config_stream)
    return ServiceConfig(**config)
