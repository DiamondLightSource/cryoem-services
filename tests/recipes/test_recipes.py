from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from cryoemservices.services.bfactor_setup import BFactorParameters
from cryoemservices.services.cryolo import CryoloParameters
from cryoemservices.services.ctffind import CTFParameters
from cryoemservices.services.denoise import DenoiseParameters
from cryoemservices.services.extract import ExtractParameters
from cryoemservices.services.extract_class import ExtractClassParameters
from cryoemservices.services.icebreaker import IceBreakerParameters
from cryoemservices.services.membrain_seg import MembrainSegParameters
from cryoemservices.services.motioncorr import MotionCorrParameters
from cryoemservices.services.postprocess import PostProcessParameters
from cryoemservices.services.select_classes import SelectClassesParameters
from cryoemservices.services.select_particles import SelectParticlesParameters
from cryoemservices.services.tomo_align import TomoParameters
from cryoemservices.wrappers.class2d_wrapper import Class2DParameters
from cryoemservices.wrappers.class3d_wrapper import Class3DParameters
from cryoemservices.wrappers.refine3d_wrapper import RefineParameters

try:
    from cryoemservices.services.node_creator import NodeCreatorParameters
except ImportError:

    class NodeCreatorParameters(BaseModel):
        job_type: str
        input_file: str
        output_file: str


class ISPyBParameters(BaseModel):
    dcid: int
    image_number: int
    ispyb_command: str
    message: str
    movie_id: int
    movie_path: str
    program_id: int
    status: str
    store_result: str
    tomogram_id: int


class MurfeyParameters(BaseModel):
    program_id: int
    session_id: int


FIXTURE_DIR = Path(__file__).parent.parent.parent.resolve()
known_services = {
    "BFactor": BFactorParameters,
    "Class2DWrapper": Class2DParameters,
    "Class3DWrapper": Class3DParameters,
    "CrYOLO": CryoloParameters,
    "CTFFind": CTFParameters,
    "Denoise": DenoiseParameters,
    "EMISPyB": ISPyBParameters,
    "Extract": ExtractParameters,
    "ExtractClass": ExtractClassParameters,
    "IceBreaker": IceBreakerParameters,
    "Images": BaseModel,
    "MembrainSeg": MembrainSegParameters,
    "MotionCorr": MotionCorrParameters,
    "Murfey": MurfeyParameters,
    "NodeCreator": NodeCreatorParameters,
    "PostProcess": PostProcessParameters,
    "RefineWrapper": RefineParameters,
    "SelectClasses": SelectClassesParameters,
    "SelectParticles": SelectParticlesParameters,
    "TomoAlign": TomoParameters,
}


@pytest.mark.datafiles(
    FIXTURE_DIR / "recipes/em-spa-bfactor.json",
    FIXTURE_DIR / "recipes/em-spa-class2d.json",
    FIXTURE_DIR / "recipes/em-spa-class3d.json",
    FIXTURE_DIR / "recipes/em-spa-extract.json",
    FIXTURE_DIR / "recipes/em-spa-preprocess.json",
    FIXTURE_DIR / "recipes/em-spa-refine.json",
    FIXTURE_DIR / "recipes/em-tomo-align.json",
    FIXTURE_DIR / "recipes/em-tomo-preprocess.json",
)
def test_spa_preprocess_recipe(datafiles):
    """Test for the service names and parameter keys in the recipes"""
    for recipe_name in datafiles.glob("em-*"):
        with open(recipe_name, "r") as json_data:
            spa_preprocess_recipe = json.load(json_data)

        recipe_steps = spa_preprocess_recipe.keys()
        for step in recipe_steps:
            if step == "start":
                continue

            service_name = spa_preprocess_recipe[step]["service"]
            assert service_name in list(known_services.keys())

            model_type = known_services[service_name]
            if not model_type:
                continue

            model_parameters = model_type.model_fields
            service_parameters = (
                spa_preprocess_recipe[step]
                .get(
                    "job_parameters", spa_preprocess_recipe[step].get("parameters", {})
                )
                .keys()
            )

            for parameter in service_parameters:
                assert parameter in model_parameters
