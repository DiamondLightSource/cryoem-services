from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import ispyb
import ispyb.sqlalchemy as models

# import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import pylatex
import sqlalchemy.orm
from gemmi import cif
from sqlalchemy.sql import func


class SessionResults:
    """
    session report
    - basic info e.g. pixel size, dose
    - collection stats e.g. number of grids, micrographs
    - preprocessing stats e.g. mean particle count, ctf max res, particle diameter
    - 2d classification example
    - 3d resolution and angular distribution (and fraction of nyquist)
    - refined resolution and bfactor
    - symmetry estimate

    internal only
    - comparison of resolutions to other sessions
    - comparison of res vs pixel size
    """

    def __init__(self, dc_id: int):
        # Ispyb IDs
        self.dc_id: int = dc_id
        self.ispyb_sessionmaker = sqlalchemy.orm.sessionmaker(
            bind=sqlalchemy.create_engine(
                ispyb.sqlalchemy.url(), connect_args={"use_pure": True}
            )
        )
        with self.ispyb_sessionmaker() as session:
            self.image_directory = (
                session.query(models.DataCollection)
                .filter(models.DataCollection.dataCollectionId == dc_id)
                .one()
                .imageDirectory
            )
        self.visit_name = self.image_directory.split("/")[5]
        self.raw_name = self.image_directory.split("/")[6]

        self.number_of_grid_squares = len(
            list(Path(self.image_directory).glob("GridSquare*"))
        )

        # Get all processing jobs
        self.autoproc_ids = []
        self.processing_stages = []
        with self.ispyb_sessionmaker() as session:
            self.processing_jobs = (
                session.query(models.ProcessingJob)
                .filter(models.ProcessingJob.dataCollectionId == dc_id)
                .all()
            )
            for job in self.processing_jobs:
                self.processing_stages.append(job.recipe)
                self.autoproc_ids.append(
                    session.query(models.AutoProcProgram)
                    .filter(
                        models.AutoProcProgram.processingJobId == job.processingJobId
                    )
                    .one()
                    .autoProcProgramId,
                )

        # Collection parameters
        self.pixel_size: float = 0
        self.image_size: tuple = (0, 0)
        self.voltage: int = 300
        self.c2_aperture: float = 50
        self.slit_width: float = 20
        self.magnification: int = 0
        self.total_dose: float = 0
        self.exposure_time: float = 0
        self.dose_per_frame: float = 0
        self.frame_count: int = 0

        # Preprocessing results
        self.micrograph_count: int = 0
        self.example_micrographs: List[str] = []
        self.example_picks: List[str] = []
        self.mean_picks: float = 0
        self.median_ctf_resolution: float = 0
        self.particle_diameter: float = 0

        # Classification results
        self.class2d_batches: int = 0
        self.binned_pixel_size: float = 0
        self.example_class2d: List[str] = []
        self.class3d_batch: int = 0
        self.provided_symmetry: str = "C1"
        self.class3d_images: List[str] = []
        self.class3d_particles: List[int] = []
        self.class3d_resolution: List[float] = []
        self.class3d_completeness: List[float] = []
        self.class3d_flat_x: np.array = np.array([])
        self.class3d_flat_y: np.array = np.array([])
        self.class3d_flat_z: np.array = np.array([])
        self.class3d_angdist: str = ""
        self.refined_batch: int = 0
        self.refined_symmetry: List[str] = []
        self.refined_resolution: List[float] = []
        self.refined_completeness: List[float] = []
        self.bfactor: List[float] = []

    def write_report(self):
        doc = pylatex.Document(
            geometry_options={
                "tmargin": "2cm",
                "bmargin": "2cm",
                "lmargin": "2cm",
                "rmargin": "2cm",
            }
        )
        doc.preamble.append(
            pylatex.Command(
                "title", f"Auto-processing report for {self.visit_name} {self.raw_name}"
            )
        )
        doc.preamble.append(pylatex.Command("date", pylatex.NoEscape(r"\today")))
        doc.append(pylatex.NoEscape(r"\maketitle"))

        with doc.create(pylatex.Section("Data collection parameters")):
            doc.append("The following parameters were set during data collection:\n")
            with doc.create(pylatex.Table(position="h!")) as table_environment:
                table_environment.append(pylatex.NoEscape(r"\centering"))
                table_environment.append(pylatex.NoEscape(r"\label{collection_params}"))
                table_environment.add_caption("Parameters used for data collection")
                with doc.create(pylatex.Tabular("|c|c|")) as table:
                    table.add_hline()
                    table.add_row(("Parameter", "Value"))
                    table.add_hline()
                    table.add_row(("Voltage", f"{self.voltage} keV"))
                    table.add_row(("Magnification", self.magnification))
                    table.add_row(
                        (
                            "Pixel size",
                            pylatex.NoEscape(str(self.pixel_size * 10) + r" $\AA$"),
                        )
                    )
                    table.add_row(
                        ("Image size", f"{self.image_size[0]} x {self.image_size[1]}")
                    )
                    table.add_row(("Exposure time", f"{self.exposure_time} s"))
                    table.add_row(("Number of frames", self.frame_count))
                    table.add_row(
                        (
                            "Dose per frame",
                            pylatex.NoEscape(
                                str(self.dose_per_frame) + r" $e^- / \AA ^2$"
                            ),
                        )
                    )
                    table.add_row(
                        (
                            "C2 aperture size",
                            pylatex.NoEscape(str(self.c2_aperture) + r" ${\mu}$m"),
                        )
                    )
                    table.add_row(("Slit width", f"{self.slit_width} eV"))
                    table.add_hline()

        with doc.create(pylatex.Section("Pre-processing")):
            doc.append(
                f"A total of {self.micrograph_count} micrographs were then collected "
                f"across {self.number_of_grid_squares} grid squares.\n"
            )
            doc.append(
                pylatex.NoEscape(
                    rf"These had a median CTF max resolution of {self.median_ctf_resolution} $\AA$"
                )
            )
            doc.append(
                pylatex.NoEscape(
                    f"\nParticle picking gave a mean of {self.mean_picks} particles "
                    "per micrograph, and an estimated particle diameter of "
                    rf"{self.particle_diameter} $\AA$."
                )
            )

            with doc.create(pylatex.Figure(position="h")) as micrograph_image:
                micrograph_image.add_image(self.example_micrographs[0], width="200px")
                micrograph_image.append(pylatex.NoEscape(r"\hspace{20px}"))
                micrograph_image.add_image(self.example_micrographs[1], width="200px")
                micrograph_image.add_caption(
                    "The first and last motion corrected micrographs"
                )

            with doc.create(pylatex.Figure(position="h")) as pick_image:
                pick_image.add_image(self.example_picks[0], width="200px")
                pick_image.append(pylatex.NoEscape(r"\hspace{20px}"))
                pick_image.add_image(self.example_picks[1], width="200px")
                pick_image.add_caption(
                    "The first and last motion micrographs, with particle picks overlaid"
                )

        with doc.create(pylatex.Section("Particle classification")):
            doc.append(
                pylatex.NoEscape(
                    "Before classification the particles are binned to a pixel size "
                    r"that gives a Nyquist frequency of around 8.5 $\AA$. "
                    "The binned pixel size for this collection was "
                    rf"{self.binned_pixel_size} $\AA$."
                )
            )
            doc.append(pylatex.NoEscape("\n\n"))
            doc.append(
                pylatex.NoEscape(
                    f"{self.class2d_batches} batches of 2D classification were run, "
                    "with 50,000 particles in each batch. "
                    r"Figure \ref{class3d_table} shows some examples of the classes "
                    "which were generated. "
                )
            )
            doc.append(pylatex.NoEscape("\n"))

            with doc.create(pylatex.Figure(position="h")) as class2d_image:
                for i in range(min(len(self.example_class2d), 18)):
                    # Display up to 18 examples of 2d classes
                    class2d_image.add_image(self.example_class2d[i], width="50px")
                    class2d_image.append(pylatex.NoEscape(r"\hspace{1px}"))
                class2d_image.add_caption(
                    "The most populous two classes from some 2D classification batches"
                )
                class2d_image.append(pylatex.NoEscape(r"\label{class2d_images}"))

            doc.append(
                pylatex.NoEscape(
                    f"3D classification was run up to {self.class3d_batch} particles "
                    f"using a symmetry of {self.provided_symmetry}. "
                    r"The classes produced are given in table \ref{class3d_table}"
                )
            )
            doc.append(pylatex.NoEscape("\n"))
            with doc.create(pylatex.Table(position="h!")) as table_environment:
                table_environment.append(pylatex.NoEscape(r"\centering"))
                table_environment.append(pylatex.NoEscape(r"\label{class3d_table}"))
                table_environment.add_caption("3D classification results")
                with doc.create(pylatex.Tabular("|c|c|c|c|")) as table:
                    table.add_hline()
                    table.add_row(
                        (
                            "Class number",
                            "Number of particles",
                            "Resolution",
                            "Fourier completeness",
                        )
                    )
                    table.add_hline()
                    for class3d in range(len(self.class3d_particles)):
                        table.add_row(
                            (
                                class3d + 1,
                                self.class3d_particles[class3d],
                                pylatex.NoEscape(
                                    rf"{self.class3d_resolution[class3d]} $\AA$"
                                ),
                                self.class3d_completeness[class3d],
                            )
                        )
                    table.add_hline()

            # with doc.create(pylatex.Figure(position="h")) as projections_3d:
            #    plt.imshow(self.class3d_flat_x)
            #    projections_3d.add_plot(width="75px")
            #    plt.imshow(self.class3d_flat_y)
            #    projections_3d.add_plot(width="75px")
            #    plt.imshow(self.class3d_flat_z)
            #    projections_3d.add_plot(width="75px")
            #    projections_3d.add_caption("Projections of the best 3D class")

            with doc.create(pylatex.Figure(position="h")) as angdist_image:
                angdist_image.add_image(self.class3d_angdist, width="200px")
                angdist_image.add_caption(
                    "The distribution of particle angles for the best 3D class"
                )

            if self.refined_batch:
                doc.append(pylatex.NoEscape("\n\n"))
                doc.append(
                    pylatex.NoEscape(
                        r"Refinement of the 3D structure was run using "
                        rf"a pixel size of {self.pixel_size * 20} $\AA$ "
                        f"and {self.refined_batch} particles."
                    )
                )
                doc.append(pylatex.NoEscape("\n"))
                if len(self.refined_symmetry) == 1:
                    refine1 = 0
                else:
                    refine1 = np.where(self.refined_symmetry == "C1")[0][0]

                doc.append(
                    pylatex.NoEscape(
                        rf"A final resolution of {self.refined_resolution[refine1]} $\AA$ "
                        f"was obtained with completeness {self.refined_completeness[refine1]}."
                    )
                )
                doc.append(pylatex.NoEscape("\n"))
                doc.append(
                    f"An estimated B-factor is {self.bfactor[refine1]}, "
                    f"but we caution that masks are not optimised and "
                    f"the performance of earlier automated processing steps "
                    f"will affect the results, "
                    f"so you should be able to improve upon this.\n"
                )

                if len(self.refined_symmetry) == 2:
                    symm_refine = 1 - refine1
                    doc.append(
                        pylatex.NoEscape(
                            "Following refinement, we estimate that the symmetry "
                            f"of the sample is {self.refined_symmetry[symm_refine]}. "
                            "Using this symmetry refinement gives a final resolution "
                            rf"of {self.refined_resolution[refine1]} $\AA$ "
                            f"and completeness {self.refined_completeness[refine1]}."
                        )
                    )
                    doc.append(pylatex.NoEscape("\n"))

        doc.generate_pdf("report")

    def gather_preprocessing_ispyb_results(self):
        # Basic collection information
        with self.ispyb_sessionmaker() as session:
            data_collection = (
                session.query(models.DataCollection)
                .filter(models.DataCollection.dataCollectionId == self.dc_id)
                .one()
            )
            self.pixel_size = data_collection.pixelSizeOnImage
            self.image_size = (data_collection.imageSizeX, data_collection.imageSizeY)
            self.total_dose = data_collection.totalExposedDose
            self.exposure_time = data_collection.exposureTime
            self.voltage = data_collection.voltage
            self.c2_aperture = data_collection.c2aperture
            self.magnification = data_collection.magnification
            self.slit_width = data_collection.slitGapHorizontal

            # Collection stats
            self.micrograph_count = (
                session.query(func.count(models.MotionCorrection.motionCorrectionId))
                .filter(models.MotionCorrection.dataCollectionId == self.dc_id)
                .one()[0]
            )

            first_mc_id = (
                session.query(models.MotionCorrection)
                .filter(models.MotionCorrection.dataCollectionId == self.dc_id)
                .all()
            )
            self.dose_per_frame = first_mc_id[0].dosePerFrame
            self.frame_count = first_mc_id[0].lastFrame
            self.example_micrographs = [
                first_mc_id[0].micrographSnapshotFullPath,
                first_mc_id[-1].micrographSnapshotFullPath,
            ]

            self.example_picks = [
                session.query(models.ParticlePicker)
                .filter(
                    models.ParticlePicker.firstMotionCorrectionId
                    == first_mc_id[0].motionCorrectionId
                )
                .one()
                .summaryImageFullPath,
                session.query(models.ParticlePicker)
                .filter(
                    models.ParticlePicker.firstMotionCorrectionId
                    == first_mc_id[-1].motionCorrectionId
                )
                .one()
                .summaryImageFullPath,
            ]

            # Preprocessing information
            preprocessing_loc = np.where(
                np.array(self.processing_stages) == "em-spa-preprocess"
            )[0][0]
            preprocess_program_id = self.autoproc_ids[preprocessing_loc]
            if preprocess_program_id:
                self.median_ctf_resolution = (
                    session.query(
                        func.percentile_disc(0.5)
                        .within_group(models.CTF.estimatedResolution)
                        .over(partition_by=models.CTF.autoProcProgramId)
                    )
                    .filter(models.CTF.autoProcProgramId == preprocess_program_id)
                    .first()[0]
                )

                self.mean_picks = (
                    session.query(func.avg(models.ParticlePicker.numberOfParticles))
                    .filter(models.ParticlePicker.programId == preprocess_program_id)
                    .one()[0]
                )

                self.particle_diameter = (
                    session.query(models.ParticlePicker)
                    .filter(models.ParticlePicker.programId == preprocess_program_id)
                    .filter(models.ParticlePicker.particleDiameter > 0)
                    .first()
                    .particleDiameter
                )
            else:
                print("Cannot find preprocessing job")

    def gather_classification_ispyb_results(self):
        with self.ispyb_sessionmaker() as session:
            # 2D classification information
            class2d_loc = np.where(
                np.array(self.processing_stages) == "em-spa-class2d"
            )[0][0]
            class2d_program_id = self.autoproc_ids[class2d_loc]
            if class2d_program_id:
                class2d_all_groups = (
                    session.query(models.ParticleClassificationGroup)
                    .filter(
                        models.ParticleClassificationGroup.programId
                        == class2d_program_id
                    )
                    .all()
                )

                self.example_class2d = []
                if class2d_all_groups:
                    for class2d_group in class2d_all_groups:
                        class2d_classes = (
                            session.query(models.ParticleClassification)
                            .filter(
                                models.ParticleClassification.particleClassificationGroupId
                                == class2d_group.particleClassificationGroupId
                            )
                            .filter(models.ParticleClassification.selected == 1)
                            .order_by(
                                models.ParticleClassification.particlesPerClass.desc()
                            )
                            .all()
                        )

                        if class2d_classes:
                            self.example_class2d.append(
                                class2d_classes[0].classImageFullPath
                            )
                            self.example_class2d.append(
                                class2d_classes[1].classImageFullPath
                            )
            else:
                print("Cannot find Class2D job")

            if self.example_class2d:
                # Find the classification pixel size
                class2d_job = Path(self.example_class2d[0]).parent
                class2d_doc = cif.read_file(f"{class2d_job}/run_it025_data.star")
                self.binned_pixel_size = float(
                    list(
                        class2d_doc.find_block("optics").find_loop("_rlnImagePixelSize")
                    )[0]
                )
                self.class2d_batches = len(list(class2d_job.parent.glob("job*")))

            # 3D classification information
            class3d_loc = np.where(
                np.array(self.processing_stages) == "em-spa-class3d"
            )[0][0]
            class3d_program_id = self.autoproc_ids[class3d_loc]
            if class3d_program_id:
                class3d_group = (
                    session.query(models.ParticleClassificationGroup)
                    .filter(
                        models.ParticleClassificationGroup.programId
                        == class3d_program_id
                    )
                    .first()
                )

                if class3d_group:
                    self.class3d_batch = class3d_group.numberOfParticlesPerBatch
                    self.provided_symmetry = class3d_group.symmetry

                    class3d_classes = (
                        session.query(models.ParticleClassification)
                        .filter(
                            models.ParticleClassification.particleClassificationGroupId
                            == class3d_group.particleClassificationGroupId
                        )
                        .all()
                    )

                    if class3d_classes:
                        for class3d in class3d_classes:
                            self.class3d_particles.append(class3d.particlesPerClass)
                            self.class3d_resolution.append(class3d.estimatedResolution)
                            self.class3d_completeness.append(
                                class3d.overallFourierCompleteness
                            )
                            self.class3d_images.append(class3d.classImageFullPath)

                        if len(self.class3d_resolution) > 0:
                            best_image = self.class3d_images[
                                self.class3d_resolution == min(self.class3d_resolution)
                            ]
                            self.class3d_angdist = (
                                str(Path(best_image).parent / Path(best_image).stem)
                                + "_angdist.jpeg"
                            )
                            with mrcfile.open(best_image) as mrc:
                                image3d = mrc.data
                            self.class3d_flat_x = np.sum(image3d, axis=0)
                            self.class3d_flat_y = np.sum(image3d, axis=1)
                            self.class3d_flat_z = np.sum(image3d, axis=2)
            else:
                print("Cannot find Class3D job")

            # Refinement information
            refine_loc = np.where(np.array(self.processing_stages) == "em-spa-refine")[
                0
            ][0]
            refine_program_id = self.autoproc_ids[refine_loc]
            if refine_program_id:
                refine_group = (
                    session.query(models.ParticleClassificationGroup)
                    .filter(
                        models.ParticleClassificationGroup.programId
                        == refine_program_id
                    )
                    .all()
                )

                if refine_group:
                    self.refined_batch = refine_group[0].numberOfParticlesPerBatch

                    for refine3d in refine_group:
                        refine_class = (
                            session.query(models.ParticleClassification)
                            .filter(
                                models.ParticleClassification.particleClassificationGroupId
                                == refine3d.particleClassificationGroupId
                            )
                            .one()
                        )
                        self.refined_symmetry.append(refine3d.symmetry)
                        self.refined_resolution.append(refine_class.estimatedResolution)
                        self.refined_completeness.append(
                            refine_class.overallFourierCompleteness
                        )
                        self.bfactor.append(refine_class.bFactorFitLinear)
            else:
                print("Cannot find refinement job")


def run():
    parser = argparse.ArgumentParser(
        description="Generate a report for a data collection"
    )
    parser.add_argument(
        "--dc_id",
        help="Data collection ID",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--internal",
        help="Run comparison to other sessions",
        default=False,
        action="store_true",
        required=False,
    )
    args = parser.parse_args()

    results = SessionResults(args.dc_id)
    results.gather_preprocessing_ispyb_results()
    results.gather_classification_ispyb_results()
    results.write_report()


if __name__ == "__main__":
    run()
