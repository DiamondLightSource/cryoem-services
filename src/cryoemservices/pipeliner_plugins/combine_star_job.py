from __future__ import annotations

from pathlib import Path

from pipeliner.data_structure import NODE_PARTICLEGROUPMETADATA, SELECT_DIR
from pipeliner.display_tools import mini_montage_from_starfile
from pipeliner.job_options import BooleanJobOption, IntJobOption, StringJobOption
from pipeliner.nodes import Node
from pipeliner.pipeliner_job import ExternalProgram, PipelinerJob

try:
    from pipeliner.pipeliner_job import PipelinerCommand
except ImportError:
    PipelinerCommand = None

COMBINE_STAR_NAME = "combine_star_files"


class ProcessStarFiles(PipelinerJob):
    PROCESS_NAME = "combine_star_files_job"
    OUT_DIR = SELECT_DIR

    def __init__(self):
        super().__init__()

        self.jobinfo.display_name = "Particle star file merging"
        self.jobinfo.short_desc = "Combine and split star files of particles"
        self.jobinfo.long_desc = (
            "Combine star files of particles, then optionally split them again."
        )

        self.jobinfo.programs = [ExternalProgram(command=COMBINE_STAR_NAME)]

        self.joboptions["files_to_process"] = StringJobOption(
            label="The star files from which to combine particles",
            is_required=True,
            help_text="The names of the star files, separated by spaces.",
        )
        self.joboptions["do_split"] = BooleanJobOption(
            label="Whether to split the combined star file", default_value=False
        )
        self.joboptions["n_files"] = IntJobOption(
            label="Number of files to split the combined file into",
            default_value=-1,
            help_text=(
                "Provide either the number of files to split into,"
                " or the number of particles per file."
            ),
            deactivate_if=[("do_split", "is", "False")],
        )
        self.joboptions["split_size"] = IntJobOption(
            label="Number of particles to put in each split",
            default_value=-1,
            help_text=(
                "Provide either the number of files to split into,"
                " or the number of particles per file."
            ),
            deactivate_if=[("do_split", "is", "False")],
        )

        self.set_joboption_order(
            ["files_to_process", "do_split", "n_files", "split_size"]
        )

    def get_commands(self):
        """Construct the command for combining and splitting star files"""
        command = [COMBINE_STAR_NAME]
        file_list = self.joboptions["files_to_process"].get_string().split(" ")
        command.extend(file_list)
        command.extend(["--output_dir", str(self.output_dir)])

        if self.joboptions["do_split"].get_boolean():
            command.extend(["--split"])

            if (
                self.joboptions["n_files"].get_number() <= 0
                and self.joboptions["split_size"].get_number() <= 0
            ):
                raise ValueError(
                    "ERROR: When splitting the combined STAR file into subsets,"
                    " set n_files or split_size to a positive value"
                )

            if self.joboptions["n_files"].get_number() > 0:
                command.extend(["--n_files", self.joboptions["n_files"].get_string()])

            if self.joboptions["split_size"].get_number() > 0:
                command.extend(
                    ["--split_size", self.joboptions["split_size"].get_string()]
                )

        # Add files as input nodes, as long as they are not also the output node
        for particle_file in file_list:
            if particle_file != str(Path(self.output_dir) / "particles_all.star"):
                self.input_nodes.append(Node(particle_file, NODE_PARTICLEGROUPMETADATA))
        self.output_nodes.append(
            Node(
                str(Path(self.output_dir) / "particles_all.star"),
                NODE_PARTICLEGROUPMETADATA,
            )
        )

        if PipelinerCommand is None:
            return [command]
        pipeliner_commands = [PipelinerCommand([command], relion_control=False)]
        return pipeliner_commands

    def create_output_nodes(self):
        self.add_output_node(
            "particles_all.star", NODE_PARTICLEGROUPMETADATA, ["relion"]
        )

    def post_run_actions(self):
        """Find any output files produced by the splitting"""
        output_files = Path(self.output_dir).glob("particles_split*.star")
        for split in output_files:
            self.output_nodes.append(Node(str(split), NODE_PARTICLEGROUPMETADATA))

    def create_results_display(self):
        with open(
            Path(self.output_dir) / "class_averages.star", "r"
        ) as all_classes, open(
            Path(self.output_dir) / ".class_display_tmp.star", "w"
        ) as display_classes:
            for line in range(200):
                class_line = all_classes.readline()
                if not class_line:
                    break
                display_classes.write(class_line)
        with open(
            Path(self.output_dir) / "particles_all.star", "r"
        ) as all_particles, open(
            Path(self.output_dir) / ".particles_display_tmp.star", "w"
        ) as display_particles:
            for line in range(200):
                particles_line = all_particles.readline()
                if not particles_line:
                    break
                display_particles.write(particles_line)
        output_dobs = [
            mini_montage_from_starfile(
                starfile=str(Path(self.output_dir) / ".class_display_tmp.star"),
                block="",
                column="_rlnReferenceImage",
                outputdir=self.output_dir,
                nimg=100,
                title="Selected 2D classes",
            ),
            mini_montage_from_starfile(
                starfile=str(Path(self.output_dir) / ".particles_display_tmp.star"),
                block="particles",
                column="_rlnImageName",
                outputdir=self.output_dir,
                nimg=20,
                title="Examples of selected particles",
            ),
        ]
        (Path(self.output_dir) / ".class_display_tmp.star").unlink()
        (Path(self.output_dir) / ".particles_display_tmp.star").unlink()
        return output_dobs
