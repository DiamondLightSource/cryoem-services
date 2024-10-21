from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run():
    parser = argparse.ArgumentParser(
        description="Create a new project to manually continue from auto-processing"
    )
    parser.add_argument(
        "-p",
        "--project",
        help="Path to original project directory",
        dest="project",
    )
    parser.add_argument(
        "-d",
        "--destination",
        help="Path to directory where new project should be created",
        dest="destination",
        default=".",
    )
    parser.add_argument(
        "--hardlink-cutoff",
        type=int,
        help="File size in bytes past which to hardlink rather than copy",
        default=5e6,
    )
    args = parser.parse_args()

    required_dirs = (".Nodes", ".TMP_runfiles")

    ignore_file_extensions = {"Class2D": [".jpeg"]}

    project_path = Path(args.project).resolve()
    destination_path = Path(args.destination).resolve()
    destination_path.mkdir(exist_ok=True)

    for f in project_path.glob("*"):
        if (f.is_dir() and f.name in required_dirs) or f.is_file():
            subprocess.run(
                [
                    "rsync",
                    "--recursive",
                    str(f),
                    (
                        str(destination_path)
                        if f.is_dir()
                        else str(destination_path / f.relative_to(project_path))
                    ),
                ]
            )
        elif f.is_dir():
            job_type_dir = destination_path / f.relative_to(project_path)
            job_type_dir.mkdir(exist_ok=True)
            for job_dir in f.glob("*"):
                if not job_dir.is_symlink():
                    new_job_dir = destination_path / job_dir.relative_to(project_path)
                    new_job_dir.mkdir(exist_ok=True)
                    for jf in job_dir.glob("*"):
                        if jf.is_file():
                            if jf.suffix in ignore_file_extensions.get(f.name, []):
                                continue
                            file_linked = False
                            if jf.stat().st_size > args.hardlink_cutoff:
                                try:
                                    jf.hardlink_to(
                                        f"{destination_path / jf.relative_to(project_path)}"
                                    )
                                    file_linked = True
                                except OSError:
                                    file_linked = False
                            if not file_linked:
                                subprocess.run(
                                    [
                                        "rsync",
                                        str(jf),
                                        f"{destination_path / jf.relative_to(project_path)}",
                                    ]
                                )
                        else:
                            # All subdirectories in the job get linked
                            try:
                                (
                                    destination_path / jf.relative_to(project_path)
                                ).symlink_to(jf)
                            except FileExistsError:
                                pass
                else:
                    # Case of symlinks in the job directory
                    job_dir_source = job_dir.resolve()
                    try:
                        (destination_path / job_type_dir / job_dir.name).symlink_to(
                            job_dir_source
                        )
                    except FileExistsError:
                        pass
        elif f.is_symlink():
            # Case of symlinks in the project directory
            source = f.resolve()
            try:
                (destination_path / f.relative_to(project_path)).symlink_to(source)
            except FileExistsError:
                pass
