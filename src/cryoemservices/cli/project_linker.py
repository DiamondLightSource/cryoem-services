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
    parser.add_argument(
        "--skip-class2d",
        dest="skip_class2d",
        action="store_true",
        default=False,
        help="Optionally skip copying 2D classification stages, for a smaller project",
    )
    args = parser.parse_args()

    required_dirs = (".Nodes", ".TMP_runfiles")

    ignore_file_extensions = {"Class2D": [".jpeg"]}
    skip_class2d_ignore_jobs = {"Class2D": 1, "IceBreaker": 10}

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
        elif f.is_symlink():
            # Case of symlinks in the project directory
            source = f.resolve()
            try:
                (destination_path / f.relative_to(project_path)).symlink_to(
                    destination_path / source.relative_to(project_path)
                    if source.is_relative_to(project_path)
                    else source
                )
            except FileExistsError:
                pass
        elif f.is_dir():
            job_type_dir = destination_path / f.relative_to(project_path)
            job_type_dir.mkdir(exist_ok=True)
            for job_dir in f.glob("*"):
                if (
                    args.skip_class2d
                    and skip_class2d_ignore_jobs.get(job_type_dir.name, "")
                    and job_dir.name.startswith("job")
                ):
                    # Skip copying Class2D and IceBreaker particle jobs if requested
                    job_number = int(job_dir.name[3:])
                    if job_number >= skip_class2d_ignore_jobs[job_type_dir.name]:
                        continue

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
                    # Case of symlinks used as job aliases
                    job_dir_source = job_dir.resolve()
                    try:
                        (destination_path / job_type_dir / job_dir.name).symlink_to(
                            destination_path / job_dir_source.relative_to(project_path)
                        )
                    except FileExistsError:
                        pass

    if args.skip_class2d and (destination_path / "short_pipeline.star").is_file():
        # Set the pipeline which excludes 2D jobs as the default
        (destination_path / "default_pipeline.star").unlink()
        (destination_path / "short_pipeline.star").rename(
            destination_path / "default_pipeline.star"
        )
