from __future__ import annotations

import sys

from cryoemservices.cli import project_linker


def test_project_linker(tmp_path):
    """Test that Relion-style projects are linked as expected"""

    original_project = tmp_path / "original_project"
    new_project = tmp_path / "new_project"

    # Some sample files and the expected behaviour of the linker on them
    dummy_project_files = {
        ".Nodes/NodeType/job/node": "copy",
        "MotionCorr/job002/file.star": "copy",
        "MotionCorr/job002/Movies/movie.mrc": "parent_link",
        "Class2D/job010/class.mrc": "copy",
        "Class2D/job010/class.jpeg": "ignore",
        "default_pipeline.star": "copy",
        ".gui_projectdir.star": "copy",
        "MotionCorr/MC_link": "symlink",
        "top_level_symlink": "symlink",
    }

    # Make a sample project to copy from
    for dummy_file in dummy_project_files.keys():
        (original_project / dummy_file).parent.mkdir(exist_ok=True, parents=True)
        if dummy_project_files[dummy_file] == "symlink":
            symlink_target = (original_project / dummy_file).parent / "link_target"
            symlink_target.mkdir()
            (original_project / dummy_file).symlink_to(symlink_target)
        else:
            (original_project / dummy_file).touch()
    external_link_target = tmp_path / "external_link_target"
    (original_project / "external_link").symlink_to(external_link_target)

    # Run the project linker
    sys.argv = [
        "relipy.link",
        "--project",
        str(original_project),
        "--destination",
        str(new_project),
    ]
    project_linker.run()

    # Check all the expected new files got made
    for dummy_file in dummy_project_files.keys():
        assert (new_project / dummy_file).exists()
        if dummy_project_files[dummy_file] == "copy":
            assert (new_project / dummy_file).is_file()
        elif dummy_project_files[dummy_file] == "parent_link":
            assert (new_project / dummy_file).parent.is_symlink()
        elif dummy_project_files[dummy_file] == "symlink":
            assert (new_project / dummy_file).is_symlink()
            assert (new_project / dummy_file).resolve() == (
                new_project / dummy_file
            ).parent / "link_target"
        elif dummy_project_files[dummy_file] == "ignore":
            assert not (new_project / dummy_file).exists()
        else:
            raise RuntimeError("Unknown file creation option")
    assert (new_project / "external_link").is_symlink()
    assert (new_project / "external_link").resolve() == external_link_target
