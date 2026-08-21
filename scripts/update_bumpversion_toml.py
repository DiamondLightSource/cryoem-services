"""
Script to update the '.bumpversion.toml' file whenever it is run as part of the pre-commit checks.
"""

import ast
from pathlib import Path

cwd = Path(__file__).parent.parent  # Start with 'cryoem-services' as the working dir


# Get the version without needing to install cryoemservices
def get_version(path: Path):
    tree = ast.parse(path.read_text())

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__version__":
                    return ast.literal_eval(node.value)
    raise ValueError(f"Could not find '__version__' in {path}")


version = get_version(cwd / "src/cryoemservices/__init__.py")

# Find Helm charts
helm_charts = [
    str(file.relative_to(cwd)) for file in sorted((cwd / "Helm").glob("**/Chart.yaml"))
]

# Construct lines to write to file
bumpversion_toml_lines = [
    "[tool.bumpversion]",
    f'current_version = "{version}"',
    "commit = true",
    "tag = true",
    "",
]
# Add sections for 'pyproject.toml' and 'cryoemservices.__init__'
bumpversion_toml_lines.extend(
    [
        "[[tool.bumpversion.files]]",
        'filename = "pyproject.toml"',
        "search = 'version = \"{current_version}\"'",
        "replace = 'version = \"{new_version}\"'",
        "",
        "[[tool.bumpversion.files]]",
        'filename = "src/cryoemservices/__init__.py"',
        "search = '__version__ = \"{current_version}\"'",
        "replace = '__version__ = \"{new_version}\"'",
        "",
    ]
)

# Iteratively add sections for Helm charts
for file in helm_charts:
    bumpversion_toml_lines.extend(
        [
            "[[tool.bumpversion.files]]",
            f'filename = "{file}"',
            "search = 'version: {current_version}'",
            "replace = 'version: {new_version}'",
            "",
        ]
    )

# Write to file
bumpversion_toml_file = cwd / ".bumpversion.toml"
with open(bumpversion_toml_file, "w") as f:
    f.write("\n".join(bumpversion_toml_lines))
