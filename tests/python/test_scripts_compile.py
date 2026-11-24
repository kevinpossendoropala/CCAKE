"""Every Python file in the repository must at least be valid Python."""

import subprocess

import pytest

from conftest import REPO

SEARCHED = ("utilities", "misc", "plotting_scripts", "tests", "ci")


def python_files():
    """Files git knows about (tracked, or new and not ignored); a plain search otherwise."""
    listed = subprocess.run(
        [
            "git",
            "-C",
            str(REPO),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "*.py",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if listed.returncode == 0 and listed.stdout.strip():
        names = listed.stdout.split("\n")
        return sorted(name for name in names if name and (REPO / name).is_file())
    found = []
    for directory in SEARCHED:
        found += [str(path.relative_to(REPO)) for path in (REPO / directory).rglob("*.py")]
    return sorted(found)


FILES = python_files()


def test_there_are_python_files():
    assert FILES, "found no Python files to check"


@pytest.mark.parametrize("name", FILES)
def test_compiles(name):
    source = (REPO / name).read_text(encoding="utf-8")
    compile(source, name, "exec")
