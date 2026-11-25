"""Shared helpers for the Python tests.

The scripts under utilities/ are stand-alone programs, not a package, so the
tests either import one by its path or run it as a subprocess, the way a user
would.
"""

import importlib.util
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]

IC_COLUMNS = 17  # columns of the ccake initial-condition format


def import_script(relative_path):
    """Import a stand-alone script as a module, without running its main()."""
    path = REPO / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_script(relative_path, *arguments, cwd):
    """Run a script with the current interpreter and return the finished process."""
    return subprocess.run(
        [sys.executable, str(REPO / relative_path), *map(str, arguments)],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def read_ic(path):
    """Header line and data rows (lists of tokens) of an initial-condition file."""
    lines = pathlib.Path(path).read_text().splitlines()
    return lines[0], [line.split() for line in lines[1:] if line.strip()]


@pytest.fixture
def repo():
    return REPO
