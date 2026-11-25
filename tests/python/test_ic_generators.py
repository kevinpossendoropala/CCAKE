"""The initial-condition generators write files that ccake can read.

These tests check the file format (header, number of rows and columns, finite
numbers) and a few exact properties. They do not judge the physics.
"""

import math

import pytest

from conftest import IC_COLUMNS, import_script, read_ic, run_script


def assert_well_formed(rows, expected_rows):
    assert len(rows) == expected_rows
    assert {len(row) for row in rows} == {IC_COLUMNS}
    for row in rows:
        for token in row:
            assert math.isfinite(float(token)), token


def test_long_generator(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (tmp_path / "long_data").mkdir()
    finished = run_script("utilities/long_ic_generator.py", "1.0", cwd=run_dir)
    assert finished.returncode == 0, finished.stdout + finished.stderr
    header, rows = read_ic(tmp_path / "long_data" / "ic_long_tau_1.0.dat")
    assert header == "#0 0.1 0.1 0.01 0 -0.1 -0.1 -10"
    assert_well_formed(rows, 2001)
    assert rows[0][2] == "-10.0" and rows[-1][2] == "10.0"


def test_bjorken_generator(tmp_path):
    pytest.importorskip("pandas")  # the script imports it
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (tmp_path / "bjorken").mkdir()
    finished = run_script("utilities/bjorken_ic_generator.py", "1.0", cwd=run_dir)
    assert finished.returncode == 0, finished.stdout + finished.stderr
    header, rows = read_ic(tmp_path / "bjorken" / "bjorken_tau_1.0.dat")
    assert header == "#0 0.1 0.1 0.001 0 -0.1 -0.1 -10"
    assert_well_formed(rows, 2001)
    # a uniform slab at rest
    assert {row[3] for row in rows} == {"10"}
    assert {row[9] for row in rows} == {"0"}


def test_ideal_gubser_functions():
    gubser = import_script("utilities/ideal_gubser_ic_generator.py")
    # the fluid is at rest on the axis
    assert gubser.velocity_x(0.0, 0.0) == 0
    assert gubser.velocity_y(0.0, 0.0) == 0
    # the flow is radial: mirrored points have mirrored velocities
    for x, y in ((0.5, 0.0), (1.0, 2.0), (3.0, -4.0)):
        r = math.hypot(x, y)
        assert gubser.velocity_x(-x, r) == -gubser.velocity_x(x, r)
        assert gubser.velocity_y(-y, r) == -gubser.velocity_y(y, r)
    # the energy density is positive and falls off away from the axis
    assert gubser.eps_a(0.0) > gubser.eps_a(1.0) > gubser.eps_a(5.0) > 0


def test_ideal_gubser_generator(tmp_path):
    profile = tmp_path / "profile.dat"
    profile.write_text(
        "# x y T ux uy pixx piyy pixy pizz\n"
        "0.0 0.0 0.2 0 0 0 0 0 0\n"
        "0.5 0.0 0.2 0 0 0 0 0 0\n"
        "0.0 -1.5 0.2 0 0 0 0 0 0\n"
    )
    output = tmp_path / "ic.dat"
    finished = run_script("utilities/ideal_gubser_ic_generator.py", profile, output, cwd=tmp_path)
    assert finished.returncode == 0, finished.stdout + finished.stderr
    header, rows = read_ic(output)
    assert header.startswith("#0 ")
    assert_well_formed(rows, 3)
    assert [row[:2] for row in rows] == [["0.0", "0.0"], ["0.5", "0.0"], ["0.0", "-1.5"]]


def test_gubser_converter(tmp_path):
    profile = tmp_path / "profile.dat"
    profile.write_text(
        "# x y T ux uy pixx piyy pixy pizz\n"
        "-5.0 -5.0 0.01 -0.9 -0.9 1e-6 2e-6 3e-6 4e-6\n"
        "0.0 0.0 0.2 0.0 0.0 0.0 0.0 0.0 0.0\n"
    )
    output = tmp_path / "ic.dat"
    finished = run_script("utilities/gubser_ic_converter.py", profile, output, cwd=tmp_path)
    assert finished.returncode == 0, finished.stdout + finished.stderr
    header, rows = read_ic(output)
    assert header == "#0 0.05 0.05 0.1 0 -5 -5 -0.1"
    assert_well_formed(rows, 2)
    # positions, velocities and shear components are copied through unchanged
    assert rows[0][0:2] == ["-5.0", "-5.0"]
    assert rows[0][7:9] == ["-0.9", "-0.9"]
    assert [rows[0][i] for i in (11, 12, 14, 16)] == ["1e-6", "3e-6", "2e-6", "4e-6"]
    # a hotter cell has a larger energy density
    assert float(rows[1][3]) > float(rows[0][3]) > 0
