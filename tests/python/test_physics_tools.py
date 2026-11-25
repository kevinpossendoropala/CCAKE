"""The comparison code of the physics checks (tests/physics/physics_tools.py).

Small surfaces and snapshots with known answers, written in the CCAKE output formats.
"""

import math

import numpy as np
import pytest
from conftest import import_script

pt = import_script("tests/physics/physics_tools.py")


def write_surface(path, dimension, rows, header="1.5 0 -0.25"):
    """A freeze_out.dat with the given {column name: value} per row."""
    layout = pt.SURFACE_COLUMNS[dimension]
    lines = ["# " + header]
    for row in rows:
        values = ["0"] * (layout["ncols"] - 1)
        for name, value in row.items():
            values[layout[name]] = repr(value)
        lines.append(" ".join(values + ["1"]))
    path.write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize("dimension", [2, 3])
def test_surface_sums_and_volume_weighted_means(tmp_path, dimension):
    rows = [
        {"V": 1.0, "T": 0.15, "rhoB": 0.5, "tau": 2.0},
        {"V": 3.0, "T": 0.11, "rhoB": 0.1, "tau": 6.0},
    ]
    write_surface(tmp_path / "freeze_out.dat", dimension, rows)
    header, data = pt.read_surface(tmp_path / "freeze_out.dat", dimension)
    assert header == [1.5, 0.0, -0.25]
    summary = pt.surface_summary(data, dimension)
    assert summary["elements"] == 2
    assert summary["sum V"] == pytest.approx(4.0)
    assert summary["sum rhoB V"] == pytest.approx(0.5 + 0.3)
    assert summary["<T>_V"] == pytest.approx((0.15 + 3 * 0.11) / 4)
    assert summary["<tau>_V"] == pytest.approx(5.0)
    assert summary["width tau"] == pytest.approx(math.sqrt((9 + 3 * 1) / 4))
    assert ("<eta>_V" in summary) == (dimension == 3)


def test_a_surface_of_the_other_dimension_is_refused(tmp_path):
    write_surface(tmp_path / "freeze_out.dat", 3, [{"V": 1.0}])
    with pytest.raises(ValueError, match="columns"):
        pt.read_surface(tmp_path / "freeze_out.dat", 2)


def test_elementwise_differences(tmp_path):
    write_surface(tmp_path / "a.dat", 2, [{"V": 1.0, "T": 0.15}, {"V": 2.0, "T": 0.14}])
    write_surface(tmp_path / "b.dat", 2, [{"V": 1.0, "T": 0.16}, {"V": 2.5, "T": 0.14}])
    _, a = pt.read_surface(tmp_path / "a.dat", 2)
    _, b = pt.read_surface(tmp_path / "b.dat", 2)
    diff = pt.elementwise(a, b, 2)
    assert diff["T"] == pytest.approx(0.01)
    assert diff["V"] == pytest.approx(0.5)
    assert diff["rhoB"] == 0.0
    assert pt.elementwise(a, b[:1], 2) is None


def test_last_row_of_conservation(tmp_path):
    path = tmp_path / "conservation.dat"
    path.write_text("t Eloss S Btotal Stotal Qtotal\n1 0 10 2 0 0\n2 -1 12 2 0 0\n")
    last = pt.read_conservation(path)
    assert last == {"t": 2.0, "Eloss": -1.0, "S": 12.0, "Btotal": 2.0, "Stotal": 0.0, "Qtotal": 0.0}


def write_reference(path):
    """A Gubser reference on the diagonal: two times, e = 1 + r, pixy = -r."""
    lines = [
        "# made up",
        "# tau x y eta e rhoB rhoS rhoQ ux uy ueta Pi pixx pixy pixeta piyy piyeta pietaeta",
    ]
    for tau in (1.0, 1.1):
        for x in np.linspace(0.1, 4.0, 40):
            r = math.hypot(x, x)
            values = [
                tau,
                x,
                x,
                0,
                1 + r,
                2 + r,
                0,
                0,
                0.1 * r,
                0.1 * r,
                0,
                0,
                -0.1 * r,
                -r,
                0,
                0,
                0,
                0,
            ]
            lines.append(" ".join(repr(float(v)) for v in values))
    path.write_text("\n".join(lines) + "\n")


def test_gubser_measures_are_zero_for_the_reference_itself(tmp_path):
    write_reference(tmp_path / "reference.dat")
    reference = pt.read_gubser_reference(tmp_path / "reference.dat")
    assert sorted(reference) == [1.0, 1.1]
    ref = reference[1.0]
    columns = {
        "x": ref["x"],
        "y": ref["y"],
        "e": ref["e"],
        "rhoB": ref["rhoB"],
        "ux": ref["ux"],
        "uy": ref["uy"],
        "pixx": ref["pixx"] / pt.HBARC,
        "piyy": ref["piyy"] / pt.HBARC,
        "pixy": ref["pixy"] / pt.HBARC,
    }
    measures = pt.gubser_measures(columns, ref)
    assert all(value == pytest.approx(0.0, abs=1e-12) for value in measures.values())

    columns["e"] = columns["e"] * 1.01  # one per cent everywhere
    assert pt.gubser_measures(columns, ref)["max |de/e|"] == pytest.approx(0.01)
