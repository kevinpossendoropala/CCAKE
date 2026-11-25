"""The checks of tests/regression/run_smoke.py must be able to fail."""

import pytest

from conftest import import_script

smoke = import_script("tests/regression/run_smoke.py")


def test_accepts_a_good_table():
    rows = [["0", "1.5", "conformal"], ["1", "-2e-3", "conformal"]]
    assert smoke.check_table("file", rows, "table") == 3


@pytest.mark.parametrize("bad", ["nan", "-nan", "inf", "-inf"])
def test_rejects_non_finite_numbers(bad):
    rows = [["0", "1.5"], ["1", bad]]
    with pytest.raises(smoke.CheckFailed):
        smoke.check_table("file", rows, "table")


def test_rejects_rows_of_different_length():
    with pytest.raises(smoke.CheckFailed):
        smoke.check_table("file", [["0", "1"], ["0", "1", "2"]], "table")


def test_rejects_an_empty_table():
    with pytest.raises(smoke.CheckFailed):
        smoke.check_table("file", [], "table")
