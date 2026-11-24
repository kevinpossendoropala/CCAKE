"""utilities/trento2ccake.py: reading a TRENTo grid and writing the ccake format."""

import pytest

from conftest import IC_COLUMNS, import_script, read_ic, run_script

GRID = [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.5]]


def write_grid(path, grid, header="# event 0\n# b = 1.0\n"):
    path.write_text(header + "\n".join(" ".join(str(v) for v in row) for row in grid) + "\n")


def test_read_trento_block(tmp_path):
    trento = import_script("utilities/trento2ccake.py")
    source = tmp_path / "grid.dat"
    write_grid(source, GRID)
    rows, nx, ny = trento.read_trento_block(source)
    assert (nx, ny) == (3, 4)
    assert rows == GRID


def test_read_trento_block_skips_blank_lines(tmp_path):
    trento = import_script("utilities/trento2ccake.py")
    source = tmp_path / "grid.dat"
    source.write_text("# header\n\n1 2\n\n3 4\n")
    rows, nx, ny = trento.read_trento_block(source)
    assert rows == [[1.0, 2.0], [3.0, 4.0]]
    assert (nx, ny) == (2, 2)


def test_read_trento_block_rejects_ragged_rows(tmp_path):
    trento = import_script("utilities/trento2ccake.py")
    source = tmp_path / "grid.dat"
    source.write_text("1 2 3\n4 5\n")
    with pytest.raises(RuntimeError):
        trento.read_trento_block(source)


def test_read_trento_block_rejects_empty_file(tmp_path):
    trento = import_script("utilities/trento2ccake.py")
    source = tmp_path / "grid.dat"
    source.write_text("# only a header\n")
    with pytest.raises(RuntimeError):
        trento.read_trento_block(source)


def test_conversion(tmp_path):
    source = tmp_path / "grid.dat"
    write_grid(source, GRID)
    output = tmp_path / "ic.dat"
    finished = run_script("utilities/trento2ccake.py", source, 0.5, 0.5, output, cwd=tmp_path)
    assert finished.returncode == 0, finished.stdout + finished.stderr
    header, rows = read_ic(output)
    assert header == "#0 0.5 0.5 0.1 0 -0.5 -0.5 -0.5"
    assert len(rows) == 12
    assert {len(row) for row in rows} == {IC_COLUMNS}
    # one output row per cell, row index along x, column index along y
    for ix in range(3):
        for iy in range(4):
            row = rows[ix * 4 + iy]
            assert float(row[0]) == -0.5 + ix * 0.5
            assert float(row[1]) == -0.5 + iy * 0.5
            assert float(row[3]) == GRID[ix][iy] * 75  # the script's fixed normalization


def test_missing_input_is_an_error(tmp_path):
    finished = run_script(
        "utilities/trento2ccake.py",
        tmp_path / "nothing.dat",
        0.5,
        0.5,
        tmp_path / "ic.dat",
        cwd=tmp_path,
    )
    assert finished.returncode == 1
    assert "does not exist" in finished.stdout
