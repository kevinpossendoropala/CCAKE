#!/usr/bin/env python3
"""End-to-end smoke test: a short ideal 1+1D run of ccake.

It generates the initial condition with utilities/long_ic_generator.py, runs
ccake for a fixed number of time steps, and checks that the run finished and
that what it wrote is well formed: the expected files exist, every row of a
file has the same number of columns, and no number is NaN or infinite.

It does not judge the physics. There is no tolerance in this file.
"""

import argparse
import math
import os
import pathlib
import subprocess
import sys

N_PARTICLES = 2001  # rows written by long_ic_generator.py
IC_COLUMNS = 17  # columns of the ccake initial-condition format
CONSERVATION_HEADER = "t Eloss S Btotal Stotal Qtotal"


class CheckFailed(Exception):
    pass


def check(condition, message):
    if not condition:
        raise CheckFailed(message)


def run(command, cwd, env=None, log=None):
    """Run a command, write its output to a log file, return the exit code."""
    with open(log, "w") as handle:
        return subprocess.run(
            command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT
        ).returncode


def data_rows(path, skip=0):
    """Rows of a whitespace-separated text file, as lists of tokens."""
    with open(path) as handle:
        lines = handle.read().splitlines()
    return [line.split() for line in lines[skip:] if line.strip()]


def check_table(path, rows, what):
    """Same number of columns in every row; every number finite."""
    check(rows, "%s: no data rows in %s" % (what, path))
    widths = {len(row) for row in rows}
    check(
        len(widths) == 1,
        "%s: rows of %s have different numbers of columns: %s" % (what, path, sorted(widths)),
    )
    for number, row in enumerate(rows):
        for column, token in enumerate(row):
            try:
                value = float(token)
            except ValueError:
                continue  # a label, such as the EoS name
            check(
                math.isfinite(value),
                "%s: %s in %s, data row %d, column %d" % (what, token, path, number, column),
            )
    return widths.pop()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ccake", required=True, help="path to the ccake executable")
    parser.add_argument("--source-dir", required=True, help="root of the CCAKE repository")
    parser.add_argument("--work-dir", required=True, help="scratch directory for this run")
    parser.add_argument("--steps", type=int, default=20, help="time steps to run")
    args = parser.parse_args()

    source = pathlib.Path(args.source_dir).resolve()
    work = pathlib.Path(args.work_dir).resolve()
    run_dir = work / "run"
    out_dir = work / "out"
    for directory in (run_dir, out_dir, work / "long_data"):
        directory.mkdir(parents=True, exist_ok=True)
    for stale in list(out_dir.glob("system_state_*.dat")) + [out_dir / "conservation.dat"]:
        if stale.exists():
            stale.unlink()

    # 1. initial condition (the generator writes to ../long_data of its working directory)
    code = run(
        [sys.executable, str(source / "utilities" / "long_ic_generator.py"), "1.0"],
        cwd=run_dir,
        log=work / "ic_generator.log",
    )
    check(
        code == 0,
        "long_ic_generator.py exited with code %d (see %s)" % (code, work / "ic_generator.log"),
    )
    ic_file = work / "long_data" / "ic_long_tau_1.0.dat"
    check(ic_file.is_file(), "the generator did not write %s" % ic_file)
    ic_rows = data_rows(ic_file, skip=1)
    check(
        len(ic_rows) == N_PARTICLES,
        "initial condition has %d rows, expected %d" % (len(ic_rows), N_PARTICLES),
    )
    check(
        check_table(ic_file, ic_rows, "initial condition") == IC_COLUMNS,
        "initial condition does not have %d columns" % IC_COLUMNS,
    )

    # 2. configuration
    template = (source / "tests" / "regression" / "long_1d.yaml").read_text()
    config = work / "long_1d.yaml"
    config.write_text(template.replace("@IC_FILE@", str(ic_file)))

    # 3. the run
    env = dict(os.environ, CCAKE_MAX_STEPS=str(args.steps))
    code = run(
        [str(pathlib.Path(args.ccake).resolve()), str(config), str(out_dir)],
        cwd=run_dir,
        env=env,
        log=work / "ccake.log",
    )
    check(code == 0, "ccake exited with code %d (see %s)" % (code, work / "ccake.log"))

    # 4. what it wrote
    conservation = out_dir / "conservation.dat"
    check(conservation.is_file(), "ccake did not write %s" % conservation)
    with open(conservation) as handle:
        header = handle.readline().strip()
    check(
        header == CONSERVATION_HEADER,
        "conservation.dat header is %r, expected %r" % (header, CONSERVATION_HEADER),
    )
    conservation_rows = data_rows(conservation, skip=1)
    check_table(conservation, conservation_rows, "conservation.dat")

    snapshots = sorted(out_dir.glob("system_state_*.dat"))
    check(snapshots, "ccake wrote no system_state_*.dat snapshot")
    widths = set()
    for snapshot in snapshots:
        rows = data_rows(snapshot, skip=1)  # the first line is the time
        check(
            len(rows) == N_PARTICLES,
            "%s has %d particles, expected %d" % (snapshot.name, len(rows), N_PARTICLES),
        )
        widths.add(check_table(snapshot, rows, "snapshot"))
    check(len(widths) == 1, "snapshots disagree on the number of columns: %s" % sorted(widths))

    print(
        "smoke run OK: %d steps, %d snapshots of %d particles x %d columns, "
        "%d conservation rows"
        % (args.steps, len(snapshots), N_PARTICLES, widths.pop(), len(conservation_rows))
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CheckFailed as failure:
        print("smoke run FAILED: %s" % failure)
        sys.exit(1)
