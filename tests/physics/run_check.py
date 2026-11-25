#!/usr/bin/env python3
"""Run the physics checks of tests/physics/ and compare them with their references.

    run_check.py --ccake build/ccake --work /tmp/physics gubser_bsq pbpb_2d
    run_check.py --ccake build/ccake --work /tmp/physics --github      (every check a runner can do)

Each check: its inputs are found and verified (checks.yaml), its config is written with the
input paths, CCAKE runs, and the result is compared with the reference. A Markdown report
goes to --report (default: standard output), the numbers to <work>/<check>/result.json.

A quantity is judged only if tests/physics/tolerances.yaml gives it a tolerance; otherwise it
is measured and shown. The exit code is non-zero if a run fails, an output is missing, an
input does not match its checksum, or a judged quantity is outside its tolerance.

--update-reference writes the result of this run as the new reference of a surface check
(do it only with a deliberate change of the physics, and say why in the commit).

--capped runs each check for only its capped_steps time steps (checks.yaml) and checks
that the run ended and that its files are well formed. Nothing is compared with a
reference. The coverage build uses it: a full run takes hours without optimisation.
"""

import argparse
import hashlib
import json
import lzma
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
import urllib.request

import numpy as np
import yaml

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import physics_tools as pt  # noqa: E402


# ---------------------------------------------------------------------------- inputs
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cache_dir():
    return pathlib.Path(
        os.environ.get("CCAKE_PHYSICS_CACHE", pathlib.Path.home() / ".cache" / "ccake-physics")
    )


def fetch_file(spec, check_dir, destination):
    """Put one input file at destination (copy, link or download) and verify it."""
    expected = spec["sha256"]
    if "repo" in spec:
        source = check_dir / spec["repo"]
        if source.suffix == ".xz":
            with lzma.open(source) as fin, open(destination, "wb") as fout:
                shutil.copyfileobj(fin, fout)
        else:
            shutil.copyfile(source, destination)
    else:
        candidates = []
        if "local" in spec:
            local = pathlib.Path(spec["local"])
            if os.environ.get("CCAKE_PHYSICS_DATA"):
                candidates.append(pathlib.Path(os.environ["CCAKE_PHYSICS_DATA"]) / local.name)
            candidates.append(local)
        cached = cache_dir() / expected[:16] / destination.name
        candidates.append(cached)
        found = next((c for c in candidates if c.is_file()), None)
        if found is None:
            if "url" not in spec:
                raise FileNotFoundError(
                    "input %s is not at %s and has no download address"
                    % (destination.name, " or ".join(str(c) for c in candidates))
                )
            cached.parent.mkdir(parents=True, exist_ok=True)
            partial = cached.with_name(cached.name + ".part")
            print("downloading", spec["url"], flush=True)
            request = urllib.request.Request(spec["url"], headers={"User-Agent": "python-requests"})
            with urllib.request.urlopen(request, timeout=600) as r, open(partial, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            if sha256(partial) != expected:
                raise ValueError("download of %s does not match its SHA-256" % spec["url"])
            partial.rename(cached)
            found = cached
        os.symlink(found.resolve(), destination)
    got = sha256(destination)
    if got != expected:
        raise ValueError("%s: SHA-256 %s, expected %s" % (destination, got, expected))


def prepare_inputs(check, check_dir, work):
    paths = {}
    inputs = work / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    for name, spec in check["inputs"].items():
        if "directory" in spec:
            target = inputs / name
            target.mkdir(exist_ok=True)
            for filename, file_spec in spec["directory"].items():
                fetch_file(file_spec, check_dir, target / filename)
        else:
            source_name = pathlib.Path(spec.get("repo") or spec.get("local") or name).name
            target = inputs / source_name.removesuffix(".xz")
            fetch_file(spec, check_dir, target)
        paths[name] = target
    return paths


# ------------------------------------------------------------------------------- run
def run_ccake(ccake, check_dir, work, paths, threads, max_steps=None):
    config = (check_dir / "config.yaml").read_text()
    for name, path in paths.items():
        config = config.replace("@INPUT:%s@" % name, str(path))
    missing = re.findall(r"@INPUT:(\w+)@", config)
    if missing:
        raise ValueError("config.yaml uses inputs checks.yaml does not list: %s" % missing)
    (work / "config.yaml").write_text(config)
    results = work / "results"
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_PROC_BIND="false")
    if max_steps is not None:
        env["CCAKE_MAX_STEPS"] = str(max_steps)
    start = time.time()
    with open(work / "ccake.log", "w") as log:
        code = subprocess.call(
            [str(pathlib.Path(ccake).resolve()), "config.yaml", "results"],
            cwd=work,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return code, time.time() - start, results


# --------------------------------------------------------------------------- compare
def judge(name, quantity, value, reference, tolerances):
    """(difference, tolerance, verdict) of one quantity."""
    tolerance = tolerances.get(name, {}).get(quantity)
    if reference is None:
        difference = None
    elif isinstance(reference, (int, float)):
        difference = pt.relative_difference(reference, value)
    else:
        difference = None
    if tolerance is None or difference is None:
        verdict = "measured"
    else:
        verdict = "pass" if abs(difference) <= tolerance else "FAIL"
    return difference, tolerance, verdict


def compare_surface(name, check, check_dir, results, tolerances):
    dim = check["dimension"]
    header, data = pt.read_surface(results / "freeze_out.dat", dim)
    summary = pt.surface_summary(data, dim)
    final = pt.read_conservation(results / "conservation.dat")
    ref_dir = check_dir / check["reference"]
    ref_summary = None
    if (ref_dir / "summary.json").is_file():
        ref_summary = json.loads((ref_dir / "summary.json").read_text())
    rows = []
    for quantity, value in summary.items():
        ref = ref_summary["surface"].get(quantity) if ref_summary else None
        rows.append((quantity, ref, value) + judge(name, quantity, value, ref, tolerances))
    for quantity in ("S", "Btotal", "Stotal", "Qtotal", "Eloss", "t"):
        value = final[quantity]
        ref = ref_summary["conservation"].get(quantity) if ref_summary else None
        label = "final %s (conservation.dat)" % quantity
        rows.append((label, ref, value) + judge(name, label, value, ref, tolerances))
    lines = [
        "Self-check of the column map: sum n_B V = %.6g, final B in conservation.dat = %.6g"
        % (summary["sum rhoB V"], final["Btotal"])
    ]
    element = None
    ref_surface = ref_dir / "freeze_out.dat.xz"
    if ref_surface.is_file():
        with lzma.open(ref_surface, "rt") as f:
            tmp = results / "reference_freeze_out.dat"
            tmp.write_text(f.read())
        _, ref_data = pt.read_surface(tmp, dim)
        tmp.unlink()
        element = pt.elementwise(ref_data, data, dim)
        if element is None:
            lines.append("Element by element: not compared, the number of elements differs.")
        else:
            worst = ", ".join("%s %.3g" % kv for kv in element.items() if kv[1] > 0)
            lines.append(
                "Element by element, largest |difference| per column: %s"
                % (worst or "none (identical)")
            )
    else:
        lines.append("Element by element: no reference surface in the repository.")
    return rows, lines, {"surface": summary, "conservation": final, "elementwise": element}


def compare_gubser(name, check, check_dir, results, tolerances):
    reference = pt.read_gubser_reference(check_dir / check["reference"])
    previous = {}
    if (check_dir / "previous.json").is_file():
        previous = json.loads((check_dir / "previous.json").read_text())
    snapshots = {}
    for path in results.glob("system_state_*.dat"):
        with open(path) as f:
            snapshots[round(float(f.readline()), 3)] = path
    rows = []
    measures = {}
    for tau in check["times"]:
        path = snapshots.get(round(tau, 3))
        if path is None:
            raise FileNotFoundError("no snapshot at tau = %g in %s" % (tau, results))
        _, columns = pt.read_state(path)
        values = pt.gubser_measures(columns, reference[round(tau, 6)])
        measures["%.1f" % tau] = values
        for quantity, value in values.items():
            label = "tau=%.1f %s" % (tau, quantity)
            ref = previous.get("%.1f" % tau, {}).get(quantity)
            tolerance = tolerances.get(name, {}).get(quantity)
            if tolerance is None:
                verdict = "measured"
            else:
                verdict = "pass" if value <= tolerance else "FAIL"
            rows.append((label, ref, value, None, tolerance, verdict))
    lines = [
        "Measures against the semi-analytic solution along the diagonal phi = pi/4; the"
        " reference column is the run of 2026-09-14 (analysis/numerical_checks)."
    ]
    return rows, lines, {"measures": measures}


def check_capped(check, results):
    """A step-capped run: form of the output only (physics_tools.well_formed)."""
    surface = check.get("capped_surface", check["kind"] == "surface")
    counts = pt.well_formed(results, check["dimension"], surface)
    rows = [(what, None, number, None, None, "measured") for what, number in counts.items()]
    lines = [
        "Capped at %d time steps: the output is well formed. Nothing was compared with a"
        " reference." % check["capped_steps"]
    ]
    return rows, lines, {"capped": counts}


# ---------------------------------------------------------------------------- report
def fmt(x):
    if x is None:
        return ""
    if isinstance(x, float):
        return "%.6g" % x
    return str(x)


def report(name, check, info, rows, lines):
    out = ["## %s" % name, "", check["description"].strip(), ""]
    out.append(
        "Run: exit code %d, %.0f s on %d OpenMP thread(s)."
        % (info["exit"], info["seconds"], info["threads"])
    )
    out.append("")
    if rows:
        out.append("| Quantity | Reference | This run | Relative difference | Tolerance | |")
        out.append("|---|---|---|---|---|---|")
        for quantity, ref, value, diff, tol, verdict in rows:
            out.append(
                "| %s | %s | %s | %s | %s | %s |"
                % (quantity, fmt(ref), fmt(value), fmt(diff), fmt(tol), verdict)
            )
        out.append("")
    out += lines
    out.append("")
    return "\n".join(out)


def update_reference(check, check_dir, results, data):
    ref_dir = check_dir / check["reference"]
    ref_dir.mkdir(parents=True, exist_ok=True)
    keep = {"surface": data["surface"], "conservation": data["conservation"]}
    (ref_dir / "summary.json").write_text(json.dumps(keep, indent=1, sort_keys=True) + "\n")
    shutil.copyfile(results / "conservation.dat", ref_dir / "conservation.dat")
    if check.get("github"):
        with open(results / "freeze_out.dat", "rb") as fin:
            with lzma.open(ref_dir / "freeze_out.dat.xz", "wb", preset=9) as fout:
                shutil.copyfileobj(fin, fout)


# ------------------------------------------------------------------------------ main
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checks", nargs="*", help="names from checks.yaml")
    parser.add_argument("--github", action="store_true", help="every check a runner can do")
    parser.add_argument("--all", action="store_true", help="every check")
    parser.add_argument("--ccake", required=True, help="the ccake executable")
    parser.add_argument("--work", required=True, help="a new or empty directory for the runs")
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    parser.add_argument("--report", help="append the Markdown report to this file")
    parser.add_argument("--update-reference", action="store_true")
    parser.add_argument(
        "--capped", action="store_true", help="capped_steps time steps only, form of the output"
    )
    args = parser.parse_args()
    if args.capped and args.update_reference:
        parser.error("a capped run cannot be a reference")

    checks = yaml.safe_load((HERE / "checks.yaml").read_text())
    tolerances = yaml.safe_load((HERE / "tolerances.yaml").read_text()) or {}
    # a check of kind "run" has no reference: it exists only capped
    usable = {
        n: c
        for n, c in checks.items()
        if ("capped_steps" in c if args.capped else c["kind"] != "run")
    }
    names = list(args.checks)
    if args.all:
        names = list(usable)
    elif args.github:
        names = [n for n, c in usable.items() if c.get("github")]
    if not names:
        parser.error("name a check, or give --github or --all")
    for name in names:
        if name not in checks:
            parser.error("checks.yaml has no check %r" % name)
        if name not in usable:
            parser.error(
                "%s has no capped_steps" % name
                if args.capped
                else "%s has no reference: run it with --capped" % name
            )

    failed = False
    for name in names:
        check = checks[name]
        check_dir = HERE / name
        work = pathlib.Path(args.work).resolve() / name
        work.mkdir(parents=True, exist_ok=False)
        info = {"exit": -1, "seconds": 0.0, "threads": args.threads}
        rows, lines, data = [], [], {}
        try:
            paths = prepare_inputs(check, check_dir, work)
            info["exit"], info["seconds"], results = run_ccake(
                args.ccake,
                check_dir,
                work,
                paths,
                args.threads,
                check["capped_steps"] if args.capped else None,
            )
            if info["exit"] != 0:
                raise RuntimeError("ccake exited with code %d (see %s)" % (info["exit"], work))
            if args.capped:
                rows, lines, data = check_capped(check, results)
            else:
                compare = compare_surface if check["kind"] == "surface" else compare_gubser
                rows, lines, data = compare(name, check, check_dir, results, tolerances)
            if args.update_reference and check["kind"] == "surface":
                update_reference(check, check_dir, results, data)
                lines.append("The reference was replaced by this run.")
        except Exception as error:  # report it, then go on with the next check
            failed = True
            lines.append("**Error:** %s" % error)
        if any(row[5] == "FAIL" for row in rows):
            failed = True
        (work / "result.json").write_text(
            json.dumps({"run": info, **data}, indent=1, default=float) + "\n"
        )
        text = report(name, check, info, rows, lines)
        if args.report:
            with open(args.report, "a") as f:
                f.write(text + "\n")
        print(text, flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
