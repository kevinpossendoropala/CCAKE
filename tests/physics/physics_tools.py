"""Readers and comparisons for the physics checks of tests/physics/.

Only numpy is needed. Units are those of the CCAKE output files unless stated.
"""

import math

import numpy as np

HBARC = 0.1973269804  # GeV fm

# freeze_out.dat, 0-based columns (src/output.cpp, Output::print_freeze_out). The 2+1D
# layout has two velocity and two position columns and three shear components fewer.
SURFACE_COLUMNS = {
    2: {
        "gamma": 3,
        "ux": 4,
        "uy": 5,
        "V": 6,
        "bulk": 7,
        "tau": 13,
        "x": 14,
        "y": 15,
        "s": 16,
        "e": 17,
        "T": 18,
        "muB": 19,
        "muS": 20,
        "muQ": 21,
        "w": 22,
        "cs2": 23,
        "rhoB": 24,
        "rhoS": 25,
        "rhoQ": 26,
        "ncols": 40,
    },
    3: {
        "gamma": 4,
        "ux": 5,
        "uy": 6,
        "ueta": 7,
        "V": 8,
        "bulk": 9,
        "tau": 17,
        "x": 18,
        "y": 19,
        "eta": 20,
        "s": 21,
        "e": 22,
        "T": 23,
        "muB": 24,
        "muS": 25,
        "muQ": 26,
        "w": 27,
        "cs2": 28,
        "rhoB": 29,
        "rhoS": 30,
        "rhoQ": 31,
        "ncols": 45,
    },
}

# system_state_<n>.dat, 0-based columns (Output::print_system_state). Column 38 is text.
STATE_COLUMNS = {
    "x": 2,
    "y": 3,
    "e": 10,
    "rhoB": 11,
    "pixx": 25,
    "piyy": 26,
    "pixy": 29,
    "ux": 33,
    "uy": 34,
}


# ----------------------------------------------------------------------------- surfaces
def read_surface(path, dimension):
    """Returns (header charges B, S, Q at the start of the run, data array)."""
    with open(path) as f:
        header = f.readline().lstrip("#").split()
    layout = SURFACE_COLUMNS[dimension]
    ncols = layout["ncols"]
    # the last column is the name of the equation of state; it is not read
    data = np.loadtxt(path, comments="#", usecols=range(ncols - 1), ndmin=2)
    with open(path) as f:
        f.readline()
        first = f.readline().split()
    if first and len(first) != ncols:
        raise ValueError(
            "%s has %d columns, the %d+1D layout has %d" % (path, len(first), dimension, ncols)
        )
    return [float(x) for x in header], data


def read_conservation(path):
    """The last row of conservation.dat as a dict (t, Eloss, S, Btotal, Stotal, Qtotal)."""
    with open(path) as f:
        names = f.readline().split()
        last = None
        for line in f:
            if line.strip():
                last = line
    return dict(zip(names, (float(x) for x in last.split())))


def surface_summary(data, dimension):
    """Integrated and volume-weighted quantities of a freeze-out surface."""
    col = SURFACE_COLUMNS[dimension]
    volume = data[:, col["V"]]
    total = float(volume.sum())
    out = {"elements": float(len(data)), "sum V": total}
    for charge in ("rhoB", "rhoS", "rhoQ"):
        out["sum %s V" % charge] = float((data[:, col[charge]] * volume).sum())
    names = ["T", "muB", "tau", "gamma", "bulk", "e", "cs2"]
    if dimension == 3:
        names.append("eta")
    for name in names:
        values = data[:, col[name]]
        mean = float((values * volume).sum() / total) if total else math.nan
        var = float((volume * (values - mean) ** 2).sum() / total) if total else math.nan
        out["<%s>_V" % name] = mean
        out["width %s" % name] = math.sqrt(max(var, 0.0))
    return out


def elementwise(reference, run, dimension):
    """Largest absolute difference per named column, element by element.

    Only meaningful when both surfaces have the same elements in the same order.
    """
    if reference.shape != run.shape:
        return None
    col = SURFACE_COLUMNS[dimension]
    return {
        name: float(np.max(np.abs(run[:, i] - reference[:, i])))
        for name, i in col.items()
        if name != "ncols"
    }


def relative_difference(reference, value):
    if reference == value:
        return 0.0
    scale = max(abs(reference), abs(value))
    return (value - reference) / scale if scale else math.nan


# ------------------------------------------------------------------------------ Gubser
def read_state(path):
    """One system_state_<n>.dat: (time, dict of columns)."""
    with open(path) as f:
        time = float(f.readline())
    cols = sorted(STATE_COLUMNS.values())
    data = np.loadtxt(path, skiprows=1, usecols=cols, ndmin=2)
    index = {c: k for k, c in enumerate(cols)}
    return time, {name: data[:, index[c]] for name, c in STATE_COLUMNS.items()}


def diagonal(columns, phi_tol=1e-2):
    """Rows on the diagonal phi = pi/4 (where pi^xy is largest), sorted in r."""
    phi = np.arctan2(columns["y"], columns["x"])
    keep = np.abs(phi - np.pi / 4) < phi_tol
    r = np.hypot(columns["x"][keep], columns["y"][keep])
    order = np.argsort(r)
    out = {name: values[keep][order] for name, values in columns.items()}
    out["r"] = r[order]
    return out


def gubser_measures(snapshot_columns, reference_rows):
    """The measures of the 2026-09 check (analysis/numerical_checks/plot_gubser_bsq.py).

    e, rhoB, ux: largest relative difference for r < 3 fm. pixx: largest difference for
    r < 3 fm over the largest |pixx| of the reference. pixy: the same for r < 4.5 fm.
    Shear in GeV/fm^3 (the snapshot holds fm^-4).
    """
    sim = diagonal(snapshot_columns)
    for name in ("pixx", "piyy", "pixy"):
        sim[name] = sim[name] * HBARC
    ref = reference_rows
    core = sim["r"] < 3.0
    out = {}
    for name in ("e", "rhoB", "ux"):
        at = np.interp(sim["r"][core], ref["r"], ref[name])
        out["max |d%s/%s|" % (name, name)] = float(np.max(np.abs(sim[name][core] / at - 1.0)))
    ref_pixx = np.interp(sim["r"][core], ref["r"], ref["pixx"])
    out["max |dpixx|/max|pixx|"] = float(
        np.max(np.abs(sim["pixx"][core] - ref_pixx)) / np.max(np.abs(ref["pixx"]))
    )
    inside = sim["r"] < 4.5
    ref_pixy = np.interp(sim["r"][inside], ref["r"], ref["pixy"])
    out["max |dpixy|/max|pixy|"] = float(
        np.max(np.abs(sim["pixy"][inside] - ref_pixy)) / np.max(np.abs(ref["pixy"]))
    )
    return out


def read_gubser_reference(path):
    """reference_diagonal.dat: {tau: {column: array}} with r added."""
    with open(path) as f:
        names = None
        for line in f:
            if line.startswith("# tau "):
                names = line[2:].split()
                break
    data = np.loadtxt(path, comments="#", ndmin=2)
    out = {}
    for tau in np.unique(data[:, 0]):
        rows = data[data[:, 0] == tau]
        cols = {name: rows[:, k] for k, name in enumerate(names)}
        r = np.hypot(cols["x"], cols["y"])
        order = np.argsort(r)
        cols = {name: values[order] for name, values in cols.items()}
        cols["r"] = r[order]
        out[round(float(tau), 6)] = cols
    return out
