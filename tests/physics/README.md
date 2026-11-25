# Physics checks

Full runs of CCAKE, compared with an analytic solution or with a reference run. They take
minutes to an hour each, so they are not part of the unit tests: the workflow
`.github/workflows/physics.yml` runs them nightly, by hand, and on pull requests that change
the code, and `delta_submit.sh` runs them on Delta.

| Check | Compared with | Where |
|---|---|---|
| `gubser_bsq` | semi-analytic BSQ viscous Gubser flow (VGCC, Ingles et al. arXiv:2503.20021), as in arXiv:2511.22852 Fig. 7 | GitHub, Delta |
| `pbpb_2d` | a reference run: Pb+Pb 5.02 TeV, 2+1D, TRENTo+ICCING, B/S/Q, Houston EoS | GitHub, Delta |
| `auau77_3d` | a reference run: Au+Au 7.7 GeV, 3+1D, freeze-in initial condition, AMPT parton sources, bh_new_1000 EoS | Delta (its inputs are not public yet) |

`checks.yaml` describes each check and its inputs, with their SHA-256.

## Running them

```bash
# here: ccake from a build, a new directory for the runs
python3 tests/physics/run_check.py --ccake build/ccake --work /tmp/physics gubser_bsq pbpb_2d
python3 tests/physics/run_check.py --ccake build/ccake --work /tmp/physics --github

# on Delta, through SLURM (the account is yours to give)
tests/physics/delta_submit.sh --account ACCOUNT --ccake build/ccake --work /path/new_dir \
    --threads 16 --time 2:00:00 --python /path/to/python --all
```

Python needs numpy and PyYAML. Inputs are found in the repository (`.xz` files), at the local
path in `checks.yaml` (Delta) or under `$CCAKE_PHYSICS_DATA`, or downloaded into
`$CCAKE_PHYSICS_CACHE` (default `~/.cache/ccake-physics`). The report is Markdown; the numbers
of each check are in `<work>/<check>/result.json`.

## What is compared

**Surface checks** (`pbpb_2d`, `auau77_3d`), against `<check>/reference/summary.json`:
number of surface elements; sum of V = u.dSigma; sum of n_B V, n_S V, n_Q V; the
volume-weighted mean and width of T, mu_B, tau, gamma, bulk pressure, energy density, c_s^2
(and eta_s in 3D); the last row of `conservation.dat`. Element by element, the largest
difference per column is reported where the reference surface is in the repository
(`pbpb_2d`). The report also prints sum n_B V next to the final net baryon number, a check
that the column map is right.

**Gubser** (`gubser_bsq`), along the diagonal phi = pi/4 at tau = 1.0 ... 1.5 fm, the measures
of the 2026-09 check in `analysis/numerical_checks/plot_gubser_bsq.py`: largest relative
difference of e, n_B, u^x for r < 3 fm; largest difference of pi^xx (r < 3 fm) and pi^xy
(r < 4.5 fm) over the largest |pi| of the solution. `gubser_bsq/previous.json` holds the
values of the run of 2026-09-14, shown next to each new run.

## Tolerances

`tolerances.yaml` decides what is judged. It is empty: every quantity is measured and shown,
none can fail a run. A check fails only if CCAKE stops with an error, an output is missing,
or an input does not match its checksum.

## References

The surface references were made with commit `696999cd` on Delta, one OpenMP thread. A change
that moves the physics on purpose replaces them in the same pull request and says why:

```bash
python3 tests/physics/run_check.py --ccake build/ccake --work /tmp/new_reference \
    --threads 1 --update-reference pbpb_2d auau77_3d
```
