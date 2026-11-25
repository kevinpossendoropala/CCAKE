#!/usr/bin/env bash
# Submit the physics checks to SLURM (written for Delta, works on any SLURM cluster).
#
#   tests/physics/delta_submit.sh --account ACCOUNT --ccake PATH --work DIR \
#       [--threads N] [--time H:MM:SS] [--python PYTHON] [--dependency JOBID] \
#       [run_check.py arguments: check names, --all, --update-reference ...]
#
# The executable is copied into DIR first, so a rebuild does not change a waiting job.
# Memory is 2 GB per thread. The report is DIR/report.md, the job log DIR/slurm-<id>.out.
# PYTHON needs numpy and PyYAML (default: python3 on the PATH).
set -euo pipefail

account="" ccake="" work="" threads=16 limit="2:00:00" python=python3 dependency=""
passthrough=()
while [ $# -gt 0 ]; do
  case "$1" in
    --account) account=$2; shift 2 ;;
    --ccake) ccake=$2; shift 2 ;;
    --work) work=$2; shift 2 ;;
    --threads) threads=$2; shift 2 ;;
    --time) limit=$2; shift 2 ;;
    --python) python=$2; shift 2 ;;
    --dependency) dependency=$2; shift 2 ;;
    *) passthrough+=("$1"); shift ;;
  esac
done
if [ -z "$account" ] || [ -z "$ccake" ] || [ -z "$work" ]; then
  sed -n '2,12p' "$0" >&2
  exit 2
fi
case "$limit" in
  *:*:*) ;;
  *) echo "--time must be H:MM:SS (SLURM reads 06:00 as six minutes)" >&2; exit 2 ;;
esac
if [ -e "$work" ]; then
  echo "$work exists; give a new directory" >&2
  exit 2
fi

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$work"
work=$(cd "$work" && pwd)
cp "$ccake" "$work/ccake"
printf '%q ' "${passthrough[@]}" > "$work/arguments"

extra=()
if [ -n "$dependency" ]; then extra+=(--dependency="afterok:$dependency"); fi
sbatch --parsable "${extra[@]}" \
  --job-name=ccake-physics --account="$account" --partition=cpu \
  --nodes=1 --ntasks=1 --cpus-per-task="$threads" --mem="$((2 * threads))G" \
  --time="$limit" --output="$work/slurm-%j.out" <<EOF
#!/usr/bin/env bash
set -uo pipefail
ulimit -c 0
cd "$work"
eval "args=(\$(cat arguments))"
"$python" "$here/run_check.py" --ccake "$work/ccake" --work "$work/runs" \
  --threads "$threads" --report "$work/report.md" "\${args[@]}"
EOF
