#!/usr/bin/env bash
# Run the CCAKE tests locally, the way the GitHub workflows run them.
#
#   ci/run_all_tests.sh [options]
#
# Stages, in order (each one reported PASS, FAIL or skipped at the end):
#   build      configure and build with the CI settings (Release, -Wall -Wextra, x86-64-v3,
#              tests on) in --build DIR
#   tests      CTest with --threads OpenMP threads, then again with 1 thread
#              (the "quarantine" tests, known issues, run separately and never fail the script)
#   python     pytest tests/python
#   lint       pre-commit hooks and ci/actionlint.sh (skipped if pre-commit is not installed)
#   capped     with --capped: every physics check for a few time steps (tests/physics/README.md)
#   physics    with --physics: the full physics checks a GitHub runner can do, compared with
#              their references (minutes to an hour)
#   coverage   with --coverage: a second build without optimisation and with --coverage, all
#              tests and the capped physics checks, then a gcovr report (as .github/workflows/
#              coverage.yml does; needs gcovr)
#
# Options:
#   --prefix DIR     where Kokkos and Cabana are installed (ci/install-deps.sh DIR); added to
#                    CMAKE_PREFIX_PATH. Not needed if CMake finds them already.
#   --build DIR      build directory (default: build-tests). --coverage uses DIR-coverage.
#   --threads N      OpenMP threads of the first test pass and the physics runs (default 4)
#   --jobs N         parallel compile jobs (default: all cores)
#   --arch ARCH      CCAKE_ARCH (default x86-64-v3, as in CI; "native" is faster on one machine)
#   --python PATH    Python with numpy, pytest and PyYAML (default: python3 on the PATH)
#   --gcov PATH      gcov matching the compiler, for --coverage (default: gcov)
#   --skip-build     use what is already built in --build DIR
#   --capped, --physics, --coverage   the optional stages above
#
# The exit code is 0 only if every stage that ran passed.
set -uo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/.." && pwd)

prefix="" build="build-tests" threads=4 jobs=$(nproc) arch="x86-64-v3" python=python3 gcov=gcov
skip_build=0 do_capped=0 do_physics=0 do_coverage=0
while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) prefix=$2; shift 2 ;;
    --build) build=$2; shift 2 ;;
    --threads) threads=$2; shift 2 ;;
    --jobs) jobs=$2; shift 2 ;;
    --arch) arch=$2; shift 2 ;;
    --python) python=$2; shift 2 ;;
    --gcov) gcov=$2; shift 2 ;;
    --skip-build) skip_build=1; shift ;;
    --capped) do_capped=1; shift ;;
    --physics) do_physics=1; shift ;;
    --coverage) do_coverage=1; shift ;;
    -h|--help) sed -n '2,36p' "$0"; exit 0 ;;
    *) echo "unknown option $1 (see --help)" >&2; exit 2 ;;
  esac
done

build=$(mkdir -p "$build" && cd "$build" && pwd)
logs="$build/test-logs"
mkdir -p "$logs"
ulimit -c 0  # no core dumps from tests that are meant to abort
prefix_args=()
[ -n "$prefix" ] && prefix_args=(-DCMAKE_PREFIX_PATH="$prefix")

declare -a names=() results=()
record() { names+=("$1"); results+=("$2"); }
# run_stage NAME COMMAND...: runs it with its output in test-logs/NAME.log
run_stage() {
  local name=$1; shift
  local start=$SECONDS
  echo "== $name"
  if "$@" > "$logs/$name.log" 2>&1; then
    record "$name" "PASS ($((SECONDS - start)) s)"
  else
    record "$name" "FAIL ($((SECONDS - start)) s, see $logs/$name.log)"
    tail -20 "$logs/$name.log"
  fi
}

configure_and_build() {
  local dir=$1; shift
  cmake -S "$repo" -B "$dir" -DCMAKE_BUILD_TYPE=Release ${prefix_args[@]+"${prefix_args[@]}"} \
        -DCCAKE_ARCH="$arch" -DENABLE_TESTING=ON -DPython3_EXECUTABLE="$(command -v "$python")" "$@" &&
  cmake --build "$dir" -j "$jobs"
}

physics_checks() {  # physics_checks WORK_DIR CCAKE [run_check.py arguments]
  local work=$1 ccake=$2; shift 2
  rm -rf "$work"
  "$python" "$repo/tests/physics/run_check.py" --ccake "$ccake" --work "$work" \
            --threads "$threads" --report "$work.md" "$@"
}

# ---------------------------------------------------------------------------- stages
if [ $skip_build -eq 1 ]; then
  record build "skipped (--skip-build)"
else
  run_stage build configure_and_build "$build" -DCMAKE_CXX_FLAGS="-Wall -Wextra"
fi

if [ -x "$build/ccake" ]; then
  run_stage tests_threads env OMP_NUM_THREADS="$threads" OMP_PROC_BIND=false \
            ctest --test-dir "$build" --output-on-failure --timeout 600 -LE quarantine
  run_stage tests_1thread env OMP_NUM_THREADS=1 \
            ctest --test-dir "$build" --output-on-failure -j "$jobs" --timeout 600 -LE quarantine
  # known issues: expected to fail, reported but never counted
  if env OMP_NUM_THREADS=2 ctest --test-dir "$build" -L quarantine --timeout 600 \
         > "$logs/known_issues.log" 2>&1; then
    if grep -q "No tests were found" "$logs/known_issues.log"; then
      record known_issues "none"
    else
      record known_issues "all pass (move them back into the main suite?)"
    fi
  else
    record known_issues "fail as expected (not counted)"
  fi
else
  record tests "not run: no build in $build"
fi

run_stage python env PYTHONDONTWRITEBYTECODE=1 \
          "$python" -m pytest "$repo/tests/python" -q -p no:cacheprovider

if command -v pre-commit > /dev/null; then
  run_stage lint bash -c "cd '$repo' && pre-commit run --all-files && ci/actionlint.sh"
else
  record lint "skipped (pre-commit is not installed; pip install pre-commit)"
fi

if [ $do_capped -eq 1 ]; then
  run_stage capped physics_checks "$build/physics-capped" "$build/ccake" --capped --all
fi
if [ $do_physics -eq 1 ]; then
  run_stage physics physics_checks "$build/physics" "$build/ccake" --github
fi

if [ $do_coverage -eq 1 ]; then
  cov="$build-coverage"
  coverage_run() {
    configure_and_build "$cov" -DCCAKE_OPTIMIZE=OFF \
        -DCMAKE_CXX_FLAGS_RELEASE="-O0 -g --coverage -fprofile-update=atomic -DNDEBUG" &&
    find "$cov" -name '*.gcda' -delete &&
    env OMP_NUM_THREADS=2 OMP_PROC_BIND=false ctest --test-dir "$cov" -j 2 --timeout 1200 -LE quarantine &&
    { env OMP_NUM_THREADS=2 ctest --test-dir "$cov" -j 2 --timeout 1200 -L quarantine || true; } &&
    physics_checks "$cov/physics-capped" "$cov/ccake" --capped --all &&
    (cd "$repo" && gcovr --root . --object-directory "$cov" --filter 'src/' --filter 'include/' \
        --gcov-executable "$gcov" --exclude-throw-branches \
        --gcov-suspicious-hits-threshold 1099511627776 --print-summary \
        --txt "$cov/coverage.txt" --cobertura "$cov/coverage.xml" --html "$cov/coverage.html")
  }
  run_stage coverage coverage_run
  [ -f "$cov/coverage.txt" ] && tail -3 "$logs/coverage.log" | sed 's/^/   /'
fi

# --------------------------------------------------------------------------- summary
echo
echo "Summary (logs in $logs):"
failed=0
for i in "${!names[@]}"; do
  printf '  %-15s %s\n' "${names[$i]}" "${results[$i]}"
  case "${results[$i]}" in FAIL*) failed=1 ;; esac
done
exit $failed
