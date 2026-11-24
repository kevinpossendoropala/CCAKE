#!/usr/bin/env bash
# Build and install the two dependencies of CCAKE that Linux distributions do not
# ship in a usable version: Kokkos and Cabana.
#
#   ci/install-deps.sh <install prefix>
#
# Then configure CCAKE with  -DCMAKE_PREFIX_PATH=<install prefix>.
# The same script is used by the CI workflows and can be used on a laptop.
#
# Environment variables:
#   CC, CXX     compilers to use (default: whatever CMake finds)
#   JOBS        parallel build jobs (default: number of cores)
#   WORK_DIR    where to download and build (default: a temporary directory)
#   KOKKOS_EXTRA_ARGS, CABANA_EXTRA_ARGS   extra arguments for the two cmake calls
#
# The other dependencies come from the system: a C++17 compiler with OpenMP,
# CMake >= 3.16, git, HDF5 with C++ bindings, GSL and yaml-cpp >= 0.8.
set -euo pipefail

# Versions CCAKE is developed and run with. The commit is checked after the
# download, so a moved tag cannot slip in different code.
KOKKOS_VERSION=4.1.00
KOKKOS_COMMIT=1a3ea28f6e97b4c9dd2c8ceed53ad58ed5f94dfe
CABANA_VERSION=0.8.0
CABANA_COMMIT=44c43577c905626d1d38c257f89c305dad6e7e37

if [ $# -ne 1 ]; then
  echo "usage: $0 <install prefix>" >&2
  exit 2
fi
mkdir -p "$1"
PREFIX=$(cd "$1" && pwd)
JOBS=${JOBS:-$(nproc)}
WORK_DIR=${WORK_DIR:-$(mktemp -d)}
mkdir -p "$WORK_DIR"

# fetch <name> <repository url> <tag> <expected commit>
fetch() {
  local name=$1 url=$2 tag=$3 commit=$4
  if [ ! -d "$WORK_DIR/$name/.git" ]; then
    git clone --quiet --depth 1 --branch "$tag" "$url" "$WORK_DIR/$name"
  fi
  local got
  got=$(git -C "$WORK_DIR/$name" rev-parse HEAD)
  if [ "$got" != "$commit" ]; then
    echo "ERROR: $name $tag is at commit $got, expected $commit" >&2
    exit 1
  fi
}

echo "== Kokkos $KOKKOS_VERSION -> $PREFIX"
fetch kokkos https://github.com/kokkos/kokkos.git "$KOKKOS_VERSION" "$KOKKOS_COMMIT"
# CCAKE names both Kokkos::OpenMP and Kokkos::Serial, so both backends are needed.
# shellcheck disable=SC2086
cmake -S "$WORK_DIR/kokkos" -B "$WORK_DIR/kokkos-build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_CXX_EXTENSIONS=OFF \
  -DKokkos_ENABLE_OPENMP=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_TESTS=OFF \
  ${KOKKOS_EXTRA_ARGS:-}
cmake --build "$WORK_DIR/kokkos-build" -j "$JOBS"
cmake --install "$WORK_DIR/kokkos-build"

echo "== Cabana $CABANA_VERSION -> $PREFIX"
fetch cabana https://github.com/ECP-copa/Cabana.git "$CABANA_VERSION" "$CABANA_COMMIT"
# CCAKE uses Cabana::Core only. MPI and HDF5 are switched off on purpose: with
# MPI found, every build of CCAKE would need MPI too, and Cabana refuses to
# configure when it finds HDF5 without MPI.
# shellcheck disable=SC2086
cmake -S "$WORK_DIR/cabana" -B "$WORK_DIR/cabana-build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_PREFIX_PATH="$PREFIX" \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_DISABLE_FIND_PACKAGE_MPI=ON \
  -DCMAKE_DISABLE_FIND_PACKAGE_HDF5=ON \
  -DCabana_ENABLE_GRID=OFF \
  -DCabana_ENABLE_TESTING=OFF \
  -DCabana_ENABLE_EXAMPLES=OFF \
  -DCabana_ENABLE_PERFORMANCE_TESTING=OFF \
  ${CABANA_EXTRA_ARGS:-}
cmake --build "$WORK_DIR/cabana-build" -j "$JOBS"
cmake --install "$WORK_DIR/cabana-build"

echo "== done. Configure CCAKE with -DCMAKE_PREFIX_PATH=$PREFIX"
