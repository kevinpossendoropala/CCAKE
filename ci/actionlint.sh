#!/usr/bin/env bash
# Check the GitHub workflow files with actionlint (https://github.com/rhysd/actionlint).
#
#   ci/actionlint.sh [workflow files...]      default: .github/workflows/*.yml
#
# Downloads the pinned release for Linux x86-64 once, checks its checksum, and
# runs it. ACTIONLINT_DIR chooses where the program is kept.
set -euo pipefail

VERSION=1.7.12
SHA256=8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8

DIR=${ACTIONLINT_DIR:-${TMPDIR:-/tmp}/actionlint-$VERSION}
if [ ! -x "$DIR/actionlint" ]; then
  mkdir -p "$DIR"
  archive=actionlint_${VERSION}_linux_amd64.tar.gz
  curl -fsSL -o "$DIR/$archive" \
    "https://github.com/rhysd/actionlint/releases/download/v$VERSION/$archive"
  echo "$SHA256  $DIR/$archive" | sha256sum -c - >/dev/null
  tar -xzf "$DIR/$archive" -C "$DIR" actionlint
fi

if [ $# -eq 0 ]; then
  set -- .github/workflows/*.yml
fi
"$DIR/actionlint" -color=false "$@"
echo "actionlint: $# file(s) checked, no problems"
