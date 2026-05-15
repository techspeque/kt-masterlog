#!/usr/bin/env bash
#
# Show or bump the package version.
#
# The single source of truth is __version__ in src/kt_masterlog/__init__.py.
# Hatchling reads from there at build time (see [tool.hatch.version] in
# pyproject.toml), so the wheel metadata and the runtime __version__ are
# always identical by construction.
#
# Usage:
#   bash scripts/version.sh                 # show current version
#   bash scripts/version.sh 0.2.0           # bump to 0.2.0
#   bash scripts/version.sh 0.2.0-rc1       # pre-release tags allowed
set -euo pipefail

cd "$(dirname "$0")/.."

INIT="src/kt_masterlog/__init__.py"

read_version() {
    grep -E '^__version__ = ' "$INIT" | sed -E 's/__version__ = "([^"]+)"/\1/'
}

if [[ $# -eq 0 ]]; then
    echo "Current version: $(read_version)"
    echo "  source: $INIT"
    exit 0
fi

NEW="$1"

# Permissive semver: X.Y.Z optionally followed by -prerelease or .build
if ! [[ "$NEW" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][a-zA-Z0-9.]+)?$ ]]; then
    echo "Error: '$NEW' is not a valid version." >&2
    echo "       Expected X.Y.Z, optionally followed by -suffix (e.g. 0.2.0-rc1)." >&2
    exit 1
fi

OLD=$(read_version)
echo "Bumping $OLD -> $NEW"

# sed -i.bak works on both GNU sed (Linux) and BSD sed (macOS).
sed -i.bak -E "s/^__version__ = \"[^\"]+\"/__version__ = \"${NEW}\"/" "$INIT"
rm -f "${INIT}.bak"

NOW=$(read_version)
if [[ "$NOW" != "$NEW" ]]; then
    echo "ERROR: version bump did not apply." >&2
    exit 1
fi

echo "Done. If uv.lock is committed, refresh it with: uv lock"
