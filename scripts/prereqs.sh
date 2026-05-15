#!/usr/bin/env bash
#
# Install everything needed to develop kt-masterlog.
#
# Idempotent: safe to re-run. Bootstraps uv if missing, installs Python 3.12
# via uv, then syncs the dev dependency group into .venv/.
#
# Usage:
#   bash scripts/prereqs.sh
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Checking for uv"
if ! command -v uv >/dev/null 2>&1; then
    echo "    uv not found — installing from astral.sh"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Make uv visible to the rest of this script run.
    export PATH="${HOME}/.local/bin:${PATH}"
    if ! command -v uv >/dev/null 2>&1; then
        echo "ERROR: uv install completed but uv is not on PATH."
        echo "       Add \$HOME/.local/bin to your PATH and re-run."
        exit 1
    fi
fi
echo "    uv $(uv --version | awk '{print $2}')"

echo "==> Installing Python 3.12 (no-op if already present)"
uv python install 3.12

echo "==> Syncing dependencies (dev group)"
uv sync --group dev

echo ""
echo "Done. The project venv is at .venv/. Use 'uv run <cmd>' to invoke tools,"
echo "or activate it with: source .venv/bin/activate"
