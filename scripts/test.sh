#!/usr/bin/env bash
#
# Run the test suite with coverage.
#
# Any extra arguments are forwarded to pytest, so you can do things like:
#   bash scripts/test.sh -k bayesian
#   bash scripts/test.sh tests/test_callbacks.py
#   bash scripts/test.sh -x --pdb
set -euo pipefail

cd "$(dirname "$0")/.."

uv run pytest tests/ \
    -v \
    --cov=kt_masterlog \
    --cov-report=term-missing \
    "$@"
