#!/usr/bin/env bash
#
# Single CI entry point — invoked identically by the GitHub workflow and by
# contributors validating locally before a push. Keeping one source of truth
# eliminates "passes locally, fails in CI" drift.
#
# Usage:
#   bash scripts/ci.sh
set -euo pipefail

cd "$(dirname "$0")/.."

echo "============================================================"
echo "  CI: prereqs"
echo "============================================================"
bash scripts/prereqs.sh

echo ""
echo "============================================================"
echo "  CI: tests"
echo "============================================================"
bash scripts/test.sh

echo ""
echo "============================================================"
echo "  CI passed"
echo "============================================================"
