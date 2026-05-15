#!/usr/bin/env bash
#
# CHANGELOG.md helper — wraps git-cliff for common operations.
#
# Usage:
#   bash scripts/changelog.sh                # preview the [Unreleased] section
#   bash scripts/changelog.sh --regenerate   # rewrite CHANGELOG.md from full
#                                            #   git history (idempotent)
#   bash scripts/changelog.sh --tag vX.Y.Z   # preview what vX.Y.Z's release
#                                            #   notes will look like
#
# release.sh invokes git-cliff directly during the release flow; this
# script exists so you can preview or regenerate the changelog any time
# without doing a release.
set -euo pipefail

cd "$(dirname "$0")/.."

case "${1:-}" in
    --regenerate)
        uv run git-cliff --output CHANGELOG.md
        echo "CHANGELOG.md regenerated from git history."
        ;;
    --tag)
        if [[ -z "${2:-}" ]]; then
            echo "Usage: bash scripts/changelog.sh --tag vX.Y.Z" >&2
            exit 1
        fi
        uv run git-cliff --tag "$2" --unreleased
        ;;
    "")
        uv run git-cliff --unreleased
        ;;
    *)
        echo "Unknown option: $1" >&2
        echo "Usage: bash scripts/changelog.sh [--regenerate | --tag vX.Y.Z]" >&2
        exit 1
        ;;
esac
