#!/usr/bin/env bash
#
# Prepare a release locally.
#
# All work is LOCAL and REVERSIBLE — this script never publishes to PyPI.
# Publishing is handled by .github/workflows/release.yml, which fires when
# the version tag is pushed to GitHub. This separation means:
#   - destructive ops (commit, tag) are reviewable on your machine
#   - PyPI uploads are auditable in CI logs and tied to the tag
#   - the tag is the single source of truth for what was released
#
# Pipeline:
#   1. Validate environment (clean tree, on main, tag doesn't exist)
#   2. Run full CI (prereqs + tests + coverage)
#   3. Bump __version__
#   4. Refresh uv.lock if present
#   5. Regenerate CHANGELOG.md from conventional commits
#   6. Commit the version bump + changelog
#   7. Create an annotated git tag (vX.Y.Z)
#   8. Build wheel + sdist into dist/ (sanity check; CI rebuilds and uploads)
#
# After the script finishes successfully, push the commit and tag to
# trigger the publish workflow:
#   git push origin main
#   git push origin vX.Y.Z
#
# Usage:
#   bash scripts/release.sh 0.2.0
#   bash scripts/release.sh 0.2.0-rc1
set -euo pipefail

cd "$(dirname "$0")/.."

NEW_VERSION="${1:-}"

if [[ -z "$NEW_VERSION" ]]; then
    echo "Usage: bash scripts/release.sh <version>" >&2
    exit 1
fi

TAG="v${NEW_VERSION}"

# ---------- Safety checks ----------

echo "==> [1/7] Pre-flight checks"

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Error: working tree has uncommitted changes." >&2
    git status --short >&2
    exit 1
fi

BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$BRANCH" != "main" ]]; then
    echo "Error: must release from 'main' (currently on '$BRANCH')." >&2
    exit 1
fi

if git rev-parse --verify --quiet "refs/tags/${TAG}" >/dev/null; then
    echo "Error: tag $TAG already exists locally." >&2
    exit 1
fi

if git ls-remote --exit-code --tags origin "refs/tags/${TAG}" >/dev/null 2>&1; then
    echo "Error: tag $TAG already exists on origin." >&2
    exit 1
fi

echo "    clean tree, on main, $TAG is available"

# ---------- CI ----------

echo ""
echo "==> [2/8] CI (prereqs + tests)"
bash scripts/ci.sh

# ---------- Version bump ----------

echo ""
echo "==> [3/8] Bumping version"
bash scripts/version.sh "$NEW_VERSION"

# ---------- uv.lock refresh ----------

echo ""
echo "==> [4/8] Refreshing uv.lock"
if [[ -f uv.lock ]]; then
    uv lock
    echo "    uv.lock refreshed"
else
    echo "    no uv.lock present, skipping"
fi

# ---------- CHANGELOG regeneration ----------

echo ""
echo "==> [5/8] Regenerating CHANGELOG.md from commits"
# git-cliff reads all tags in history and renders Keep-a-Changelog format.
# --tag treats unreleased commits as part of the requested version.
uv run git-cliff --tag "$TAG" --output CHANGELOG.md
echo "    CHANGELOG.md updated"

# ---------- Commit ----------

echo ""
echo "==> [6/8] Committing version bump + changelog"
git add src/kt_masterlog/__init__.py CHANGELOG.md
[[ -f uv.lock ]] && git add uv.lock
git commit -m "release: ${NEW_VERSION}"

# ---------- Tag ----------

echo ""
echo "==> [7/8] Tagging $TAG"
git tag -a "$TAG" -m "${NEW_VERSION}"

# ---------- Local build (sanity check) ----------

echo ""
echo "==> [8/8] Building wheel + sdist (local sanity check)"
rm -rf dist/
uv build

echo ""
echo "Built artifacts:"
ls -1 dist/

# ---------- Next steps ----------

echo ""
echo "============================================================"
echo "  Release prepared. To publish, push the commit and tag:"
echo "============================================================"
echo "  git push origin main"
echo "  git push origin $TAG"
echo ""
echo "Pushing the tag triggers .github/workflows/release.yml, which"
echo "rebuilds and uploads to PyPI via trusted publishing."
echo ""
echo "Or undo locally (before pushing) with:"
echo "  git tag -d $TAG"
echo "  git reset --hard HEAD~1"
echo "  rm -rf dist/"
