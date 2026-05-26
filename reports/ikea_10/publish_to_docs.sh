#!/usr/bin/env bash
# Sync report/ -> ../docs/ for GitHub Pages.
# Excludes dev-only files (build script, scratch frames, pycache).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
rsync -a --delete \
  --exclude='build.py' \
  --exclude='extract_images.py' \
  --exclude='publish_to_docs.sh' \
  --exclude='scratch/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.gitkeep' \
  --exclude='report.html' \
  ./ ../docs/
# build.py writes docs/index.html directly; nothing to rename.
# Run build.py beforehand if HTML changed.
echo "published $(find ../docs -type f | wc -l) files to docs/"
du -sh ../docs/
