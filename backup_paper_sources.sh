#!/usr/bin/env bash
#
# Back up the paper LaTeX sources that are deliberately kept OUT of this repo.
#
# paper_acl/ and hubness_arr_submission_august/ are gitignored on purpose (the
# repo is the public code artifact; the paper source is not part of it). That
# means those directories exist only on local disk. This script snapshots their
# *source* files into a timestamped tarball in the repo's parent directory,
# alongside the existing .bundle backups.
#
# Usage:  ./backup_paper_sources.sh [destination-dir]
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$(dirname "$REPO")}"

SOURCES=(
  "hubness_arr_submission_august"   # ARR submission build (self-contained)
  "paper_acl"                       # working copy, notes, figures, scripts
)

STAMP="$(date +%Y%m%d-%H%M)"
HASH="$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo nogit)"
ARCHIVE="$DEST/paper-sources-$STAMP-$HASH.tar.gz"

cd "$REPO"

for d in "${SOURCES[@]}"; do
  [ -d "$d" ] || { echo "missing source dir: $d" >&2; exit 1; }
done

LIST="$(mktemp)"
trap 'rm -f "$LIST"' EXIT

# Keep: .tex .bib .sty .bst figures, scripts, READMEs/notes, experiment JSON.
# Drop: LaTeX build artifacts, caches, packaged site zips, stray temp blobs.
# Experiment run logs (*_run.log, E1_norm*.log) are results, not build noise -
# they are matched first so the generic *.log exclusion does not eat them.
find "${SOURCES[@]}" \
    \( -name '__pycache__' -o -name '.ipynb_checkpoints' \) -prune -o \
    -type f \( -name '*_run.log' -o -name 'E?_norm*.log' \) -print -o \
    -type f \
      ! -name '*.aux' ! -name '*.log' ! -name '*.bbl' ! -name '*.blg' \
      ! -name '*.out' ! -name '*.fls' ! -name '*.fdb_latexmk' \
      ! -name '*.synctex.gz' ! -name '*.pyc' ! -name '*.tmp' \
      ! -name '*.zip' ! -name '.DS_Store' \
      ! -name 'ziFSe3Hz' \
    -print |
# A .pdf sitting next to a .tex of the same name is a build output, not a
# figure (e.g. hubness_acl_v14.pdf); figures under figures/ have no .tex twin.
while IFS= read -r f; do
  case "$f" in
    *.pdf) [ -e "${f%.pdf}.tex" ] && continue ;;
  esac
  printf '%s\n' "$f"
done | LC_ALL=C sort > "$LIST"

tar -czf "$ARCHIVE" -T "$LIST"

echo "archive: $ARCHIVE"
echo "files:   $(wc -l < "$LIST" | tr -d ' ')"
echo "size:    $(du -h "$ARCHIVE" | cut -f1)"
