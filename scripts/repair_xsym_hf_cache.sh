#!/usr/bin/env bash
# Replace macOS Git-LFS "XSym" stub files in HF hub snapshots with real blob content.
# Use when cache was copied via rsync from a Mac clone without materialized LFS files.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HUB="${HUGGINGFACE_HUB_CACHE:-${SCRIPT_DIR}/cache/huggingface/hub}"

repair_repo() {
  local repo_dir="$1"
  [[ -d "${repo_dir}/snapshots" ]] || return 0
  local fixed=0 skipped=0
  while IFS= read -r -d '' xsym; do
    local blob_rel blob
    blob_rel="$(sed -n '4p' "${xsym}" | tr -d '[:space:]')"
    if [[ -z "${blob_rel}" ]]; then
      skipped=$((skipped + 1))
      continue
    fi
    blob="$(python3 - "${xsym}" "${blob_rel}" <<'PY'
import os, sys
xsym, rel = sys.argv[1], sys.argv[2]
print(os.path.normpath(os.path.join(os.path.dirname(xsym), rel)))
PY
)"
    if [[ ! -f "${blob}" ]]; then
      echo "  missing blob for ${xsym}: ${blob}" >&2
      skipped=$((skipped + 1))
      continue
    fi
    if head -c 4 "${blob}" | grep -q 'XSym'; then
      echo "  blob still XSym (run hf download): ${blob}" >&2
      skipped=$((skipped + 1))
      continue
    fi
    cp -f "${blob}" "${xsym}"
    fixed=$((fixed + 1))
  done < <(find "${repo_dir}/snapshots" -type f -print0 | while IFS= read -r -d '' f; do
    head -c 4 "${f}" 2>/dev/null | grep -q 'XSym' && printf '%s\0' "${f}"
  done)
  echo "$(basename "${repo_dir}"): fixed=${fixed} skipped=${skipped}"
}

for repo in "${HUB}"/models--nvidia--Kimodo-*; do
  [[ -d "${repo}" ]] || continue
  repair_repo "${repo}"
done
