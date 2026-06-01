#!/usr/bin/env bash
# Копирует уже скачанные Kimodo diffusion из ~/.cache/huggingface/hub
# в engineAI-studio-pro/cache/huggingface/hub (для переноса на Linux-сервер).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${1:-${HOME}/.cache/huggingface/hub}"
DEST="${SCRIPT_DIR}/cache/huggingface/hub"

mkdir -p "${DEST}"

for repo in models--nvidia--Kimodo-SMPLX-RP-v1 \
            models--nvidia--Kimodo-SOMA-RP-v1 \
            models--nvidia--Kimodo-SOMA-SEED-v1 \
            models--nvidia--Kimodo-G1-RP-v1; do
  if [[ -d "${SRC}/${repo}" ]]; then
    echo "Copying ${repo}..."
    rsync -a "${SRC}/${repo}/" "${DEST}/${repo}/"
    du -sh "${DEST}/${repo}"
  else
    echo "Skip (not found): ${SRC}/${repo}"
  fi
done

echo ""
echo "Done. Total cache:"
du -sh "${DEST}"
echo "On Linux: source ./activate_cuda.sh  (sets HF_HOME to ${SCRIPT_DIR}/cache/huggingface)"
echo "If copied from macOS Git LFS without pull, run: ./scripts/repair_xsym_hf_cache.sh"
