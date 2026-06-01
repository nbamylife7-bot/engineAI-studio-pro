#!/usr/bin/env bash
# Download gated nvidia/Kimodo-* checkpoints (requires HF access approval + token).
# Usage: ./scripts/download_kimodo_models.sh [--force]
set -euo pipefail

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/activate_cuda.sh"

if [[ -z "${HF_TOKEN:-}" ]] && ! hf auth whoami &>/dev/null; then
  echo "Hugging Face login required for gated Kimodo models." >&2
  echo "  1. Accept license: https://huggingface.co/nvidia/Kimodo-SMPLX-RP-v1" >&2
  echo "  2. hf auth login   # or export HF_TOKEN=hf_..." >&2
  exit 1
fi

MODELS=(
  nvidia/Kimodo-SMPLX-RP-v1
  nvidia/Kimodo-SOMA-RP-v1
  nvidia/Kimodo-SOMA-SEED-v1
  nvidia/Kimodo-G1-RP-v1
)

for repo in "${MODELS[@]}"; do
  echo "Downloading ${repo}..."
  if [[ "${FORCE}" -eq 1 ]]; then
    hf download "${repo}" --cache-dir "${HUGGINGFACE_HUB_CACHE}" --force-download --max-workers 4
  else
    hf download "${repo}" --cache-dir "${HUGGINGFACE_HUB_CACHE}" --max-workers 4
  fi
done

"${SCRIPT_DIR}/scripts/repair_xsym_hf_cache.sh"
echo "Done."
