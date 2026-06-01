#!/usr/bin/env bash
# Download matbee/kimodo-llm2vec-nf4 (~5 GB). Public repo; hf auth not required.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/activate_cuda.sh"

if ! command -v hf >/dev/null 2>&1; then
  echo "Hugging Face CLI required: pip install -U 'huggingface_hub[cli]'" >&2
  exit 1
fi

DEST="${LLM2VEC_LOCAL_BASE:-${SCRIPT_DIR}/models/kimodo-llm2vec-nf4}"
mkdir -p "$(dirname "${DEST}")"

# HF Xet acceleration (activate_cuda.sh sets HF_XET_HIGH_PERFORMANCE=1)
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

echo "Downloading matbee/kimodo-llm2vec-nf4 -> ${DEST}"
echo "(HF login not required; hf auth login is only for gated Kimodo diffusion checkpoints)"

if ! hf download matbee/kimodo-llm2vec-nf4 --local-dir "${DEST}"; then
  echo "" >&2
  echo "If download fails (401/gated), try: hf auth login" >&2
  exit 1
fi

if [[ ! -d "${DEST}/supervised_adapter" ]]; then
  echo "Warning: ${DEST}/supervised_adapter not found" >&2
  exit 1
fi

echo "OK: NF4 model ready"
echo "  LLM2VEC_LOCAL_BASE=${DEST}"
echo "  LLM2VEC_LOCAL_PEFT=${DEST}/supervised_adapter"
