#!/usr/bin/env bash
# Text encoder API on GPU: matbee/kimodo-llm2vec-nf4 (~5 GB VRAM steady).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/activate_cuda.sh"

if [[ ! -f "${LLM2VEC_LOCAL_BASE}/config.json" ]]; then
  echo "NF4 model not found at ${LLM2VEC_LOCAL_BASE}" >&2
  echo "Run: ./download_nf4.sh" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LLM2VEC_QUANTIZE=nf4
export LLM2VEC_LOCAL_BASE
export LLM2VEC_LOCAL_PEFT
export TEXT_ENCODER=llm2vec_nf4
export TEXT_ENCODER_DEVICE="${TEXT_ENCODER_DEVICE:-cuda:0}"
export TEXT_ENCODER_DTYPE=bfloat16
export TEXT_ENCODER_QUANTIZATION=none
export HF_ENABLE_PARALLEL_LOADING=YES

echo "Loading NF4 from ${LLM2VEC_LOCAL_BASE}"
cd "${KIMODO_REPO}"
exec kimodo_textencoder --text-encoder llm2vec_nf4 "$@"
