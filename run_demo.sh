#!/usr/bin/env bash
# Demo с NF4 text encoder в том же процессе (как в доке matbee: TEXT_ENCODER_MODE=local).
# Нужно: ./download_nf4.sh, GPU ~12–16+ GB VRAM (NF4 ~5 GB + diffusion).
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
export LLM2VEC_DEVICE_MAP="${LLM2VEC_DEVICE_MAP:-cuda:0}"
export TEXT_ENCODER=llm2vec_nf4
export TEXT_ENCODER_MODE=local
export TEXT_ENCODER_DEVICE="${TEXT_ENCODER_DEVICE:-cuda:0}"
export TEXT_ENCODER_DTYPE=bfloat16
export TEXT_ENCODER_QUANTIZATION=none

export GMR_ROOT="${GMR_ROOT:-${SCRIPT_DIR}/web-version/gmr}"
export SMPLX_BODY_MODELS="${SMPLX_BODY_MODELS:-${GMR_ROOT}/assets/body_models}"
export KIMODO_T800="${KIMODO_T800:-1}"
export KIMODO_T800_SKIN="${KIMODO_T800_SKIN:-white}"
export KIMODO_T800_HIDE_HUMAN_MESH="${KIMODO_T800_HIDE_HUMAN_MESH:-0}"
export KIMODO_DEMO_MODEL="${KIMODO_DEMO_MODEL:-Kimodo-SMPLX-RP-v1}"

echo "Kimodo demo + NF4 local encoder on ${TEXT_ENCODER_DEVICE}"
echo "T800: enable 'Show T800 robot (retargeted)' in Visualize → Body options (on by default)."
echo "If model load fails (VRAM): terminal 1: ./run_textencoder.sh  terminal 2: ./run_demo_api.sh"
exec kimodo_demo --model "${KIMODO_DEMO_MODEL}" "$@"
