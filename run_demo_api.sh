#!/usr/bin/env bash
# Demo via API: NF4 text encoder in a separate process (lower VRAM in demo).
# Terminal 1: ./run_textencoder.sh
# Terminal 2: ./run_demo_api.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/activate_cuda.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TEXT_ENCODER=llm2vec_nf4
export TEXT_ENCODER_MODE=api
export TEXT_ENCODER_URL="${TEXT_ENCODER_URL:-http://127.0.0.1:9550/}"

export GMR_ROOT="${GMR_ROOT:-${SCRIPT_DIR}/web-version/gmr}"
export SMPLX_BODY_MODELS="${SMPLX_BODY_MODELS:-${GMR_ROOT}/assets/body_models}"
export KIMODO_T800="${KIMODO_T800:-1}"
export KIMODO_T800_SKIN="${KIMODO_T800_SKIN:-white}"
export KIMODO_T800_IK_SAFETY="${KIMODO_T800_IK_SAFETY:-1}"
export KIMODO_T800_SMOOTH="${KIMODO_T800_SMOOTH:-1}"
export KIMODO_DEMO_MODEL="${KIMODO_DEMO_MODEL:-Kimodo-SMPLX-RP-v1}"

cd "${KIMODO_REPO}"
exec kimodo_demo --model "${KIMODO_DEMO_MODEL}" "$@"
