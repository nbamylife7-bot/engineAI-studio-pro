#!/usr/bin/env bash
# Kimodo interactive demo on Apple Silicon (MPS).
# Mirrors the project-root run_demo.sh env vars; keep both in sync.
#
# Text encoder: LLM2Vec (Meta-Llama-3-8B) via API — start ./run_textencoder.sh first.
# F2LLM is not used on Mac.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PARENT_ACTIVATE="${REPO_ROOT}/../activate_kimodo.sh"

if [[ -f "${PARENT_ACTIVATE}" ]]; then
  # shellcheck source=/dev/null
  source "${PARENT_ACTIVATE}"
else
  echo "activate_kimodo.sh not found at ${PARENT_ACTIVATE}" >&2
  exit 1
fi

KIMODO_BIN="${KIMODO_BIN:-/opt/miniconda3/envs/kimodo/bin}"

# Unsupported MPS ops fall back to CPU (PyTorch).
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TEXT_ENCODER="${TEXT_ENCODER:-llm2vec}"
export TEXT_ENCODER_MODE="${TEXT_ENCODER_MODE:-local}"
export TEXT_ENCODER_URL="${TEXT_ENCODER_URL:-http://127.0.0.1:9550/}"
export TEXT_ENCODER_DEVICE=cpu
export TEXT_ENCODER_DTYPE=float16

export KIMODO_DEMO_MODEL="${KIMODO_DEMO_MODEL:-Kimodo-SMPLX-RP-v1}"

# Mac memory: CPU skinning, smaller chunks, lazy cache for 6s+ clips (180 frames @ 30fps).
export KIMODO_SKIN_DEVICE="${KIMODO_SKIN_DEVICE:-cpu}"
export KIMODO_SKIN_CHUNK_SIZE="${KIMODO_SKIN_CHUNK_SIZE:-32}"
export KIMODO_LAZY_SKIN_FRAMES="${KIMODO_LAZY_SKIN_FRAMES:-300}"
export KIMODO_T800_SKIN="${KIMODO_T800_SKIN:-white}"

exec "${KIMODO_BIN}/kimodo_demo" --model "${KIMODO_DEMO_MODEL}" "$@"
