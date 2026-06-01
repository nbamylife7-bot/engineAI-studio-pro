#!/usr/bin/env bash
# Activate kimodo-cuda environment (Linux + NVIDIA).
# Usage: source ./activate_cuda.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${KIMODO_CUDA_ENV:-kimodo-cuda}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}" 2>/dev/null || {
    echo "Environment ${ENV_NAME} not found. Run: ./install.sh" >&2
    return 1 2>/dev/null || exit 1
  }
elif [[ -f "${SCRIPT_DIR}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/.venv/bin/activate"
else
  echo "No conda and no .venv. Run ./install.sh" >&2
  return 1 2>/dev/null || exit 1
fi

export ENGINEAI_STUDIO_ROOT="${SCRIPT_DIR}"
export KIMODO_CUDA_ROOT="${ENGINEAI_STUDIO_ROOT}"
export KIMODO_REPO="${SCRIPT_DIR}/kimodo"
# Editable pip may still point at the old directory name
_legacy_kimodo="${SCRIPT_DIR}/kimodo-metal-mps-support-main"
if [[ -d "${KIMODO_REPO}" ]] && [[ ! -e "${_legacy_kimodo}" ]]; then
  ln -sfn kimodo "${_legacy_kimodo}"
fi

# WSL2: libcuda from Windows host (required for torch.cuda)
if [[ -d /usr/lib/wsl/lib ]]; then
  case ":${LD_LIBRARY_PATH:-}:" in
    *:/usr/lib/wsl/lib:*) ;;
    *) export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" ;;
  esac
fi

# Use HF cache under ./cache/huggingface without blocking on hub API (see load_model.py)
export LOCAL_CACHE="${LOCAL_CACHE:-true}"
export GMR_ROOT="${SCRIPT_DIR}/web-version/gmr"
export SMPLX_BODY_MODELS="${GMR_ROOT}/assets/body_models"
export KIMODO_T800_IK_SAFETY="${KIMODO_T800_IK_SAFETY:-1}"

# Local diffusion cache (nvidia/Kimodo-*), see copy_kimodo_checkpoints.sh
export HF_HOME="${HF_HOME:-${SCRIPT_DIR}/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
# Faster HF downloads (Xet); see huggingface_hub docs
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

# matbee/kimodo-llm2vec-nf4 (https://huggingface.co/matbee/kimodo-llm2vec-nf4)
export LLM2VEC_DEVICE_MAP="${LLM2VEC_DEVICE_MAP:-cuda:0}"
export LLM2VEC_QUANTIZE="${LLM2VEC_QUANTIZE:-nf4}"
export LLM2VEC_LOCAL_BASE="${LLM2VEC_LOCAL_BASE:-${SCRIPT_DIR}/models/kimodo-llm2vec-nf4}"
export LLM2VEC_LOCAL_PEFT="${LLM2VEC_LOCAL_PEFT:-${LLM2VEC_LOCAL_BASE}/supervised_adapter}"

export TEXT_ENCODER="${TEXT_ENCODER:-llm2vec_nf4}"
export TEXT_ENCODER_DEVICE="${TEXT_ENCODER_DEVICE:-cuda:0}"
export TEXT_ENCODER_DTYPE="${TEXT_ENCODER_DTYPE:-bfloat16}"
export TEXT_ENCODER_QUANTIZATION="${TEXT_ENCODER_QUANTIZATION:-none}"

if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/.env"
  set +a
fi
