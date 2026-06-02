#!/usr/bin/env bash
# Install NVIDIA GEM-SMPL (GENMO) as an optional sidecar for video → SMPL-X motion import.
# License: GENMO code is NVIDIA OneWay Noncommercial — research/evaluation only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GEM_ROOT="${KIMODO_GEM_ROOT:-${ROOT}/vendor/GENMO}"

if [[ ! -d "${GEM_ROOT}/.git" ]]; then
  echo "Cloning NVlabs/GENMO into ${GEM_ROOT} …"
  mkdir -p "$(dirname "${GEM_ROOT}")"
  git clone --depth 1 https://github.com/NVlabs/GENMO.git "${GEM_ROOT}"
else
  echo "GENMO already cloned at ${GEM_ROOT}"
fi

cd "${GEM_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv …"
  pip install uv
fi

if [[ ! -d .venv ]]; then
  uv venv .venv --python 3.10
fi
# shellcheck source=/dev/null
source .venv/bin/activate

_GEM_PY="$(pwd)/.venv/bin/python"
_gem_install_torch() {
  if command -v uv >/dev/null 2>&1; then
    if [[ "${GEM_TORCH_INDEX}" == *nightly* ]]; then
      uv pip install --python "${_GEM_PY}" --pre torch torchvision --index-url "${GEM_TORCH_INDEX}"
    else
      uv pip install --python "${_GEM_PY}" torch torchvision --index-url "${GEM_TORCH_INDEX}"
    fi
  else
    if [[ "${GEM_TORCH_INDEX}" == *nightly* ]]; then
      pip install --pre torch torchvision --index-url "${GEM_TORCH_INDEX}"
    else
      pip install torch torchvision --index-url "${GEM_TORCH_INDEX}"
    fi
  fi
}

GEM_TORCH_INDEX="${PYTORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu124}"
if [[ "${PYTORCH_CUDA_INDEX:-}" == *cu128* ]]; then
  echo "Using PYTORCH_CUDA_INDEX=${PYTORCH_CUDA_INDEX}"
elif command -v nvidia-smi >/dev/null 2>&1 || [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
  _smi="$(command -v nvidia-smi 2>/dev/null || echo /usr/lib/wsl/lib/nvidia-smi)"
  if "${_smi}" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | grep -q '^12\.'; then
    GEM_TORCH_INDEX="https://download.pytorch.org/whl/nightly/cu128"
    echo "Detected Blackwell GPU (sm_120); using PyTorch nightly cu128."
  fi
fi

echo "Installing GEM-SMPL dependencies (may pull default PyPI torch) …"
bash scripts/install_env.sh

echo "Installing PyTorch from ${GEM_TORCH_INDEX} (after install_env) …"
_gem_install_torch

echo ""
echo "=== GEM-SMPL install (manual steps) ==="
echo "1. SMPL-X body model: register at https://smpl-x.is.tue.mpg.de/"
echo "   Place SMPLX_NEUTRAL.npz under:"
echo "   ${GEM_ROOT}/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz"
echo ""
echo "2. HMR2 + ViTPose checkpoints (for video demo preprocessing):"
echo "   bash ${ROOT}/scripts/download_gem_checkpoints.sh"
echo ""
echo "3. GEM checkpoint (auto-downloads on first run, or):"
echo "   huggingface-cli download nvidia/GEM-X gem_smpl.ckpt --local-dir inputs/pretrained"
echo ""
echo "4. YOLOv8x at ${GEM_ROOT}/yolov8x.pt — default person detector (KIMODO_GEM_DETECTOR=yolov8)"
echo "   YOLOX+ByteTrack — optional (KIMODO_GEM_DETECTOR=yolox); needs: uv pip install onnxruntime-gpu"
echo ""
echo "Add to your .env:"
echo "  KIMODO_GEM_ROOT=${GEM_ROOT}"
echo "  KIMODO_GEM=1"
echo ""
echo "Test CLI:"
echo "  source ${ROOT}/activate_cuda.sh"
echo "  python ${ROOT}/scripts/gem_video_to_amass.py --video /path/to/clip.mp4 --static-cam"
