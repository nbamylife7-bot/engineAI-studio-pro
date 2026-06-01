#!/usr/bin/env bash
# Установка Kimodo для NVIDIA CUDA (Linux).
# Чистая система: сначала sudo ./install_system_deps.sh — см. docs/LINUX_FRESH_INSTALL.md
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${KIMODO_CUDA_ENV:-kimodo-cuda}"
KIMODO_REPO="${SCRIPT_DIR}/kimodo-metal-mps-support-main"
INSTALL_SYSTEM="${INSTALL_SYSTEM_DEPS:-0}"
PYTORCH_CUDA_INDEX="${PYTORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu124}"

usage() {
  cat <<EOF
Usage: ./install.sh [options]

  --system-deps     sudo apt: build-essential, cmake, simde, git, nvidia (Ubuntu/Debian)
  --skip-torch      не переустанавливать PyTorch
  -h, --help        эта справка

Перед первой установкой на чистом Linux:
  docs/LINUX_FRESH_INSTALL.md
EOF
}

SKIP_TORCH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --system-deps) INSTALL_SYSTEM=1; shift ;;
    --skip-torch) SKIP_TORCH=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ ! -d "${KIMODO_REPO}" ]]; then
  echo "Не найден ${KIMODO_REPO}." >&2
  exit 1
fi

if [[ "${INSTALL_SYSTEM}" == "1" ]]; then
  if [[ -x "${SCRIPT_DIR}/install_system_deps.sh" ]]; then
    echo "==> Системные зависимости (sudo)..."
    sudo "${SCRIPT_DIR}/install_system_deps.sh"
  else
    echo "install_system_deps.sh not found" >&2
    exit 1
  fi
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: nvidia-smi не найден. Установите драйвер NVIDIA и перезагрузитесь." >&2
  echo "         См. docs/LINUX_FRESH_INSTALL.md" >&2
fi

# --- Conda ---
if ! command -v conda >/dev/null 2>&1; then
  echo "Conda не найдена."
  echo "Установите Miniconda: https://docs.conda.io/en/latest/miniconda.html"
  echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
  echo "  bash Miniconda3-latest-Linux-x86_64.sh"
  exit 1
fi

# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
  echo "==> Conda env ${ENV_NAME} (python 3.10)..."
  conda create -n "${ENV_NAME}" python=3.10 -y
fi
conda activate "${ENV_NAME}"

# SIMDe для MotionCorrection (Linux)
if [[ -d /usr/include/simde ]]; then
  export CMAKE_PREFIX_PATH="/usr:${CMAKE_PREFIX_PATH:-}"
elif [[ -d /usr/local/include/simde ]]; then
  export CMAKE_PREFIX_PATH="/usr/local:${CMAKE_PREFIX_PATH:-}"
elif [[ -d "${SCRIPT_DIR}/vendor/simde" ]]; then
  export CMAKE_PREFIX_PATH="${SCRIPT_DIR}/vendor/simde:${CMAKE_PREFIX_PATH:-}"
  echo "==> SIMDe: vendor/simde (без apt libsimde-dev)"
else
  echo "WARNING: libsimde-dev не найден. sudo apt install libsimde-dev" >&2
fi

if [[ "${SKIP_TORCH}" != "1" ]]; then
  echo "==> PyTorch (CUDA) from ${PYTORCH_CUDA_INDEX}..."
  pip install --upgrade pip wheel setuptools
  pip install torch torchvision --index-url "${PYTORCH_CUDA_INDEX}"
fi

echo "==> CUDA Python deps (bitsandbytes, huggingface_hub, cmake, ...)..."
pip install -r "${SCRIPT_DIR}/requirements-cuda.txt"

echo "==> Kimodo [demo,t800] (сборка motion_correction может занять несколько минут)..."
pip install -e "${KIMODO_REPO}[demo,t800]"

echo "==> Проверка CUDA + bitsandbytes..."
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available — проверьте nvidia-smi и драйвер'
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print('CUDA OK:', name, f'(sm_{cap[0]}{cap[1]})')
x = torch.randn(32, 32, device='cuda')
(x @ x).sum().item()
print('cuda matmul OK')
import bitsandbytes as bnb
print('bitsandbytes OK:', bnb.__version__)
if cap[0] >= 12 and 'cu128' not in torch.__version__:
    print('WARNING: RTX 50xx (sm_120) needs PyTorch nightly cu128 — docs/GPU_RUNTIME.md')
"

if ! command -v hf >/dev/null 2>&1; then
  pip install -U "huggingface_hub[cli]"
fi

echo ""
echo "=============================================="
echo "Установка Kimodo CUDA завершена."
echo ""
echo "  source ./activate_cuda.sh"
echo "  ./download_nf4.sh                # ~5 GB NF4 (без hf auth)"
echo "  hf auth login && ./scripts/download_kimodo_models.sh   # gated diffusion"
echo "  ./scripts/verify_gpu_setup.sh"
echo "  ./run_demo.sh  или  run_textencoder + run_demo_api"
echo ""
echo "Документация: docs/GITHUB_INSTALL.md  docs/GPU_RUNTIME.md"
echo "=============================================="
