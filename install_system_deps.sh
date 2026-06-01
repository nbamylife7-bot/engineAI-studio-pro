#!/usr/bin/env bash
# System dependencies for a fresh Linux install (Ubuntu/Debian).
# Run: sudo ./install_system_deps.sh
# Or:  INSTALL_SYSTEM_DEPS=1 ./install.sh  (calls this script when sudo is available)

set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run with sudo: sudo ./install_system_deps.sh" >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive

echo "==> Updating package lists..."
apt-get update -qq

echo "==> Base utilities (git, curl, archives)..."
apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  wget \
  git \
  git-lfs \
  unzip \
  tar \
  pkg-config \
  software-properties-common

echo "==> C++ build (Kimodo MotionCorrection)..."
apt-get install -y --no-install-recommends \
  build-essential \
  cmake \
  ninja-build \
  libsimde-dev \
  python3-dev

git lfs install --system 2>/dev/null || git lfs install || true

echo "==> NVIDIA driver (proprietary, recommended for GPU)..."
echo "    If nvidia-smi already works, you can skip this step (Ctrl+C)."
echo "    Otherwise we install the open-driver metapackage (version depends on your repo)."
sleep 2

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "    nvidia-smi already available:"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
else
  if command -v ubuntu-drivers >/dev/null 2>&1; then
    apt-get install -y ubuntu-drivers-common
    ubuntu-drivers install --gpgpu || ubuntu-drivers autoinstall || true
    echo "    Reboot may be required after driver install: sudo reboot"
  else
    echo "    Install the NVIDIA driver manually, then run: nvidia-smi"
    echo "    https://www.nvidia.com/Download/index.aspx"
    echo "    Debian: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/"
  fi
fi

echo ""
echo "==> Done (system packages)."
echo "    Next (no sudo):"
echo "      cd engineAI-studio-pro"
echo "      ./install.sh"
echo "      ./download_nf4.sh   # no hf auth required"
