#!/usr/bin/env bash
# Системные зависимости для чистого Linux (Ubuntu/Debian).
# Запуск: sudo ./install_system_deps.sh
# Или:   INSTALL_SYSTEM_DEPS=1 ./install.sh  (вызовет этот скрипт при наличии sudo)

set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Запустите с sudo: sudo ./install_system_deps.sh" >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive

echo "==> Обновление списка пакетов..."
apt-get update -qq

echo "==> Базовые утилиты (git, curl, архивы)..."
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

echo "==> Сборка C++ (Kimodo MotionCorrection)..."
apt-get install -y --no-install-recommends \
  build-essential \
  cmake \
  ninja-build \
  libsimde-dev \
  python3-dev

# git-lfs для больших файлов HF (если понадобится)
git lfs install --system 2>/dev/null || git lfs install || true

echo "==> NVIDIA driver (проприетарный, рекомендуется для GPU)..."
echo "    Если nvidia-smi уже работает — этот шаг можно пропустить (Ctrl+C)."
echo "    Иначе ставим метапакет open-драйвера (версия зависит от репозитория)."
sleep 2

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "    nvidia-smi уже есть:"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
else
  # Ubuntu 22.04/24.04: ubuntu-drivers; на Debian — вручную из NVIDIA
  if command -v ubuntu-drivers >/dev/null 2>&1; then
    apt-get install -y ubuntu-drivers-common
    ubuntu-drivers install --gpgpu || ubuntu-drivers autoinstall || true
    echo "    После установки драйвера может потребоваться перезагрузка: sudo reboot"
  else
    echo "    Установите драйвер NVIDIA вручную, затем проверьте: nvidia-smi"
    echo "    https://www.nvidia.com/Download/index.aspx"
    echo "    Debian: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/"
  fi
fi

echo ""
echo "==> Готово (системные пакеты)."
echo "    Дальше (без sudo):"
echo "      cd engineAI-studio-pro"
echo "      ./install.sh"
echo "      ./download_nf4.sh   # без hf auth"
