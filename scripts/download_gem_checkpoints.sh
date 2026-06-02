#!/usr/bin/env bash
# Download GEM-SMPL preprocessing checkpoints (HMR2, ViTPose, YOLOv8x) from HuggingFace mirrors.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GEM_ROOT="${KIMODO_GEM_ROOT:-${ROOT}/vendor/GENMO}"
DOWNLOAD_ONNX=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --onnx) DOWNLOAD_ONNX=1; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "${GEM_ROOT}/inputs/checkpoints/hmr2" "${GEM_ROOT}/inputs/checkpoints/vitpose"
mkdir -p "${GEM_ROOT}/inputs/pretrained"

source "${ROOT}/activate_cuda.sh" 2>/dev/null || true

HMR2_OUT="${GEM_ROOT}/inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt"
VITPOSE_OUT="${GEM_ROOT}/inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth"
GEM_CKPT_OUT="${GEM_ROOT}/inputs/pretrained/gem_smpl.ckpt"
YOLO_OUT="${GEM_ROOT}/yolov8x.pt"

_hf_download() {
  local repo="$1" file="$2" dest="$3"
  if command -v hf >/dev/null 2>&1; then
    hf download "${repo}" "${file}" --local-dir "$(dirname "${dest}")"
    if [[ -f "$(dirname "${dest}")/${file}" && "$(dirname "${dest}")/${file}" != "${dest}" ]]; then
      mv -f "$(dirname "${dest}")/${file}" "${dest}"
    fi
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "${repo}" "${file}" --local-dir "$(dirname "${dest}")"
  else
    python3 -m pip install -q huggingface_hub
    python3 -c "
from huggingface_hub import hf_hub_download
from pathlib import Path
p = hf_hub_download('${repo}', '${file}')
Path('${dest}').parent.mkdir(parents=True, exist_ok=True)
import shutil
shutil.copy(p, '${dest}')
"
  fi
}

if [[ ! -f "${HMR2_OUT}" ]] || [[ "$(stat -c%s "${HMR2_OUT}" 2>/dev/null || echo 0)" -lt 1000000 ]]; then
  echo "Downloading HMR2 checkpoint …"
  _hf_download "zju3dv/GVHMR" "hmr2/epoch=10-step=25000.ckpt" "${HMR2_OUT}" || \
    _hf_download "nvidia/GEM-X" "hmr2/epoch=10-step=25000.ckpt" "${HMR2_OUT}" || true
fi

if [[ ! -f "${VITPOSE_OUT}" ]] || [[ "$(stat -c%s "${VITPOSE_OUT}" 2>/dev/null || echo 0)" -lt 1000000 ]]; then
  echo "Downloading ViTPose checkpoint …"
  _hf_download "zju3dv/GVHMR" "vitpose/vitpose-h-multi-coco.pth" "${VITPOSE_OUT}" || true
fi

if [[ ! -f "${GEM_CKPT_OUT}" ]] || [[ "$(stat -c%s "${GEM_CKPT_OUT}" 2>/dev/null || echo 0)" -lt 1000000 ]]; then
  echo "Downloading gem_smpl.ckpt …"
  _hf_download "nvidia/GEM-X" "gem_smpl.ckpt" "${GEM_CKPT_OUT}" || true
fi

if [[ ! -f "${YOLO_OUT}" ]] || [[ "$(stat -c%s "${YOLO_OUT}" 2>/dev/null || echo 0)" -lt 100000000 ]]; then
  echo "Downloading YOLOv8x (~131 MB) …"
  if command -v wget >/dev/null 2>&1; then
    wget -q --show-progress -O "${YOLO_OUT}" \
      "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x.pt" || true
  fi
fi

if [[ "${DOWNLOAD_ONNX}" -eq 1 ]]; then
  echo "Downloading GEM-SMPL ONNX bundle (webcam path; optional) …"
  mkdir -p "${GEM_ROOT}/inputs/onnx"
  _hf_download "nvidia/GEM-X" "gem_smpl/onnx/vitpose_coco17.onnx" "${GEM_ROOT}/inputs/onnx/vitpose_coco17.onnx" || true
fi

echo ""
echo "Checkpoint status:"
ls -lah "${HMR2_OUT}" 2>/dev/null || echo "  MISSING: ${HMR2_OUT}"
ls -lah "${VITPOSE_OUT}" 2>/dev/null || echo "  MISSING: ${VITPOSE_OUT}"
ls -lah "${GEM_CKPT_OUT}" 2>/dev/null || echo "  MISSING: ${GEM_CKPT_OUT}"
if [[ -f "${YOLO_OUT}" ]] && [[ "$(stat -c%s "${YOLO_OUT}" 2>/dev/null || echo 0)" -ge 100000000 ]]; then
  ls -lah "${YOLO_OUT}"
else
  echo "ERROR: YOLOv8x required for video demo: ${YOLO_OUT}" >&2
  exit 1
fi
