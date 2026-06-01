#!/usr/bin/env bash
# Check CUDA, PyTorch GPU, bitsandbytes, Kimodo, motion_correction, T800 deps.
# Run: source ./activate_cuda.sh && ./scripts/verify_gpu_setup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/activate_cuda.sh"

FAIL=0
ok() { echo "  OK   $*"; }
warn() { echo "  WARN $*"; }
bad() { echo "  FAIL $*"; FAIL=1; }
note() { echo "  NOTE $*"; }

echo "=== EngineAI Studio Pro — GPU check ==="
echo "Python: $(python -V 2>&1)"
echo "LD_LIBRARY_PATH (WSL): ${LD_LIBRARY_PATH:-<unset>}"
echo ""

echo "=== 1. NVIDIA driver / CUDA runtime ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1 | ok
else
  if [[ -d /usr/lib/wsl/lib ]] && [[ -f /usr/lib/wsl/lib/libcuda.so.1 ]]; then
    note "no nvidia-smi in WSL, but libcuda in /usr/lib/wsl/lib (Windows driver) — OK for WSL2"
  else
    bad "nvidia-smi missing and no WSL libcuda — install NVIDIA driver"
  fi
fi

echo ""
echo "=== 2. PyTorch CUDA ==="
python <<'PY'
import sys
import torch

print(f"  torch {torch.__version__}")
if not torch.cuda.is_available():
    print("  FAIL torch.cuda.is_available() is False")
    sys.exit(1)
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(f"  OK   GPU: {name} (sm_{cap[0]}{cap[1]})")
try:
    x = torch.randn(64, 64, device="cuda")
    y = (x @ x).sum().item()
    print(f"  OK   cuda matmul: {y:.4f}")
except Exception as e:
    print(f"  FAIL cuda matmul: {e}")
    sys.exit(1)
# Blackwell RTX 50xx needs recent PyTorch (cu128 nightly)
if cap[0] >= 12 and "cu128" not in torch.__version__ and "+cu12" in torch.__version__:
    print("  WARN RTX 50xx (sm_120): use PyTorch nightly cu128, see docs/GPU.md")
PY
[[ $? -eq 0 ]] || FAIL=1

echo ""
echo "=== 3. bitsandbytes (NF4) ==="
python -c "import bitsandbytes as bnb; print('  OK   bitsandbytes', bnb.__version__)" || bad "bitsandbytes import failed"

echo ""
echo "=== 4. Kimodo + motion_correction ==="
(cd "${KIMODO_REPO}" && python <<'PY')
import sys
try:
    import kimodo
    print("  OK   kimodo import")
except ImportError as e:
    print(f"  FAIL kimodo: {e}")
    sys.exit(1)
try:
    import motion_correction
    print("  OK   motion_correction (C++ postprocess)")
except ImportError as e:
    print(f"  FAIL motion_correction: {e}")
    print("       sudo apt install libsimde-dev && pip install -e kimodo --no-deps --no-build-isolation")
    sys.exit(1)
from kimodo.retarget import is_t800_available, missing_t800_dependencies
if is_t800_available():
    print("  OK   T800 / GMR dependencies")
else:
    print("  WARN T800:", ", ".join(missing_t800_dependencies()))
PY
[[ $? -eq 0 ]] || FAIL=1

echo ""
echo "=== 5. Text encoder device (NF4 = GPU only) ==="
(cd "${KIMODO_REPO}" && python <<'PY')
import os
import torch
from kimodo.device_utils import resolve_text_encoder_device

os.environ.setdefault("LLM2VEC_QUANTIZE", "nf4")
dev = resolve_text_encoder_device("cuda")
print(f"  OK   resolve_text_encoder_device -> {dev}")
if dev == "cpu":
    print("  FAIL NF4 cannot run on CPU with current config")
    raise SystemExit(1)
PY
[[ $? -eq 0 ]] || FAIL=1

echo ""
echo "=== 6. Model files (optional) ==="
NF4="${LLM2VEC_LOCAL_BASE:-${SCRIPT_DIR}/models/kimodo-llm2vec-nf4}/config.json"
if [[ -f "${NF4}" ]] && [[ $(stat -c%s "${NF4}" 2>/dev/null || echo 0) -gt 200 ]]; then
  ok "NF4 config: ${NF4}"
else
  note "NF4 not found — run: ./download_nf4.sh"
fi
SMPLX_CFG="${HUGGINGFACE_HUB_CACHE:-${SCRIPT_DIR}/cache/huggingface/hub}/models--nvidia--Kimodo-SMPLX-RP-v1"
if find "${SMPLX_CFG}" -name config.yaml -size +500c 2>/dev/null | grep -q .; then
  ok "Kimodo-SMPLX-RP-v1 in HF cache"
else
  warn "Kimodo-SMPLX-RP-v1 missing — hf auth login && ./scripts/download_kimodo_models.sh"
fi

echo ""
if [[ "${FAIL}" -eq 0 ]]; then
  echo "=== GPU runtime OK (models — see section 6) ==="
else
  echo "=== Critical errors — see docs/INSTALL.md ==="
  exit 1
fi
