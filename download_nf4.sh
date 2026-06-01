#!/usr/bin/env bash
# Скачать matbee/kimodo-llm2vec-nf4 (~5 ГБ) — репозиторий открытый, hf auth не обязателен.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/activate_cuda.sh"

if ! command -v hf >/dev/null 2>&1; then
  echo "Нужен Hugging Face CLI: pip install -U 'huggingface_hub[cli]'" >&2
  exit 1
fi

DEST="${LLM2VEC_LOCAL_BASE:-${SCRIPT_DIR}/models/kimodo-llm2vec-nf4}"
mkdir -p "$(dirname "${DEST}")"

# hf_transfer ускоряет загрузку, но опционален (pip install hf_transfer)
if python3 -c "import hf_transfer" 2>/dev/null; then
  export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
else
  export HF_HUB_ENABLE_HF_TRANSFER=0
fi

echo "Downloading matbee/kimodo-llm2vec-nf4 -> ${DEST}"
echo "(авторизация HF не требуется; hf auth login нужен только для чекпоинтов Kimodo diffusion)"

if ! hf download matbee/kimodo-llm2vec-nf4 --local-dir "${DEST}"; then
  echo "" >&2
  echo "Если загрузка отклонена (401/gated), попробуйте: hf auth login" >&2
  exit 1
fi

if [[ ! -d "${DEST}/supervised_adapter" ]]; then
  echo "Warning: ${DEST}/supervised_adapter not found" >&2
  exit 1
fi

echo "OK: NF4 model ready"
echo "  LLM2VEC_LOCAL_BASE=${DEST}"
echo "  LLM2VEC_LOCAL_PEFT=${DEST}/supervised_adapter"
