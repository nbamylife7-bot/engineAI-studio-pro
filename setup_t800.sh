#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/activate_cuda.sh"

pip install -e "${KIMODO_REPO}[t800]"
echo "GMR_ROOT=${GMR_ROOT}"
