#!/usr/bin/env bash
# Create GitHub repo engineAI-studio-pro and push main (requires GITHUB_USER + GITHUB_TOKEN).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SCRIPT_DIR}"

REPO_NAME="${GITHUB_REPO_NAME:-engineAI-studio-pro}"
BRANCH="${GITHUB_BRANCH:-main}"

if [[ -f "${SCRIPT_DIR}/.env.github" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/.env.github"
  set +a
fi

: "${GITHUB_USER:?Set GITHUB_USER (export or .env.github)}"
: "${GITHUB_TOKEN:?Set GITHUB_TOKEN (export or .env.github)}"

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  git init -b "${BRANCH}"
fi

export GIT_AUTHOR_NAME="${GIT_AUTHOR_NAME:-${GITHUB_USER}}"
export GIT_AUTHOR_EMAIL="${GIT_AUTHOR_EMAIL:-${GITHUB_USER}@users.noreply.github.com}"
export GIT_COMMITTER_NAME="${GIT_COMMITTER_NAME:-${GIT_AUTHOR_NAME}}"
export GIT_COMMITTER_EMAIL="${GIT_COMMITTER_EMAIL:-${GIT_AUTHOR_EMAIL}}"
# Per-repo identity only (do not touch global git config)
git config user.name "${GIT_AUTHOR_NAME}"
git config user.email "${GIT_AUTHOR_EMAIL}"

git add -A
if git diff --cached --quiet; then
  echo "Nothing to commit (working tree clean)."
else
  git commit -m "$(cat <<EOF
Initial release: EngineAI Studio Pro

Kimodo CUDA demo, NF4 text encoder, T800 GMR retargeting, install scripts and docs.
EOF
)"
fi

API="https://api.github.com/repos/${GITHUB_USER}/${REPO_NAME}"
code="$(curl -s -o /tmp/gh_create_repo.json -w "%{http_code}" \
  -X POST \
  -H "Authorization: Bearer ${GITHUB_TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  "https://api.github.com/user/repos" \
  -d "{\"name\":\"${REPO_NAME}\",\"description\":\"Kimodo motion generation on NVIDIA CUDA with NF4 encoder and EngineAI T800 retargeting\",\"private\":false,\"auto_init\":false}")"

if [[ "${code}" == "201" ]]; then
  echo "Created https://github.com/${GITHUB_USER}/${REPO_NAME}"
elif [[ "${code}" == "422" ]]; then
  echo "Repository likely already exists (HTTP 422)."
else
  echo "Create repo returned HTTP ${code}:" >&2
  cat /tmp/gh_create_repo.json >&2
  exit 1
fi

REMOTE_URL="https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${REPO_NAME}.git"
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "https://github.com/${GITHUB_USER}/${REPO_NAME}.git"
else
  git remote add origin "https://github.com/${GITHUB_USER}/${REPO_NAME}.git"
fi

git push "${REMOTE_URL}" "${BRANCH}:${BRANCH}" 2>/dev/null || git push -u origin "${BRANCH}"

# Drop token from stored remote URL
git remote set-url origin "https://github.com/${GITHUB_USER}/${REPO_NAME}.git"

echo ""
echo "Published: https://github.com/${GITHUB_USER}/${REPO_NAME}"
