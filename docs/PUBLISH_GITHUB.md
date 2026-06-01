# Publish **engineAI-studio-pro** to GitHub

## One-time setup

1. Create a [Personal Access Token](https://github.com/settings/tokens) with scope **`repo`** (classic) or fine-grained access to create/push repositories.

2. Export credentials (do **not** commit these):

```bash
export GITHUB_USER="your_github_username"
export GITHUB_TOKEN="ghp_xxxxxxxx"   # or github_pat_...
```

Optional file (gitignored):

```bash
cp .env.github.example .env.github
# edit .env.github, then:
set -a && source .env.github && set +a
```

3. From the project root:

```bash
./scripts/publish_to_github.sh
```

This will:

- `git init` (if needed) and commit tracked files
- Create `https://github.com/$GITHUB_USER/engineAI-studio-pro` if it does not exist
- Push branch `main`

## Manual push

```bash
git init
git add -A
git status   # confirm models/, cache/, .env are NOT staged
git commit -m "Initial release: EngineAI Studio Pro (Kimodo CUDA + T800)"

git remote add origin "https://github.com/YOUR_GITHUB_USERNAME/engineAI-studio-pro.git"
git branch -M main
git push -u origin main
```

Use SSH instead:

```bash
git remote add origin git@github.com:YOUR_GITHUB_USERNAME/engineAI-studio-pro.git
```

## Repository settings (recommended)

- **Description:** Kimodo motion generation on NVIDIA CUDA with NF4 encoder and EngineAI T800 retargeting
- **Topics:** `cuda`, `kimodo`, `robotics`, `motion-generation`, `engineai`
- **Do not** commit: `models/`, `cache/`, `.env`, SMPL-X body weights (see `.gitignore`)

## Rename local folder (optional)

After clone the directory is `engineAI-studio-pro`. Your existing checkout may still be named `kimodo-cuda-nvidia`:

```bash
mv kimodo-cuda-nvidia engineAI-studio-pro
cd engineAI-studio-pro
```

Scripts use their own directory path; renaming is cosmetic.
