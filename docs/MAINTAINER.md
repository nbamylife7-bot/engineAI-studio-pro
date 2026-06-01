# Publishing the repository

## Do not commit

Already in `.gitignore`: `cache/`, `models/`, `.env`, `.env.github`, `body_models/`, `vendor/simde/`, `*.egg-info/`, large gifs under `kimodo/assets/`.

Before push:

```bash
git status
./scripts/verify_gpu_setup.sh   # optional, on a GPU machine
```

Never commit tokens in `.env.github`.

## Push to GitHub

1. Create a fine-grained or classic PAT on GitHub with **repo** scope (for private repos) or public_repo (for public).

2. Configure credentials (gitignored):

```bash
cp .env.github.example .env.github
# Edit: GITHUB_USER=your_username
#       GITHUB_TOKEN=ghp_...
```

3. Publish:

```bash
./scripts/publish_to_github.sh
```

The script commits staged changes (if any), creates `engineAI-studio-pro` on your account if missing, pushes `main`, and removes the token from the stored remote URL.

Manual alternative:

```bash
git remote add origin https://github.com/YOUR_USER/engineAI-studio-pro.git
git push -u origin main
```

Use a credential helper or `gh auth login` instead of embedding the token in the remote URL.

## Repository size

About ~180 MB in git (code + T800 meshes + G1 assets). Users download model weights separately.
