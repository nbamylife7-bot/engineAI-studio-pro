# EngineAI Studio Pro

**engineAI-studio-pro** — Linux + **NVIDIA GPU** studio for Kimodo motion generation, NF4 text encoding, and **EngineAI T800** retargeting.

Text encoder: [matbee/kimodo-llm2vec-nf4](https://huggingface.co/matbee/kimodo-llm2vec-nf4) (~5 GB disk, ~5 GB VRAM).

**Weights are not in this repository.** After clone: `./download_nf4.sh` and (for diffusion) `./scripts/download_kimodo_models.sh`.

## Install from GitHub (zero to demo)

| Step | Doc |
|------|-----|
| Clone → driver → conda → `./install.sh` → download models → run | **[docs/GITHUB_INSTALL.md](docs/GITHUB_INSTALL.md)** |
| Full Linux checklist | **[docs/LINUX_FRESH_INSTALL.md](docs/LINUX_FRESH_INSTALL.md)** |
| GPU vs CPU, WSL2, RTX 50xx | **[docs/GPU_RUNTIME.md](docs/GPU_RUNTIME.md)** |

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/engineAI-studio-pro.git
cd engineAI-studio-pro
sudo ./install_system_deps.sh   # once
./install.sh
source ./activate_cuda.sh
./scripts/verify_gpu_setup.sh

./download_nf4.sh
hf auth login                    # gated nvidia/Kimodo-* only
./scripts/download_kimodo_models.sh

./run_demo.sh                    # http://127.0.0.1:7860
```

**RTX 50xx (Blackwell, sm_120):** PyTorch **nightly cu128** — [docs/GPU_RUNTIME.md](docs/GPU_RUNTIME.md).

**12 GB VRAM:** `./run_textencoder.sh` + `./run_demo_api.sh`.

## Requirements

| | |
|--|--|
| GPU | NVIDIA **≥12 GB VRAM** (16+ GB for single-process `run_demo.sh`) |
| OS | Ubuntu 22.04/24.04 or Debian 12 x86_64; **WSL2** supported |
| Driver | `nvidia-smi` on bare metal, or WSL2 + Windows NVIDIA driver |
| Python | 3.10 (conda env `kimodo-cuda`, set via `KIMODO_CUDA_ENV`) |
| HF | Not required for matbee NF4; **yes** for `nvidia/Kimodo-*` diffusion |

## Verify GPU

```bash
source ./activate_cuda.sh
./scripts/verify_gpu_setup.sh
```

## Publish / update GitHub

See [docs/PUBLISH_GITHUB.md](docs/PUBLISH_GITHUB.md) and `./scripts/publish_to_github.sh`.

## Environment

`source ./activate_cuda.sh` sets `ENGINEAI_STUDIO_ROOT`, `HF_HOME=./cache/huggingface`, WSL `LD_LIBRARY_PATH`, NF4 paths. See `.env.example`.

## Disk (full install)

| | ~size |
|--|--------|
| NF4 | ~5 GB |
| One Kimodo checkpoint | ~1.1 GB |
| conda + PyTorch | 8–12 GB |
| **Total** | **~40–50 GB** |

## T800 robot

1. Model **Kimodo-SMPLX-RP-v1** in the UI.
2. **Visualize → Show T800 robot (retargeted)**.
3. SMPL-X body models: `web-version/gmr/assets/body_models/` (not in git — download separately).

```bash
./setup_t800.sh
./run_demo.sh
```

## Layout

```
engineAI-studio-pro/
  docs/GITHUB_INSTALL.md
  docs/GPU_RUNTIME.md
  install.sh / activate_cuda.sh
  scripts/verify_gpu_setup.sh
  kimodo-metal-mps-support-main/
  web-version/gmr/
  models/          # gitignored
  cache/           # gitignored
```

## Mac

Use the main Mac Kimodo project, not this CUDA package.
