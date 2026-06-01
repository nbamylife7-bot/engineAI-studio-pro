# Install **engineAI-studio-pro** from GitHub (NVIDIA GPU, from scratch)

Guide for cloning [engineAI-studio-pro](https://github.com/YOUR_GITHUB_USERNAME/engineAI-studio-pro) and running the Kimodo + T800 demo on Linux with an NVIDIA GPU.

## Requirements

- **GPU:** NVIDIA with **≥12 GB VRAM** (16+ GB for `run_demo.sh` with local NF4)
- **OS:** Ubuntu 22.04/24.04 or Debian 12, x86_64 (WSL2 OK — [GPU_RUNTIME.md](GPU_RUNTIME.md))
- **Disk:** ~40–50 GB
- **RAM:** 16+ GB
- **Hugging Face account** for gated `nvidia/Kimodo-*` models

## 1. Clone

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/engineAI-studio-pro.git
cd engineAI-studio-pro
```

> Repository does **not** include model weights (`models/`, `cache/` are gitignored).

## 2. NVIDIA driver

```bash
nvidia-smi
```

If missing:

```bash
sudo ./install_system_deps.sh
sudo reboot
```

See [LINUX_FRESH_INSTALL.md](LINUX_FRESH_INSTALL.md) §2.

## 3. Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
"$HOME/miniconda3/bin/conda" init bash
exec bash
```

## 4. Install EngineAI Studio Pro

```bash
cd engineAI-studio-pro
sudo ./install_system_deps.sh
./install.sh
```

**RTX 50xx (Blackwell):**

```bash
export PYTORCH_CUDA_INDEX=https://download.pytorch.org/whl/nightly/cu128
./install.sh
```

Verify:

```bash
source ./activate_cuda.sh
./scripts/verify_gpu_setup.sh
```

## 5. Download models

### NF4 text encoder (no HF login)

```bash
./download_nf4.sh
```

### Kimodo diffusion (gated)

1. Accept license: https://huggingface.co/nvidia/Kimodo-SMPLX-RP-v1  
2. Read token: https://huggingface.co/settings/tokens  

```bash
hf auth login
./scripts/download_kimodo_models.sh
```

If cache was copied from Mac and files are ~1 KB:

```bash
./scripts/repair_xsym_hf_cache.sh
```

### SMPL-X body models (T800 human mesh)

Download from [SMPL-X](https://smpl-x.is.tue.mpg.de/) into:

`web-version/gmr/assets/body_models/smplx/`

## 6. Run demo

```bash
source ./activate_cuda.sh
./run_demo.sh
```

Open http://127.0.0.1:7860 → Load model → **Kimodo-SMPLX-RP-v1** → Generate.

**12 GB VRAM:**

```bash
./run_textencoder.sh    # terminal 1
./run_demo_api.sh         # terminal 2
```

## 7. T800

- Model: **Kimodo-SMPLX-RP-v1**
- **Show T800 robot (retargeted)** after Generate
- `./setup_t800.sh` if T800 extras were not installed

## Related docs

- [LINUX_FRESH_INSTALL.md](LINUX_FRESH_INSTALL.md)
- [GPU_RUNTIME.md](GPU_RUNTIME.md)
- [PUBLISH_GITHUB.md](PUBLISH_GITHUB.md) — maintainers: push updates
