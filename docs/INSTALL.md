# Installing EngineAI Studio Pro (Linux, NVIDIA)

Guide for a clean machine or new server. Assume you cloned `engineAI-studio-pro` and work from the repository root.

![Demo screenshot](sceen.png)

**Tested on:** NVIDIA RTX 50xx (Blackwell), **12 GB VRAM**. Single-process demo uses about **7–7.5 GB** GPU memory — **8 GB** may work in theory but is not guaranteed; use the two-process mode if you hit OOM.

## 0. Expected result

After all steps:

1. Conda environment `kimodo-cuda` with PyTorch (CUDA), bitsandbytes, and editable `kimodo` package.
2. Directory `models/kimodo-llm2vec-nf4/` — NF4 encoder (~5 GB).
3. Cache `cache/huggingface/hub/` — at least `nvidia/Kimodo-SMPLX-RP-v1` for the demo.
4. `./run_demo.sh` opens the interactive demo at http://127.0.0.1:7860.

**Weights are not in git** — only code, scripts, and T800 assets. Download models separately (normal).

---

## 1. GPU and driver check

On bare Linux:

```bash
nvidia-smi
```

You should see your GPU and driver version. If the command is missing, install the driver first:

```bash
sudo ./install_system_deps.sh
sudo reboot
nvidia-smi
```

`install_system_deps.sh` installs build tools, `cmake`, `libsimde-dev`, `git`, `git-lfs`, and can help with the NVIDIA driver on Ubuntu.

### WSL2

Install the driver in **Windows**. Inside WSL, `nvidia-smi` may be missing — that is not always a problem: PyTorch can still see the GPU via `/usr/lib/wsl/lib/libcuda.so`. `activate_cuda.sh` adds that path to `LD_LIBRARY_PATH`. Verify with:

```bash
source ./activate_cuda.sh
./scripts/verify_gpu_setup.sh
```

---

## 2. Miniconda (if needed)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
"$HOME/miniconda3/bin/conda" init bash
exec bash
```

---

## 3. Project install

```bash
cd engineAI-studio-pro
./install.sh
```

This creates `kimodo-cuda` (Python 3.10), installs PyTorch with CUDA (default **cu124**), dependencies from `requirements-cuda.txt`, and `pip install -e ./kimodo[demo,t800]`. Building `motion_correction` can take several minutes — requires `libsimde-dev` from step 1.

All-in-one with sudo:

```bash
./install.sh --system-deps
```

### RTX 50xx (Blackwell, sm_120)

On RTX 5090/5070 etc., stable PyTorch cu124 often reports that sm_120 is unsupported. Before `./install.sh`:

```bash
export PYTORCH_CUDA_INDEX=https://download.pytorch.org/whl/nightly/cu128
./install.sh
```

See also [GPU.md](GPU.md).

### Post-install check

```bash
source ./activate_cuda.sh
./scripts/verify_gpu_setup.sh
```

Expect: `torch.cuda`, matmul, `bitsandbytes`, `import kimodo`, `motion_correction`.

---

## 4. Models

### 4.1. NF4 text encoder (no HF login)

```bash
source ./activate_cuda.sh
./download_nf4.sh
```

[matbee/kimodo-llm2vec-nf4](https://huggingface.co/matbee/kimodo-llm2vec-nf4) is public. Files go to `models/kimodo-llm2vec-nf4/`.

### 4.2. Kimodo diffusion (Hugging Face login)

1. Register at https://huggingface.co  
2. Accept the model license, e.g. https://huggingface.co/nvidia/Kimodo-SMPLX-RP-v1  
3. Create a Read token: https://huggingface.co/settings/tokens  

```bash
hf auth login
./scripts/download_kimodo_models.sh
```

Default cache: `cache/huggingface/` (set by `activate_cuda.sh`).

If you copied a cache from macOS and logs show `config.yaml missing` with ~1 KB files — broken XSym stubs instead of LFS:

```bash
./scripts/repair_xsym_hf_cache.sh
./scripts/download_kimodo_models.sh
```

Copy from another machine:

```bash
./copy_kimodo_checkpoints.sh /path/to/huggingface/hub
./scripts/repair_xsym_hf_cache.sh
```

### 4.3. SMPL-X for human mesh and T800

SMPL-X body models are **not in git** (license). Download from https://smpl-x.is.tue.mpg.de/ and extract to:

`web-version/gmr/assets/body_models/smplx/`

Without this, the demo may use SOMA or a reduced human mesh; full SMPL-X + T800 needs these files.

---

## 5. Run the demo

```bash
source ./activate_cuda.sh
./run_demo.sh
```

Browser: http://127.0.0.1:7860  

1. **Load model** → `Kimodo-SMPLX-RP-v1`  
2. Enter a prompt → **Generate**  
3. Robot: **Visualize → Show T800 robot (retargeted)** (after generation)

### Low VRAM (~8–12 GB)

If `./run_demo.sh` runs out of memory, split encoder and demo (works well on 12 GB; also try on 8 GB):

Terminal 1:

```bash
source ./activate_cuda.sh
./run_textencoder.sh
```

Terminal 2:

```bash
source ./activate_cuda.sh
./run_demo_api.sh
```

Encoder listens on port 9550; the demo calls it via API so the demo process mostly holds diffusion weights.

### T800

GMR dependencies are usually installed with `[t800]` during `install.sh`. If not:

```bash
./setup_t800.sh
```

---

## 6. Troubleshooting

**`CUDA not available`**  
Driver, reboot, on WSL run `source ./activate_cuda.sh`. Reinstall torch with CUDA.

**`no kernel image` / sm_120 warning**  
Use PyTorch nightly **cu128** (RTX 50xx section above).

**`Some modules are dispatched on the CPU or the disk` when loading NF4**  
Not enough VRAM in one process — use `run_textencoder.sh` + `run_demo_api.sh`. Check `TEXT_ENCODER_DEVICE=cuda:0` and `LLM2VEC_DEVICE_MAP=cuda:0` (`activate_cuda.sh` sets these).

**`No module named motion_correction`**  
`sudo apt install libsimde-dev`, then:

```bash
source ./activate_cuda.sh
pip install -e ./kimodo --no-build-isolation
```

**`config.yaml is missing` / weights ~1 KB**  
`./scripts/repair_xsym_hf_cache.sh`, then download the model again.

**HF 401 / gated**  
`hf auth login` and accept the license on the model page.

**Human disappears after Generate, only robot remains**  
Enable **Show Mesh** or ensure `KIMODO_T800_HIDE_HUMAN_MESH` is not set to `1`.

**Horizon EngineAI logos**  
Controlled by `KIMODO_HORIZON_LOGO*` in `.env.example` (sky branding N/E/S/W).

---

## 7. Directory layout

```
engineAI-studio-pro/
  install.sh, activate_cuda.sh, run_demo.sh
  download_nf4.sh
  scripts/verify_gpu_setup.sh
  scripts/download_kimodo_models.sh
  scripts/publish_to_github.sh
  kimodo/                 # Kimodo package sources
  web-version/gmr/        # T800 retarget
  models/                 # created by download_nf4.sh (not in git)
  cache/                  # HF hub (not in git)
  docs/INSTALL.md         # this file
  docs/GPU.md
```

---

## 8. Versions (reference build)

- Python 3.10, conda env `kimodo-cuda`
- PyTorch 2.x + CUDA 12.4 (or cu128 nightly on RTX 50xx)
- bitsandbytes ≥ 0.43
- NVIDIA driver ≥ 525

Kimodo is from NVIDIA; NF4 encoder from matbee. See Hugging Face for model terms of use.
