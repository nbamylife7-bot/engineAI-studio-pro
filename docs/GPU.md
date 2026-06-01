# GPU and memory

Summary: diffusion and the NF4 encoder should run on NVIDIA GPU. Some post-processing and T800 retarget run on CPU — that is expected.

## Verify setup

```bash
source ./activate_cuda.sh
./scripts/verify_gpu_setup.sh
```

## GPU workloads

- Kimodo diffusion (denoiser, DDIM)
- NF4 LLM2Vec (`LLM2VEC_QUANTIZE=nf4`, `TEXT_ENCODER_DEVICE=cuda:0`)
- bitsandbytes cannot run NF4 on CPU — if you see “dispatched on CPU/disk”, check VRAM and the two-process mode

Load order in a single process: NF4 first, then diffusion (`kimodo/model/load_model.py`). Otherwise `device_map=auto` may place weights on CPU.

## VRAM

Measured on **RTX 50xx / 12 GB**: one-process demo peaks around **7–7.5 GB** (NF4 + diffusion). **8 GB** may be enough in theory — not fully tested; keep the two-process fallback.

| Mode | Commands | Notes |
|------|----------|--------|
| Single process | `./run_demo.sh` | ~7–7.5 GB observed; tested on 12 GB RTX 50xx |
| Two processes | `./run_textencoder.sh` + `./run_demo_api.sh` | If OOM on 8–12 GB or you want more headroom |

## CPU (expected)

- Checkpoint load via `map_location=cpu`, then `.to(cuda)`
- `motion_correction` (C++)
- T800 / GMR retarget
- Embedding disk cache, export

## WSL2

Driver lives in Windows. `activate_cuda.sh` sets `LD_LIBRARY_PATH=/usr/lib/wsl/lib`.

## RTX 50xx (sm_120)

Stable cu124 has no kernels for Blackwell. Install:

```bash
export PYTORCH_CUDA_INDEX=https://download.pytorch.org/whl/nightly/cu128
./install.sh
```

Or torch only:

```bash
pip uninstall -y torch torchvision
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

## Useful variables

| Variable | Default |
|----------|---------|
| `CUDA_VISIBLE_DEVICES` | `0` |
| `TEXT_ENCODER_DEVICE` | `cuda:0` |
| `LLM2VEC_DEVICE_MAP` | `cuda:0` |
| `TEXT_ENCODER_MODE` | `local` in `run_demo.sh`, `api` in `run_demo_api.sh` |
| `KIMODO_FAST_MATMUL` | `1` (TF32 on RTX 30xx+) |

See also `.env.example`.
