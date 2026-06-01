# GPU vs CPU: что где выполняется

Kimodo CUDA рассчитан на **NVIDIA GPU**. Ниже — что обязательно на GPU, что на CPU по задумке, и как проверить окружение.

## Быстрая проверка

```bash
source ./activate_cuda.sh
./scripts/verify_gpu_setup.sh
```

## Обязательно на GPU (NVIDIA)

| Компонент | Где | Переменные / примечание |
|-----------|-----|-------------------------|
| **Diffusion (Kimodo denoiser)** | `cuda` | `device` в `load_model()` → `Kimodo.to(cuda)` |
| **NF4 text encoder (matbee)** | `cuda:0` | `LLM2VEC_QUANTIZE=nf4`, `TEXT_ENCODER_DEVICE=cuda:0`; bitsandbytes **не работает на CPU** |
| **Генерация motion** | GPU | DDIM + denoiser forward |
| **Inference encode (local)** | GPU | `LLM2VecEncoder` с `device_map={"": 0}` (см. `text_encoder_quantization.py`) |

Порядок загрузки в одном процессе: **сначала NF4, потом diffusion** (`load_model.py` → `_load_kimodo_nf4_text_encoder_first`), чтобы `device_map=auto` не сбрасывал веса на CPU/диск.

## Режимы запуска и VRAM

| Режим | Скрипты | VRAM в demo-процессе |
|--------|---------|----------------------|
| **Один процесс** | `./run_demo.sh` | NF4 (~5 GB) + diffusion (~4–8 GB) — нужно **≥12 GB**, лучше **16 GB** |
| **Два процесса** | `./run_textencoder.sh` + `./run_demo_api.sh` | В demo только diffusion — **рекомендуется для 12 GB** |

## Намеренно на CPU (нормально)

| Компонент | Почему |
|-----------|--------|
| **Загрузка checkpoint** | `map_location="cpu"` в `loading.py` — веса потом переносятся на GPU |
| **Embedding cache (диск)** | NumPy на диске; при hit тензор снова на GPU для denoiser |
| **LLM2Vec encode output** | В `llm2vec.py` embeddings `.cpu()` для кэша/батча (можно оптимизировать позже) |
| **T800 retarget (GMR)** | NumPy + MuJoCo IK в `retarget/t800.py` — CPU; результат только для визуализации |
| **Postprocess motion_correction** | C++ extension, CPU (после GPU diffusion) |
| **Viser UI / экспорт NPZ** | CPU |

## WSL2

- Драйвер на **Windows**; в Linux часто нет `nvidia-smi`, но есть `/usr/lib/wsl/lib/libcuda.so`.
- [`activate_cuda.sh`](../activate_cuda.sh) выставляет `LD_LIBRARY_PATH=/usr/lib/wsl/lib`.

## RTX 50xx (Blackwell, sm_120)

Стабильный PyTorch **cu124** не содержит ядер для sm_120. Нужен **nightly cu128**:

```bash
source ./activate_cuda.sh
pip uninstall -y torch torchvision
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
./scripts/verify_gpu_setup.sh
```

Или при установке:

```bash
export PYTORCH_CUDA_INDEX=https://download.pytorch.org/whl/nightly/cu128
./install.sh --skip-torch   # если env уже есть — только переустановить torch
```

## Переменные GPU

| Переменная | По умолчанию | Назначение |
|------------|--------------|------------|
| `CUDA_VISIBLE_DEVICES` | `0` | Какая GPU |
| `TEXT_ENCODER_DEVICE` | `cuda:0` | Локальный NF4 |
| `LLM2VEC_DEVICE_MAP` | `cuda:0` | Не использовать `auto` при нехватке VRAM |
| `TEXT_ENCODER_MODE` | `local` в `run_demo.sh`, `api` в `run_demo_api.sh` | Где крутится encoder |

## Что делать, если «уехало на CPU»

1. Сообщение про `dispatched on the CPU or the disk` → мало VRAM: **`run_demo_api.sh`** + **`run_textencoder.sh`**.
2. `CUDA not available` → драйвер / WSL `LD_LIBRARY_PATH` / переустановить PyTorch с CUDA.
3. NF4 на CPU → проверьте `LLM2VEC_QUANTIZE=nf4` и `TEXT_ENCODER_DEVICE=cuda:0`.
