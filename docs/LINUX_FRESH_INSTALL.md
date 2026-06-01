# EngineAI Studio Pro (`engineAI-studio-pro`): установка с нуля на Linux

Чеклист для **только что установленной** Ubuntu 22.04/24.04 или Debian 12 (x86_64, NVIDIA GPU).

Краткая версия для публикации на GitHub: **[GITHUB_INSTALL.md](GITHUB_INSTALL.md)**.

## 1. Железо и ОС

| Требование | Минимум |
|------------|---------|
| GPU NVIDIA | **≥12 GB VRAM** (лучше 16+ GB для demo + diffusion) |
| ОС | Ubuntu 22.04/24.04 LTS или Debian 12, **x86_64** |
| Диск | **~25–40 GB** свободно (модели + conda + кэш HF) |
| RAM | **16+ GB** системной памяти |
| Интернет | для драйвера, conda, Hugging Face |

## 2. NVIDIA: драйвер (обязательно до PyTorch)

Проверка:

```bash
nvidia-smi
```

Должны быть видны GPU, версия драйвера и память. Если команды нет:

### Ubuntu (рекомендуется)

```bash
sudo apt update
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers install --gpgpu
# или: sudo ubuntu-drivers autoinstall
sudo reboot
```

После перезагрузки снова: `nvidia-smi`.

### Debian / вручную

- [Драйвер NVIDIA](https://www.nvidia.com/Download/index.aspx) под вашу карту, или
- [CUDA Installation Guide (Linux)](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

**Для PyTorch cu124** обычно достаточно драйвера **≥525** (лучше **535/550+**). Отдельный полный CUDA Toolkit на диск **не обязателен** — runtime идёт с колесом PyTorch.

### WSL2 (Windows + Linux)

- Драйвер ставится в **Windows**; в WSL часто **нет** `nvidia-smi`, но PyTorch видит GPU через `/usr/lib/wsl/lib/libcuda.so`.
- После `source ./activate_cuda.sh` путь к libcuda подставляется автоматически.
- Проверка: `./scripts/verify_gpu_setup.sh` (см. [GPU_RUNTIME.md](GPU_RUNTIME.md)).

## 3. Системные пакеты (один раз)

Из папки `engineAI-studio-pro`:

```bash
sudo ./install_system_deps.sh
```

Ставит: `git`, `curl`, `wget`, `build-essential`, `cmake`, `ninja`, `libsimde-dev`, `python3-dev`, `git-lfs`, при необходимости — драйвер NVIDIA.

Вручную (эквивалент):

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build git git-lfs curl wget \
  libsimde-dev python3-dev ca-certificates
```

## 4. Miniconda (Python 3.10)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
"$HOME/miniconda3/bin/conda" init bash
# перезайти в shell или: source ~/.bashrc
```

## 5. Kimodo + PyTorch CUDA + bitsandbytes

```bash
cd /path/to/engineAI-studio-pro
./install.sh
source ./activate_cuda.sh
```

`install.sh` создаёт env `kimodo-cuda`, ставит:

- PyTorch **cu124** по умолчанию (см. [pytorch.org](https://pytorch.org/get-started/locally/))
- `bitsandbytes`, `accelerate` — NF4 encoder
- `huggingface_hub`, `hf-transfer` — загрузка моделей
- `kimodo[demo,t800]` — демо и T800

**RTX 50xx (Blackwell, compute capability 12.0):** стабильный cu124 не поддерживает sm_120. Перед `./install.sh`:

```bash
export PYTORCH_CUDA_INDEX=https://download.pytorch.org/whl/nightly/cu128
./install.sh
```

Проверка окружения:

```bash
source ./activate_cuda.sh
./scripts/verify_gpu_setup.sh
```

## 6. Hugging Face CLI

```bash
pip install -U "huggingface_hub[cli]"
```

### NF4 text encoder — **без логина**

[matbee/kimodo-llm2vec-nf4](https://huggingface.co/matbee/kimodo-llm2vec-nf4) — **открытый** репозиторий (`gated: false`). Достаточно:

```bash
./download_nf4.sh
```

`hf auth login` для этой модели **не нужен**.

### Когда нужен `hf auth login`

| Модель | Авторизация |
|--------|-------------|
| matbee/kimodo-llm2vec-nf4 | **Нет** |
| Kimodo diffusion (например `Kimodo-SMPLX-RP-v1`) | **Да**, при первой генерации с HF |
| [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | **Да**, если качаете полный bf16 вместо matbee |

```bash
hf auth login   # токен Read: https://huggingface.co/settings/tokens
```

### Скачать NF4 text encoder (~5 GB)

```bash
./download_nf4.sh
```

Ускорение загрузки (опционально):

```bash
export HF_XET_HIGH_PERFORMANCE=1   # в activate_cuda.sh по умолчанию
```

Кэш по умолчанию в этом репозитории: `cache/huggingface/` (`activate_cuda.sh` → `HF_HOME`).

### Kimodo diffusion (gated, nvidia/*)

```bash
hf auth login
# принять лицензию на https://huggingface.co/nvidia/Kimodo-SMPLX-RP-v1
./scripts/download_kimodo_models.sh
```

Если копировали кэш с Mac и видите `config.yaml missing` или файлы ~1 KB — это XSym-заглушки:

```bash
./scripts/repair_xsym_hf_cache.sh
```

Перенос кэша с другой машины (опционально):

```bash
./copy_kimodo_checkpoints.sh /path/to/huggingface/hub
./scripts/repair_xsym_hf_cache.sh
```

## 7. Запуск

```bash
source ./activate_cuda.sh
./download_nf4.sh          # если ещё не скачали

# Вариант A: demo + NF4 local (один процесс)
./run_demo.sh

# Вариант B: два терминала
./run_textencoder.sh       # :9550
./run_demo_api.sh          # :7860
```

## 8. T800 (опционально)

```bash
./setup_t800.sh
export KIMODO_T800=1
./run_demo_api.sh
```

## 9. Типичные проблемы

| Симптом | Решение |
|---------|---------|
| `CUDA not available` | Драйвер / WSL `LD_LIBRARY_PATH`, PyTorch с CUDA |
| `no kernel image` / sm_120 | PyTorch **nightly cu128** — [GPU_RUNTIME.md](GPU_RUNTIME.md) |
| `No module named bitsandbytes` | `pip install bitsandbytes` в env `kimodo-cuda` |
| `SIMDe headers not found` | `sudo apt install libsimde-dev` или `vendor/simde`, пересобрать kimodo |
| `config.yaml missing` / 1 KB weights | `./scripts/repair_xsym_hf_cache.sh` + `download_kimodo_models.sh` |
| HF 401 / gated | `hf auth login` для `nvidia/Kimodo-*`; NF4 matbee без логина |
| OOM / CPU+disk dispatch | `run_textencoder.sh` + `run_demo_api.sh`; `LLM2VEC_DEVICE_MAP=cuda:0` |
| NF4 на CPU | `TEXT_ENCODER_DEVICE=cuda:0` |
| `motion_correction` missing | `libsimde-dev` + `pip install -e kimodo-metal-mps-support-main` |

## 10. Версии (ориентир, 2026)

| Компонент | Версия |
|-----------|--------|
| Python | 3.10 |
| PyTorch | 2.x + **cu124** (RTX 50xx: **cu128 nightly**) |
| bitsandbytes | ≥0.43 |
| transformers | 5.1.0 (в kimodo) |
| peft | ≥0.18 |
| NVIDIA driver | ≥525 (для CUDA 12.x) |
