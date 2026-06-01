# Quick Start

This page provides a quick introduction to motion generation with Kimodo. For detailed explanations, we recommend reviewing the full documentation pages linked in each section.

Before running these commands, follow the [installation guide](installation.md) to install Kimodo in a virtual environment or using Docker.

## Overview: Kimodo Models
Motion generation can be performed with several trained Kimodo models that vary by skeleton and training dataset.

> Note: models will be downloaded automatically when attempting to generate from the CLI or Interactive Demo, so there is no need to download them manually

| Model | Skeleton | Training Data | Release Date | Hugging Face | License |
|-------|------|------|-------------|-------------|----|
| **Kimodo-SOMA-RP-v1** | [SOMA](https://github.com/NVlabs/SOMA-X) | [Bones Rigplay 1](https://bones.studio/datasets#rp01) | March 16, 2026 | [Link](https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1) | [NVIDIA Open Model](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |
| **Kimodo-G1-RP-v1** | [Unitree G1](https://github.com/unitreerobotics/unitree_mujoco/tree/main/unitree_robots/g1) | [Bones Rigplay 1](https://bones.studio/datasets#rp01) | March 16, 2026  | [Link](https://huggingface.co/nvidia/Kimodo-G1-RP-v1) | [NVIDIA Open Model](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |
| **Kimodo-SOMA-SEED-v1** | [SOMA](https://github.com/NVlabs/SOMA-X) | [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed) | March 16, 2026  | [Link](https://huggingface.co/nvidia/Kimodo-SOMA-SEED-v1) | [NVIDIA Open Model](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |
| **Kimodo-G1-SEED-v1** | [Unitree G1](https://github.com/unitreerobotics/unitree_mujoco/tree/main/unitree_robots/g1) | [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed) | March 16, 2026  | [Link](https://huggingface.co/nvidia/Kimodo-G1-SEED-v1) | [NVIDIA Open Model](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |
| **Kimodo-SMPLX-RP-v1** | [SMPL-X](https://github.com/vchoutas/smplx) | [Bones Rigplay 1](https://bones.studio/datasets#rp01) | March 16, 2026  | [Link](https://huggingface.co/nvidia/Kimodo-SMPLX-RP-v1) | [NVIDIA R&D Model](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-internal-scientific-research-and-development-model-license/) |

By default, we recommend using the models trained on the full Bones Rigplay dataset (700 hours of mocap) for your motion generation needs.
The models trained on BONES-SEED use 288 hours of [publicly available mocap data](https://huggingface.co/datasets/bones-studio/seed) so are less capable, but are useful for comparing your own trained models on the same dataset. Soon, we will be releasing a benchmark to make it easy to compare motion generation models trained on BONES-SEED.

### Recommended Hardware
Kimodo requires  ~17GB of VRAM to generate locally, due primarily to the size of the text embedding model.

The model has been most extensively tested on GeForce RTX 3090, GeForce RTX 4090, and NVIDIA A100 GPUs, but it should work on other recent cards with sufficient VRAM.

## Run Text-Encoder Service
Motion generation relies on embedding the input text prompt, which becomes the input to Kimodo. Although it is fine to run the CLI commands and demo on their own, it may be preferred to start the _text encoder service_ in the background, which can be shared across all motion generation requests. This is much more efficient when making many consecutive CLI calls, as it avoids needing to instantiate the large text encoder every time.

To start the text encoder service:
```bash
kimodo_textencoder
```

The first run of the service will take a while as it downloads the embedding model. We recommend running this in the background or in a separate terminal where it will stay open and usable by other scripts.

If you are using the Docker set up, the service can alternatively be started in the container with:
```bash
docker compose up text-encoder
```

> Note: when the text encoder is initialized, the transformers library will report several unexpected and missing layers for LLM2Vec. These are expected and can be safely ignored.

## Command-Line Text-to-Motion Generation
**[CLI Documentation](../user_guide/cli.md)**

You can generate motions from the command line using the generate script:

```bash
kimodo_gen "A person walks forward." \
    --model Kimodo-SOMA-RP-v1 \
    --duration 5.0 \
    --output output
```

The `--model` command corresponds to the model name in the table above. The output motion will be saved using the stem name given by `--output` in the Kimodo [output format](../user_guide/output_formats.md). For a detailed description of all generation arguments, including how to generate motion with constraints, see the full [CLI documentation](../user_guide/cli.md).

If you set up Kimodo with Docker, you can instead run generation inside the Docker container, replacing `kimodo_gen XXX` with `docker compose run --rm demo kimodo_gen XXX`. If you will be running generation multiple times, it is better to start the `demo` container (e.g., in another terminal or in the background), and then run commands inside it with `docker compose exec demo kimodo_gen XXX`.


## Interactive Motion Authoring Demo
**[Demo Documentation](../interactive_demo/index.md)**

The demo allows easily generating motions with an intuitive control interface for text prompting and constraints.

The demo can be started with:
```bash
kimodo_demo
```

The demo is a webapp that will run on [http://localhost:7860](http://localhost:7860). Open this URL in your browser to access the interface.

If you are using Docker, the demo can be launched with:
```bash
docker compose up demo
```
or if you want to start the demo and text encoder service (explained below) at the same time, use:
```bash
docker compose up
```

<details>
<summary>Additional Tips for Docker</summary>

You may find the following commands useful if running Kimodo within the Docker containers. In the example commands below, you can also replace `demo` by `text-encoder`:

**Check logs:**

```bash
docker compose logs demo
```

**Stop service:**

```bash
docker compose stop demo
```

**Restart service:**

```bash
docker compose restart demo
```

**Stop and remove everything:**

```bash
docker compose down
```

</details>
