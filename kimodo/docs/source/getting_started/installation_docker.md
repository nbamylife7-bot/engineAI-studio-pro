# Installation With Docker

> Note: the first time building and running with Docker can take several minutes, please be patient.

## Clone Modified Viser Library
The interactive demo relies on [a fork of Viser](https://github.com/nv-tlabs/kimodo-viser) that implements a timeline interface and more. Clone it within the `kimodo` directory before building with Docker using:
```bash
git clone https://github.com/nv-tlabs/kimodo-viser.git
```

## Quick Install

Before running Docker, make sure your Hugging Face token is available at
`~/.cache/huggingface/token` on the host, for example by running
`hf auth login` once outside the container (see the [Installation](installation.md) instructions).

The easiest way to build and immediately run the interactive demo webapp (with the text-encoder service) in one command is:

```bash
docker compose up -d --build
```

## Step-by-Step Installation

Alternatively, you can first build with:

```bash
docker compose build
```

This builds `text-encoder` and `demo` containers corresponding to the text encoding service and the interactive motion authoring webapp, respectively. Please see the [quick start guide](quick_start.md) for more information on these.

<details>

<summary>Advanced Configuration of Dependencies</summary>

This repo uses:
- `docker_requirements.in`: human-maintained, top-level dependencies
- `docker_requirements.txt`: pinned lockfile (automatically generated)

Notes:
- We keep a lockfile for **reproducible Docker builds** (so a rebuild next week pulls the same deps).
- The lockfile intentionally **omits `torch`/CUDA wheels** because the Docker base image
  (`nvcr.io/nvidia/pytorch`) already provides a tested PyTorch build (avoids slow installs and CUDA mismatches).

</details>
<br>

After building, you will need to manually start the text-encoder service before doing any motion generation:
```bash
docker compose up text-encoder
```
Note, the first time running this command will take a long time as the Llama-based text encoder is downloaded.

Finally, to start the interactive demo:
```bash
docker compose up demo
```

For more information on using the Docker setup, see the [Quick Start](quick_start.md) guide next.
