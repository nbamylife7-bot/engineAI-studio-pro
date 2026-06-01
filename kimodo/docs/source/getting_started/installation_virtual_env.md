# Installation With Virtual Environment

> Note: the repo was tested with Python 3.10+ and PyTorch 2.0+.

## Create Enviroment
We recommend setting up a separate virtual environment for Kimodo to avoid dependency conflicts.

### Using venv
```bash
python -m venv venv
source venv/bin/activate
```

### Using Conda
```bash
conda create -n kimodo python=3.10
conda activate kimodo
```

## Install Dependencies

### Install PyTorch
First, make sure to install a version of [PyTorch](https://pytorch.org/get-started/locally/) that works with your system and CUDA version. We suggest anything over PyTorch 2.0. We strongly suggest using a GPU-capable version of PyTorch to generate motions in a reasonable amount of time.

### (Optional) Clone Modified Viser Library
The interactive demo relies on [a fork of Viser](https://github.com/nv-tlabs/kimodo-viser) that implements a timeline interface and more. If you want to have an editable install of this version of Viser (i.e., you expect to modify it), clone and install it within the `kimodo` directory using:
```bash
git clone https://github.com/nv-tlabs/kimodo-viser.git
pip install -e kimodo-viser
```

### Install Kimodo
Next, install Kimodo run this command from the base of repo:
```bash
pip install -e .
```
This results in a single editable install for Kimodo and the MotionCorrection package.

If you plan to use the demo, you can instead run:
```bash
pip install -e ".[all]"
```
This will install our [Viser fork](https://github.com/nv-tlabs/kimodo-viser) (if not already installed in the previous step) and the [SOMA body model](https://github.com/NVlabs/SOMA-X).

Next, head over to the [Quick Start](quick_start.md) page to test out your installation by generating some motions.
