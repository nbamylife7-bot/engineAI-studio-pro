<p align="center">
  <img src="./assets/banner.png" alt="Banner" width="100%">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-76B900.svg" alt="License"></a>
  <a href="https://research.nvidia.com/labs/sil/projects/kimodo/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
  <a href="https://research.nvidia.com/labs/sil/projects/kimodo/docs/index.html"><img src="https://img.shields.io/badge/docs-online-green.svg" alt="Documentation"></a>
</p>

## Overview

Kimodo is a **ki**nematic **mo**tion **d**iffusi**o**n model trained on a large-scale (700 hours) commercially-friendly optical motion capture dataset. The model generates high-quality 3D human and robot motions, and is controlled through text prompts and an extensive set of constraints such as full-body pose keyframes, end-effector positions/rotations, 2D paths, and 2D waypoints. Full details of the model architecture and training are available in the [technical report](https://research.nvidia.com/labs/sil/projects/kimodo/assets/kimodo_tech_report.pdf).

This repository provides:
- **Inference**: code and CLI to generate motions on both human and robot skeletons
- **Interactive Demo**: easily author motions with a timeline interface of text prompts and kinematic controls
- **Annotations**: [additional text descriptions](https://huggingface.co/datasets/nvidia/SEED-Timeline-Annotations) for the [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed) dataset, including fine-grained temporal descriptions
- _[Coming Soon]_ **Benchmark**: test cases and evaluation code built on the [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed) dataset to evaluate motion generation models based on text and constraint-following abilities

<div align="center">
  <img src="assets/teaser.gif" width="1280">
</div>

## Kimodo Models

Several variations of Kimodo-v1 are available trained on various skeletons and datasets. All models support text-to-motion and kinematic controls.

> Note: models will be downloaded automatically when attempting to generate from the CLI or Interactive Demo, so there is no need to download them manually

| Model | Skeleton | Training Data | Release Date | Hugging Face | License |
|:-------|:-------------|:------:|:------:|:-------------:|:-------------:|
| **Kimodo-SOMA-RP-v1** | [SOMA](https://github.com/NVlabs/SOMA-X) | [Bones Rigplay 1](https://bones.studio/datasets#rp01) | March 16, 2026 | [Link](https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1) | [NVIDIA Open Model](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |
| **Kimodo-G1-RP-v1** | [Unitree G1](https://github.com/unitreerobotics/unitree_mujoco/tree/main/unitree_robots/g1) | [Bones Rigplay 1](https://bones.studio/datasets#rp01) | March 16, 2026  | [Link](https://huggingface.co/nvidia/Kimodo-G1-RP-v1) | [NVIDIA Open Model](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |
| **Kimodo-SOMA-SEED-v1** | [SOMA](https://github.com/NVlabs/SOMA-X) | [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed) | March 16, 2026  | [Link](https://huggingface.co/nvidia/Kimodo-SOMA-SEED-v1) | [NVIDIA Open Model](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |
| **Kimodo-G1-SEED-v1** | [Unitree G1](https://github.com/unitreerobotics/unitree_mujoco/tree/main/unitree_robots/g1) | [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed) | March 16, 2026  | [Link](https://huggingface.co/nvidia/Kimodo-G1-SEED-v1) | [NVIDIA Open Model](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |
| **Kimodo-SMPLX-RP-v1** | [SMPL-X](https://github.com/vchoutas/smplx) | [Bones Rigplay 1](https://bones.studio/datasets#rp01) | March 16, 2026  | [Link](https://huggingface.co/nvidia/Kimodo-SMPLX-RP-v1) | [NVIDIA R&D Model](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-internal-scientific-research-and-development-model-license/) |

By default, we recommend using the models trained on the full Bones Rigplay 1 dataset (700 hours of mocap) for your motion generation needs.
The models trained on BONES-SEED use 288 hours of [publicly available mocap data](https://huggingface.co/datasets/bones-studio/seed) so are less capable, but are useful for comparing your own trained models on the same dataset. Soon, we will be releasing a benchmark to make it easy to compare motion generation models trained on BONES-SEED.

## Getting Started

Please see the full documentation for detailed installation instructions, how to use the CLI and Interactive Demo, and other practical tips for generating motions with Kimodo:

**[Full Documentation](https://research.nvidia.com/labs/sil/projects/kimodo/docs)**
- [Quick Start Guide](https://research.nvidia.com/labs/sil/projects/kimodo/docs/getting_started/quick_start.html)
- [Installation Instructions](https://research.nvidia.com/labs/sil/projects/kimodo/docs/getting_started/installation.html)
- [Interactive Motion Authoring Demo](https://research.nvidia.com/labs/sil/projects/kimodo/docs/interactive_demo/index.html)
- [Command-Line Interface](https://research.nvidia.com/labs/sil/projects/kimodo/docs/user_guide/cli.html)
- [API Reference](https://research.nvidia.com/labs/sil/projects/kimodo/docs/api_reference/index.html)

Some notes on installation environment:
- Kimodo requires ~17GB of VRAM to generate locally, primarily due to the text embedding model
- The model has been most extensively tested on GeForce RTX 3090, GeForce RTX 4090, and NVIDIA A100 GPUs, but should work on other recent cards with sufficient VRAM
- This repo was developed on Linux, though Windows should work especially if using Docker

Before getting started with motion generation, please review the [best practices](https://research.nvidia.com/labs/sil/projects/kimodo/docs/key_concepts/limitations.html) and be aware of [model limitations](https://research.nvidia.com/labs/sil/projects/kimodo/docs/key_concepts/limitations.html#limitations).

## Interactive Motion Authoring Demo

<div align="center">
  <img src="assets/demo_screenshot.png" width="1000">
</div>

</br>

**[Demo Documentation and Tutorial](https://research.nvidia.com/labs/sil/projects/kimodo/docs/interactive_demo/index.html)**

The web-based interactive demo provides an intuitive interface for generating motions with any of the Kimodo model variations. After installation, the demo can be launched with the `kimodo_demo` command. It runs locally on http://127.0.0.1:7860. Open this URL in your browser to access the interface (or use port forwarding if set up on a server).

### Demo Features
- **Multiple Characters**: Supports generating with the SOMA, G1, and SMPL-X versions of Kimodo
- **Text Prompts**: Enter one or more natural language descriptions of desired motions on the timeline
- **Timeline Editor**: Add and edit keyframes and constrained intervals on multiple constraint tracks
- **Constraint Types**:
  - Full-Body: Complete joint position constraints at specific frames
  - 2D Root: Define waypoints or full paths to follow on the ground plane
  - End-Effectors: Control hands and feet positions/rotations
- **Constraint Editing**: Editing mode allows for re-posing of constraints or adjusting waypoints
- **3D Visualization**: Real-time rendering of generated motions with skeleton and skinned mesh options
- **Playback Controls**: Preview generated motions with adjustable playback speed
- **Multiple Samples**: Generate and compare multiple motion variations
- **Examples**: Load pre-existing examples to better understand Kimodo's capabilities
- **Export**: Save constraints and generated motions for later use

## Command-Line Interface

**[CLI Documentation and Examples](https://research.nvidia.com/labs/sil/projects/kimodo/docs/user_guide/cli.html)**

Motions can also be generated directly from the command line with the `kimodo_gen` command or by running `python -m kimodo.scripts.generate` directly.

**Key Arguments:**
- `prompt`: A single text description or sequence of texts for the desired motion (required)
- `--model`: Which Kimodo model to use for generation
- `--duration`: Motion duration in seconds
- `--num_samples`: Number of motion variations to generate
- `--constraints`: Constraint file to control the generated motion (e.g., saved from the web demo)
- `--diffusion_steps`: Number of denoising steps
- `--cfg_type` / `--cfg_weight`: Classifier-free guidance (`nocfg`, `regular` with one weight, or `separated` with two weights for text vs. constraints); see the [CLI docs](https://research.nvidia.com/labs/sil/projects/kimodo/docs/user_guide/cli.html#classifier-free-guidance-cfg)
- `--no-postprocess`: Flag to disable foot skate and constraint cleanup post-processing
- `--seed`: Random seed for reproducible results

The script supports different output formats depending on which skeleton is used. By default, a custom NPZ format is saved that is compatible with the web demo.
For Kimodo-G1 models, the motion can be saved in the standard MuJoCo qpos CSV format.
For Kimodo-SMPLX, motion can be saved in the standard AMASS npz format for compability with existing pipelines.

### Default NPZ Output Format
Generated motions are saved as NPZ files containing:
- `posed_joints`: Global joint positions `[T, J, 3]`
- `global_rot_mats`: Global joint rotation matrices `[T, J, 3, 3]`
- `local_rot_mats`: Local (parent-relative) joint rotation matrices `[T, J, 3, 3]`
- `foot_contacts`: Foot contact labels [left heel, left toe, right heel, right toes] `[T, 4]`
- `smooth_root_pos`: Smoothed root representations outputted from the model `[T, 3]`
- `root_positions`: The (non-smoothed) trajectory of the actual root joint (e.g., pelvis) `[T, 3]`
- `global_root_heading`: The heading direction output from the model `[T, 2]`

`T` the number of frames and `J` the number of joints.

## Low-Level Python API

**[Model API Documentation](https://research.nvidia.com/labs/sil/projects/kimodo/docs/api_reference/model.html#kimodo.model.kimodo_model.Kimodo.__call__)**

For maximum flexibility, the low-level model inference API can be called directly, rather than going through our high-level CLI.
This allows for advanced model configuration including classifier-free guidance weights and parameters related to transitions in multi-prompt sequences.

## Downstream Robotics Applications of Kimodo

### Visualizing G1 Motions with MuJoCo

<div align="center">
  <img src="assets/mujoco_result.gif" width="800">
</div>

After generating motions on the G1 robot skeleton and saving to the MuJoCo qpos CSV file format, they can be easily used and visualized within MuJoCo.
A minimal visualization script is available with:
```
python -m kimodo.scripts.mujoco_load
```
Make sure to edit the script to correctly point to your CSV file and install Mujoco before running this.

### Tracking Generated Motions with ProtoMotions

<div align="center">
  <img src="assets/protomotions_results.gif" width="1280">
</div>

[ProtoMotions](https://github.com/NVlabs/ProtoMotions) is a GPU-accelerated simulation and learning framework for training physically simulated digital humans and humanoid robots. The Kimodo NPZ and CSV output formats are both compatible with ProtoMotions making it easy to train physics-based policies with generated motions from Kimodo. ProtoMotions supports outputs on both the SOMA skeleton and Unitree G1

After generating motions with Kimodo, head over to the [ProtoMotions docs](https://github.com/NVlabs/ProtoMotions?tab=readme-ov-file#-motion-authoring-with-kimodo) to see how to import them.

### Retargeting Motions to Other Robots with GMR

<div align="center">
  <img src="assets/gmr_results.gif" width="1280">
</div>

Motions generated by Kimodo-SMPLX can be retargeted to other robots using [General Motion Retargeting (GMR)](https://github.com/YanjieZe/GMR).
GMR supports the AMASS NPZ format out of the box, so simply generate motions with Kimodo and use `--output` to save; the AMASS NPZ is written to `stem_amass.npz` (single sample) or in the output folder (multiple samples). Then, use the [SMPL-X to Robot script](https://github.com/YanjieZe/GMR?tab=readme-ov-file#retargeting-from-smpl-x-amass-omomo-to-robot) in GMR to retarget to any supported robot. For example:
```
# run within GMR codebase
python scripts/smplx_to_robot.py --smplx_file /path/to/saved/amass_format.npz --robot booster_t1
```

## Timeline Annotations for BONES-SEED

As detailed in the [tech report](https://research.nvidia.com/labs/sil/projects/kimodo/assets/kimodo_tech_report.pdf), Kimodo is trained using fine-grained temporal text annotations of mocap clips.
While the full [Rigplay 1](https://bones.studio/datasets#rp01) dataset is proprietary, we have released the temporal segmentations for the public [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed) subset.
These annotations are already included in the BONES-SEED dataset, but the standalone labels and additional information about them is [available on HuggingFace](https://huggingface.co/datasets/nvidia/SEED-Timeline-Annotations).


## Related Humanoid Work at NVIDIA
Kimodo is part of a larger effort to enable humanoid motion data for robotics, physical AI, and other applications.

Check out these related works:
* [SOMA Body Model](https://github.com/NVlabs/SOMA-X) - a unified parameteric human body model
* [BONES-SEED Dataset](https://huggingface.co/datasets/bones-studio/seed) - a large scale human(oid) motion capture dataset in SOMA and G1 format
* [ProtoMotions](https://github.com/NVlabs/ProtoMotions) - simulation and learning framework for training physically simulated human(oid)s
* [SOMA Retargeter](https://github.com/NVIDIA/soma-retargeter) - SOMA to G1 retargeting tool
* [GEM](https://github.com/NVlabs/GEM-X) - human motion reconstruction from video
* [GEAR SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl) - humanoid behavior foundation model for physical robots

## Citation

If you use this code in your research, please cite:

```bibtex
@article{Kimodo2026,
  title={Kimodo: Scaling Controllable Human Motion Generation},
  author={Rempe, Davis and Petrovich, Mathis and Yuan, Ye and Zhang, Haotian and Peng, Xue Bin and Jiang, Yifeng and Wang, Tingwu and Iqbal, Umar and Minor, David and de Ruyter, Michael and Li, Jiefeng and Tessler, Chen and Lim, Edy and Jeong, Eugene and Wu, Sam and Hassani, Ehsan and Huang, Michael and Yu, Jin-Bey and Chung, Chaeyeon and Song, Lina and Dionne, Olivier and Kautz, Jan and Yuen, Simon and Fidler, Sanja},
  journal={arXiv},
  year={2026}
}
```

## License

This codebase is licensed under [Apache-2.0](LICENSE). Note that model checkpoints and data are licensed separately as indicated on the HuggingFace download pages.

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

## Acknowledgments

This project builds upon excellent open-source projects:
- [Viser](https://github.com/nerfstudio-project/viser) for 3D motion authoring demo
- [LLM2Vec](https://github.com/McGill-NLP/llm2vec) for text encoding

## Contact

For questions or issues, plese open an issue on this repository or reach out directly to the authors.

---
