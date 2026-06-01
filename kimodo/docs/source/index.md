# Kimodo Documentation

<div class="hero">
  <div class="hero-title">Kimodo</div>
  <div class="hero-subtitle">
    Scaling controllable human motion generation
  </div>
  <div class="hero-actions">
    <a href="getting_started/installation.html">Get Started</a>
    <a class="secondary" href="interactive_demo/index.html">Interactive Demo</a>
    <a class="secondary" href="https://github.com/nv-tlabs/kimodo">GitHub</a>
    <a class="secondary" href="https://research.nvidia.com/labs/sil/projects/kimodo/assets/kimodo_tech_report.pdf">Tech Report</a>
  </div>
</div>


## Overview

Kimodo is a **ki**nematic **mo**tion **d**iffusi**o**n model trained on a large-scale (700 hours) commercially-friendly optical motion capture dataset. The model generates high-quality 3D human and robot motions, and is controlled through text prompts and an extensive set of constraints such as full-body pose keyframes, end-effector positions/rotations, 2D paths, and 2D waypoints. See the [project page](https://research.nvidia.com/labs/sil/projects/kimodo/) for details.

## Highlights

<div class="card-grid">
  <div class="card">
    <h3>Controlled Generation</h3>
    <p>Text prompts combined with full-body, root, and end-effector constraints.</p>
  </div>
  <div class="card">
    <h3>Human(oid) Support</h3>
    <p>Model variations for both digital humans and humanoid robots.</p>
  </div>
  <div class="card">
    <h3>Interactive Demo</h3>
    <p>Timeline editing, real-time 3D visualization, and example presets.</p>
  </div>
</div>

## Quick links

- [Installation](getting_started/installation.md)
- [Quick Start](getting_started/quick_start.md)
- [Command Line Interface](user_guide/cli.md)
- [Interactive Demo](interactive_demo/index.md)
- [Project Structure](project_structure.md)

```{toctree}
:maxdepth: 3
:caption: Getting Started
:hidden:

getting_started/installation
getting_started/quick_start
```

```{toctree}
:maxdepth: 2
:caption: User Guide
:hidden:

interactive_demo/index
user_guide/cli
user_guide/constraints
user_guide/output_formats
user_guide/configuration
```

```{toctree}
:maxdepth: 2
:caption: Key Concepts
:hidden:

key_concepts/model
key_concepts/limitations
key_concepts/motion_representation
key_concepts/constraints
key_concepts/skeleton
```

```{toctree}
:maxdepth: 2
:caption: Reference
:hidden:

project_structure
project_info
api_reference/index
```
