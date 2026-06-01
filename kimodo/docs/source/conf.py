# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = "Kimodo"
copyright = "2026, NVIDIA"
author = "NVIDIA"

version = ""
release = ""

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx_copybutton",
    "myst_parser",
    "sphinx_design",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": False,
}
autodoc_typehints = "none"

autosummary_generate = True

# Avoid initialization issues for optional native libs
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")


class Mock:
    """Mock class for imports that can't be satisfied."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return Mock()

    def __getattr__(self, name):
        if name in ("__file__", "__path__"):
            return "/dev/null"
        if name == "__version__":
            # Some libraries (e.g. safetensors) parse torch.__version__ with
            # packaging.version.Version, so this must be a valid PEP 440 string.
            return "0.0.0"
        if name == "__signature__":
            return None
        if name == "__mro_entries__":
            return lambda bases: ()
        return Mock()

    def __getitem__(self, name):
        return Mock()

    def __iter__(self):
        return iter([])

    def __or__(self, other):
        return Mock()

    def __ror__(self, other):
        return Mock()


mock_modules = [
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.optim",
    "torch.distributed",
    "torch.cuda",
    "torch.utils",
    "torch.utils.data",
    "lightning",
    "lightning.fabric",
    "lightning_fabric",
    "pytorch_lightning",
    "tensordict",
    "pydantic",
    "pydantic.dataclasses",
    "pydantic_core",
    "mujoco",
    "isaacgym",
    "isaacgymenvs",
    "genesis",
    "omni",
    "wandb",
    "hydra",
    "omegaconf",
    "tqdm",
    "trimesh",
    "pyvista",
    "smplx",
    "smpl",
    "scipy",
    "scipy.spatial",
    "scipy.spatial.transform",
    "peft",
    "transformers",
    "safetensors",
    "safetensors.torch",
    "sklearn",
    "PIL",
    "cv2",
    "rich",
    "rich.progress",
    "skimage",
    "imageio",
    "openmesh",
    "gym",
    "easydict",
    "dm_control",
    "dm_control.mjcf",
    "dm_control.mujoco",
    "matplotlib",
    "matplotlib.pyplot",
]

for mod in mock_modules:
    sys.modules[mod] = Mock()

autodoc_mock_imports = mock_modules

templates_path = ["_templates"]
exclude_patterns = ["api_reference/_generated/**"]

language = "en"

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"

# -- Options for HTML output -------------------------------------------------

html_theme = "nvidia_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/logo-placeholder.svg"
html_show_sourcelink = False

html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
}

toc_object_entries_show_parents = "hide"

htmlhelp_basename = "Kimododoc"

# -- Options for intersphinx -------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# Generate heading anchors so cross-doc links like path.md#fragment resolve (local ids).
myst_heading_anchors = 4

# Required so `:::{dropdown}` and other fenced directives in .md files are parsed (not shown as plain text).
myst_enable_extensions = ["colon_fence"]


def setup(app):
    app.add_css_file("custom.css")
