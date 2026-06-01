# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EngineAI logo billboards in the sky band (N/E/S/W), above the ground horizon."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import viser.transforms as tf

if TYPE_CHECKING:
    import viser
    from viser import ImageHandle


def sky_height_for_distance(distance: float, *, elevation_deg: float) -> float:
    """Place logo center above the ground-line horizon at ``elevation_deg``."""
    return float(distance) * math.tan(math.radians(float(elevation_deg)))


def _prepare_logo_rgba(logo_path: Path) -> np.ndarray:
    """RGBA with transparent fringe; flip rows for viser/Three.js texture upload."""
    from PIL import Image

    with Image.open(logo_path) as img:
        rgba = np.array(img.convert("RGBA"), dtype=np.uint8, copy=True)

    # Viser uses THREE.TextureLoader (flipY): match screen-up with numpy row 0 = top.
    rgba = np.flipud(rgba).copy()
    # Mirror so text reads from the stage (not through the back face).
    rgba = np.fliplr(rgba).copy()

    rgb = rgba[:, :, :3].astype(np.int16)
    alpha = rgba[:, :, 3].astype(np.int16)
    # Drop near-white halos if present.
    near_white = (rgb.min(axis=2) > 245) & (alpha > 0)
    rgba[near_white, 3] = 0
    # Soften low-alpha fringe onto transparent.
    fringe = (alpha > 0) & (alpha < 32)
    rgba[fringe, 3] = 0

    return rgba


def _billboard_wxyz_toward_origin(position: np.ndarray) -> np.ndarray:
    """Orientation for ``add_image`` so the logo faces the origin, upright (Y-up)."""
    pos = np.asarray(position, dtype=np.float64).reshape(3)
    outward = pos / (np.linalg.norm(pos) + 1e-12)

    # Viser image mesh applies Rx(pi); world normal = -R[:,2] when R maps group -> world.
    z_axis = outward
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x_axis = np.cross(world_up, z_axis)
    norm_x = np.linalg.norm(x_axis)
    if norm_x < 1e-8:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        x_axis /= norm_x
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis) + 1e-12

    rotation = np.stack([x_axis, y_axis, z_axis], axis=1)
    return tf.SO3.from_matrix(rotation).wxyz


def add_engineai_horizon_logos(
    client: "viser.ClientHandle",
    logo_path: Path,
    *,
    distance: float,
    height: float,
    logo_height_m: float = 5.0,
) -> list["ImageHandle"]:
    """Sky branding: four billboards far from the stage, facing the origin."""
    if not logo_path.is_file():
        return []

    rgba = _prepare_logo_rgba(logo_path)
    img_h, img_w = rgba.shape[:2]
    aspect = float(img_w) / float(max(img_h, 1))
    render_height = float(logo_height_m)
    render_width = render_height * aspect

    d = float(distance)
    # Slightly below the sky-band default so logos sit closer to the horizon line.
    y = float(height) * 0.88
    placements = (
        ("north", np.array([0.0, y, -d], dtype=np.float64)),
        ("south", np.array([0.0, y, d], dtype=np.float64)),
        ("east", np.array([d, y, 0.0], dtype=np.float64)),
        ("west", np.array([-d, y, 0.0], dtype=np.float64)),
    )

    handles: list[ImageHandle] = []
    for name, position in placements:
        handle = client.scene.add_image(
            f"/horizon/engineai_{name}",
            rgba,
            render_width=render_width,
            render_height=render_height,
            format="png",
            cast_shadow=False,
            receive_shadow=False,
            wxyz=_billboard_wxyz_toward_origin(position),
            position=tuple(float(v) for v in position),
        )
        handles.append(handle)
    return handles
