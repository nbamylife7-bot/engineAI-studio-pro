# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export utilities for converting internal motion representations into common file formats.

This module is intended to hold lightweight serialization / export helpers that can be reused
outside of interactive demos.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from kimodo.geometry import matrix_to_quaternion as _matrix_to_quaternion
from kimodo.skeleton import SkeletonBase


def _strip_end_site_blocks(bvh_text: str) -> str:
    """Remove all 'End Site { ... }' blocks from BVH text so output matches original format.

    bvhio adds an End Site for every leaf joint when writing; we do not set EndSite on joints, so we
    post-process the string to remove these blocks for Blender/original compatibility.
    """
    lines = bvh_text.splitlines(keepends=True)
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if "End Site" in line:
            # Skip this line and the following block { ... }; brace-count to find closing }
            i += 1
            if i < len(lines) and "{" in lines[i]:
                i += 1
                depth = 1
                while i < len(lines) and depth > 0:
                    if "{" in lines[i]:
                        depth += 1
                    if "}" in lines[i]:
                        depth -= 1
                    i += 1
            continue
        result.append(line)
        i += 1
    return "".join(result)


def _coerce_batch(name: str, x: torch.Tensor, *, expected_ndim: int) -> torch.Tensor:
    """Coerce (T, ...) or (1, T, ...) into (T, ...)."""
    if x.ndim == expected_ndim:
        return x
    if x.ndim == expected_ndim + 1:
        if int(x.shape[0]) != 1:
            raise ValueError(
                f"{name} has batch dimension B={int(x.shape[0])}, but BVH export " "only supports a single clip (B==1)."
            )
        return x[0]
    raise ValueError(f"{name} must have shape (T, ...) or (1, T, ...); got {tuple(x.shape)}")


def motion_to_bvh(
    local_rot_mats: torch.Tensor,
    root_positions: torch.Tensor,
    *,
    skeleton: "SkeletonBase",
    fps: float,
) -> str:
    """Convert local rotations and root positions to BVH format; return UTF-8 string.

    Args:
        local_rot_mats: (T, J, 3, 3) or (1, T, J, 3, 3) local rotation matrices.
        root_positions: (T, 3) or (1, T, 3) root joint positions (e.g. from posed joints).
        skeleton: Skeleton with bone_order_names, bvh_neutral_joints, etc.
        fps: Frames per second for the motion.

    Notes:
        BVH is plain-text. Root is named "Root" with ZYX rotation order; leaf joints
        have no End Site block.
    """
    try:
        import bvhio  # type: ignore[import-not-found]
        import glm  # type: ignore[import-not-found]
        from SpatialTransform import Pose  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "BVH export requires `bvhio` (and its deps `PyGLM` + `SpatialTransform`). "
            "Install with: `pip install bvhio`."
        ) from e

    local_rot_mats = local_rot_mats.detach()
    root_positions = root_positions.detach()
    # SOMA: accept either somaskel30 (convert to 77) or somaskel77 (use as-is)
    if skeleton.name == "somaskel30":
        local_rot_mats = skeleton.to_SOMASkeleton77(local_rot_mats)
        skeleton = skeleton.somaskel77

    local_rot_mats, _ = skeleton.from_standard_tpose(local_rot_mats)

    neutral = skeleton.bvh_neutral_joints.detach().cpu().numpy()
    joint_names = list(skeleton.bone_order_names)
    parents = skeleton.joint_parents.detach().cpu().numpy().astype(int)
    root_idx = int(skeleton.root_idx)

    local_rot_mats = _coerce_batch("local_rot_mats", local_rot_mats, expected_ndim=4)
    T, J = local_rot_mats.shape[:2]
    q_wxyz = _matrix_to_quaternion(local_rot_mats).detach().cpu().numpy()  # [T, J, 4]

    root_xyz = _coerce_batch("root_positions", root_positions, expected_ndim=2)
    root_xyz = root_xyz.detach().cpu().numpy()  # [T, 3]

    # Build BVH hierarchy: Root (wrapper at origin) -> Hips (pelvis with offset in meters) -> ...
    # Offsets are in meters to match the original format.
    children: dict[int, list[int]] = {i: [] for i in range(J)}
    for i, p in enumerate(parents):
        if p >= 0:
            children[int(p)].append(int(i))

    _ROOT_CHANNELS = [
        "Xposition",
        "Yposition",
        "Zposition",
        "Zrotation",
        "Yrotation",
        "Xrotation",
    ]
    _JOINT_CHANNELS = ["Zrotation", "Yrotation", "Xrotation"]

    # Scale from meters to centimeters (match original BVH scale).
    neutral = neutral * 100
    root_xyz = root_xyz * 100

    # Hips offset from Root: use skeleton neutral; if root is at origin (zeros), use a
    # nominal pelvis height so the hierarchy is non-degenerate in Blender.
    hips_offset = neutral[root_idx]
    if (hips_offset == 0).all():
        hips_offset = np.array([0.0, 100.0, 0.0], dtype=neutral.dtype)  # 1 m in cm

    def _make_joint(i: int) -> "bvhio.BvhJoint":
        name = joint_names[i]
        j = bvhio.BvhJoint(name, offset=glm.vec3(0, 0, 0))
        if i == root_idx:
            # Hips: offset from Root (origin) in cm
            off = hips_offset
            j.Offset = glm.vec3(float(off[0]), float(off[1]), float(off[2]))
            j.Channels = _ROOT_CHANNELS.copy()
        else:
            p = int(parents[i])
            off = neutral[i] - neutral[p]
            j.Offset = glm.vec3(float(off[0]), float(off[1]), float(off[2]))
            j.Channels = _JOINT_CHANNELS.copy()

        for c in children[i]:
            j.Children.append(_make_joint(c))
        return j

    # Wrapper Root at origin; single child is Hips (skeleton root).
    root_wrapper = bvhio.BvhJoint("Root", offset=glm.vec3(0.0, 0.0, 0.0))
    root_wrapper.Channels = _ROOT_CHANNELS.copy()
    root_wrapper.Children.append(_make_joint(root_idx))
    root_joint = root_wrapper

    # Populate keyframes: Root = identity/zero, Hips = root motion, others = local rotation.
    bvh_layout = root_joint.layout()
    name_to_id = {n: idx for idx, n in enumerate(joint_names)}
    ordered_joint_ids = []
    for bj, _, _ in bvh_layout:
        if bj.Name == "Root":
            ordered_joint_ids.append(None)
        else:
            ordered_joint_ids.append(name_to_id[bj.Name])

    bvh_joints = [bj for bj, _, _ in bvh_layout]
    for bj in bvh_joints:
        bj.Keyframes = [None] * T  # type: ignore[list-item]

    identity_quat = glm.quat(1.0, 0.0, 0.0, 0.0)
    zero_vec = glm.vec3(0.0, 0.0, 0.0)
    for t in range(T):
        for bj, jid in zip(bvh_joints, ordered_joint_ids):
            if jid is None:
                position = zero_vec
                rotation = identity_quat
            elif jid == root_idx:
                pos = root_xyz[t]
                position = glm.vec3(float(pos[0]), float(pos[1]), float(pos[2]))
                qw, qx, qy, qz = q_wxyz[t, jid]
                rotation = glm.quat(float(qw), float(qx), float(qy), float(qz))
            else:
                position = zero_vec
                qw, qx, qy, qz = q_wxyz[t, jid]
                rotation = glm.quat(float(qw), float(qx), float(qy), float(qz))
            bj.Keyframes[t] = Pose(position, rotation)  # type: ignore[index]

    container = bvhio.BvhContainer(root_joint, frameCount=T, frameTime=1.0 / float(fps))
    with tempfile.NamedTemporaryFile(mode="w", suffix=".bvh", delete=False, encoding="utf-8") as f:
        tmp_path = f.name
    try:
        bvhio.writeBvh(tmp_path, container, percision=6)
        bvh_text = Path(tmp_path).read_text(encoding="utf-8")
        return _strip_end_site_blocks(bvh_text)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def motion_to_bvh_bytes(
    local_rot_mats: torch.Tensor,
    root_positions: torch.Tensor,
    *,
    skeleton: "SkeletonBase",
    fps: float,
) -> bytes:
    """Convert local rotations and root positions to BVH bytes (UTF-8).

    Convenience wrapper around :func:`motion_to_bvh`.
    """
    return motion_to_bvh(local_rot_mats, root_positions, skeleton=skeleton, fps=fps).encode("utf-8")


def save_motion_bvh(
    path: str | Path,
    local_rot_mats: torch.Tensor,
    root_positions: torch.Tensor,
    *,
    skeleton: "SkeletonBase",
    fps: float,
) -> None:
    """Write local rotations and root positions to a BVH file at the given path."""
    Path(path).write_text(
        motion_to_bvh(local_rot_mats, root_positions, skeleton=skeleton, fps=fps),
        encoding="utf-8",
    )
