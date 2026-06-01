# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert kimodo motion to AMASS/SMPL-X compatible parameters (axis-angle, Y-up or Z-up)."""

import os

import einops
import numpy as np
import torch

from kimodo.assets import skeleton_asset_path
from kimodo.geometry import matrix_to_axis_angle
from kimodo.tools import ensure_batched, to_numpy, to_torch


@ensure_batched(local_rot_mats=5, root_positions=3, lengths=1)
def get_amass_parameters(
    local_rot_mats,
    root_positions,
    skeleton,
    z_up=True,
    *,
    preserve_lateral: bool = False,
):
    """Convert local rot mats and root positions to AMASS-style trans and pose_body; optional z_up
    coordinate transform.

    Our method generates motions with Y-up and +Z forward; if z_up=True, transform to Z-up and +Y
    forward as in AMASS.

    When ``preserve_lateral=True`` (EngineAI T800 retargeting), use a direct Y-up→Z-up axis swap
    without the default ``rot_z_180`` correction. The default path flips AMASS X and makes the
    retargeted robot walk opposite to the Kimodo human unless the viewer flips X again.
    """
    # Remove the root offset; SMPL-X FK adds pelvis offset back.
    pelvis_offset = to_numpy(skeleton.neutral_joints[skeleton.root_idx])
    trans = to_numpy(root_positions) - pelvis_offset

    root_rot_mats = to_numpy(local_rot_mats[:, :, 0])
    rot_mats_cpu = to_torch(local_rot_mats, device="cpu", dtype=torch.float32)
    local_rot_axis_angle = to_numpy(matrix_to_axis_angle(rot_mats_cpu))
    pose_body = einops.rearrange(local_rot_axis_angle[:, :, 1:], "b t j d -> b t (j d)")

    # Optionally convert from Y-up to Z-up coordinates.
    if z_up:
        if preserve_lateral:
            # Kimodo (+X right, +Y up, +Z forward) -> AMASS Z-up (+X right, +Y forward, +Z up).
            y_up_to_z_up = np.array(
                [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                dtype=np.float32,
            )
        else:
            y_up_to_z_up = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            )
            # 180-degree rotation around +Z to keep forward as +Y.
            rot_z_180 = np.array(
                [
                    [-1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            y_up_to_z_up = np.matmul(rot_z_180, y_up_to_z_up)
        root_rot_mats = np.matmul(y_up_to_z_up, root_rot_mats)
        trans = np.matmul(trans + pelvis_offset, y_up_to_z_up.T) - pelvis_offset

    root_orient = to_numpy(matrix_to_axis_angle(to_torch(root_rot_mats, device="cpu", dtype=torch.float32)))
    return trans, root_orient, pose_body


class AMASSConverter:
    def __init__(
        self,
        fps,
        skeleton,
        beta_path=str(skeleton_asset_path("smplx22", "beta.npy")),
        mean_hands_path=str(skeleton_asset_path("smplx22", "mean_hands.npy")),
    ):
        self.fps = fps
        self.skeleton = skeleton
        # Load betas
        if os.path.exists(beta_path):
            # only use first 16 betas to match AMASS
            betas = np.load(beta_path)[:16]
        else:
            betas = np.zeros(16)

        # Load mean hands
        if os.path.exists(mean_hands_path):
            mean_hands = np.load(mean_hands_path)
        else:
            mean_hands = np.zeros(90)

        self.default_frame_params = {
            "pose_jaw": np.zeros(3),
            "pose_eye": np.zeros(6),
            "pose_hand": mean_hands,
        }
        self.output_dict_base = {
            "gender": "neutral",
            "surface_model_type": "smplx",
            "betas": betas,
            "num_betas": len(betas),
            "mocap_frame_rate": float(fps),
        }

    def convert_save_npz(self, output: dict, npz_path, z_up=True, *, preserve_lateral: bool = False):
        trans, root_orient, pose_body = get_amass_parameters(
            output["local_rot_mats"],
            output["root_positions"],
            self.skeleton,
            z_up=z_up,
            preserve_lateral=preserve_lateral,
        )
        nb_frames = trans.shape[-2]

        amass_output_base = self.output_dict_base.copy()
        for key, val in self.default_frame_params.items():
            amass_output_base[key] = einops.repeat(val, "d -> t d", t=nb_frames)

        amass_output_base["mocap_time_length"] = nb_frames / self.fps
        self.save_npz(trans, root_orient, pose_body, amass_output_base, npz_path)

    def save_npz(self, trans, root_orient, pose_body, base_output, npz_path):
        shape = trans.shape
        if len(shape) == 3 and shape[0] == 1:
            # if only one motion, squeeze the data
            trans = trans[0]
            root_orient = root_orient[0]
            pose_body = pose_body[0]
            shape = trans.shape
        if len(shape) == 2:
            amass_output = {
                "trans": trans,
                "root_orient": root_orient,
                "pose_body": pose_body,
            } | base_output
            np.savez(npz_path, **amass_output)

        elif len(shape) == 3:
            # real batch of motions
            npz_path_base, ext = os.path.splitext(npz_path)
            for i in range(shape[0]):
                npz_path_i = npz_path_base + "_" + str(i).zfill(2) + ext
                self.save_npz(trans[i], root_orient[i], pose_body[i], base_output, npz_path_i)


# amass_output = {
#     "gender": "neutral",
#     "surface_model_type": "smplx",
#     "mocap_frame_rate": float(fps),
#     "mocap_time_length": len(motion) / float(fps)
#     "trans": trans,
#     "betas": betas,
#     "num_betas": len(betas),
#     "root_orient": np.array([T, 3]), # axis angle
#     "pose_body": np.array([T, 63]), # 63=21*3, axis angle 21 = 22 - root
#     "pose_hand": np.array([T, 90]), # 90=30*3=15*2*3 axis angle (load from mean_hands)
#     "pose_jaw": np.array([T, 3]), # all zeros is fine
#     "pose_eye": np.array([T, 6]), # all zeros is fine`
# }
