# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pickle

import numpy as np
import torch

import motion_correction


def _to_cpu_float32_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32, copy=False)


def _assign_from_numpy(tensor: torch.Tensor, values: np.ndarray) -> None:
    restored = torch.from_numpy(values.reshape(tensor.shape))
    tensor.copy_(restored.to(device=tensor.device, dtype=tensor.dtype))


def correct_motion(
    hipTranslations,
    jointRotations,
    contacts,
    hipTranslationsInput,
    rotationsInput,
    constraint_masks,
    contact_threshold,
    root_margin,
    working_rig,
    has_double_ankle_joints=False,
):
    joint_names = [x.name for x in working_rig]
    joint_parents = [
        joint_names.index(working_rig[i].parent) if working_rig[i].parent in joint_names else -1
        for i in range(len(working_rig))
    ]
    joint_ref_translations = [list(x.t_pose_translation) for x in working_rig]
    joint_ref_rotations = [list(x.t_pose_rotation) for x in working_rig]

    left_hand_idx = [i for i in range(len(joint_names)) if working_rig[i].retarget_tag == "LeftHand"]
    if len(left_hand_idx) != 1:
        raise RuntimeError(f"correct_motion: Expected exactly one joint with LeftHand tag")
    left_hand_idx = left_hand_idx[0]

    right_hand_idx = [i for i in range(len(joint_names)) if working_rig[i].retarget_tag == "RightHand"]
    if len(right_hand_idx) != 1:
        raise RuntimeError(f"correct_motion: Expected exactly one joint with RightHand tag")
    right_hand_idx = right_hand_idx[0]

    left_foot_idx = [i for i in range(len(joint_names)) if working_rig[i].retarget_tag == "LeftFoot"]
    if len(left_foot_idx) != 1:
        raise RuntimeError(f"correct_motion: Expected exactly one joint with LeftFoot tag")
    left_foot_idx = left_foot_idx[0]

    right_foot_idx = [i for i in range(len(joint_names)) if working_rig[i].retarget_tag == "RightFoot"]
    if len(right_foot_idx) != 1:
        raise RuntimeError(f"correct_motion: Expected exactly one joint with RightFoot tag")
    right_foot_idx = right_foot_idx[0]

    end_frame = hipTranslations.shape[1]

    default_mask = torch.zeros(hipTranslations.shape[1], dtype=torch.float32, device=hipTranslations.device)
    root_mask = constraint_masks.get("Root", default_mask)
    full_body_mask = constraint_masks.get("FullBody", default_mask)
    left_hand_mask = constraint_masks.get("LeftHand", default_mask)
    right_hand_mask = constraint_masks.get("RightHand", default_mask)
    left_foot_mask = constraint_masks.get("LeftFoot", default_mask)
    right_foot_mask = constraint_masks.get("RightFoot", default_mask)

    batch_size = hipTranslations.shape[0]

    full_body_mask_np = _to_cpu_float32_numpy(full_body_mask)
    left_hand_mask_np = _to_cpu_float32_numpy(left_hand_mask)
    right_hand_mask_np = _to_cpu_float32_numpy(right_hand_mask)
    left_foot_mask_np = _to_cpu_float32_numpy(left_foot_mask)
    right_foot_mask_np = _to_cpu_float32_numpy(right_foot_mask)
    root_mask_np = _to_cpu_float32_numpy(root_mask)

    for b in range(batch_size):
        hipTranslationsCorrected = _to_cpu_float32_numpy(hipTranslations[b, :end_frame]).flatten()
        rotationsCorrected = _to_cpu_float32_numpy(jointRotations[b, :end_frame]).flatten()

        hipTranslationsInput_flat = _to_cpu_float32_numpy(hipTranslationsInput).flatten()
        rotationsInput_flat = _to_cpu_float32_numpy(rotationsInput).flatten()
        ctcs = _to_cpu_float32_numpy(contacts[b]).flatten()

        motion_correction.correct_motion(
            hipTranslationsCorrected,
            rotationsCorrected,
            hipTranslationsInput_flat,
            rotationsInput_flat,
            full_body_mask_np,
            left_hand_mask_np,
            right_hand_mask_np,
            left_foot_mask_np,
            right_foot_mask_np,
            root_mask_np,
            np.asarray(ctcs, dtype=np.float32),
            joint_parents,
            joint_ref_translations,
            joint_ref_rotations,
            left_hand_idx,
            right_hand_idx,
            left_foot_idx,
            right_foot_idx,
            contact_threshold,
            root_margin,
            has_double_ankle_joints,
        )

        _assign_from_numpy(hipTranslations[b, :end_frame], hipTranslationsCorrected)
        _assign_from_numpy(jointRotations[b, :end_frame], rotationsCorrected)
