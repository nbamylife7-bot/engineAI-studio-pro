"""Retarget Kimodo SMPL-X motion to EngineAI T800 qpos via GMR."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable

import numpy as np

from kimodo.exports.smplx import AMASSConverter
from kimodo.skeleton import SkeletonBase

from .gmr_bootstrap import bootstrap_gmr, is_t800_available


def retarget_character_motion(
    motion,
    skeleton: SkeletonBase,
    fps: float,
    *,
    flatten_feet: bool = False,
    auto_ground: bool = True,
    status: Callable[[str], None] | None = None,
) -> tuple[list[np.ndarray], int]:
    """Convert a demo ``CharacterMotion`` to T800 MuJoCo qpos frames."""
    if not is_t800_available():
        from .gmr_bootstrap import missing_t800_dependencies

        missing = ", ".join(missing_t800_dependencies())
        raise RuntimeError(f"T800 retargeting is unavailable: {missing}")

    bootstrap_gmr()
    from scripts.smplx_npz_to_robot import convert_smplx_amass_npz

    status_fn = status or (lambda _msg: None)
    local_rot_mats = motion.joints_local_rot.detach().cpu().numpy()
    root_positions = motion.joints_pos[:, skeleton.root_idx, :].detach().cpu().numpy()

    converter = AMASSConverter(skeleton=skeleton, fps=fps)
    with tempfile.TemporaryDirectory(prefix="kimodo_t800_") as tmp_dir:
        npz_path = Path(tmp_dir) / "motion_amass.npz"
        pkl_path = Path(tmp_dir) / "motion_t800.pkl"
        converter.convert_save_npz(
            {
                "local_rot_mats": local_rot_mats,
                "root_positions": root_positions,
            },
            str(npz_path),
            z_up=True,
        )
        qpos_frames, motion_fps = convert_smplx_amass_npz(
            str(npz_path),
            tgt_fps=max(1, int(round(fps))),
            auto_ground=auto_ground,
            flatten_feet=flatten_feet,
            robot="t800",
            output_path=pkl_path,
            status=status_fn,
        )
    return qpos_frames, motion_fps
