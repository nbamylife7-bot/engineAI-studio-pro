"""Retarget Kimodo SMPL-X motion to EngineAI T800 qpos via GMR."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable

import numpy as np

from kimodo.demo.config import (
    KIMODO_T800_IK_SAFETY,
    KIMODO_T800_IK_STRIDE,
    KIMODO_T800_SMOOTH,
    KIMODO_T800_SMOOTH_WINDOW,
)
from kimodo.exports.smplx import AMASSConverter
from kimodo.skeleton import SkeletonBase

from .gmr_bootstrap import bootstrap_gmr, is_t800_available


def _odd_window(window: int, n_frames: int) -> int:
    """Largest odd window <= ``window`` that fits ``n_frames``."""
    w = min(int(window), n_frames if n_frames % 2 == 1 else n_frames - 1)
    if w % 2 == 0:
        w -= 1
    return w


def smooth_t800_qpos_frames(
    qpos_frames: list[np.ndarray],
    *,
    window: int = 13,
    polyorder: int = 3,
) -> list[np.ndarray]:
    """Remove standing IK ping-pong from retargeted qpos (median-3 + Savitzky-Golay).

    GMR IK amplifies small SMPL-X per-frame noise ~4-5x in the knees/ankles, producing a
    high-frequency frame-to-frame oscillation while the human is nearly static. ``ik_safety_break``
    does not affect this (mink only raises on limit violations). A median-3 filter removes the
    alternating A/B/A ping-pong without distorting real (monotonic) motion; a light Savitzky-Golay
    pass smooths the residual. Applied once on the qpos data, so viser playback stays exact at full
    fps. Joint angles only are filtered for pose; the root translation/orientation get a light pass.
    """
    if len(qpos_frames) < 5:
        return qpos_frames

    from scipy.signal import medfilt, savgol_filter

    Q = np.asarray(qpos_frames, dtype=np.float64)
    T, nq = Q.shape
    out = Q.copy()
    w = _odd_window(window, T)
    can_savgol = w >= 5 and w > polyorder

    # Joints (dof): kill ping-pong, then light smooth.
    if nq > 7:
        dof = medfilt(Q[:, 7:], kernel_size=(3, 1))
        if can_savgol:
            dof = savgol_filter(dof, window_length=w, polyorder=polyorder, axis=0)
        out[:, 7:] = dof

    # Root position (incl. small standing z bounce): light smooth only.
    if can_savgol:
        out[:, :3] = savgol_filter(Q[:, :3], window_length=w, polyorder=polyorder, axis=0)

    # Root orientation: light smooth + renormalize (rotations are tiny while standing).
    if nq >= 7 and can_savgol:
        quat = savgol_filter(Q[:, 3:7], window_length=w, polyorder=polyorder, axis=0)
        norm = np.linalg.norm(quat, axis=1, keepdims=True)
        norm[norm < 1e-8] = 1.0
        out[:, 3:7] = quat / norm

    return [out[i].copy() for i in range(T)]


def export_motion_amass_npz(
    motion,
    skeleton: SkeletonBase,
    fps: float,
    npz_path: str,
) -> None:
    """Write a CharacterMotion to an AMASS SMPL-X NPZ (torch/GPU step, main process)."""
    local_rot_mats = motion.joints_local_rot.detach().cpu().numpy()
    root_positions = motion.joints_pos[:, skeleton.root_idx, :].detach().cpu().numpy()
    converter = AMASSConverter(skeleton=skeleton, fps=fps)
    # Standard AMASS Z-up export (no preserve_lateral — that breaks T800 facing vs the human).
    converter.convert_save_npz(
        {
            "local_rot_mats": local_rot_mats,
            "root_positions": root_positions,
        },
        str(npz_path),
        z_up=True,
    )


def retarget_npz_to_qpos(
    npz_path: str,
    fps: float,
    *,
    flatten_feet: bool = False,
    auto_ground: bool = True,
    ik_safety_break: bool = True,
    ik_stride: int = 1,
    smooth: bool = True,
    smooth_window: int = 13,
    output_pkl: str | None = None,
    status: Callable[[str], None] | None = None,
) -> tuple[list[np.ndarray], int]:
    """CPU-only step: AMASS NPZ -> smoothed T800 qpos. Safe to run in a worker process."""
    bootstrap_gmr()
    from scripts.smplx_npz_to_robot import convert_smplx_amass_npz

    status_fn = status or (lambda _msg: None)
    out_pkl = Path(output_pkl) if output_pkl else Path(npz_path).with_suffix(".t800.pkl")
    qpos_frames, motion_fps = convert_smplx_amass_npz(
        str(npz_path),
        tgt_fps=max(1, int(round(fps))),
        auto_ground=auto_ground,
        flatten_feet=flatten_feet,
        robot="t800",
        ik_safety_break=ik_safety_break,
        ik_stride=ik_stride,
        output_path=out_pkl,
        status=status_fn,
    )
    if smooth and qpos_frames:
        status_fn("Smoothing standing IK jitter …")
        qpos_frames = smooth_t800_qpos_frames(qpos_frames, window=smooth_window)
    return qpos_frames, motion_fps


def prepare_t800_human_frames(npz_path: str, fps: float) -> dict:
    """Torch SMPL-X FK in the main process → picklable payload for worker IK (option A)."""
    bootstrap_gmr()
    from scripts.smplx_npz_to_robot import prepare_smplx_human_frames

    return prepare_smplx_human_frames(str(npz_path), tgt_fps=max(1, int(round(fps))))


def retarget_prepared_to_qpos(
    payload: dict,
    fps: float,
    *,
    flatten_feet: bool = False,
    auto_ground: bool = True,
    ik_safety_break: bool = True,
    ik_stride: int = 1,
    smooth: bool = True,
    smooth_window: int = 13,
    output_pkl: str | None = None,
    status: Callable[[str], None] | None = None,
) -> tuple[list[np.ndarray], int]:
    """Pure-CPU IK from prepared SMPL-X frames (no torch). Runs in a worker process."""
    bootstrap_gmr()
    from scripts.smplx_npz_to_robot import retarget_human_frames

    status_fn = status or (lambda _msg: None)
    out_pkl = Path(output_pkl) if output_pkl else None
    qpos_frames, motion_fps = retarget_human_frames(
        payload["human_frames"],
        aligned_fps=payload["aligned_fps"],
        tgt_fps=payload.get("tgt_fps", max(1, int(round(fps)))),
        height=payload["height"],
        auto_ground=auto_ground,
        flatten_feet=flatten_feet,
        robot="t800",
        ik_safety_break=ik_safety_break,
        ik_stride=ik_stride,
        output_path=out_pkl,
        status=status_fn,
    )
    if smooth and qpos_frames:
        status_fn("Smoothing standing IK jitter …")
        qpos_frames = smooth_t800_qpos_frames(qpos_frames, window=smooth_window)
    return qpos_frames, motion_fps


def retarget_character_motion(
    motion,
    skeleton: SkeletonBase,
    fps: float,
    *,
    flatten_feet: bool = False,
    auto_ground: bool = True,
    smooth: bool | None = None,
    ik_stride: int | None = None,
    status: Callable[[str], None] | None = None,
) -> tuple[list[np.ndarray], int]:
    """Convert a demo ``CharacterMotion`` to T800 MuJoCo qpos frames (single sample)."""
    if not is_t800_available():
        from .gmr_bootstrap import missing_t800_dependencies

        missing = ", ".join(missing_t800_dependencies())
        raise RuntimeError(f"T800 retargeting is unavailable: {missing}")

    do_smooth = KIMODO_T800_SMOOTH if smooth is None else bool(smooth)
    stride = KIMODO_T800_IK_STRIDE if ik_stride is None else int(ik_stride)

    with tempfile.TemporaryDirectory(prefix="kimodo_t800_") as tmp_dir:
        npz_path = Path(tmp_dir) / "motion_amass.npz"
        pkl_path = Path(tmp_dir) / "motion_t800.pkl"
        export_motion_amass_npz(motion, skeleton, fps, str(npz_path))
        qpos_frames, motion_fps = retarget_npz_to_qpos(
            str(npz_path),
            fps,
            flatten_feet=flatten_feet,
            auto_ground=auto_ground,
            ik_safety_break=KIMODO_T800_IK_SAFETY,
            ik_stride=stride,
            smooth=do_smooth,
            smooth_window=KIMODO_T800_SMOOTH_WINDOW,
            output_pkl=str(pkl_path),
            status=status,
        )
    return qpos_frames, motion_fps
