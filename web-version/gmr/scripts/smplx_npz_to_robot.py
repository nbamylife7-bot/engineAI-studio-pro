"""AMASS / Kimodo SMPL-X NPZ → T800 robot motion PKL.

Supports Kimodo exports in AMASS SMPL-X layout (pose_body, root_orient, trans,
pose_hand, mocap_frame_rate, …). Kimodo native NPZ (posed_joints) is detected
and rejected with a clear message — use AMASS export from Kimodo instead.
"""

from __future__ import annotations

import os
import pathlib
import pickle
from typing import Callable, Literal

import numpy as np

import scripts.bvh_to_robot as bvr
import scripts.fbx_to_robot as fr
import scripts.t800_foot_postprocess as foot_pp
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.smpl import (
    _as_scalar_fps,
    get_smplx_data_offline_fast,
    load_smplx_file,
)

AMASS_SMPLX_KEYS = ("pose_body", "root_orient", "trans")
KIMODO_NATIVE_KEYS = ("posed_joints", "joint_names")

NpzKind = Literal["amass_smplx", "kimodo_native", "unknown"]


def detect_npz_kind(npz_path: str) -> NpzKind:
    data = np.load(npz_path, allow_pickle=True)
    keys = set(data.files)
    if all(k in keys for k in AMASS_SMPLX_KEYS):
        return "amass_smplx"
    if any(k in keys for k in KIMODO_NATIVE_KEYS):
        return "kimodo_native"
    return "unknown"


def describe_npz_kind(kind: NpzKind) -> str:
    if kind == "amass_smplx":
        return "AMASS SMPL-X (Kimodo export compatible)"
    if kind == "kimodo_native":
        return "Kimodo native joints NPZ (not supported — re-export as AMASS SMPL-X)"
    return "unknown NPZ layout"


def resolve_smplx_body_models_path() -> pathlib.Path:
    env = os.environ.get("SMPLX_BODY_MODELS", "").strip()
    if env:
        path = pathlib.Path(env).expanduser()
        if path.is_dir():
            return path
        raise FileNotFoundError(
            f"SMPLX_BODY_MODELS={env} is not a directory. "
            "Point it at the folder that contains smplx/SMPLX_NEUTRAL.pkl (or .npz)."
        )

    default = pathlib.Path(__file__).resolve().parent.parent / "assets" / "body_models"
    smplx_dir = default / "smplx"
    if smplx_dir.is_dir():
        return default
    raise FileNotFoundError(
        "SMPL-X body models not found. Download from https://smpl-x.is.tue.mpg.de/ "
        f"and install under {default}/smplx/ "
        "(SMPLX_NEUTRAL.pkl or .npz, plus MALE/FEMALE if needed), "
        "or set SMPLX_BODY_MODELS to your models directory."
    )


def resolve_kimodo_t800_ik_safety_break() -> bool:
    """Kimodo T800 retarget: True avoids standing IK ping-pong (GMR default); False was legacy Kimodo."""
    return os.environ.get("KIMODO_T800_IK_SAFETY", "1").strip().lower() not in ("0", "false", "no")


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical interpolation of wxyz quaternions (numpy, single pair)."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        n = np.linalg.norm(out)
        return out / n if n > 1e-8 else q0
    theta0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin0 = np.sin(theta0)
    s0 = np.sin((1.0 - t) * theta0) / sin0
    s1 = np.sin(t * theta0) / sin0
    return s0 * q0 + s1 * q1


def _interpolate_qpos_frames(key_idx: list[int], key_qpos: list[np.ndarray], n_frames: int) -> list[np.ndarray]:
    """Upsample sparsely-solved qpos (root pos lerp, root quat slerp, joints lerp) to ``n_frames``."""
    if len(key_idx) <= 1 or key_idx[-1] <= 0:
        return [key_qpos[0].copy() for _ in range(n_frames)]
    out: list[np.ndarray] = []
    seg = 0
    for f in range(n_frames):
        while seg < len(key_idx) - 2 and key_idx[seg + 1] <= f:
            seg += 1
        i0, i1 = key_idx[seg], key_idx[seg + 1]
        q0, q1 = key_qpos[seg], key_qpos[seg + 1]
        t = 0.0 if i1 == i0 else float(np.clip((f - i0) / (i1 - i0), 0.0, 1.0))
        q = q0.copy()
        q[:3] = (1.0 - t) * q0[:3] + t * q1[:3]
        if q.size >= 7:
            q[3:7] = _quat_slerp(q0[3:7], q1[3:7], t)
        if q.size > 7:
            q[7:] = (1.0 - t) * q0[7:] + t * q1[7:]
        out.append(q)
    return out


def prepare_smplx_human_frames(
    npz_path: str,
    *,
    tgt_fps: int = 30,
    human_height: float = 0.0,
    status: Callable[[str], None] = lambda _msg: None,
) -> dict:
    """SMPL-X forward kinematics (torch) → picklable per-frame body transforms.

    This is the only torch step. Run it in the main process so worker processes need no torch /
    SMPL-X model: they consume the returned ``human_frames`` (plain numpy) and only solve IK.
    """
    kind = detect_npz_kind(npz_path)
    if kind == "kimodo_native":
        raise RuntimeError(
            "This NPZ is Kimodo native format (posed_joints). "
            "In Kimodo, export as AMASS SMPL-X NPZ instead (pose_body / root_orient / trans)."
        )
    if kind != "amass_smplx":
        raise RuntimeError(
            "Unrecognized NPZ layout. Expected AMASS SMPL-X keys: "
            + ", ".join(AMASS_SMPLX_KEYS)
        )

    body_models = resolve_smplx_body_models_path()
    status(f"Loading AMASS SMPL-X {pathlib.Path(npz_path).name} …")
    smplx_data, body_model, smplx_output, detected_height = load_smplx_file(
        npz_path, body_models
    )
    src_fps = _as_scalar_fps(
        smplx_data["mocap_frame_rate"] if "mocap_frame_rate" in smplx_data.files else None
    )
    num_frames = int(smplx_data["pose_body"].shape[0])
    status(
        f"Parsed {num_frames} frames @ {src_fps:.0f} fps → retarget @ {int(tgt_fps)} fps …"
    )

    human_frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=int(tgt_fps)
    )
    if not human_frames:
        raise RuntimeError("No frames after SMPL-X forward kinematics / FPS alignment.")

    height = float(human_height) if human_height > 0 else float(detected_height)
    return {
        "human_frames": human_frames,
        "aligned_fps": aligned_fps,
        "height": height,
        "tgt_fps": int(tgt_fps),
        "src_fps": float(src_fps),
        "num_frames": num_frames,
    }


def retarget_human_frames(
    human_frames: list,
    *,
    aligned_fps: float,
    tgt_fps: int = 30,
    height: float,
    auto_ground: bool = True,
    flatten_feet: bool = False,
    robot: str = "t800",
    ik_safety_break: bool | None = None,
    ik_stride: int = 1,
    output_path: pathlib.Path | None = None,
    status: Callable[[str], None] = lambda _msg: None,
) -> tuple[list[np.ndarray], int]:
    """Pure-CPU IK: prepared SMPL-X body transforms → T800 qpos. Safe to run in a worker (no torch)."""
    use_ik_safety = (
        resolve_kimodo_t800_ik_safety_break()
        if ik_safety_break is None
        else bool(ik_safety_break)
    )
    retargeter = GMR(
        actual_human_height=height,
        src_human="smplx",
        tgt_robot=robot,
        ik_safety_break=use_ik_safety,
        verbose=False,
    )

    if auto_ground:
        ground = bvr.estimate_ground_offset(retargeter, human_frames)
        retargeter.set_ground_offset(ground)

    n_frames = len(human_frames)
    stride = max(1, int(ik_stride))
    if stride > 1 and n_frames > 2 * stride:
        key_idx = list(range(0, n_frames, stride))
        if key_idx[-1] != n_frames - 1:
            key_idx.append(n_frames - 1)
        key_qpos: list[np.ndarray] = []
        for i in key_idx:
            qpos = retargeter.retarget(human_frames[i], frame_index=i)
            if flatten_feet:
                qpos = foot_pp.postprocess_robot_qpos_feet(retargeter.model, qpos, flatten=True)
            key_qpos.append(qpos.copy())
        qpos_frames = _interpolate_qpos_frames(key_idx, key_qpos, n_frames)
    else:
        qpos_frames = []
        for i, frame in enumerate(human_frames):
            qpos = retargeter.retarget(frame, frame_index=i)
            if flatten_feet:
                qpos = foot_pp.postprocess_robot_qpos_feet(retargeter.model, qpos, flatten=True)
            qpos_frames.append(qpos.copy())

    motion_fps = int(round(aligned_fps)) if aligned_fps else int(tgt_fps)
    if output_path is not None:
        motion = fr._build_motion_data_from_qpos_list(qpos_frames, motion_fps)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as fh:
            pickle.dump(motion, fh)
        status(f"Saved {output_path.name} ({len(qpos_frames)} frames @ {motion_fps} fps).")
    return qpos_frames, motion_fps


def convert_smplx_amass_npz(
    npz_path: str,
    *,
    tgt_fps: int = 30,
    human_height: float = 0.0,
    auto_ground: bool = True,
    flatten_feet: bool = False,
    robot: str = "t800",
    ik_safety_break: bool | None = None,
    ik_stride: int = 1,
    output_path: pathlib.Path,
    status: Callable[[str], None] = lambda _msg: None,
) -> tuple[list[np.ndarray], int]:
    payload = prepare_smplx_human_frames(
        npz_path, tgt_fps=int(tgt_fps), human_height=human_height, status=status
    )
    return retarget_human_frames(
        payload["human_frames"],
        aligned_fps=payload["aligned_fps"],
        tgt_fps=payload["tgt_fps"],
        height=payload["height"],
        auto_ground=auto_ground,
        flatten_feet=flatten_feet,
        robot=robot,
        ik_safety_break=ik_safety_break,
        ik_stride=ik_stride,
        output_path=output_path,
        status=status,
    )
