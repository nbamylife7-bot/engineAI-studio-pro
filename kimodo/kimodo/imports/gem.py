# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional NVIDIA GEM-SMPL (GENMO) video → AMASS SMPL-X → Kimodo motion tensors.

GEM runs in a separate Python env (``KIMODO_GEM_ROOT``). This module converts GEM outputs
for the existing T800 GMR retarget pipeline (AMASS NPZ layout).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Union

import numpy as np
import torch

from kimodo.assets import skeleton_asset_path
from kimodo.geometry import axis_angle_to_matrix, matrix_to_axis_angle
from kimodo.skeleton import SMPLXSkeleton22, SkeletonBase

PathLike = Union[str, Path]

_DEFAULT_GEM_ROOT = Path(os.environ.get("ENGINEAI_STUDIO_ROOT", os.getcwd())) / "vendor" / "GENMO"
_GEM_DEMO_SCRIPT = Path("scripts/demo/demo_smpl_hpe.py")
_YOLO_WEIGHTS_NAME = "yolov8x.pt"
_MIN_YOLO_BYTES = 100_000_000
# GVHMR regressors (from zju3dv/GVHMR gem|hmr4d/utils/body_model)
_BODY_MODEL_DIR_REL = Path("gem/utils/body_model")
_BODY_MODEL_FILES: dict[str, int] = {
    "smplx2smpl_sparse.pt": 100_000,
    "smpl_coco17_J_regressor.pt": 100_000,
    "smpl_neutral_J_regressor.pt": 100_000,
    "smplx_verts437.pt": 1_000,
    "coco_aug_dict.pth": 500,
}

# Optional axis remap (off by default). Kimodo ``get_amass_parameters`` is for Kimodo FK,
# not GVHMR/GEM ``body_params_global`` (see GMR ``load_gvhmr_pred_file``).
_GEM_YUP_TO_AMASS_ZUP = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float32,
)
_GEM_ROT_Z_180 = np.array(
    [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)
_GEM_TO_AMASS_R = (_GEM_ROT_Z_180 @ _GEM_YUP_TO_AMASS_ZUP).astype(np.float32)
# 180° about +Y — fixes person facing +Z vs Studio/GMR −Z.
_GEM_YAW_180_YUP = np.array(
    [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
    dtype=np.float32,
)


def resolve_gem_root() -> Path | None:
    raw = os.environ.get("KIMODO_GEM_ROOT", "").strip()
    if raw:
        root = Path(raw).expanduser().resolve()
        return root if root.is_dir() else None
    candidate = _DEFAULT_GEM_ROOT.resolve()
    return candidate if candidate.is_dir() else None


def resolve_gem_python(gem_root: Path | None = None) -> Path | None:
    override = os.environ.get("KIMODO_GEM_PYTHON", "").strip()
    if override:
        path = Path(override).expanduser()
        return path if path.is_file() else None
    root = gem_root or resolve_gem_root()
    if root is None:
        return None
    for rel in (".venv/bin/python", ".venv/Scripts/python.exe"):
        candidate = root / rel
        if candidate.is_file():
            return candidate
    return None


def _gem_body_model_paths(gem_root: Path) -> dict[str, Path]:
    bm = gem_root / _BODY_MODEL_DIR_REL
    return {name: bm / name for name in _BODY_MODEL_FILES}


def _gem_video_asset_paths(gem_root: Path) -> dict[str, Path]:
    paths = {
        "yolo": gem_root / _YOLO_WEIGHTS_NAME,
        "hmr2": gem_root / "inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt",
        "vitpose": gem_root / "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth",
        "gem_ckpt": gem_root / "inputs/pretrained/gem_smpl.ckpt",
    }
    paths.update(_gem_body_model_paths(gem_root))
    return paths


def is_gem_runner_available() -> bool:
    root = resolve_gem_root()
    if root is None:
        return False
    py = resolve_gem_python(root)
    if py is None:
        return False
    return (root / _GEM_DEMO_SCRIPT).is_file()


def is_gem_video_ready() -> bool:
    """True when GENMO demo script and all video-pipeline checkpoints are present."""
    root = resolve_gem_root()
    if root is None or not is_gem_runner_available():
        return False
    assets = _gem_video_asset_paths(root)
    yolo = assets["yolo"]
    if not yolo.is_file() or yolo.stat().st_size < _MIN_YOLO_BYTES:
        return False
    for key, path in assets.items():
        if key == "yolo":
            continue
        if path.name in _BODY_MODEL_FILES:
            min_bytes = _BODY_MODEL_FILES[path.name]
        else:
            min_bytes = 1_000_000
        if not path.is_file() or path.stat().st_size < min_bytes:
            return False
    return True


def gem_video_readiness_error() -> str | None:
    """Human-readable reason when :func:`is_gem_video_ready` is false."""
    root = resolve_gem_root()
    if root is None:
        return "KIMODO_GEM_ROOT not set or GENMO not cloned (run scripts/install_gem.sh)."
    if not is_gem_runner_available():
        return f"GEM demo missing under {root} or .venv Python not found."
    missing: list[str] = []
    for name, path in _gem_video_asset_paths(root).items():
        if name == "yolo":
            min_bytes = _MIN_YOLO_BYTES
        elif path.name in _BODY_MODEL_FILES:
            min_bytes = _BODY_MODEL_FILES[path.name]
        else:
            min_bytes = 1_000_000
        if not path.is_file() or path.stat().st_size < min_bytes:
            missing.append(str(path))
    if missing:
        return (
            "GEM video checkpoints missing or incomplete:\n  "
            + "\n  ".join(missing)
            + "\nRun: bash scripts/download_gem_checkpoints.sh"
        )
    return None


def resolve_gem_detector() -> str:
    """``yolov8`` (default) or ``yolox`` — passed to GENMO ``demo_smpl_hpe.py --detector``."""
    raw = os.environ.get("KIMODO_GEM_DETECTOR", "yolov8").strip().lower()
    if raw in ("yolox", "yolox_bytetrack", "bytetrack"):
        return "yolox"
    return "yolov8"


def gem_yolox_readiness_error() -> str | None:
    """Return error when YOLOX path is selected but onnxruntime is missing in GENMO venv."""
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return (
            "YOLOX+ByteTrack needs onnxruntime in the GENMO venv. "
            "In vendor/GENMO: uv pip install onnxruntime-gpu  "
            "(see docs/INSTALL.md Steps 8–10)."
        )
    return None


def _tensor_to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _env_yaw_180_enabled() -> bool:
    raw = os.environ.get("KIMODO_GEM_YAW_180", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _env_bbox_translation_enabled() -> bool:
    return os.environ.get("KIMODO_GEM_BBOX_TRANSL", "").strip().lower() in ("1", "true", "yes")


def _apply_gem_rigid_yup(
    trans: np.ndarray,
    root_orient: np.ndarray,
    r_fix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r = torch.as_tensor(r_fix, dtype=torch.float32)
    root_mats = axis_angle_to_matrix(torch.as_tensor(root_orient, dtype=torch.float32))
    root_mats = torch.einsum("ij,tjk->tik", r, root_mats)
    root_out = matrix_to_axis_angle(root_mats).numpy().astype(np.float32)
    trans_out = torch.as_tensor(trans, dtype=torch.float32) @ r.T
    return trans_out.numpy().astype(np.float32), root_out


def _apply_gem_yaw_180_yup(
    trans: np.ndarray,
    root_orient: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return _apply_gem_rigid_yup(trans, root_orient, _GEM_YAW_180_YUP)


def _gem_global_yup_to_amass_zup(
    trans: np.ndarray,
    root_orient: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Match :func:`kimodo.exports.smplx.get_amass_parameters` (``z_up=True``)."""
    return _apply_gem_rigid_yup(trans, root_orient, _GEM_TO_AMASS_R)


def _preprocess_bbx_path_for_smpl_pt(smpl_params_pt: Path) -> Path | None:
    """``demo_smpl_hpe`` writes ``preprocess/bbx_{yolov8|yolox}.pt`` next to ``smpl_params.pt``."""
    pre = smpl_params_pt.parent / "preprocess"
    det = resolve_gem_detector()
    for name in (f"bbx_{det}.pt", "bbx_yolov8.pt", "bbx_yolox.pt", "bbx.pt"):
        candidate = pre / name
        if candidate.is_file():
            return candidate
    return None


def _apply_bbox_image_translation(
    trans: np.ndarray,
    bbx_path: Path,
    *,
    person_height_m: float = 1.7,
) -> np.ndarray:
    """Add coarse world translation from YOLO bbox center motion (static-camera heuristic)."""
    try:
        bbx = torch.load(bbx_path, map_location="cpu", weights_only=False)
    except TypeError:
        bbx = torch.load(bbx_path, map_location="cpu")
    bbx = _tensor_to_numpy(bbx).astype(np.float32)
    if bbx.ndim != 2 or bbx.shape[1] < 3 or bbx.shape[0] != trans.shape[0]:
        return trans

    cx, cy, size = bbx[:, 0], bbx[:, 1], np.maximum(bbx[:, 2], 1.0)
    meters_per_px = person_height_m / size
    dx = (cx - cx[0]) * meters_per_px
    dz = -(cy - cy[0]) * meters_per_px

    out = trans.copy()
    out[:, 0] = out[:, 0] + dx.astype(np.float32)
    out[:, 2] = out[:, 2] + dz.astype(np.float32)
    return out


def _mean_betas(betas) -> np.ndarray:
    arr = _tensor_to_numpy(betas).astype(np.float32)
    if arr.ndim >= 2:
        arr = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
    arr = arr.reshape(-1)
    if arr.shape[0] < 16:
        arr = np.pad(arr, (0, 16 - arr.shape[0]))
    return arr[:16].astype(np.float32)


def gem_pt_to_amass_npz(
    smpl_params_pt: PathLike,
    npz_out: PathLike,
    *,
    fps: float = 30.0,
    use_global: bool = True,
    amass_z_up: bool = False,
    bbox_translation: bool | None = None,
    yaw_180: bool | None = None,
) -> Path:
    """Convert GEM ``smpl_params.pt`` to AMASS-layout SMPL-X NPZ for GMR / Studio.

    By default writes ``body_params_global`` as GEM/GVHMR produced them (Y-up world, same as GMR
    ``load_gvhmr_pred_file``). Set ``amass_z_up=True`` only for experiments.

    ``yaw_180`` (default on, env ``KIMODO_GEM_YAW_180=0`` to disable) rotates global root by 180°
    about +Y so the character faces Studio/GMR forward instead of backward.
    """
    pt_path = Path(smpl_params_pt)
    if not pt_path.is_file():
        raise FileNotFoundError(f"GEM output not found: {pt_path}")

    try:
        pred = torch.load(pt_path, map_location="cpu", weights_only=False)
    except TypeError:
        pred = torch.load(pt_path, map_location="cpu")
    key = "body_params_global" if use_global else "body_params_incam"
    if key not in pred:
        alt = "body_params_incam" if use_global else "body_params_global"
        if alt in pred:
            key = alt
        else:
            raise KeyError(f"{pt_path} has no body_params_global or body_params_incam")

    bp = pred[key]
    body_pose = _tensor_to_numpy(bp["body_pose"]).astype(np.float32)
    if body_pose.ndim == 3:
        body_pose = body_pose.reshape(body_pose.shape[0], -1)
    root_orient = _tensor_to_numpy(bp["global_orient"]).astype(np.float32)
    trans = _tensor_to_numpy(bp["transl"]).astype(np.float32)
    betas = _mean_betas(bp.get("betas", np.zeros(10, dtype=np.float32)))

    num_frames = int(body_pose.shape[0])
    if root_orient.shape[0] != num_frames or trans.shape[0] != num_frames:
        raise ValueError(
            f"Frame count mismatch in {pt_path}: body={num_frames}, "
            f"root={root_orient.shape[0]}, trans={trans.shape[0]}"
        )
    if body_pose.shape[1] != 63:
        raise ValueError(f"Expected pose_body width 63 (21 SMPL joints), got {body_pose.shape[1]}")

    if amass_z_up and use_global:
        trans, root_orient = _gem_global_yup_to_amass_zup(trans, root_orient)

    use_yaw = _env_yaw_180_enabled() if yaw_180 is None else yaw_180
    if use_yaw and use_global:
        trans, root_orient = _apply_gem_yaw_180_yup(trans, root_orient)

    use_bbox = _env_bbox_translation_enabled() if bbox_translation is None else bbox_translation
    if use_bbox and use_global:
        bbx_path = _preprocess_bbx_path_for_smpl_pt(pt_path)
        if bbx_path is not None:
            trans = _apply_bbox_image_translation(trans, bbx_path)
        else:
            print(
                f"[GEM] KIMODO_GEM_BBOX_TRANSL=1 but no bbox cache under {pt_path.parent / 'preprocess'}; "
                "skipping bbox root translation.",
                flush=True,
            )

    mean_hands_path = skeleton_asset_path("smplx22", "mean_hands.npy")
    if mean_hands_path.is_file():
        mean_hands = np.load(mean_hands_path).astype(np.float32)
    else:
        mean_hands = np.zeros(90, dtype=np.float32)
    pose_hand = np.tile(mean_hands.reshape(1, -1), (num_frames, 1))

    out_path = Path(npz_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        gender="neutral",
        surface_model_type="smplx",
        mocap_frame_rate=float(fps),
        mocap_time_length=num_frames / float(fps),
        trans=trans,
        root_orient=root_orient,
        pose_body=body_pose,
        pose_hand=pose_hand,
        pose_jaw=np.zeros((num_frames, 3), dtype=np.float32),
        pose_eye=np.zeros((num_frames, 6), dtype=np.float32),
        betas=betas,
        num_betas=len(betas),
    )
    return out_path


def amass_npz_to_kimodo_tensors(
    npz_path: PathLike,
    skeleton: SkeletonBase,
    *,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """AMASS SMPL-X NPZ → Kimodo ``joints_pos`` / ``joints_rot`` / ``joints_local_rot``."""
    data = np.load(npz_path, allow_pickle=True)
    for key in ("pose_body", "root_orient", "trans"):
        if key not in data.files:
            raise ValueError(f"AMASS NPZ missing {key!r}: {npz_path}")

    if not isinstance(skeleton, SMPLXSkeleton22):
        raise TypeError(
            f"Video import requires SMPL-X skeleton (got {type(skeleton).__name__}). "
            "Switch the demo model to Kimodo-SMPLX-*."
        )

    device = torch.device(device)
    skeleton = skeleton.to(device)

    num_frames = int(data["pose_body"].shape[0])
    root_aa = torch.as_tensor(data["root_orient"], dtype=torch.float32, device=device)
    body_aa = torch.as_tensor(data["pose_body"], dtype=torch.float32, device=device).reshape(
        num_frames, 21, 3
    )
    all_aa = torch.cat([root_aa.unsqueeze(1), body_aa], dim=1)
    local_rot = axis_angle_to_matrix(all_aa.unsqueeze(0))

    trans = torch.as_tensor(data["trans"], dtype=torch.float32, device=device)
    pelvis_offset = skeleton.neutral_joints[skeleton.root_idx].to(device=device, dtype=torch.float32)
    root_positions = (trans + pelvis_offset).unsqueeze(0)

    joints_rot, joints_pos, joints_local_rot = skeleton.fk(local_rot, root_positions)
    return joints_pos[0], joints_rot[0], joints_local_rot[0]


def run_gem_video_hpe(
    video_path: PathLike,
    output_root: PathLike,
    *,
    static_cam: bool = True,
    detector: str | None = None,
    yolo_period: int | None = None,
    ckpt_path: str | None = None,
    timeout_sec: float | None = None,
) -> Path:
    """Run GEM-SMPL ``demo_smpl_hpe.py`` in the external GENMO env; return ``smpl_params.pt`` path."""
    video = Path(video_path).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")

    gem_root = resolve_gem_root()
    if gem_root is None:
        raise RuntimeError(
            "GEM (GENMO) not found. Clone https://github.com/NVlabs/GENMO and run scripts/install_gem.sh, "
            "or set KIMODO_GEM_ROOT to the repo path."
        )
    gem_python = resolve_gem_python(gem_root)
    if gem_python is None:
        raise RuntimeError(
            f"No Python for GEM under {gem_root}. Run scripts/install_gem.sh or set KIMODO_GEM_PYTHON."
        )
    demo_script = gem_root / _GEM_DEMO_SCRIPT
    if not demo_script.is_file():
        raise FileNotFoundError(f"GEM demo script missing: {demo_script}")

    yolo_weights = gem_root / _YOLO_WEIGHTS_NAME
    if not yolo_weights.is_file() or yolo_weights.stat().st_size < _MIN_YOLO_BYTES:
        raise FileNotFoundError(
            f"YOLO weights missing or incomplete: {yolo_weights} (~131 MB).\n"
            "GENMO downloads this at runtime from GitHub; if that fails, copy manually:\n"
            f"  scp yolov8x.pt user@host:{yolo_weights}\n"
            "Or: bash scripts/download_gem_checkpoints.sh (when GitHub/HF is reachable)."
        )

    out_root = Path(output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(gem_python),
        str(demo_script),
        "--video",
        str(video),
        "--output_root",
        str(out_root),
        "--no_render",
    ]
    readiness_err = gem_video_readiness_error()
    if readiness_err:
        raise RuntimeError(readiness_err)

    det = resolve_gem_detector() if detector is None else detector.strip().lower()
    if det not in ("yolov8", "yolox"):
        det = "yolov8"
    if det == "yolox":
        yolox_err = gem_yolox_readiness_error()
        if yolox_err:
            raise RuntimeError(yolox_err)

    if static_cam:
        cmd.append("--static_cam")
    cmd.extend(["--detector", det])
    if det == "yolox":
        period = yolo_period
        if period is None:
            period = int(os.environ.get("KIMODO_GEM_YOLOX_PERIOD", "1"))
        cmd.extend(["--yolo-period", str(max(1, int(period)))])
    ckpt = ckpt_path or os.environ.get("KIMODO_GEM_CKPT", "").strip()
    if ckpt:
        cmd.extend(["--ckpt_path", ckpt])

    env = os.environ.copy()
    env["PYTHONPATH"] = str(gem_root) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    if timeout_sec is None:
        raw = os.environ.get("KIMODO_GEM_TIMEOUT_SEC", "1800").strip()
        timeout_sec = float(raw) if raw else None

    print(f"[GEM] Running video HPE: {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(
            cmd,
            cwd=str(gem_root),
            env=env,
            check=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"GEM video inference timed out after {timeout_sec}s") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"GEM video inference failed (exit {exc.returncode})") from exc

    smpl_pt = out_root / video.stem / "smpl_params.pt"
    if not smpl_pt.is_file():
        raise FileNotFoundError(
            f"GEM finished but {smpl_pt} was not created. Check GEM logs under {out_root}."
        )
    return smpl_pt


def import_video_to_amass_npz(
    video_path: PathLike,
    work_dir: PathLike,
    *,
    static_cam: bool = True,
    detector: str | None = None,
    yolo_period: int | None = None,
    fps: float = 30.0,
) -> Path:
    """Full GEM subprocess + AMASS NPZ export."""
    work = Path(work_dir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    smpl_pt = run_gem_video_hpe(
        video_path,
        work / "gem_out",
        static_cam=static_cam,
        detector=detector,
        yolo_period=yolo_period,
    )
    amass_path = work / f"{Path(video_path).stem}_amass.npz"
    gem_pt_to_amass_npz(smpl_pt, amass_path, fps=fps)
    return amass_path
