#!/usr/bin/env python3
"""CLI: video → GEM-SMPL (GENMO) → AMASS SMPL-X NPZ (for GMR / Kimodo T800 retarget)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_KIMODO = _ROOT / "kimodo"
if str(_KIMODO) not in sys.path:
    sys.path.insert(0, str(_KIMODO))

from kimodo.imports.gem import (  # noqa: E402
    gem_pt_to_amass_npz,
    gem_video_readiness_error,
    import_video_to_amass_npz,
    is_gem_video_ready,
    resolve_gem_detector,
    run_gem_video_hpe,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", help="Input .mp4 / .mov / .avi")
    parser.add_argument(
        "--smpl-params-pt",
        help="Skip GEM inference; convert existing smpl_params.pt to AMASS NPZ",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="",
        help="Output .npz path (default: <video_stem>_amass.npz next to video or cwd)",
    )
    parser.add_argument(
        "--work-dir",
        default="",
        help="Temp dir for GEM outputs (default: system temp)",
    )
    parser.add_argument(
        "--static-cam",
        action="store_true",
        default=True,
        help="Assume static camera (default: on)",
    )
    parser.add_argument(
        "--moving-cam",
        action="store_true",
        help="Disable static camera (GEM uses moving-camera path when available)",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="AMASS mocap_frame_rate")
    parser.add_argument(
        "--bbox-transl",
        action="store_true",
        help="Add horizontal root motion from YOLO bbox drift (or set KIMODO_GEM_BBOX_TRANSL=1)",
    )
    parser.add_argument(
        "--no-yaw-180",
        action="store_true",
        help="Do not rotate global root 180° about Y (default: on; env KIMODO_GEM_YAW_180=0)",
    )
    parser.add_argument(
        "--detector",
        choices=("yolov8", "yolox"),
        default="",
        help="Person detector (default: KIMODO_GEM_DETECTOR or yolov8)",
    )
    parser.add_argument(
        "--yolo-period",
        type=int,
        default=0,
        help="YOLOX: detect every N frames (0 = use KIMODO_GEM_YOLOX_PERIOD or 1)",
    )
    args = parser.parse_args()
    yaw_180 = not args.no_yaw_180
    detector = args.detector.strip().lower() if args.detector else resolve_gem_detector()
    yolo_period = args.yolo_period if args.yolo_period > 0 else None

    if args.smpl_params_pt:
        pt = Path(args.smpl_params_pt).expanduser().resolve()
        out = Path(args.output).expanduser().resolve() if args.output else pt.with_name(f"{pt.parent.name}_amass.npz")
        gem_pt_to_amass_npz(
            pt, out, fps=args.fps, bbox_translation=args.bbox_transl or None, yaw_180=yaw_180
        )
        print(f"Wrote {out}")
        return 0

    if not args.video:
        parser.error("Provide --video or --smpl-params-pt")

    readiness_err = gem_video_readiness_error()
    if readiness_err:
        print(readiness_err, file=sys.stderr)
        return 1
    if not is_gem_video_ready():
        print("GEM video pipeline is not ready.", file=sys.stderr)
        return 1

    video = Path(args.video).expanduser().resolve()
    if not video.is_file():
        print(f"Video not found: {video}", file=sys.stderr)
        return 1

    static_cam = not args.moving_cam
    if args.work_dir:
        work = Path(args.work_dir).expanduser().resolve()
        work.mkdir(parents=True, exist_ok=True)
        smpl_pt = run_gem_video_hpe(
            video,
            work / "gem_out",
            static_cam=static_cam,
            detector=detector,
            yolo_period=yolo_period,
        )
        out = (
            Path(args.output).expanduser().resolve()
            if args.output
            else work / f"{video.stem}_amass.npz"
        )
        gem_pt_to_amass_npz(
            smpl_pt, out, fps=args.fps, bbox_translation=args.bbox_transl or None, yaw_180=yaw_180
        )
    else:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="kimodo_gem_cli_") as tmp:
            out_path = import_video_to_amass_npz(
                video,
                tmp,
                static_cam=static_cam,
                detector=detector,
                yolo_period=yolo_period,
                fps=args.fps,
            )
            out = Path(args.output).expanduser().resolve() if args.output else video.with_name(f"{video.stem}_amass.npz")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(Path(out_path).read_bytes())

    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
