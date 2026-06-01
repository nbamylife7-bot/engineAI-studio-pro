"""T800 motion quality audit (GMR ``motion_quality``) for Kimodo demo playback."""

from __future__ import annotations

import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .gmr_bootstrap import bootstrap_gmr, is_t800_available

QualityVerdict = Literal["pass", "warn", "fail"]

# Ankle roll is often pinned near a limit by GMR / foot flattening — not actionable in demo QA.
_PINNED_LIMIT_JOINTS = frozenset({"J05_ANKLE_ROLL_L", "J11_ANKLE_ROLL_R"})

# Leg / root joints checked for IK ping-pong (pelvis vs foot target conflict).
_OSCILLATION_JOINTS: tuple[tuple[str, int], ...] = (
    ("J03_KNEE_PITCH_L", 3),
    ("J09_KNEE_PITCH_R", 9),
    ("J04_ANKLE_PITCH_L", 4),
    ("J10_ANKLE_PITCH_R", 10),
)

# Sphere-bound pairs that are usually mesh overlap, not real interpenetration.
_STRUCTURAL_COLLISION_PREFIXES = (
    ("LINK_BASE", "LINK_HIP_ROLL"),
    ("LINK_BASE", "LINK_HIP_PITCH"),
)


@dataclass
class T800QualityRecord:
    character_name: str
    pkl_path: str
    report_path: str
    verdict: QualityVerdict
    score: int
    summary: dict[str, Any]
    report: dict[str, Any]

    @property
    def verdict_label(self) -> str:
        return {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}[self.verdict]


def _safe_character_slug(name: str) -> str:
    slug = re.sub(r"[^\w\-]+", "_", name.strip())
    return slug or "character"


def quality_output_dir(client_id: int) -> Path:
    base = os.environ.get("KIMODO_T800_QUALITY_DIR", "").strip()
    root = Path(base).expanduser() if base else Path.home() / ".cache" / "kimodo_demo" / "t800_quality"
    return (root / str(client_id)).resolve()


def motion_pkl_bytes(qpos_frames: list[np.ndarray], fps: float) -> bytes:
    """Serialize retargeted T800 qpos frames to GMR motion PKL bytes."""
    return pickle.dumps(build_motion_data_from_qpos(qpos_frames, fps), protocol=pickle.HIGHEST_PROTOCOL)


def build_motion_data_from_qpos(qpos_frames: list[np.ndarray], fps: float) -> dict[str, Any]:
    bootstrap_gmr()
    from general_motion_retargeting.motion_quality import build_motion_data

    root_pos = np.asarray([qpos[:3] for qpos in qpos_frames], dtype=np.float64)
    root_rot = np.asarray([qpos[3:7][[1, 2, 3, 0]] for qpos in qpos_frames], dtype=np.float64)
    dof_pos = np.asarray([qpos[7:] for qpos in qpos_frames], dtype=np.float64)
    return build_motion_data(
        fps=fps,
        root_pos=root_pos,
        root_rot=root_rot,
        dof_pos=dof_pos,
    )


def _t800_quality_config(dof_count: int):
    bootstrap_gmr()
    import mujoco as mj
    from general_motion_retargeting.motion_quality import MotionQualityConfig
    from general_motion_retargeting.params import ROBOT_XML_DICT
    from kimodo.viz.t800_rig import load_t800_mj_model

    model = load_t800_mj_model()
    xml_path = ROBOT_XML_DICT["t800"].resolve()
    lower: list[float] = []
    upper: list[float] = []
    names: list[str] = []
    for i in range(dof_count):
        qpos_adr = 7 + i
        joint_id = 0
        for j in range(model.njnt):
            if int(model.jnt_qposadr[j]) == qpos_adr:
                joint_id = j
                break
        lo, hi = [float(v) for v in model.jnt_range[joint_id]]
        lower.append(lo)
        upper.append(hi)
        joint_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id) or f"J{i:02d}"
        names.append(str(joint_name))

    return MotionQualityConfig(
        robot="t800",
        joint_names=names,
        joint_lower_limits=np.asarray(lower, dtype=np.float64),
        joint_upper_limits=np.asarray(upper, dtype=np.float64),
        model_path=str(xml_path),
        # Kimodo demo: slightly looser GMR defaults; we post-filter advisory hits below.
        jump_threshold_rad=0.55,
        velocity_threshold_rad_s=18.0,
        limit_margin_rad=0.02,
        floor_clearance=0.0,
        collision_margin=-0.02,
        max_collision_pairs=30,
    )


def _is_structural_collision(issue: dict[str, Any]) -> bool:
    body_a = str(issue.get("body_a", ""))
    body_b = str(issue.get("body_b", ""))
    pair = (body_a, body_b)
    rev = (body_b, body_a)
    for a_prefix, b_prefix in _STRUCTURAL_COLLISION_PREFIXES:
        if (pair[0].startswith(a_prefix) and pair[1].startswith(b_prefix)) or (
            rev[0].startswith(a_prefix) and rev[1].startswith(b_prefix)
        ):
            return True
    clearance = float(issue.get("clearance_m", 0.0))
    return clearance > -0.035


def _filter_limit_pressure(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for issue in issues:
        name = str(issue.get("joint_name", ""))
        if name in _PINNED_LIMIT_JOINTS:
            dist = float(issue.get("distance_to_limit_rad", 0.0))
            if dist <= 0.012:
                continue
        filtered.append(issue)
    return filtered


def _filter_floor_anomalies(
    issues: list[dict[str, Any]],
    floor_summary: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not issues:
        return []
    if not floor_summary:
        return issues
    median_z = float(floor_summary.get("median_support_z", 0.0))
    # GMR retarget often sits ~5–8 cm below MuJoCo z=0; only flag worse-than-typical grounding.
    threshold = median_z - 0.025
    return [issue for issue in issues if float(issue.get("support_min_z", 0.0)) < threshold]


def _filter_collisions(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [issue for issue in issues if not _is_structural_collision(issue)]


def _kimodo_velocity_spikes(
    dof_pos: np.ndarray,
    fps: float,
    joint_names: list[str],
) -> list[dict[str, Any]]:
    """Step velocity only (avoids np.gradient overshoot used by raw GMR audit)."""
    if dof_pos.shape[0] < 2 or fps <= 0:
        return []
    step_vel = np.zeros_like(dof_pos, dtype=np.float64)
    step_vel[1:] = np.diff(dof_pos, axis=0) * float(fps)
    step_vel[0] = step_vel[1]
    threshold = 18.0
    frames, joints = np.where(np.abs(step_vel) >= threshold)
    issues: list[dict[str, Any]] = []
    for frame, joint_idx in zip(frames, joints):
        issues.append(
            {
                "kind": "velocity_spike",
                "frame": int(frame) + 1,
                "joint_index": int(joint_idx),
                "joint_name": joint_names[joint_idx],
                "velocity_rad_s": float(step_vel[frame, joint_idx]),
                "abs_velocity_rad_s": float(abs(step_vel[frame, joint_idx])),
            }
        )
    return sorted(issues, key=lambda item: item["abs_velocity_rad_s"], reverse=True)


def _detect_oscillation_segments(
    qpos_frames: list[np.ndarray],
    *,
    fps: float,
    window: int = 14,
    min_flips: int = 5,
    min_amplitude_rad: float = 0.028,
    min_root_z_amplitude_m: float = 0.003,
) -> list[dict[str, Any]]:
    if len(qpos_frames) < window + 1:
        return []

    dof = np.asarray([q[7:] for q in qpos_frames], dtype=np.float64)
    root_z = np.asarray([q[2] for q in qpos_frames], dtype=np.float64)
    segments: list[dict[str, Any]] = []

    def _scan_signal(name: str, values: np.ndarray, *, min_amplitude: float) -> None:
        if values.shape[0] < window + 1:
            return
        last_reported = -window * 2
        for start in range(0, values.shape[0] - window):
            if start < last_reported + window - 2:
                continue
            end = start + window
            chunk = values[start:end]
            diffs = np.diff(chunk)
            flips = int(np.sum(diffs[1:] * diffs[:-1] < 0))
            amplitude = float(np.max(chunk) - np.min(chunk))
            if flips >= min_flips and amplitude >= min_amplitude:
                segments.append(
                    {
                        "kind": "oscillation",
                        "signal": name,
                        "start_frame": int(start + 1),
                        "end_frame": int(end),
                        "flip_count": flips,
                        "amplitude_rad": amplitude,
                        "start_sec": round(start / fps, 3),
                        "end_sec": round((end - 1) / fps, 3),
                    }
                )
                last_reported = start

    _scan_signal("root_z", root_z, min_amplitude=min_root_z_amplitude_m)
    for joint_name, joint_idx in _OSCILLATION_JOINTS:
        if joint_idx < dof.shape[1]:
            _scan_signal(joint_name, dof[:, joint_idx], min_amplitude=min_amplitude_rad)

    return segments


def _merge_oscillation_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not segments:
        return []
    merged: list[dict[str, Any]] = []
    for seg in sorted(segments, key=lambda s: (s["start_frame"], s["signal"])):
        if not merged:
            merged.append(seg)
            continue
        prev = merged[-1]
        if seg["signal"] == prev["signal"] and seg["start_frame"] <= prev["end_frame"] + 2:
            prev["end_frame"] = max(prev["end_frame"], seg["end_frame"])
            prev["flip_count"] = max(prev["flip_count"], seg["flip_count"])
            prev["amplitude_rad"] = max(prev["amplitude_rad"], seg["amplitude_rad"])
            prev["end_sec"] = seg["end_sec"]
        else:
            merged.append(seg)
    return merged


def _build_kimodo_summary(
    report: dict[str, Any],
    qpos_frames: list[np.ndarray],
    *,
    fps: float,
    joint_names: list[str],
) -> dict[str, Any]:
    issues = report.get("issues", {})
    floor_summary = report.get("floor_summary") or {}

    limit_filtered = _filter_limit_pressure(list(issues.get("limit_pressure", [])))
    floor_filtered = _filter_floor_anomalies(list(issues.get("floor_anomalies", [])), floor_summary)
    collision_filtered = _filter_collisions(list(issues.get("candidate_collisions", [])))

    dof_pos = np.asarray([qpos[7:] for qpos in qpos_frames], dtype=np.float64)
    velocity_filtered = _kimodo_velocity_spikes(dof_pos, fps, joint_names)
    oscillations = _merge_oscillation_segments(_detect_oscillation_segments(qpos_frames, fps=fps))
    significant_osc = _significant_oscillations(oscillations)

    jumps = list(issues.get("qpos_jumps", []))
    severe_jumps = [j for j in jumps if float(j.get("abs_delta_rad", 0.0)) >= 0.55]

    return {
        "qpos_jump_count": len(jumps),
        "severe_jump_count": len(severe_jumps),
        "velocity_spike_count": len(velocity_filtered),
        "limit_pressure_count": len(limit_filtered),
        "floor_anomaly_count": len(floor_filtered),
        "candidate_collision_count": len(collision_filtered),
        "oscillation_count": len(significant_osc),
        "oscillation_raw_count": len(oscillations),
        "median_support_z": float(floor_summary.get("median_support_z", 0.0)),
        "top_jump": jumps[0] if jumps else None,
        "top_spike": velocity_filtered[0] if velocity_filtered else None,
        "oscillations": significant_osc[:8],
        "hints": _build_hints(severe_jumps, velocity_filtered, oscillations),
    }


def _significant_oscillations(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep standing IK ping-pong; drop large-amplitude leg swings during punches."""
    significant: list[dict[str, Any]] = []
    for seg in segments:
        amp = float(seg.get("amplitude_rad", 0.0))
        if seg.get("signal") == "root_z":
            if amp >= 0.003:
                significant.append(seg)
        elif amp < 0.12:
            significant.append(seg)
    return significant


def _build_hints(
    severe_jumps: list[dict[str, Any]],
    velocity_spikes: list[dict[str, Any]],
    oscillations: list[dict[str, Any]],
) -> list[str]:
    hints: list[str] = []
    for jump in severe_jumps[:2]:
        hints.append(
            f"Jump frame {jump['frame']}: {jump['joint_name']} "
            f"Δ{float(jump['abs_delta_rad']):.2f} rad"
        )

    ranked_osc = sorted(
        _significant_oscillations(oscillations),
        key=lambda seg: (float(seg.get("amplitude_rad", 0.0)), -int(seg.get("start_frame", 0))),
    )
    for seg in ranked_osc[:2]:
        unit = "m" if seg["signal"] == "root_z" else "rad"
        hints.append(
            f"IK oscillation frames {seg['start_frame']}–{seg['end_frame']}: "
            f"{seg['signal']} (~{float(seg['amplitude_rad']):.3f} {unit})"
        )

    if not severe_jumps and velocity_spikes:
        spike = velocity_spikes[0]
        hints.append(
            f"Fast motion frame {spike['frame']}: {spike['joint_name']} "
            f"{float(spike['abs_velocity_rad_s']):.1f} rad/s"
        )
    return hints


def compute_quality_verdict(
    summary: dict[str, Any],
    *,
    frame_count: int | None = None,
) -> tuple[QualityVerdict, int]:
    """Score Kimodo-adjusted metrics (``summary['kimodo']`` when present)."""
    kimodo = summary.get("kimodo")
    src = kimodo if isinstance(kimodo, dict) else summary

    jumps = int(src.get("severe_jump_count", src.get("qpos_jump_count", 0)))
    spikes = int(src.get("velocity_spike_count", 0))
    limits = int(src.get("limit_pressure_count", 0))
    floor = int(src.get("floor_anomaly_count", 0))
    collisions = int(src.get("candidate_collision_count", 0))
    oscillations = int(src.get("oscillation_count", 0))
    frames = max(int(frame_count or 0), 1)

    score = 100
    score -= min(35, jumps * 18)
    score -= min(25, oscillations * 12)
    score -= min(20, max(0, spikes - 2) * 4)
    score -= min(12, int(limits / max(frames / 4, 1)))
    score -= min(10, int(floor / max(frames / 8, 1)) * 3)
    score -= min(8, int(collisions / max(frames / 15, 1)) * 2)
    score = max(0, min(100, score))

    if jumps >= 2 or oscillations >= 2:
        return "fail", score
    if jumps == 1 or oscillations == 1 or spikes > 6 or score < 70:
        return "warn", score
    if score < 85:
        return "warn", score
    return "pass", score


def _enrich_report_for_kimodo(
    report: dict[str, Any],
    qpos_frames: list[np.ndarray],
    *,
    fps: float,
    joint_names: list[str],
) -> dict[str, Any]:
    kimodo = _build_kimodo_summary(report, qpos_frames, fps=fps, joint_names=joint_names)
    report = dict(report)
    summary = dict(report.get("summary", {}))
    summary["kimodo"] = kimodo
    report["summary"] = summary
    report["kimodo_issues"] = {
        "oscillations": kimodo.get("oscillations", []),
        "hints": kimodo.get("hints", []),
    }
    return report


def audit_t800_qpos(
    qpos_frames: list[np.ndarray],
    fps: float,
    *,
    character_name: str,
    client_id: int,
) -> T800QualityRecord:
    if not is_t800_available():
        raise RuntimeError("T800 quality audit requires GMR dependencies.")
    if len(qpos_frames) < 2:
        raise ValueError("T800 quality audit needs at least 2 qpos frames.")

    bootstrap_gmr()
    from general_motion_retargeting.motion_quality import audit_motion_quality, save_motion_pkl, write_json_report

    motion_data = build_motion_data_from_qpos(qpos_frames, fps)
    dof_count = int(motion_data["dof_pos"].shape[1])
    config = _t800_quality_config(dof_count)
    joint_names = list(config.joint_names or [])

    out_dir = quality_output_dir(client_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = _safe_character_slug(character_name)
    pkl_path = out_dir / f"{slug}.pkl"
    report_path = out_dir / f"{slug}.quality.json"

    save_motion_pkl(motion_data, pkl_path, overwrite=True)
    report = audit_motion_quality(motion_data, config=config, motion_path=pkl_path)
    report = _enrich_report_for_kimodo(report, qpos_frames, fps=float(fps), joint_names=joint_names)
    write_json_report(report, report_path, overwrite=True)

    summary = report["summary"]
    frame_count = int(report.get("schema", {}).get("frame_count", len(qpos_frames)))
    verdict, score = compute_quality_verdict(summary, frame_count=frame_count)
    return T800QualityRecord(
        character_name=character_name,
        pkl_path=str(pkl_path),
        report_path=str(report_path),
        verdict=verdict,
        score=score,
        summary=summary,
        report=report,
    )


def format_quality_markdown(
    records: dict[str, T800QualityRecord],
    *,
    errors: dict[str, str] | None = None,
) -> str:
    errors = errors or {}
    if not records and not errors:
        return "**T800 Quality:** click **Quality check** after Generate with T800."

    lines = ["**T800 Quality check** (Kimodo-calibrated)"]
    for name in sorted(set(records.keys()) | set(errors.keys())):
        if name in errors and name not in records:
            lines.append(f"- **{name}:** ERROR — {errors[name]}")
            continue
        rec = records[name]
        k = rec.summary.get("kimodo") or rec.summary
        badge = rec.verdict_label
        lines.append(
            f"- **{name}:** {badge} ({rec.score}/100) — "
            f"jumps {k.get('severe_jump_count', k.get('qpos_jump_count', 0))}, "
            f"osc {k.get('oscillation_count', 0)}, "
            f"spikes {k.get('velocity_spike_count', 0)}, "
            f"limits {k.get('limit_pressure_count', 0)}, "
            f"floor {k.get('floor_anomaly_count', 0)}, "
            f"collision {k.get('candidate_collision_count', 0)}"
        )
        for hint in (k.get("hints") or [])[:3]:
            lines.append(f"  - {hint}")
        lines.append(f"  - `.pkl`: `{rec.pkl_path}`")
        lines.append(f"  - report: `{rec.report_path}`")
        if name in errors:
            lines.append(f"  - note: audit partial — {errors[name]}")
    return "\n".join(lines)
