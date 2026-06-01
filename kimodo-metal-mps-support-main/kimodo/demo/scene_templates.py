# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ready-made scene presets: prompts (+ optional 2D root paths) for the Kimodo demo."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

TemplateId = Literal["approach_pick_turn", "circle_3m", "sprint_stop"]

TEMPLATE_LABELS: dict[str, str] = {
    "approach_pick_turn": "Approach → pick up → turn",
    "circle_3m": "Circle ~3 m",
    "sprint_stop": "Sprint → hard stop",
}


@dataclass(frozen=True)
class SceneTemplatePlan:
    template_id: str
    title: str
    description: str
    prompt_texts: tuple[str, ...]
    prompt_durations_sec: tuple[float, ...]
    num_samples: int = 1
    root2d_constraint: dict | None = None
    enable_postprocess: bool = True


def list_scene_template_ids() -> list[str]:
    return list(TEMPLATE_LABELS.keys())


def get_scene_template_plan(template_id: str, *, fps: float) -> SceneTemplatePlan:
    if template_id not in TEMPLATE_LABELS:
        raise ValueError(f"Unknown scene template: {template_id!r}")

    if template_id == "approach_pick_turn":
        return SceneTemplatePlan(
            template_id=template_id,
            title=TEMPLATE_LABELS[template_id],
            description="Three prompts: walk up, pick up, turn with object. Click **Generate**.",
            prompt_texts=(
                "A person walks forward slowly toward an object on the ground.",
                "A person bends down and picks up an object from the ground.",
                "A person turns to the left while holding an object in both hands.",
            ),
            prompt_durations_sec=(3.0, 3.0, 3.0),
        )

    if template_id == "circle_3m":
        total_sec = 8.0
        total_frames = max(30, int(round(total_sec * fps)))
        return SceneTemplatePlan(
            template_id=template_id,
            title=TEMPLATE_LABELS[template_id],
            description="One prompt + 2D circle path (~3 m diameter). Post-processing recommended.",
            prompt_texts=(
                "A person walks forward at a steady pace while following a curved path.",
            ),
            prompt_durations_sec=(total_sec,),
            root2d_constraint=build_circle_root2d_constraint(total_frames, radius_m=1.5),
        )

    if template_id == "sprint_stop":
        return SceneTemplatePlan(
            template_id=template_id,
            title=TEMPLATE_LABELS[template_id],
            description="Sprint then sudden stop — good for weight shift and T800 quality check.",
            prompt_texts=(
                "A person sprints forward at full speed with pumping arm swing.",
                "A person stops suddenly and stabilizes in a balanced wide stance.",
            ),
            prompt_durations_sec=(2.5, 2.5),
        )

    raise ValueError(f"Unhandled template: {template_id!r}")


def build_circle_root2d_constraint(total_frames: int, *, radius_m: float = 1.5) -> dict:
    """Circle in the Kimodo ground plane (X right, Z forward), starting near the origin."""
    total_frames = max(2, int(total_frames))
    frame_indices = list(range(total_frames))
    smooth_root_2d: list[list[float]] = []
    for i in range(total_frames):
        theta = 2.0 * math.pi * float(i) / float(total_frames)
        x = radius_m * math.sin(theta)
        z = radius_m * (1.0 - math.cos(theta))
        smooth_root_2d.append([x, z])
    return {
        "type": "root2d",
        "frame_indices": frame_indices,
        "smooth_root_2d": smooth_root_2d,
    }


def prompt_frame_bounds(
    durations_sec: tuple[float, ...],
    fps: float,
) -> list[tuple[int, int]]:
    """Convert per-prompt durations to timeline [start, end] frame bounds."""
    num_frames = 0
    bounds: list[tuple[int, int]] = []
    for i, duration in enumerate(durations_sec):
        n_frames = max(1, int(round(float(duration) * fps)))
        start_frame = num_frames
        if i == len(durations_sec) - 1:
            end_frame = num_frames + n_frames - 1
        else:
            end_frame = num_frames + n_frames
        bounds.append((start_frame, end_frame))
        num_frames += n_frames
    return bounds
