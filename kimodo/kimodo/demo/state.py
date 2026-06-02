# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Optional
import threading

import torch

import kimodo.viz.viser_utils as viser_utils
import viser
from kimodo.skeleton import SkeletonBase
from kimodo.viz.viser_utils import GuiElements

from .config import (
    DEFAULT_CUR_DURATION,
    DEFAULT_MODEL,
    DEFAULT_PLAYBACK_SPEED,
)


@dataclass(frozen=True)
class ModelBundle:
    model: object
    motion_rep: object
    skeleton: SkeletonBase
    model_fps: float


@dataclass
class ClientSession:
    """Per-client session data."""

    client: viser.ClientHandle
    gui_elements: GuiElements
    motions: dict  # character_name -> CharacterMotion
    constraints: dict[str, viser_utils.ConstraintSet] = field(default_factory=dict)
    timeline_data: object = None
    frame_idx: int = 0
    playing: bool = False
    playback_speed: float = DEFAULT_PLAYBACK_SPEED
    cur_duration: float = DEFAULT_CUR_DURATION
    max_frame_idx: int = 100  # will be updated based on model_fps
    updating_motions: bool = False
    edit_mode: bool = False
    model_name: str = DEFAULT_MODEL
    model_fps: float = 0.0
    skeleton: SkeletonBase | None = None
    motion_rep: object | None = None
    examples_base_dir: str = ""
    example_dict: dict[str, str] = field(default_factory=dict)
    gui_examples_dropdown: Optional[viser.GuiInputHandle] = None
    gui_save_example_path_text: Optional[viser.GuiInputHandle] = None
    gui_model_selector: Optional[viser.GuiInputHandle] = None
    last_prompt_texts: Optional[list[str]] = None
    last_prompt_embeddings: Optional[torch.Tensor] = None
    last_prompt_lengths: Optional[list[int]] = None
    edit_mode_snapshot: Optional[dict[int, dict[str, object]]] = None
    undo_drag_snapshot: Optional[dict[str, object]] = None
    show_only_current_constraint: bool = False  # False = Show All, True = Show only Current
    t800_motions: dict = field(default_factory=dict)  # character_name -> T800CharacterMotion
    t800_quality: dict = field(default_factory=dict)  # character_name -> T800QualityRecord
    t800_quality_errors: dict = field(default_factory=dict)  # character_name -> error message
    t800_bootstrap_robot: object | None = None
    t800_preload_done: threading.Event = field(default_factory=threading.Event)
    t800_preload_in_progress: bool = False
    t800_retarget_lock: threading.Lock = field(default_factory=threading.Lock)
    skin_precompute_generation: int = 0
    # Set while a multi-sample result is on screen: click a human OR robot to commit that sample.
    sample_commit_callback: object | None = None
