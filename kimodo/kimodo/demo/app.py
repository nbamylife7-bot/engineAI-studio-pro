# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import os
import shutil
import threading
import time
from typing import Optional

import numpy as np
import torch

import viser
from kimodo.assets import DEMO_ASSETS_ROOT
from kimodo.device_utils import release_device_memory, resolve_torch_device
from kimodo.model.load_model import load_model
from kimodo.model.registry import resolve_model_name
from kimodo.skeleton import SkeletonBase, SOMASkeleton30
from kimodo.tools import load_json
from kimodo.viz import viser_utils
from kimodo.viz.viser_utils import (
    Character,
    CharacterMotion,
    EEJointsKeyframeSet,
    FullbodyKeyframeSet,
    RootKeyframe2DSet,
)
from viser.theme import TitlebarButton, TitlebarConfig, TitlebarImage

from . import generation, ui
from .config import (
    DARK_THEME,
    DEFAULT_CUR_DURATION,
    DEFAULT_MODEL,
    DEFAULT_PLAYBACK_SPEED,
    DEFAULT_PROMPT,
    DEMO_UI_QUICK_START_MODAL_MD,
    EXAMPLES_ROOT_DIR,
    HF_MODE,
    KIMODO_DEFER_SKIN_PRECOMPUTE,
    KIMODO_HORIZON_LOGO,
    KIMODO_HORIZON_LOGO_DISTANCE,
    KIMODO_HORIZON_LOGO_ELEVATION_DEG,
    KIMODO_HORIZON_LOGO_HEIGHT,
    KIMODO_HORIZON_LOGO_SIZE,
    KIMODO_GEM_DEMO_MODEL,
    KIMODO_GEM_ENABLED,
    KIMODO_T800_ENABLED,
    KIMODO_T800_HIDE_HUMAN_MESH,
    KIMODO_T800_SYNC_ATTACH,
    DEFAULT_T800_SKIN,
    LIGHT_THEME,
    MAX_ACTIVE_USERS,
    MAX_DURATION,
    MAX_SESSION_MINUTES,
    MIN_DURATION,
    MODEL_EXAMPLES_DIRS,
    MODEL_NAMES,
    SERVER_NAME,
    SERVER_PORT,
)
from .embedding_cache import CachedTextEncoder
from .queue_manager import QueueManager, UserQueue
from .state import ClientSession, ModelBundle


class Demo:
    def __init__(self, default_model_name: str = DEFAULT_MODEL):
        from kimodo.device_utils import enable_fast_matmul_precision

        enable_fast_matmul_precision()
        self.device = resolve_torch_device("auto")
        print(f"Using device: {self.device}")
        self.models: dict[str, ModelBundle] = {}
        resolved = resolve_model_name(default_model_name, "Kimodo")
        if resolved not in MODEL_NAMES:
            raise ValueError(f"Unknown model '{default_model_name}'. Expected one of: {MODEL_NAMES}")
        self.default_model_name = resolved
        self.floor_len = 20.0  # meters — needed before client connect / setup_scene
        self.ensure_examples_layout()
        self.load_model(self.default_model_name)
        self._t800_warmup_thread: threading.Thread | None = None
        if KIMODO_T800_ENABLED:
            self._t800_warmup_thread = threading.Thread(
                target=self._warm_t800_cache_background,
                daemon=True,
                name="t800-cache-warmup",
            )
            self._t800_warmup_thread.start()

        # Per-client sessions
        self.client_sessions: dict[int, ClientSession] = {}
        self.start_direction_markers: dict[int, viser_utils.WaypointMesh] = {}
        self.grid_handles: dict[int, viser.GridHandle] = {}

        self.server = viser.ViserServer(
            host=SERVER_NAME,
            port=SERVER_PORT,
            label="Studio Pro",
            enable_camera_keyboard_controls=False,  # don't move the camera with the arrow keys
        )
        self.server.scene.world_axes.visible = False  # used for debugging
        self.server.scene.set_up_direction("+y")

        # Register callbacks for session handling
        self.server.on_client_connect(self.on_client_connect)
        self.server.on_client_disconnect(self.on_client_disconnect)

        if self._t800_warmup_thread is not None:
            self._t800_warmup_thread.join(timeout=180.0)
            if self._t800_warmup_thread.is_alive():
                print("T800 mesh cache warmup still running in background.")

        # HF mode: queue and session limit
        if HF_MODE:
            self.user_queue = UserQueue(MAX_ACTIVE_USERS, MAX_SESSION_MINUTES)
            self.queue_manager = QueueManager(
                queue=self.user_queue,
                server=self.server,
                setup_demo_for_client=self._setup_demo_for_client,
                cleanup_session=self._cleanup_session_for_client,
            )
        else:
            self.user_queue = None
            self.queue_manager = None

    def ensure_examples_layout(self) -> None:
        os.makedirs(EXAMPLES_ROOT_DIR, exist_ok=True)
        for model_dir in MODEL_EXAMPLES_DIRS.values():
            os.makedirs(model_dir, exist_ok=True)

        for entry in os.listdir(EXAMPLES_ROOT_DIR):
            if entry in MODEL_EXAMPLES_DIRS:
                continue
            src = os.path.join(EXAMPLES_ROOT_DIR, entry)
            if not os.path.isdir(src):
                continue
            dst = os.path.join(
                MODEL_EXAMPLES_DIRS.get(DEFAULT_MODEL, next(iter(MODEL_EXAMPLES_DIRS.values()))),
                entry,
            )
            if not os.path.exists(dst):
                shutil.move(src, dst)

    def get_examples_base_dir(self, model_name: str, absolute: bool = True) -> str:
        return MODEL_EXAMPLES_DIRS[model_name]

    def load_model(self, model_name: str) -> ModelBundle:
        if model_name in self.models:
            return self.models[model_name]

        print(f"Loading model {model_name}...")
        try:
            model = load_model(modelname=model_name, device=self.device)
        except Exception as e:
            print(f"Error loading model: {e}\nMake sure text encoder server is running!")
            raise e

        if hasattr(model, "text_encoder"):
            model.text_encoder = CachedTextEncoder(model.text_encoder, model_name=model_name)

        skeleton = model.motion_rep.skeleton
        if isinstance(skeleton, SOMASkeleton30):
            skeleton = skeleton.somaskel77.to(model.device)
        bundle = ModelBundle(
            model=model,
            motion_rep=model.motion_rep,
            skeleton=skeleton,
            model_fps=model.motion_rep.fps,
        )
        self.models[model_name] = bundle
        print(f"Model {model_name} loaded successfully")
        self.prewarm_embedding_cache(model_name, bundle.model)
        return bundle

    def prewarm_embedding_cache(self, model_name: str, model: object) -> None:
        encoder = getattr(model, "text_encoder", None)
        if not isinstance(encoder, CachedTextEncoder):
            return

        prompt_set = set()
        prompt_set.add(DEFAULT_PROMPT)

        examples_dir = MODEL_EXAMPLES_DIRS.get(model_name)
        if examples_dir and os.path.isdir(examples_dir):
            for entry in os.listdir(examples_dir):
                example_dir = os.path.join(examples_dir, entry)
                if not os.path.isdir(example_dir):
                    continue
                meta_path = os.path.join(example_dir, "meta.json")
                if not os.path.exists(meta_path):
                    continue
                try:
                    meta = load_json(meta_path)
                except Exception:
                    continue
                for prompt in meta.get("prompts_text", []):
                    if isinstance(prompt, str):
                        prompt_set.add(prompt)

        if prompt_set:
            encoder.prewarm(list(prompt_set))

    def build_constraint_tracks(
        self, client: viser.ClientHandle, skeleton: SkeletonBase
    ) -> dict[str, viser_utils.ConstraintSet]:
        return {
            "Full-Body": FullbodyKeyframeSet(
                name="Full-Body",
                server=client,
                skeleton=skeleton,
            ),
            "End-Effectors": EEJointsKeyframeSet(
                name="End-Effectors",
                server=client,
                skeleton=skeleton,
            ),
            "2D Root": RootKeyframe2DSet(
                name="2D Root",
                server=client,
                skeleton=skeleton,
            ),
        }

    def set_timeline_defaults(self, timeline, model_fps: float) -> None:
        timeline.set_defaults(
            default_text=DEFAULT_PROMPT,
            default_duration=int(DEFAULT_CUR_DURATION * model_fps - 1),
            min_duration=int(MIN_DURATION * model_fps - 1),  # 2 seconds minimum,
            max_duration=int(
                MAX_DURATION * model_fps - 1  # - NB_TRANSITION_FRAMES
            ),  # 10 seconds maximum, minus the transition frames, if needed
            default_num_frames_zoom=int(1.10 * 10 * model_fps),  # a bit more than the max
            max_frames_zoom=1000,
            fps=model_fps,
        )

    def _apply_constraint_overlay_visibility(self, session: ClientSession) -> None:
        """Apply show-all vs show-only-current-frame to constraint overlays."""
        only_frame = session.frame_idx if session.show_only_current_constraint else None
        for constraint in session.constraints.values():
            constraint.set_overlay_visibility(only_frame)

    def set_constraint_tracks_visible(self, session: ClientSession, visible: bool) -> None:
        timeline = session.client.timeline
        timeline_data = session.timeline_data
        if timeline_data.get("constraint_tracks_visible", True) == visible:
            return

        with timeline_data["keyframe_update_lock"]:
            if visible:
                for track_id, track_info in timeline_data["tracks"].items():
                    timeline.add_track(
                        track_info["name"],
                        track_type=track_info.get("track_type", "keyframe"),
                        color=track_info.get("color"),
                        height_scale=track_info.get("height_scale", 1.0),
                        uuid=track_id,
                    )

                for keyframe_id, keyframe_data in timeline_data["keyframes"].items():
                    timeline.add_keyframe(
                        track_id=keyframe_data["track_id"],
                        frame=keyframe_data["frame"],
                        value=keyframe_data.get("value"),
                        opacity=keyframe_data.get("opacity", 1.0),
                        locked=keyframe_data.get("locked", False),
                        uuid=keyframe_id,
                    )

                for interval_id, interval_data in timeline_data["intervals"].items():
                    timeline.add_interval(
                        track_id=interval_data["track_id"],
                        start_frame=interval_data["start_frame_idx"],
                        end_frame=interval_data["end_frame_idx"],
                        value=interval_data.get("value"),
                        opacity=interval_data.get("opacity", 1.0),
                        locked=interval_data.get("locked", False),
                        uuid=interval_id,
                    )
            else:
                for track_id in list(timeline_data["tracks"].keys()):
                    timeline.remove_track(track_id)

        timeline_data["constraint_tracks_visible"] = visible

    def _cleanup_session_for_client(self, client_id: int) -> None:
        """Remove session and scene state for a client (e.g. on session expiry)."""
        if client_id in self.client_sessions:
            del self.client_sessions[client_id]
        self.start_direction_markers.pop(client_id, None)
        self.grid_handles.pop(client_id, None)

    def _setup_demo_for_client(self, client: viser.ClientHandle) -> None:
        """Initialize scene, GUI, and session state for a client (no modals)."""
        self.setup_scene(client)

        model_bundle = self.load_model(self.default_model_name)

        # Initialize each empty constraint track
        constraint_tracks = self.build_constraint_tracks(client, model_bundle.skeleton)

        # Create GUI elements for this client
        (
            gui_elements,
            timeline_tracks,
            example_dict,
            gui_examples_dropdown,
            gui_save_example_path_text,
            gui_model_selector,
        ) = ui.create_gui(
            demo=self,
            client=client,
            model_name=self.default_model_name,
            model_fps=model_bundle.model_fps,
        )
        timeline_data = {
            "tracks": timeline_tracks,
            "tracks_ids": {val["name"]: key for key, val in timeline_tracks.items()},
            "keyframes": {},
            "intervals": {},
            "keyframe_update_lock": threading.Lock(),
            "keyframe_move_timers": {},
            "pending_keyframe_moves": {},  # keyframe_id -> new_frame
            "constraint_tracks_visible": True,
            "dense_path_after_release_timer": None,
        }

        # Initialize session state
        cur_duration = DEFAULT_CUR_DURATION
        max_frame_idx = int(cur_duration * model_bundle.model_fps - 1)

        session = ClientSession(
            client=client,
            gui_elements=gui_elements,
            motions={},
            constraints=constraint_tracks,
            timeline_data=timeline_data,
            frame_idx=0,
            playing=False,
            playback_speed=DEFAULT_PLAYBACK_SPEED,
            cur_duration=cur_duration,
            max_frame_idx=max_frame_idx,
            updating_motions=False,
            edit_mode=False,
            model_name=self.default_model_name,
            model_fps=model_bundle.model_fps,
            skeleton=model_bundle.skeleton,
            motion_rep=model_bundle.motion_rep,
            examples_base_dir=self.get_examples_base_dir(self.default_model_name, absolute=True),
            example_dict=example_dict,
            gui_examples_dropdown=gui_examples_dropdown,
            gui_save_example_path_text=gui_save_example_path_text,
            gui_model_selector=gui_model_selector,
        )

        self.client_sessions[client.client_id] = session

        # Initialize default character for this client
        self.add_character_motion(client, session.skeleton)
        self.preload_t800_assets(client, session)

    def _warm_t800_cache_background(self) -> None:
        from kimodo.retarget import is_t800_available
        from kimodo.viz.t800_rig import warm_t800_mesh_cache, resolve_t800_skin_mode

        if not is_t800_available():
            return
        skin = resolve_t800_skin_mode(DEFAULT_T800_SKIN)
        try:
            warm_t800_mesh_cache(skin)
            print("T800 mesh cache warmed (server startup).")
        except Exception as exc:
            print(f"T800 cache warmup failed: {exc}")

    def _update_t800_preload_ui(self, session: ClientSession, message: str) -> None:
        markdown = session.gui_elements.gui_t800_preload_markdown
        if markdown is not None:
            markdown.content = message

    def preload_t800_assets(self, client: viser.ClientHandle, session: ClientSession, *, force: bool = False) -> None:
        """Upload T800 meshes to viser once per client (path ``/character0/t800``)."""
        from kimodo.retarget import is_t800_available
        from kimodo.viz.t800_rig import (
            BOOTSTRAP_DISPLAY_SCALE,
            T800KimodoRobot,
            resolve_t800_skin_mode,
            warm_t800_mesh_cache,
        )

        if not KIMODO_T800_ENABLED:
            session.t800_preload_done.set()
            return
        if "smplx" not in session.model_name.lower():
            session.t800_preload_done.set()
            return
        if not is_t800_available():
            self._update_t800_preload_ui(session, "**T800:** unavailable (missing GMR deps).")
            session.t800_preload_done.set()
            return
        if session.t800_bootstrap_robot is not None and not force:
            self._update_t800_preload_ui(session, "**T800:** ready.")
            session.t800_preload_done.set()
            return
        if session.t800_preload_in_progress and not force:
            return

        session.t800_preload_in_progress = True
        session.t800_preload_done.clear()
        self._update_t800_preload_ui(session, "**T800:** loading robot mesh…")

        loading_notif = client.add_notification(
            title="Loading T800…",
            body="Uploading robot meshes to the viewer.",
            loading=True,
            with_close_button=False,
        )

        def on_progress(_pct: float, msg: str) -> None:
            loading_notif.body = msg
            self._update_t800_preload_ui(session, f"**T800:** {msg}")

        def _worker() -> None:
            try:
                if force:
                    self._clear_t800_bootstrap(session)

                skin = resolve_t800_skin_mode(
                    session.gui_elements.gui_t800_skin_dropdown.value
                    if session.gui_elements.gui_t800_skin_dropdown is not None
                    else DEFAULT_T800_SKIN
                )
                if self._t800_warmup_thread is not None and self._t800_warmup_thread.is_alive():
                    self._t800_warmup_thread.join(timeout=120.0)
                warm_t800_mesh_cache(skin, on_progress=on_progress)
                robot = T800KimodoRobot(
                    client,
                    "/character0/t800",
                    skin=skin,
                    display_scale=BOOTSTRAP_DISPLAY_SCALE,
                    on_progress=on_progress if skin in ("full", "transparent") else None,
                )
                show_robot = bool(session.gui_elements.gui_t800_robot_checkbox.value)
                robot.set_visibility(show_robot)
                session.t800_bootstrap_robot = robot
                self._update_t800_preload_ui(
                    session,
                    "**T800:** ready — standing pose loaded. Generate runs retarget only.",
                )
                loading_notif.title = "T800 ready"
                loading_notif.body = "Robot mesh loaded into the viewer."
            except Exception as exc:
                self._update_t800_preload_ui(session, f"**T800:** load failed — {exc}")
                loading_notif.title = "T800 preload failed"
                loading_notif.body = str(exc)
                loading_notif.color = "red"
                print(f"T800 preload failed for client {client.client_id}: {exc}")
            finally:
                session.t800_preload_in_progress = False
                session.t800_preload_done.set()
                loading_notif.loading = False
                loading_notif.with_close_button = True
                loading_notif.auto_close_seconds = 4.0

        threading.Thread(
            target=_worker,
            daemon=True,
            name=f"t800-preload-{client.client_id}",
        ).start()

    def _wait_for_t800_preload(self, session: ClientSession, *, timeout_sec: float = 300.0) -> None:
        if session.t800_preload_done.is_set():
            return
        self._update_t800_preload_ui(session, "**T800:** waiting for robot mesh…")
        session.t800_preload_done.wait(timeout=timeout_sec)

    def _consume_preloaded_t800_robot(
        self,
        session: ClientSession,
        character_name: str,
        skin: str,
    ):
        from kimodo.viz.t800_rig import resolve_t800_skin_mode

        robot = session.t800_bootstrap_robot
        if robot is None:
            return None
        if character_name != "character0":
            return None
        if robot.skin != resolve_t800_skin_mode(skin):
            return None
        session.t800_bootstrap_robot = None
        return robot

    def _clear_t800_bootstrap(self, session: ClientSession) -> None:
        bootstrap = session.t800_bootstrap_robot
        if bootstrap is None:
            return
        try:
            bootstrap.clear()
        except Exception:
            pass
        session.t800_bootstrap_robot = None

    def on_client_connect(self, client: viser.ClientHandle) -> None:
        """Initialize GUI and state for each new client."""
        print(f"Client {client.client_id} connected")

        if HF_MODE and self.queue_manager is not None:
            self.queue_manager.on_client_connect(client)
        else:
            self._setup_demo_for_client(client)
            # Quick start after GUI is ready so T800 preload notification is visible.
            with client.gui.add_modal(
                "Welcome — Quick Start",
                size="xl",
                show_close_button=True,
                save_choice="kimodo.demo.quick_start_ack",
            ) as modal:
                client.gui.add_markdown(DEMO_UI_QUICK_START_MODAL_MD)
                client.gui.add_button("Got it (don't remind me again)").on_click(lambda _event: modal.close())

    def setup_scene(self, client: viser.ClientHandle) -> None:
        self.configure_theme(client)
        client.camera.position = np.array(
            [2.7417358737841426, 1.8790455698853281, 7.675741569777456],
            dtype=np.float64,
        )
        client.camera.look_at = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        client.camera.up_direction = np.array(
            [-1.1102230246251568e-16, 1.0, 1.3596310734468913e-32],
            dtype=np.float64,
        )
        client.camera.fov = np.deg2rad(45.0)
        grid_handle = client.scene.add_grid(
            "/grid",
            width=self.floor_len,
            height=self.floor_len,
            wxyz=viser.transforms.SO3.from_x_radians(-np.pi / 2.0).wxyz,
            position=(0.0, 0.0001, 0.0),
            fade_distance=3 * self.floor_len,
            section_color=LIGHT_THEME["grid"],
            infinite_grid=True,
        )
        self.grid_handles[client.client_id] = grid_handle
        # marker for origin
        origin_waypoint = viser_utils.WaypointMesh(
            "/origin_waypoint",
            client,
            position=np.array([0.0, 0.0, 0.0]),
            heading=np.array([0.0, 1.0]),
            color=(0, 0, 255),
        )
        self.start_direction_markers[client.client_id] = origin_waypoint

        if KIMODO_HORIZON_LOGO:
            from kimodo.viz.horizon_branding import (
                add_engineai_horizon_logos,
                sky_height_for_distance,
            )

            logo_path = DEMO_ASSETS_ROOT / "engineai_logo.png"
            horizon_dist = max(KIMODO_HORIZON_LOGO_DISTANCE, self.floor_len * 2.5)
            height_override = os.environ.get("KIMODO_HORIZON_LOGO_HEIGHT", "").strip()
            horizon_height = (
                float(height_override)
                if height_override
                else sky_height_for_distance(
                    horizon_dist,
                    elevation_deg=KIMODO_HORIZON_LOGO_ELEVATION_DEG,
                )
            )
            add_engineai_horizon_logos(
                client,
                logo_path,
                distance=horizon_dist,
                height=horizon_height,
                logo_height_m=KIMODO_HORIZON_LOGO_SIZE,
            )

    def on_client_disconnect(self, client: viser.ClientHandle) -> None:
        """Clean up when client disconnects."""
        print(f"Client {client.client_id} disconnected")
        client_id = client.client_id

        if HF_MODE and self.queue_manager is not None:
            self.queue_manager.on_client_disconnect(client_id)

        self._cleanup_session_for_client(client_id)

    def set_start_direction_visible(self, client_id: int, visible: bool) -> None:
        marker = self.start_direction_markers.get(client_id)
        if marker is None:
            return
        marker.set_visible(visible)

    def client_active(self, client_id: int) -> bool:
        return client_id in self.client_sessions

    def add_character_motion(
        self,
        client: viser.ClientHandle,
        skeleton: SkeletonBase,
        joints_pos: Optional[torch.Tensor] = None,
        joints_rot: Optional[torch.Tensor] = None,
        foot_contacts: Optional[torch.Tensor] = None,
    ) -> None:
        client_id = client.client_id
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]

        new_character = self._new_character_for_motion(session, client, skeleton)
        show_skeleton = session.gui_elements.gui_viz_skeleton_checkbox.value
        show_mesh = session.gui_elements.gui_viz_skinned_mesh_checkbox.value

        # if no motion given, initialize to character default (rest) pose for one frame
        init_joints_pos, init_joints_rot = new_character.get_pose()
        if joints_pos is None:
            joints_pos = init_joints_pos[None].repeat(session.max_frame_idx + 1, 1, 1)
        if joints_rot is None:
            joints_rot = init_joints_rot[None].repeat(session.max_frame_idx + 1, 1, 1, 1)

        new_motion = CharacterMotion(new_character, joints_pos, joints_rot, foot_contacts)
        session.motions[new_character.name] = new_motion

        # put the character at the right frame
        new_motion.set_frame(session.frame_idx)
        new_motion.character.set_skinned_mesh_visibility(show_mesh)
        new_motion.character.set_skeleton_visibility(show_skeleton)

    def _new_character_for_motion(
        self,
        session: ClientSession,
        client: viser.ClientHandle,
        skeleton: SkeletonBase,
        *,
        defer_skin_precompute: bool = False,
    ) -> Character:
        """Create a viser character (mesh + skeleton) without registering motion."""
        if "g1" in session.model_name:
            mesh_mode = "g1_stl"
        elif "smplx" in session.model_name:
            mesh_mode = "smplx_skin"
        elif "soma" in session.model_name:
            if session.gui_elements.gui_use_soma_layer_checkbox.value:
                mesh_mode = "soma_layer_skin"
            else:
                mesh_mode = "soma_skin"
        else:
            raise ValueError("The model name is not recognized for skinning.")

        show_skeleton = session.gui_elements.gui_viz_skeleton_checkbox.value
        show_mesh = session.gui_elements.gui_viz_skinned_mesh_checkbox.value

        ci = len(session.motions)
        character_name = f"character{ci}"
        return Character(
            character_name,
            client,
            skeleton,
            create_skeleton_mesh=True,
            create_skinned_mesh=True,
            visible_skeleton=show_skeleton,
            visible_skinned_mesh=show_mesh,
            skinned_mesh_opacity=session.gui_elements.gui_viz_skinned_mesh_opacity_slider.value,
            show_foot_contacts=session.gui_elements.gui_viz_foot_contacts_checkbox.value,
            dark_mode=session.gui_elements.gui_dark_mode_checkbox.value,
            mesh_mode=mesh_mode,
            gui_use_soma_layer_checkbox=session.gui_elements.gui_use_soma_layer_checkbox,
        )

    def add_character_motions_batch(
        self,
        client: viser.ClientHandle,
        skeleton: SkeletonBase,
        samples: list[tuple],
        *,
        spread_factor: float = 1.0,
    ) -> None:
        """Add several generated samples: show all at once, precompute skinning in background."""
        client_id = client.client_id
        if not self.client_active(client_id) or not samples:
            return
        session = self.client_sessions[client_id]
        num_samples = len(samples)
        center_idx = num_samples // 2
        x_trans = (np.arange(num_samples) - center_idx) * spread_factor
        defer_skin = KIMODO_DEFER_SKIN_PRECOMPUTE and num_samples > 1

        show_skeleton = session.gui_elements.gui_viz_skeleton_checkbox.value
        show_mesh = session.gui_elements.gui_viz_skinned_mesh_checkbox.value

        for i, sample in enumerate(samples):
            joints_pos, joints_rot, foot_contacts = sample[0], sample[1], sample[2] if len(sample) > 2 else None
            cur_joints_pos = joints_pos.clone() if isinstance(joints_pos, torch.Tensor) else joints_pos.copy()
            cur_joints_pos[..., 0] += x_trans[i]

            new_character = self._new_character_for_motion(session, client, skeleton)
            init_joints_pos, init_joints_rot = new_character.get_pose()
            if cur_joints_pos is None:
                cur_joints_pos = init_joints_pos[None].repeat(session.max_frame_idx + 1, 1, 1)
            if joints_rot is None:
                joints_rot = init_joints_rot[None].repeat(session.max_frame_idx + 1, 1, 1, 1)

            new_motion = CharacterMotion(
                new_character,
                cur_joints_pos,
                joints_rot,
                foot_contacts,
                defer_skin_precompute=defer_skin,
            )
            session.motions[new_character.name] = new_motion
            new_motion.set_frame(session.frame_idx)
            new_motion.character.set_skinned_mesh_visibility(show_mesh)
            new_motion.character.set_skeleton_visibility(show_skeleton)

        if defer_skin:
            self._schedule_skin_precompute(session)

    def _schedule_skin_precompute(self, session: ClientSession) -> None:
        """Background full-frame skin cache so multi-sample Generate does not block per character."""
        session.skin_precompute_generation += 1
        generation_id = session.skin_precompute_generation
        client_id = session.client.client_id
        frame_idx = session.frame_idx

        def _worker() -> None:
            if not self.client_active(client_id):
                return
            motions = list(session.motions.values())
            print(f"Precomputing skinning for {len(motions)} character(s) in background…")
            for motion in motions:
                if session.skin_precompute_generation != generation_id:
                    return
                if not self.client_active(client_id):
                    return
                if motion.character.skinned_mesh is None and motion.character.skeleton_mesh is None:
                    continue
                try:
                    motion.precompute_mesh_info()
                except Exception as exc:
                    print(f"Skin precompute failed for {motion.name}: {exc}")
            if session.skin_precompute_generation != generation_id or not self.client_active(client_id):
                return
            for motion in session.motions.values():
                try:
                    motion.set_frame(frame_idx)
                except Exception:
                    pass
            print("Skin precompute finished.")

        threading.Thread(
            target=_worker,
            daemon=True,
            name=f"skin-precompute-{client_id}",
        ).start()

    def clear_motions(self, client_id: int) -> None:
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]
        # Drop previous T800 playback (and standing bootstrap) before rebuilding the human clip.
        self.clear_t800_motions(client_id)
        self._clear_t800_bootstrap(session)
        motions_to_clear = list(session.motions.values())
        session.motions.clear()
        session.skin_precompute_generation += 1
        for motion in motions_to_clear:
            motion.clear()
        session.last_prompt_texts = None
        session.last_prompt_embeddings = None
        session.last_prompt_lengths = None
        release_device_memory(self.device)

    def clear_t800_motions(self, client_id: int) -> None:
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]
        for t800_motion in list(session.t800_motions.values()):
            t800_motion.clear()
        session.t800_motions.clear()
        session.t800_quality.clear()
        session.t800_quality_errors.clear()
        self._update_t800_quality_ui(session)

    def _update_t800_quality_ui(self, session: ClientSession) -> None:
        from kimodo.retarget.t800_quality import format_quality_markdown

        markdown = session.gui_elements.gui_t800_quality_markdown
        if markdown is None:
            return
        markdown.content = format_quality_markdown(
            session.t800_quality,
            errors=session.t800_quality_errors,
        )

    def _audit_t800_character(
        self,
        session: ClientSession,
        character_name: str,
        qpos_frames: list,
        motion_fps: float,
    ) -> None:
        from kimodo.retarget.t800_quality import audit_t800_qpos

        try:
            record = audit_t800_qpos(
                qpos_frames,
                motion_fps,
                character_name=character_name,
                client_id=session.client.client_id,
            )
            session.t800_quality[character_name] = record
            session.t800_quality_errors.pop(character_name, None)
            print(
                f"  [T800 quality] {character_name}: {record.verdict_label} "
                f"({record.score}/100) -> {record.pkl_path}"
            )
        except Exception as exc:
            session.t800_quality_errors[character_name] = str(exc)
            session.t800_quality.pop(character_name, None)
            print(f"  [T800 quality] {character_name} failed: {exc}")

    def run_t800_quality_check(self, session: ClientSession, *, notify: bool = False) -> None:
        if not session.t800_motions:
            session.t800_quality.clear()
            session.t800_quality_errors.clear()
            self._update_t800_quality_ui(session)
            return

        session.t800_quality.clear()
        session.t800_quality_errors.clear()
        for character_name, t800_motion in session.t800_motions.items():
            self._audit_t800_character(
                session,
                character_name,
                t800_motion.qpos_frames,
                t800_motion.motion_fps,
            )
        self._update_t800_quality_ui(session)

        if notify and session.t800_quality:
            worst = min(
                session.t800_quality.values(),
                key=lambda rec: (rec.verdict != "pass", rec.verdict == "fail", -rec.score),
            )
            hints = (worst.summary.get("kimodo") or {}).get("hints") or []
            body = f"{worst.character_name} — {worst.score}/100"
            if hints:
                body = f"{body}. {hints[0]}"
            session.client.add_notification(
                title=f"T800 quality: {worst.verdict_label}",
                body=body,
                auto_close_seconds=4.0,
            )
        elif notify and session.t800_quality_errors:
            first = next(iter(session.t800_quality_errors.values()))
            session.client.add_notification(
                title="T800 quality failed",
                body=first[:120],
                auto_close_seconds=4.0,
            )

    def _should_show_t800(self, session: ClientSession) -> bool:
        if not KIMODO_T800_ENABLED:
            return False
        if "smplx" not in session.model_name.lower():
            return False
        return bool(session.gui_elements.gui_t800_robot_checkbox.value)

    def retarget_t800_motions(
        self,
        client: viser.ClientHandle,
        session: ClientSession,
        *,
        wait_for_previous: bool = False,
    ) -> None:
        """Run T800 retarget + viser upload off the websocket thread."""
        if not self._should_show_t800(session):
            self.clear_t800_motions(client.client_id)
            return

        from kimodo.retarget import is_t800_available, missing_t800_dependencies

        if not is_t800_available():
            missing = ", ".join(missing_t800_dependencies())
            print(f"T800 retargeting skipped: {missing}")
            return

        client_id = client.client_id
        if wait_for_previous:
            deadline = time.time() + 600.0
            while session.t800_retarget_lock.locked() and time.time() < deadline:
                time.sleep(0.05)
        if not session.t800_retarget_lock.acquire(blocking=False):
            print("T800 retarget already in progress, skipping duplicate request.")
            return

        def _worker() -> None:
            try:
                if not self.client_active(client_id):
                    return
                self._retarget_t800_impl(client, session)
            except Exception as exc:
                print(f"T800 retargeting failed: {exc}")
                import traceback

                traceback.print_exc()
            finally:
                session.t800_retarget_lock.release()

        threading.Thread(
            target=_worker,
            daemon=True,
            name=f"t800-retarget-{client_id}",
        ).start()

    def _attach_t800_for_character(
        self,
        client: viser.ClientHandle,
        session: ClientSession,
        character_name: str,
        motion,
        qpos_frames: list,
        motion_fps: float,
        *,
        skin,
        show_robot: bool,
        use_textured_skin: bool,
        textures_preloaded: bool,
        on_progress,
        reveal: bool = True,
    ) -> None:
        """Build/assign the viser T800 robot for one finished sample (main thread)."""
        from kimodo.viz.t800_rig import (
            T800CharacterMotion,
            T800KimodoRobot,
            apply_robot_display_scale,
            compute_display_scale_for_motion,
        )

        display_scale = compute_display_scale_for_motion(motion, qpos_frames, frame_idx=session.frame_idx)

        existing = session.t800_motions.get(character_name)
        if existing is not None and existing.robot.skin == skin:
            existing.replace_qpos_frames(qpos_frames, motion_fps=float(motion_fps))
            apply_robot_display_scale(existing.robot, display_scale)
            existing.set_frame(session.frame_idx)
            existing.robot.set_visibility(show_robot and reveal)
            return

        if existing is not None:
            existing.clear()
            del session.t800_motions[character_name]

        preloaded = self._consume_preloaded_t800_robot(session, character_name, skin)
        if preloaded is not None:
            apply_robot_display_scale(preloaded, display_scale)
            preloaded.set_visibility(show_robot and reveal)
            t800_motion = T800CharacterMotion(preloaded, qpos_frames, motion_fps=float(motion_fps))
            t800_motion.set_frame(session.frame_idx)
            session.t800_motions[character_name] = t800_motion
            return

        robot = T800KimodoRobot(
            client,
            f"/{character_name}/t800",
            skin=skin,
            display_scale=display_scale,
            on_progress=on_progress if use_textured_skin and not textures_preloaded else None,
        )
        robot.set_visibility(show_robot and reveal)
        t800_motion = T800CharacterMotion(robot, qpos_frames, motion_fps=float(motion_fps))
        t800_motion.set_frame(session.frame_idx)
        session.t800_motions[character_name] = t800_motion

    def _reveal_t800_robots(self, session: ClientSession, visible: bool) -> None:
        for t800_motion in session.t800_motions.values():
            t800_motion.robot.set_visibility(visible)

    def _flush_t800_attachments(
        self,
        client: viser.ClientHandle,
        session: ClientSession,
        pending: list[tuple],
        **attach_kwargs,
    ) -> None:
        """Attach all retargeted robots, then show them together."""
        show_robot = attach_kwargs.get("show_robot", True)
        sync_reveal = KIMODO_T800_SYNC_ATTACH and len(pending) > 1
        for character_name, motion, qpos_frames, motion_fps in pending:
            if not self.client_active(client.client_id):
                return
            self._attach_t800_for_character(
                client,
                session,
                character_name,
                motion,
                qpos_frames,
                motion_fps,
                reveal=not sync_reveal or not show_robot,
                **attach_kwargs,
            )
        if sync_reveal and show_robot:
            self._reveal_t800_robots(session, True)
        self._register_sample_commit_on_robots(session)

    def _register_sample_commit_on_robots(self, session: ClientSession) -> None:
        """Let the user pick a sample by clicking a robot (works when the human mesh is hidden)."""
        callback = getattr(session, "sample_commit_callback", None)
        if callback is None or len(session.t800_motions) < 2:
            return
        for character_name, t800_motion in session.t800_motions.items():
            try:
                t800_motion.robot.register_click(callback, highlight_group=character_name)
            except Exception as exc:
                print(f"Failed to attach sample-select click on {character_name}: {exc}")

    def _retarget_t800_sequential(self, client, session, motions, **attach_kwargs) -> None:
        from kimodo.retarget import retarget_character_motion

        pending: list[tuple] = []
        for character_name, motion in motions:
            if not self.client_active(client.client_id):
                return
            print(f"Retargeting {character_name} to EngineAI T800 …")
            qpos_frames, motion_fps = retarget_character_motion(
                motion,
                session.skeleton,
                session.model_fps,
                status=lambda msg: print(f"  [T800] {msg}"),
            )
            pending.append((character_name, motion, qpos_frames, motion_fps))
        self._flush_t800_attachments(client, session, pending, **attach_kwargs)

    def _retarget_t800_parallel(self, client, session, motions, **attach_kwargs) -> None:
        """Retarget all samples across a CPU process pool; build each robot as it finishes."""
        import os
        import shutil
        import tempfile
        from concurrent.futures import as_completed

        from kimodo.retarget.t800 import export_motion_amass_npz, prepare_t800_human_frames
        from kimodo.retarget.t800_parallel import (
            _worker_retarget_prepared,
            get_retarget_pool,
            resolve_worker_count,
        )
        from .config import (
            KIMODO_T800_IK_SAFETY,
            KIMODO_T800_IK_STRIDE,
            KIMODO_T800_RETARGET_WORKERS,
            KIMODO_T800_SMOOTH,
            KIMODO_T800_SMOOTH_WINDOW,
        )

        on_progress = attach_kwargs.get("on_progress")
        tmp_dir = tempfile.mkdtemp(prefix="kimodo_t800_par_")
        try:
            workers = resolve_worker_count(KIMODO_T800_RETARGET_WORKERS, len(motions))
            if on_progress is not None:
                on_progress(0.0, f"Retargeting {len(motions)} samples on {workers} workers…")
            print(f"[T800] parallel retarget: {len(motions)} samples on {workers} workers")

            pool = get_retarget_pool(workers)
            motion_by_name = dict(motions)
            futures = {}
            for character_name, motion in motions:
                if not self.client_active(client.client_id):
                    break
                npz_path = os.path.join(tmp_dir, f"{character_name}.npz")
                export_motion_amass_npz(motion, session.skeleton, session.model_fps, npz_path)
                payload = prepare_t800_human_frames(npz_path, float(session.model_fps))
                task = dict(
                    key=character_name,
                    payload=payload,
                    fps=float(session.model_fps),
                    flatten_feet=False,
                    auto_ground=True,
                    ik_safety_break=KIMODO_T800_IK_SAFETY,
                    ik_stride=KIMODO_T800_IK_STRIDE,
                    smooth=KIMODO_T800_SMOOTH,
                    smooth_window=KIMODO_T800_SMOOTH_WINDOW,
                    output_pkl=os.path.join(tmp_dir, f"{character_name}.t800.pkl"),
                )
                futures[pool.submit(_worker_retarget_prepared, task)] = character_name

            pending: list[tuple] = []
            done = 0
            for future in as_completed(futures):
                if not self.client_active(client.client_id):
                    break
                character_name, qpos_frames, motion_fps = future.result()
                motion = motion_by_name.get(character_name)
                if motion is None or character_name not in session.motions:
                    continue
                pending.append((character_name, motion, qpos_frames, motion_fps))
                done += 1
                if on_progress is not None:
                    on_progress(0.0, f"Retargeted {done}/{len(futures)}…")
            pending.sort(key=lambda item: item[0])
            self._flush_t800_attachments(client, session, pending, **attach_kwargs)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _retarget_t800_impl(self, client: viser.ClientHandle, session: ClientSession) -> None:
        from kimodo.viz.t800_rig import resolve_t800_skin_mode

        self._wait_for_t800_preload(session)
        if not self.client_active(client.client_id):
            return

        for name in list(session.t800_motions.keys()):
            if name not in session.motions:
                session.t800_motions[name].clear()
                del session.t800_motions[name]
                session.t800_quality.pop(name, None)
                session.t800_quality_errors.pop(name, None)

        show_robot = session.gui_elements.gui_t800_robot_checkbox.value
        skin = resolve_t800_skin_mode(session.gui_elements.gui_t800_skin_dropdown.value)
        use_textured_skin = skin in ("full", "transparent")
        textures_preloaded = session.t800_bootstrap_robot is not None
        loading_notif = None
        try:
            loading_notif = client.add_notification(
                title="Loading T800…",
                body="Retargeting motion to EngineAI T800…",
                loading=True,
                with_close_button=False,
            )
        except Exception:
            loading_notif = None

        def on_progress(_pct: float, msg: str) -> None:
            if loading_notif is not None:
                try:
                    loading_notif.body = msg
                except Exception:
                    pass

        try:
            from kimodo.retarget.t800_parallel import resolve_worker_count
            from .config import KIMODO_T800_RETARGET_WORKERS

            motions = list(session.motions.items())
            attach_kwargs = dict(
                skin=skin,
                show_robot=show_robot,
                use_textured_skin=use_textured_skin,
                textures_preloaded=textures_preloaded,
                on_progress=on_progress,
            )
            use_parallel = (
                len(motions) >= 2
                and resolve_worker_count(KIMODO_T800_RETARGET_WORKERS, len(motions)) > 1
            )
            if use_parallel:
                try:
                    self._retarget_t800_parallel(client, session, motions, **attach_kwargs)
                except Exception as exc:
                    print(f"Parallel T800 retarget failed ({exc}); falling back to sequential.")
                    import traceback

                    traceback.print_exc()
                    self._retarget_t800_sequential(client, session, motions, **attach_kwargs)
            else:
                self._retarget_t800_sequential(client, session, motions, **attach_kwargs)
        finally:
            session.t800_quality.clear()
            session.t800_quality_errors.clear()
            if loading_notif is not None:
                try:
                    loading_notif.title = "T800 ready"
                    loading_notif.body = "Robot mesh loaded."
                    loading_notif.loading = False
                    loading_notif.with_close_button = True
                    loading_notif.auto_close_seconds = 2.0
                except Exception:
                    pass
            if self.client_active(client.client_id):
                self._update_t800_quality_ui(session)
                self._sync_human_character_viz_from_gui(session)

    def _sync_human_character_viz_from_gui(self, session: ClientSession) -> None:
        """Re-apply Show Mesh / Show Skeleton after Generate + T800 (retarget must not leave mesh hidden)."""
        if not session.motions:
            return
        gui = session.gui_elements
        show_mesh = gui.gui_viz_skinned_mesh_checkbox.value
        show_skeleton = gui.gui_viz_skeleton_checkbox.value
        for motion in session.motions.values():
            if KIMODO_T800_HIDE_HUMAN_MESH and self._should_show_t800(session):
                show_mesh = False
            motion.character.set_skinned_mesh_visibility(show_mesh)
            motion.character.set_skeleton_visibility(show_skeleton)

    def set_t800_visibility(self, session: ClientSession, visible: bool) -> None:
        bootstrap = session.t800_bootstrap_robot
        if bootstrap is not None:
            bootstrap.set_visibility(visible)
        for t800_motion in session.t800_motions.values():
            t800_motion.robot.set_visibility(visible)

    def set_t800_skin(self, session: ClientSession, skin_label: str) -> None:
        from kimodo.viz.t800_rig import resolve_t800_skin_mode

        skin = resolve_t800_skin_mode(skin_label)
        if not session.t800_motions:
            return

        use_textured_skin = skin in ("full", "transparent")
        loading_notif = None
        if use_textured_skin:
            loading_notif = session.client.add_notification(
                title="Loading T800…",
                body="Preparing textured robot mesh.",
                loading=True,
                with_close_button=False,
            )

        def on_progress(_pct: float, msg: str) -> None:
            if loading_notif is not None:
                loading_notif.body = msg

        try:
            for t800_motion in session.t800_motions.values():
                t800_motion.robot.set_skin(
                    skin,
                    on_progress=on_progress if use_textured_skin else None,
                )
        finally:
            if loading_notif is not None:
                loading_notif.title = "T800 ready"
                loading_notif.body = "Textured robot mesh loaded."
                loading_notif.loading = False
                loading_notif.with_close_button = True
                loading_notif.auto_close_seconds = 2.0

    def compute_model_constraints_lst(
        self,
        session: ClientSession,
        model_bundle: ModelBundle,
        num_frames: int,
    ):
        return generation.compute_model_constraints_lst(session, model_bundle, num_frames, self.device)

    def generate(
        self,
        client: viser.ClientHandle,
        prompts: list[str],
        num_frames: list[int],
        num_samples: int,
        seed: int,
        diffusion_steps: int,
        cfg_weight: Optional[list[float]] = None,
        cfg_type: Optional[str] = None,
        postprocess_parameters: Optional[dict] = None,
        transitions_parameters: Optional[dict] = None,
        real_robot_rotations: bool = False,
    ) -> None:
        session = self.client_sessions[client.client_id]
        model_bundle = self.load_model(session.model_name)
        generation.generate(
            client=client,
            session=session,
            model_bundle=model_bundle,
            prompts=prompts,
            num_frames=num_frames,
            num_samples=num_samples,
            seed=seed,
            diffusion_steps=diffusion_steps,
            cfg_weight=cfg_weight,
            cfg_type=cfg_type,
            postprocess_parameters=postprocess_parameters,
            transitions_parameters=transitions_parameters,
            real_robot_rotations=real_robot_rotations,
            device=self.device,
            clear_motions=self.clear_motions,
            add_character_motions_batch=self.add_character_motions_batch,
        )
        try:
            self.retarget_t800_motions(client, session)
        except Exception as exc:
            print(f"T800 retargeting failed: {exc}")

    def switch_session_model(self, client_id: int, model_name: str) -> ModelBundle:
        """Load ``model_name`` for one client and reset its scene motions/constraints."""
        session = self.client_sessions[client_id]
        resolved = resolve_model_name(model_name)
        if session.model_name == resolved and resolved in self.models:
            return self.models[resolved]

        session.playing = False
        session.edit_mode = False

        bundle = self.load_model(resolved)
        self.clear_motions(client_id)
        with session.timeline_data["keyframe_update_lock"]:
            for constraint in list(session.constraints.values()):
                constraint.clear()
            session.constraints = self.build_constraint_tracks(session.client, bundle.skeleton)
            session.timeline_data["keyframes"] = {}
            session.timeline_data["intervals"] = {}
            session.client.timeline.clear_keyframes()
            session.client.timeline.clear_intervals()

        session.model_name = resolved
        session.model_fps = bundle.model_fps
        session.skeleton = bundle.skeleton
        session.motion_rep = bundle.motion_rep
        session.frame_idx = 0
        if "smplx" not in resolved.lower() or not KIMODO_T800_ENABLED:
            self._clear_t800_bootstrap(session)
            self.clear_t800_motions(client_id)
        self._evict_unused_models()
        return bundle

    def import_motion_from_gem_video(
        self,
        client: viser.ClientHandle,
        video_path: str,
        *,
        static_cam: bool = True,
        detector: str | None = None,
        yolo_period: int | None = None,
    ) -> None:
        """GEM-SMPL (GENMO) sidecar: video → AMASS NPZ → SMPL-X character + optional T800 retarget."""
        import tempfile

        from kimodo.imports.gem import (
            amass_npz_to_kimodo_tensors,
            gem_video_readiness_error,
            import_video_to_amass_npz,
            is_gem_video_ready,
        )

        if not KIMODO_GEM_ENABLED:
            raise RuntimeError("GEM video import is disabled (KIMODO_GEM=0).")
        if not is_gem_video_ready():
            raise RuntimeError(
                gem_video_readiness_error()
                or "GEM (GENMO) is not installed. Run scripts/install_gem.sh or set KIMODO_GEM_ROOT."
            )

        client_id = client.client_id
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]

        self.switch_session_model(client_id, KIMODO_GEM_DEMO_MODEL)
        session = self.client_sessions[client_id]
        if not self.client_active(client_id):
            return

        video = os.path.abspath(os.path.expanduser(video_path.strip()))
        if not os.path.isfile(video):
            raise FileNotFoundError(f"Video not found: {video}")

        with tempfile.TemporaryDirectory(prefix="kimodo_gem_") as tmp_dir:
            amass_path = import_video_to_amass_npz(
                video,
                tmp_dir,
                static_cam=static_cam,
                detector=detector,
                yolo_period=yolo_period,
                fps=float(session.model_fps),
            )
            joints_pos, joints_rot, _local = amass_npz_to_kimodo_tensors(
                amass_path,
                session.skeleton,
                device=self.device,
            )

        if not self.client_active(client_id):
            return

        num_frames = int(joints_pos.shape[0])
        session.cur_duration = num_frames / float(session.model_fps)
        session.max_frame_idx = max(0, num_frames - 1)
        session.frame_idx = 0

        self.clear_motions(client_id)
        self.add_character_motion(client, session.skeleton, joints_pos, joints_rot)
        try:
            self.retarget_t800_motions(client, session)
        except Exception as exc:
            print(f"T800 retargeting after GEM import failed: {exc}")
        self.set_frame(client_id, 0)

    def set_frame(self, client_id: int, frame_idx: int, update_timeline: bool = True):
        if not self.client_active(client_id):
            return

        session = self.client_sessions[client_id]

        session.frame_idx = frame_idx
        if update_timeline:
            session.client.timeline.set_current_frame(frame_idx)
        for motion in list(session.motions.values()):
            motion.set_frame(frame_idx)
        for t800_motion in list(session.t800_motions.values()):
            t800_motion.set_frame(frame_idx)
        self._apply_constraint_overlay_visibility(session)

    def run(self) -> None:
        update_counter = 0
        while True:
            last_update_time = time.time()
            if self.models:
                # the max playback speed is 2x the model fps (from gui_playback_speed_buttons)
                playback_fps = max(bundle.model_fps for bundle in self.models.values()) * 2.0
            else:
                playback_fps = 60.0

            # update each client session independently
            #   copy to a list first to avoid changing size if client disconnects
            for client_id, session in list(self.client_sessions.items()):
                update_interval = int(playback_fps / (session.playback_speed * session.model_fps))
                new_frame_idx = session.frame_idx
                if session.playing and update_counter % update_interval == 0:
                    if session.frame_idx >= session.max_frame_idx:
                        new_frame_idx = 0
                    else:
                        new_frame_idx = session.frame_idx + 1

                    # make sure the client is still active before updating the frame
                    if self.client_active(client_id):
                        self.set_frame(client_id, new_frame_idx)

            time_remaining = max(0, 1.0 / playback_fps - (time.time() - last_update_time))
            time.sleep(time_remaining)
            update_counter += 1
            update_counter %= playback_fps  # wrap around to 0 every second

    def configure_theme(
        self,
        client: viser.ClientHandle,
        dark_mode: bool = False,
        titlebar_dark_mode_checkbox_uuid: str | None = None,
    ):
        # Sync grid color with theme (light vs dark)
        theme = DARK_THEME if dark_mode else LIGHT_THEME
        grid_handle = self.grid_handles.get(client.client_id)
        if grid_handle is not None:
            grid_handle.section_color = theme["grid"]

        #
        # setup theme
        #
        buttons = (
            TitlebarButton(
                text="EngineAI",
                icon=None,
                href="https://en.engineai.com.cn/",
            ),
        )
        assets_dir = DEMO_ASSETS_ROOT
        logo_path = assets_dir / "engineai_logo.png"
        if logo_path.exists():
            logo_b64 = base64.standard_b64encode(logo_path.read_bytes()).decode("ascii")
            image = TitlebarImage(
                image_url_light=f"data:image/png;base64,{logo_b64}",
                image_url_dark=f"data:image/png;base64,{logo_b64}",
                image_alt="EngineAI",
                href="https://en.engineai.com.cn/",
            )
        else:
            image = None
        titlebar_theme = TitlebarConfig(buttons=buttons, image=image, title_text="Studio Pro")
        client.gui.set_panel_label("Studio Pro")
        client.gui.configure_theme(
            titlebar_content=titlebar_theme,
            control_layout="floating",  # "floating",  # ['floating', 'collapsible', 'fixed']
            control_width="large",  # ['small', 'medium', 'large']
            dark_mode=dark_mode,
            show_logo=False,  # hide viser logo on bottom left corner
            show_share_button=False,
            titlebar_dark_mode_checkbox_uuid=titlebar_dark_mode_checkbox_uuid,
            brand_color=(255, 120, 0),
        )
