"""EngineAI T800 robot display in Kimodo's Y-up viser scene."""

from __future__ import annotations

from functools import lru_cache
import threading
from typing import Callable, Literal, Optional

import numpy as np
import viser
import viser.transforms as tf

from kimodo.retarget.gmr_bootstrap import bootstrap_gmr

# GMR SMPL-X retargeting: MuJoCo Z-up, root motion along +Y (AMASS forward).
# Kimodo viser scene: Y-up, +Z forward.
#
# Default AMASS export applies rot_z_180, so MuJoCo root motion uses −X where Kimodo uses +X.
# Combine axis swap (MuJoCo +Y→Kimodo +Z, +Z→+Y) with an X flip so the robot matches the human.
#
# Rigid display (same pattern as ``g1_rig.py``):
#   - bake mesh vertices once: v_kimodo = M @ v_mujoco
#   - world position: p_k = M @ p_m
#   - world orientation: R_k = M @ R_m @ M.T
MUJOCO_TO_KIMODO = np.array(
    [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
MUJOCO_TO_KIMODO_POS = MUJOCO_TO_KIMODO

# Standing qpos: mesh forward is pelvis −X. With ``MUJOCO_TO_KIMODO``, rotate +90° so the robot faces Kimodo +Z like SMPL-X.
BOOTSTRAP_STANDING_YAW_RAD = np.pi / 2.0

# Scale the displayed robot uniformly about the floor so it matches the SMPL-X human beside it.
# Meshes are baked UNSCALED; the scale is applied per robot instance (vertices + root position),
# so different instances can use different scales.
#
# The robot crouches MORE than the human during motion (its standing pose is comparatively
# taller), so a single scale cannot match both the straight standing preview and the crouched
# generated motion:
#   - ROBOT_DISPLAY_SCALE (motion): tuned so robot ≈ human across generated motion (~1.77 m mean).
#     The fully-straight standing pose at this scale would be ~1.91 m, taller than the human.
#   - BOOTSTRAP_DISPLAY_SCALE (first-load preview): tuned so the straight standing pose matches the
#     human reference height (~1.77 m). On the first generation the reused bootstrap robot is
#     switched to ROBOT_DISPLAY_SCALE via ``set_display_scale``.
# Joint angles are scale-invariant, so the pose and exported PKL are unchanged.
T800_NATIVE_HEIGHT_M = 1.744
ROBOT_DISPLAY_SCALE = 1.91 / T800_NATIVE_HEIGHT_M
BOOTSTRAP_DISPLAY_SCALE = 1.77 / T800_NATIVE_HEIGHT_M

SkinMode = Literal["white", "full", "transparent"]
ProgressCallback = Callable[[float, str], None]

# Opaque gray for the default ``white`` skin (viser ``add_mesh_simple``).
T800_GRAY_RGB = (165, 168, 172)

# Bump when ``MUJOCO_TO_KIMODO`` or the mesh bake (now UNSCALED) changes so stale baked meshes are not reused.
_MESH_BAKE_VERSION = 8

# Processed T800 mesh data is reused across generations (viser handles are recreated per character).
_GLOBAL_BAKED_TRIMESH_CACHE: dict[tuple[int, int, int, str, float], object] = {}
_GLOBAL_BAKED_WHITE_MESH_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
_T800_MESH_CACHE_WARMED = False
_WARMED_T800_SKINS: set[SkinMode] = set()
_T800_MESH_CACHE_LOCK = threading.Lock()

T800_SKIN_UI_OPTIONS = ("Textured", "White", "Transparent")
T800_SKIN_UI_TO_MODE: dict[str, SkinMode] = {
    "Textured": "full",
    "White": "white",
    "Transparent": "transparent",
}
T800_SKIN_MODE_TO_UI: dict[SkinMode, str] = {v: k for k, v in T800_SKIN_UI_TO_MODE.items()}


def resolve_t800_skin_mode(label_or_mode: str) -> SkinMode:
    value = (label_or_mode or "white").strip()
    if value in T800_SKIN_UI_TO_MODE:
        return T800_SKIN_UI_TO_MODE[value]
    if value in ("white", "full", "transparent"):
        return value  # type: ignore[return-value]
    return "white"


def _mesh_has_vertices(model, mesh_id: int) -> bool:
    return int(model.mesh_vertnum[int(mesh_id)]) > 0


def _sanitize_vertices(vertices: np.ndarray) -> np.ndarray:
    verts = np.asarray(vertices, dtype=np.float64)
    if verts.size == 0:
        return verts.reshape(0, 3) if verts.ndim != 1 else verts
    if not np.all(np.isfinite(verts)):
        verts = np.nan_to_num(verts, nan=0.0, posinf=0.0, neginf=0.0)
    if float(np.max(np.abs(verts))) > 50.0:
        return np.zeros((0, 3), dtype=np.float64)
    return verts


def _bake_mesh_vertices(vertices: np.ndarray) -> np.ndarray:
    """Convert MuJoCo mesh coordinates into Kimodo-local coordinates (UNSCALED).

    The per-instance display scale is applied when the viser handles are built, not here, so a
    single cached bake can be reused by robots displayed at different heights.
    """
    verts = _sanitize_vertices(vertices)
    if verts.size == 0:
        return verts.reshape(0, 3)
    if verts.ndim == 1:
        return MUJOCO_TO_KIMODO @ verts
    return np.ascontiguousarray(verts) @ MUJOCO_TO_KIMODO.T


def _transform_position(pos_mujoco: np.ndarray) -> np.ndarray:
    return MUJOCO_TO_KIMODO @ np.asarray(pos_mujoco, dtype=np.float64)


def _get_cached_baked_trimesh(model, geom_id: int, mesh_id: int, skin: SkinMode, alpha: float):
    if not _mesh_has_vertices(model, mesh_id):
        raise ValueError(f"T800 mesh {mesh_id} has no vertices.")

    bootstrap_gmr()
    import scripts.t800_viser_robot as tvr

    key = (_MESH_BAKE_VERSION, geom_id, mesh_id, skin, alpha)
    cached = _GLOBAL_BAKED_TRIMESH_CACHE.get(key)
    if cached is not None:
        return cached

    mesh = tvr._build_trimesh_from_mujoco(model, mesh_id, geom_id, alpha=alpha)
    baked = _bake_mesh_vertices(np.asarray(mesh.vertices, dtype=np.float64))
    if baked.shape[0] == 0:
        raise ValueError(f"T800 mesh {mesh_id} has invalid vertices.")
    mesh.vertices = baked
    _GLOBAL_BAKED_TRIMESH_CACHE[key] = mesh
    return mesh


def _get_cached_white_mesh(model, mesh_id: int) -> tuple[np.ndarray, np.ndarray]:
    if not _mesh_has_vertices(model, mesh_id):
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32)

    bootstrap_gmr()
    import scripts.t800_viser_robot as tvr

    cached = _GLOBAL_BAKED_WHITE_MESH_CACHE.get((_MESH_BAKE_VERSION, mesh_id))
    if cached is not None:
        return cached

    verts, faces = tvr._mesh_arrays(model, mesh_id)
    baked_verts = _bake_mesh_vertices(verts)
    if baked_verts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32)
    baked = (
        baked_verts.astype(np.float32),
        np.asarray(faces, dtype=np.int32),
    )
    _GLOBAL_BAKED_WHITE_MESH_CACHE[(_MESH_BAKE_VERSION, mesh_id)] = baked
    return baked


def _transform_wxyz(wxyz_mujoco: np.ndarray) -> np.ndarray:
    """Map MuJoCo geom orientation into Kimodo's Y-up scene."""
    rot = tf.SO3(wxyz=np.asarray(wxyz_mujoco, dtype=np.float64)).as_matrix()
    rot_kimodo = MUJOCO_TO_KIMODO @ rot @ MUJOCO_TO_KIMODO.T
    return tf.SO3.from_matrix(rot_kimodo).wxyz


def _apply_kimodo_yaw(position: np.ndarray, wxyz: np.ndarray, yaw_rad: float) -> tuple[np.ndarray, np.ndarray]:
    if abs(yaw_rad) < 1e-9:
        return position, wxyz
    rot = tf.SO3.from_y_radians(float(yaw_rad))
    return rot @ np.asarray(position, dtype=np.float64), (rot @ tf.SO3(wxyz=wxyz)).wxyz


def warm_t800_mesh_cache(
    skin: SkinMode = "white",
    *,
    on_progress: Optional[ProgressCallback] = None,
) -> None:
    """Pre-build MuJoCo meshes for one skin mode (no textures when ``skin=white``)."""
    global _T800_MESH_CACHE_WARMED
    with _T800_MESH_CACHE_LOCK:
        if skin in _WARMED_T800_SKINS:
            if on_progress is not None:
                on_progress(100.0, "T800 mesh cache ready.")
            return

    bootstrap_gmr()
    import mujoco as mj
    import scripts.t800_viser_robot as tvr

    model = load_t800_mj_model()
    mesh_geoms: list[tuple[int, int]] = []
    for g in range(model.ngeom):
        if model.geom_type[g] != mj.mjtGeom.mjGEOM_MESH:
            continue
        mesh_id = int(model.geom_dataid[g])
        if int(model.mesh_vertnum[mesh_id]) <= 0:
            continue
        mesh_geoms.append((g, mesh_id))

    skins_to_warm: tuple[SkinMode, ...] = (skin,)
    use_textures = skin in ("full", "transparent")
    label = "textures" if use_textures else "gray mesh"

    total_steps = max(1, len(mesh_geoms))
    step = 0
    if on_progress is not None:
        on_progress(0.0, f"Loading T800 {label}…")

    alpha = tvr.TRANSPARENT_ALPHA if skin == "transparent" else 1.0
    for geom_id, mesh_id in mesh_geoms:
        try:
            if use_textures:
                _get_cached_baked_trimesh(model, geom_id, mesh_id, skin, alpha)
            else:
                _get_cached_white_mesh(model, mesh_id)
        except ValueError:
            continue
        step += 1
        if on_progress is not None:
            pct = 100.0 * float(step) / float(total_steps)
            on_progress(pct, f"Cached T800 mesh {step}/{total_steps}")

    with _T800_MESH_CACHE_LOCK:
        _WARMED_T800_SKINS.add(skin)
        _T800_MESH_CACHE_WARMED = True
    if on_progress is not None:
        on_progress(100.0, "T800 assets ready.")


@lru_cache(maxsize=1)
def load_t800_mj_model():
    bootstrap_gmr()
    import mujoco as mj
    from general_motion_retargeting.params import ROBOT_XML_DICT

    return mj.MjModel.from_xml_path(str(ROBOT_XML_DICT["t800"].resolve()))


def _make_robot_scene_class(scene_prefix: str):
    bootstrap_gmr()
    import mujoco as mj
    import scripts.t800_viser_robot as tvr

    prefix = scene_prefix.rstrip("/")

    class _PrefixedRobotScene(tvr.RobotScene):
        def __init__(
            self,
            server,
            model,
            skin: SkinMode = "white",
            *,
            initial_qpos=None,
            display_scale: float = ROBOT_DISPLAY_SCALE,
            on_progress: Optional[ProgressCallback] = None,
        ) -> None:
            self.server = server
            self.model = model
            self.data = mj.MjData(model)
            self.skin = skin
            self.display_scale = float(display_scale)
            self.geom_handles: list[tuple[int, object]] = []
            self._last_qpos: Optional[np.ndarray] = None
            self._mesh_cache: dict[tuple[int, int, int, str, float], object] = {}
            self._build_meshes(on_progress=on_progress)
            mj.mj_forward(self.model, self.data)
            start_qpos = initial_qpos if initial_qpos is not None else tvr.t800_standing_qpos(model)
            self.update_from_qpos(start_qpos)

        def _cached_trimesh(self, geom_id: int, mesh_id: int, alpha: float):
            key = (_MESH_BAKE_VERSION, geom_id, mesh_id, self.skin, alpha)
            if key not in self._mesh_cache:
                self._mesh_cache[key] = _get_cached_baked_trimesh(
                    self.model,
                    geom_id,
                    mesh_id,
                    self.skin,
                    alpha,
                )
            return self._mesh_cache[key]

        def _build_meshes(self, on_progress=None) -> None:
            m = self.model
            alpha = tvr.TRANSPARENT_ALPHA if self.skin == "transparent" else 1.0
            use_textures = self.skin in ("full", "transparent")

            mesh_geoms: list[tuple[int, int]] = []
            for g in range(m.ngeom):
                if m.geom_type[g] != mj.mjtGeom.mjGEOM_MESH:
                    continue
                mesh_id = int(m.geom_dataid[g])
                if not _mesh_has_vertices(m, mesh_id):
                    continue
                mesh_geoms.append((g, mesh_id))

            total = len(mesh_geoms)
            if on_progress is not None and use_textures:
                on_progress(0.0, "Preparing textures…")

            pending: list[tuple[int, object]] = []
            for idx, (g, mesh_id) in enumerate(mesh_geoms):
                name = f"{prefix}/geom_{g}"
                try:
                    if use_textures:
                        tm = self._cached_trimesh(g, mesh_id, alpha).copy()
                        tm.vertices = np.asarray(tm.vertices, dtype=np.float64) * self.display_scale
                        pending.append((g, self.server.scene.add_mesh_trimesh(name, tm, visible=False)))
                    else:
                        verts, faces = _get_cached_white_mesh(m, mesh_id)
                        if verts.shape[0] == 0:
                            continue
                        verts = (verts.astype(np.float64) * self.display_scale).astype(np.float32)
                        pending.append(
                            (
                                g,
                                self.server.scene.add_mesh_simple(
                                    name,
                                    vertices=verts,
                                    faces=faces,
                                    color=T800_GRAY_RGB,
                                    flat_shading=True,
                                    visible=False,
                                ),
                            )
                        )
                except ValueError:
                    continue
                if on_progress is not None and use_textures and total > 0:
                    pct = 100.0 * float(idx + 1) / float(total)
                    on_progress(pct, f"Loading textures… {idx + 1}/{total}")

            self.geom_handles = pending

            if on_progress is not None and use_textures:
                on_progress(100.0, "Textures loaded.")

        def update_from_qpos(self, qpos: np.ndarray) -> None:
            """Run MuJoCo FK only; ``T800KimodoRobot`` owns viser handle transforms."""
            self._last_qpos = np.asarray(qpos, dtype=np.float64).copy()
            n = min(len(qpos), self.model.nq)
            self.data.qpos[:n] = qpos[:n]
            mj.mj_forward(self.model, self.data)

    return _PrefixedRobotScene


class T800KimodoRobot:
    """Wrap GMR's viser T800 renderer with Kimodo scene coordinates and prefixes."""

    def __init__(
        self,
        server: viser.ViserServer | viser.ClientHandle,
        scene_prefix: str,
        *,
        skin: SkinMode = "white",
        display_scale: float = ROBOT_DISPLAY_SCALE,
        on_progress: Optional[ProgressCallback] = None,
    ) -> None:
        bootstrap_gmr()
        import scripts.t800_viser_robot as tvr

        self.server = server
        self.scene_prefix = scene_prefix.rstrip("/")
        self.skin = resolve_t800_skin_mode(skin)
        self._display_scale = float(display_scale)
        self._bootstrap_yaw_rad = BOOTSTRAP_STANDING_YAW_RAD
        self._update_lock = threading.Lock()
        self._last_applied_qpos: Optional[np.ndarray] = None
        self._visible = True
        model = load_t800_mj_model()
        scene_cls = _make_robot_scene_class(self.scene_prefix)
        standing_qpos = tvr.t800_standing_qpos(model)
        self._inner = scene_cls(
            server,
            model,
            skin=self.skin,
            initial_qpos=standing_qpos,
            display_scale=self._display_scale,
            on_progress=on_progress,
        )
        self.update_from_qpos(standing_qpos)

    def set_display_scale(self, display_scale: float) -> None:
        """Rescale the displayed robot (rebuilds mesh handles at the new scale)."""
        if abs(float(display_scale) - self._display_scale) < 1e-9:
            return
        with self._update_lock:
            self._display_scale = float(display_scale)
            self._inner.display_scale = self._display_scale
            last_qpos = self._last_applied_qpos
            for _, handle in self._inner.geom_handles:
                self.server.scene.remove_by_name(handle.name)
            self._inner.geom_handles.clear()
            self._inner._build_meshes()
            self._last_applied_qpos = None
            if last_qpos is not None:
                self._apply_pose_from_qpos(last_qpos)
                self._last_applied_qpos = last_qpos.copy()

    def set_skin(
        self,
        skin: SkinMode,
        *,
        on_progress: Optional[ProgressCallback] = None,
    ) -> None:
        if skin == self.skin:
            return
        self.skin = resolve_t800_skin_mode(skin)
        last_qpos = self._inner._last_qpos
        self._inner.set_skin(skin, on_progress=on_progress)
        if last_qpos is not None:
            self.update_from_qpos(last_qpos)

    def set_visibility(self, visible: bool) -> None:
        self._visible = bool(visible)
        for _, handle in self._inner.geom_handles:
            handle.visible = self._visible

    def clear(self) -> None:
        with self._update_lock:
            for _, handle in self._inner.geom_handles:
                self.server.scene.remove_by_name(handle.name)
            self._inner.geom_handles.clear()
            self._last_applied_qpos = None

    def disable_bootstrap_yaw(self) -> None:
        self._bootstrap_yaw_rad = 0.0

    def _apply_pose_from_qpos(self, q: np.ndarray) -> None:
        """Apply Kimodo scene transforms directly from MuJoCo FK (no intermediate handle poses)."""
        import scripts.t800_viser_robot as tvr

        self._inner.update_from_qpos(q)
        yaw = float(self._bootstrap_yaw_rad)
        data = self._inner.data
        for g, handle in self._inner.geom_handles:
            pos_k = self._display_scale * _transform_position(data.geom_xpos[g])
            wxyz_k = _transform_wxyz(tvr._mat_to_wxyz(data.geom_xmat[g]))
            if yaw != 0.0:
                pos_k, wxyz_k = _apply_kimodo_yaw(pos_k, wxyz_k, yaw)
            handle.position = np.asarray(pos_k, dtype=np.float64)
            handle.wxyz = np.asarray(wxyz_k, dtype=np.float64)
            handle.visible = self._visible

    def update_from_qpos(self, qpos: np.ndarray) -> None:
        q = np.asarray(qpos, dtype=np.float64)
        with self._update_lock:
            if not self._inner.geom_handles:
                return
            if self._last_applied_qpos is not None and self._last_applied_qpos.shape == q.shape:
                if np.allclose(q, self._last_applied_qpos, rtol=0.0, atol=0.0):
                    return
            self._apply_pose_from_qpos(q)
            self._last_applied_qpos = q.copy()


class T800CharacterMotion:
    """Per-character T800 playback synced with SMPL-X timeline."""

    def __init__(
        self,
        robot: T800KimodoRobot,
        qpos_frames: list[np.ndarray],
        *,
        motion_fps: float = 30.0,
    ) -> None:
        self.robot = robot
        self.qpos_frames = qpos_frames
        self.motion_fps = float(motion_fps)
        self.length = len(qpos_frames)
        self._last_frame_idx = -1

    def replace_qpos_frames(self, qpos_frames: list[np.ndarray], *, motion_fps: float) -> None:
        self.qpos_frames = qpos_frames
        self.motion_fps = float(motion_fps)
        self.length = len(qpos_frames)
        self._last_frame_idx = -1
        self.robot._last_applied_qpos = None

    def set_frame(self, frame_idx: int) -> None:
        if not self.qpos_frames:
            return
        idx = min(max(frame_idx, 0), self.length - 1)
        if idx == self._last_frame_idx:
            return
        self._last_frame_idx = idx
        self.robot.disable_bootstrap_yaw()
        self.robot.update_from_qpos(self.qpos_frames[idx])

    def clear(self) -> None:
        self.robot.clear()
