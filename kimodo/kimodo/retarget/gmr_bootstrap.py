"""Locate web-version GMR assets and add them to ``sys.path``."""

from __future__ import annotations

import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path

_T800_PYTHON_DEPS = (
    "mujoco",
    "mink",
    "qpsolvers",
    "smplx",
)


def _project_roots() -> list[Path]:
    kimodo_root = Path(__file__).resolve().parents[2]
    roots = [
        kimodo_root.parent,
        kimodo_root.parent.parent,
    ]
    seen: set[Path] = set()
    unique: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


@lru_cache(maxsize=1)
def resolve_gmr_root() -> Path | None:
    env = os.environ.get("GMR_ROOT", "").strip()
    if env:
        path = Path(env).expanduser().resolve()
        return path if path.is_dir() else None

    for root in _project_roots():
        candidate = root / "web-version" / "gmr"
        if candidate.is_dir():
            return candidate.resolve()
    return None


@lru_cache(maxsize=1)
def resolve_smplx_body_models_dir() -> Path | None:
    env = os.environ.get("SMPLX_BODY_MODELS", "").strip()
    if env:
        path = Path(env).expanduser().resolve()
        if path.is_dir():
            return path

    gmr_root = resolve_gmr_root()
    if gmr_root is not None:
        default = gmr_root / "assets" / "body_models"
        smplx_dir = default / "smplx"
        if smplx_dir.is_dir():
            return default

    try:
        from kimodo.assets import skeleton_asset_path

        kimodo_smplx = Path(skeleton_asset_path("smplx22", "SMPLX_NEUTRAL.npz")).resolve()
        if kimodo_smplx.is_file():
            body_models = kimodo_smplx.parent.parent / "body_models"
            smplx_dir = body_models / "smplx"
            smplx_dir.mkdir(parents=True, exist_ok=True)
            for name in ("SMPLX_NEUTRAL.npz", "SMPLX_NEUTRAL_2020.npz"):
                link = smplx_dir / name
                if not link.exists():
                    link.symlink_to(kimodo_smplx)
            return body_models
    except Exception:
        pass
    return None


def missing_t800_dependencies() -> list[str]:
    missing: list[str] = []
    for module_name in _T800_PYTHON_DEPS:
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)

    gmr_root = resolve_gmr_root()
    if gmr_root is None:
        missing.append("GMR_ROOT (web-version/gmr)")
    else:
        xml_path = gmr_root / "assets" / "t800" / "mujoco" / "t800_full_gmr.xml"
        if not xml_path.is_file():
            missing.append(str(xml_path))

    body_models = resolve_smplx_body_models_dir()
    if body_models is None:
        missing.append("SMPLX body models (SMPLX_BODY_MODELS or gmr/assets/body_models/smplx)")
    return missing


def is_t800_available() -> bool:
    return not missing_t800_dependencies()


@lru_cache(maxsize=1)
def bootstrap_gmr() -> Path:
    gmr_root = resolve_gmr_root()
    if gmr_root is None:
        raise FileNotFoundError(
            "GMR backend not found. Set GMR_ROOT to web-version/gmr or place "
            "web-version next to the Kimodo project."
        )

    for path in (gmr_root, gmr_root / "third_party"):
        entry = str(path)
        if entry not in sys.path:
            sys.path.insert(0, entry)

    body_models = resolve_smplx_body_models_dir()
    if body_models is not None and not os.environ.get("SMPLX_BODY_MODELS"):
        os.environ["SMPLX_BODY_MODELS"] = str(body_models)

    return gmr_root
