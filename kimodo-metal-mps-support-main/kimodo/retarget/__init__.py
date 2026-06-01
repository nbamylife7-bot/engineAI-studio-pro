"""Optional humanoid retargeting backends (EngineAI T800 via GMR)."""

from .gmr_bootstrap import bootstrap_gmr, is_t800_available, missing_t800_dependencies
from .t800 import retarget_character_motion
from .t800_quality import T800QualityRecord, audit_t800_qpos, format_quality_markdown

__all__ = [
    "bootstrap_gmr",
    "is_t800_available",
    "missing_t800_dependencies",
    "retarget_character_motion",
    "T800QualityRecord",
    "audit_t800_qpos",
    "format_quality_markdown",
]
