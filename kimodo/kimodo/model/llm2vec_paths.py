# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve LLM2Vec checkpoint paths (matbee/kimodo-llm2vec-nf4 on CUDA)."""

from __future__ import annotations

import os
from pathlib import Path


def uses_nf4_export() -> bool:
    return os.environ.get("LLM2VEC_QUANTIZE", "").strip().lower() in ("nf4", "4bit")


def resolve_llm2vec_paths(
    base_model_name_or_path: str,
    peft_model_name_or_path: str | None,
) -> tuple[str, str | None]:
    """Apply ``LLM2VEC_LOCAL_*`` and default matbee layout for NF4 exports."""
    base = (os.environ.get("LLM2VEC_LOCAL_BASE") or "").strip() or base_model_name_or_path
    peft = (os.environ.get("LLM2VEC_LOCAL_PEFT") or "").strip() or peft_model_name_or_path

    if uses_nf4_export() and not (os.environ.get("LLM2VEC_LOCAL_BASE") or "").strip():
        base = os.environ.get("LLM2VEC_HUB_ID", "matbee/kimodo-llm2vec-nf4").strip()

    if uses_nf4_export() and not (os.environ.get("LLM2VEC_LOCAL_PEFT") or "").strip():
        base_path = Path(base)
        if base_path.is_dir():
            peft = str(base_path / "supervised_adapter")
        else:
            peft = f"{base.rstrip('/')}/supervised_adapter"

    return base, peft
