# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimodo: text-driven and constrained motion generation model."""

from typing import Any

__all__ = [
    "AVAILABLE_MODELS",
    "DEFAULT_MODEL",
    "load_model",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .model.load_model import AVAILABLE_MODELS, DEFAULT_MODEL, load_model

        return {
            "AVAILABLE_MODELS": AVAILABLE_MODELS,
            "DEFAULT_MODEL": DEFAULT_MODEL,
            "load_model": load_model,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
