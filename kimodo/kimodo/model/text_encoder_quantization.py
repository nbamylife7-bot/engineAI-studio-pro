# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Text encoder quantization for CUDA (NF4 matbee export + optional bnb 8/4-bit)."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)

_QUANT_ALIASES = {
    "q8": "8bit",
    "int8": "8bit",
    "bf16": "none",
    "fp16": "none",
    "float16": "none",
    "bfloat16": "none",
}


def resolve_text_encoder_quantization() -> str:
    raw = (os.environ.get("TEXT_ENCODER_QUANTIZATION") or "none").strip().lower()
    return _QUANT_ALIASES.get(raw, raw)


def resolve_effective_quantization() -> str:
    """``LLM2VEC_QUANTIZE=nf4`` (matbee) overrides ``TEXT_ENCODER_QUANTIZATION``."""
    llm2vec_q = os.environ.get("LLM2VEC_QUANTIZE", "").strip().lower()
    if llm2vec_q in ("nf4", "4bit"):
        return "nf4"
    return resolve_text_encoder_quantization()


def uses_nf4(quantization: str) -> bool:
    return quantization == "nf4"


def uses_bitsandbytes_8bit(quantization: str) -> bool:
    return quantization == "8bit" and torch.cuda.is_available()


def quantization_uses_bitsandbytes(mode: str) -> bool:
    return mode in ("4bit", "nf4") or uses_bitsandbytes_8bit(mode)


def quantization_requires_cpu(mode: str) -> bool:
    return False


def build_pretrained_kwargs(*, quantization: str, dtype: str) -> Dict[str, Any]:
    if quantization in ("none", ""):
        return {"torch_dtype": getattr(torch, dtype)}

    if uses_nf4(quantization):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "LLM2VEC_QUANTIZE=nf4 requires CUDA. "
                "hf download matbee/kimodo-llm2vec-nf4 --local-dir models/kimodo-llm2vec-nf4"
            )
        # matbee NF4 is ~5 GB; "auto" offloads to CPU/disk when diffusion already uses VRAM.
        device_map = (os.environ.get("LLM2VEC_DEVICE_MAP") or "cuda:0").strip().lower()
        if device_map in ("auto",):
            device_map_arg: object = "auto"
        elif device_map in ("cuda:0", "cuda", "0", "gpu"):
            device_map_arg = {"": 0}
        else:
            device_map_arg = device_map
        logger.info(
            "Loading NF4 text encoder (~5 GB VRAM, device_map=%s). "
            "Use ./run_demo_api.sh if VRAM is tight.",
            device_map_arg,
        )
        return {"torch_dtype": getattr(torch, dtype), "device_map": device_map_arg}

    if uses_bitsandbytes_8bit(quantization):
        from transformers import BitsAndBytesConfig

        return {
            "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
            "device_map": "auto",
        }

    if quantization == "4bit":
        from transformers import BitsAndBytesConfig

        compute_dtype = getattr(torch, dtype)
        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
            "device_map": "auto",
        }

    raise ValueError(f"Unknown TEXT_ENCODER_QUANTIZATION='{quantization}'.")


def apply_runtime_quantization(model: torch.nn.Module, quantization: str) -> torch.nn.Module:
    return model


def resolve_text_encoder_load_device(requested_device: str, quantization: str) -> str:
    if quantization_uses_bitsandbytes(quantization):
        return requested_device if torch.cuda.is_available() else "cpu"
    return requested_device
