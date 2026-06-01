# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM2Vec encoder wrapper (CUDA / matbee NF4)."""

import os

import numpy as np
import torch

from ..llm2vec_paths import resolve_llm2vec_paths
from ..text_encoder_quantization import (
    build_pretrained_kwargs,
    quantization_requires_cpu,
    quantization_uses_bitsandbytes,
    resolve_effective_quantization,
)
from .llm2vec import LLM2Vec


def _patch_transformers_bnb_warmup() -> None:
    try:
        import transformers.modeling_utils as modeling_utils

        if hasattr(modeling_utils, "caching_allocator_warmup"):
            modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
    except Exception:
        pass


_patch_transformers_bnb_warmup()


class LLM2VecEncoder:
    """LLM2Vec text embeddings."""

    def __init__(
        self,
        base_model_name_or_path: str,
        peft_model_name_or_path: str,
        dtype: str,
        llm_dim: int,
    ) -> None:
        self.llm_dim = llm_dim
        self.quantization = resolve_effective_quantization()

        cache_dir = os.environ.get("HUGGINGFACE_CACHE_DIR")
        base_model_name_or_path, peft_model_name_or_path = resolve_llm2vec_paths(
            base_model_name_or_path, peft_model_name_or_path
        )

        if "TEXT_ENCODERS_DIR" in os.environ:
            base_model_name_or_path = os.path.join(
                os.environ["TEXT_ENCODERS_DIR"], base_model_name_or_path
            )
            if peft_model_name_or_path:
                peft_model_name_or_path = os.path.join(
                    os.environ["TEXT_ENCODERS_DIR"], peft_model_name_or_path
                )

        load_kwargs = build_pretrained_kwargs(quantization=self.quantization, dtype=dtype)
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir

        if quantization_uses_bitsandbytes(self.quantization) and torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model = LLM2Vec.from_pretrained(
            base_model_name_or_path=base_model_name_or_path,
            peft_model_name_or_path=peft_model_name_or_path,
            merge_peft=False,
            **load_kwargs,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def to(self, device: torch.device):
        if quantization_uses_bitsandbytes(self.quantization):
            return self
        if quantization_requires_cpu(self.quantization):
            device = torch.device("cpu")
        self.model = self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self

    def get_device(self):
        return self.model.model.device

    def __call__(self, text: list[str] | str):
        is_string = False
        if isinstance(text, str):
            text = [text]
            is_string = True

        device = self.get_device()
        with torch.no_grad():
            encoded_text = self.model.encode(
                text,
                batch_size=len(text),
                show_progress_bar=False,
                device=str(device),
            )

        assert len(encoded_text.shape)
        assert self.llm_dim == encoded_text.shape[-1]

        encoded_text = encoded_text[:, None]
        lengths = np.ones(len(encoded_text), dtype=int).tolist()

        if is_string:
            encoded_text = encoded_text[0]
            lengths = lengths[0]

        encoded_text = torch.as_tensor(encoded_text, device=device)
        return encoded_text, lengths
