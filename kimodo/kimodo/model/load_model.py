# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load Kimodo diffusion models from local checkpoints or Hugging Face."""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf

from ..device_utils import preferred_text_encoder_dtype, resolve_torch_device, resolve_text_encoder_device
from .llm2vec_paths import uses_nf4_export
from .loading import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    DEFAULT_TEXT_ENCODER_URL,
    MODEL_NAMES,
    TMR_MODELS,
    get_env_var,
    instantiate_from_dict,
)
from .registry import get_model_info, resolve_model_name

DEFAULT_TEXT_ENCODER = "llm2vec"
TEXT_ENCODER_PRESETS = {
    "f2llm_v2_8b": {
        "target": "kimodo.model.AutoEmbeddingEncoder",
        "kwargs": {
            "model_name_or_path": "codefuse-ai/F2LLM-v2-8B",
            "dtype": "bfloat16",
            "llm_dim": 4096,
            "query_prefix": "",
        },
    },
    "llm2vec": {
        "target": "kimodo.model.LLM2VecEncoder",
        "kwargs": {
            "base_model_name_or_path": "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            "peft_model_name_or_path": "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
            "dtype": "bfloat16",
            "llm_dim": 4096,
        },
    },
    "llm2vec_nf4": {
        "target": "kimodo.model.LLM2VecEncoder",
        "kwargs": {
            "base_model_name_or_path": "matbee/kimodo-llm2vec-nf4",
            "peft_model_name_or_path": "matbee/kimodo-llm2vec-nf4/supervised_adapter",
            "dtype": "bfloat16",
            "llm_dim": 4096,
        },
    },
}


def _default_text_encoder_preset() -> str:
    if uses_nf4_export():
        return "llm2vec_nf4"
    return DEFAULT_TEXT_ENCODER


def _resolve_hf_model_path(modelname: str) -> Path:
    """Resolve model name to a local path, using Hugging Face cache or CHECKPOINT_DIR."""
    try:
        repo_id = MODEL_NAMES[modelname]
    except KeyError:
        raise ValueError(f"Model '{modelname}' not found. Available models: {MODEL_NAMES.keys()}")

    local_cache = get_env_var("LOCAL_CACHE", "False").lower() == "true"
    if not local_cache:
        snapshot_dir = snapshot_download(repo_id=repo_id)  # will check online no matter what
        return Path(snapshot_dir)

    try:
        snapshot_dir = snapshot_download(repo_id=repo_id, local_files_only=True)  # will check local cache only
        return Path(snapshot_dir)
    except Exception:
        # if local cache is not found, download from online
        try:
            snapshot_dir = snapshot_download(repo_id=repo_id)
            return Path(snapshot_dir)
        except Exception:
            raise RuntimeError(f"Could not resolve model '{modelname}' from Hugging Face (repo: {repo_id}). ") from None


def _build_api_text_encoder_conf(text_encoder_url: str) -> dict:
    return {
        "_target_": "kimodo.model.text_encoder_api.TextEncoderAPI",
        "url": text_encoder_url,
    }


def _build_local_text_encoder_conf(device: Optional[str]) -> dict:
    text_encoder_name = get_env_var("TEXT_ENCODER", _default_text_encoder_preset())
    if text_encoder_name not in TEXT_ENCODER_PRESETS:
        available = ", ".join(sorted(TEXT_ENCODER_PRESETS))
        raise ValueError(f"Unknown TEXT_ENCODER='{text_encoder_name}'. Available: {available}")

    preset = TEXT_ENCODER_PRESETS[text_encoder_name]
    kwargs = dict(preset["kwargs"])
    text_encoder_device = resolve_text_encoder_device(device)
    kwargs["dtype"] = preferred_text_encoder_dtype(
        text_encoder_device,
        override=get_env_var("TEXT_ENCODER_DTYPE"),
    )
    return {
        "_target_": preset["target"],
        **kwargs,
    }


def _select_text_encoder_conf(text_encoder_url: str, device: Optional[str]) -> dict:
    # TEXT_ENCODER_MODE options:
    # - "api": force TextEncoderAPI
    # - "local": force local bundled encoder preset
    # - "auto": try API first, fallback to local if unreachable
    mode = get_env_var("TEXT_ENCODER_MODE", "auto").lower()
    if mode == "local":
        return _build_local_text_encoder_conf(device)
    if mode == "api":
        return _build_api_text_encoder_conf(text_encoder_url)

    # NF4 is loaded locally on GPU; do not prefer a remote API that may be bf16.
    if uses_nf4_export():
        return _build_local_text_encoder_conf(device)

    api_conf = _build_api_text_encoder_conf(text_encoder_url)
    try:
        text_encoder = instantiate_from_dict(api_conf)
        # Probe availability early so inference doesn't fail later.
        text_encoder(["healthcheck"])
        return api_conf
    except Exception as error:
        print(
            "Text encoder service is unreachable, falling back to bundled local "
            f"encoder. ({type(error).__name__}: {error})"
        )
        return _build_local_text_encoder_conf(device)


def _load_kimodo_nf4_text_encoder_first(model_cfg: Dict[str, Any], resolved_device: str):
    """Instantiate NF4 LLM2Vec before diffusion weights to avoid CPU/disk offload."""
    from kimodo.model.kimodo_model import Kimodo

    cfg = dict(model_cfg)
    cfg.pop("_target_", None)
    te_cfg = cfg.pop("text_encoder")
    denoiser_cfg = cfg.pop("denoiser")
    num_base_steps = cfg.pop("num_base_steps")
    cfg_type = cfg.pop("cfg_type", "separated")
    if cfg:
        print(f"Note: unused Kimodo config keys: {sorted(cfg.keys())}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Loading NF4 text encoder on GPU before diffusion checkpoint…")
    text_encoder = instantiate_from_dict(te_cfg)
    denoiser = instantiate_from_dict(denoiser_cfg)
    return Kimodo(
        denoiser=denoiser,
        text_encoder=text_encoder,
        num_base_steps=num_base_steps,
        device=resolved_device,
        cfg_type=cfg_type,
    )


def load_model(
    modelname=None,
    device=None,
    eval_mode: bool = True,
    default_family: Optional[str] = "Kimodo",
    return_resolved_name: bool = False,
):
    """Load a kimodo model by name (e.g. 'g1', 'soma').

    Resolution of partial/full names (e.g. Kimodo-SOMA-RP-v1, SOMA) is done
    inside this function using default_family when the name is not a known
    short key.

    Args:
        modelname: Model identifier; uses DEFAULT_MODEL if None. Can be a short key,
            a full name (e.g. Kimodo-SOMA-RP-v1), or a partial name; unknown names
            are resolved via resolve_model_name using default_family.
        device: Target device for the model (e.g. 'cuda', 'cpu').
        eval_mode: If True, set model to eval mode.
        default_family: Used when modelname is not in AVAILABLE_MODELS to resolve
            partial names ("Kimodo" for demo/generation, "TMR" for embed script).
            Default "Kimodo".
        return_resolved_name: If True, return (model, resolved_short_key). If False,
            return only the model.

    Returns:
        Loaded model in eval mode, or (model, resolved short key) if
        return_resolved_name is True.

    Raises:
        ValueError: If modelname is not in AVAILABLE_MODELS and cannot be resolved.
        FileNotFoundError: If config.yaml is missing in the checkpoint folder.
    """
    if modelname is None:
        modelname = DEFAULT_MODEL
    if modelname not in AVAILABLE_MODELS:
        if default_family is not None:
            modelname = resolve_model_name(modelname, default_family)
        else:
            raise ValueError(
                f"""The model is not recognized.
            Please choose between: {AVAILABLE_MODELS}"""
            )

    resolved_modelname = modelname

    # In case, we specify a custom checkpoint directory
    configured_checkpoint_dir = get_env_var("CHECKPOINT_DIR")
    if configured_checkpoint_dir:
        print(f"CHECKPOINT_DIR is set to {configured_checkpoint_dir}, checking the local cache...")
        # Checkpoint folders are named by display name (e.g. Kimodo-SOMA-RP-v1)
        info = get_model_info(modelname)
        checkpoint_folder_name = info.display_name if info is not None else modelname
        model_path = Path(configured_checkpoint_dir) / checkpoint_folder_name
        if not model_path.exists() and modelname != checkpoint_folder_name:
            # Fallback: try short_key for backward compatibility
            model_path = Path(configured_checkpoint_dir) / modelname
        if not model_path.exists():
            print(f"Model folder not found at '{model_path}', downloading it from Hugging Face...")
            model_path = _resolve_hf_model_path(modelname)
    else:
        # Otherwise, we load the model from the local cache or download it from Hugging Face.
        model_path = _resolve_hf_model_path(modelname)

    model_config_path = model_path / "config.yaml"
    if not model_config_path.exists():
        raise FileNotFoundError(f"The model checkpoint folder exists but config.yaml is missing: {model_config_path}")

    model_conf = OmegaConf.load(model_config_path)

    if modelname in TMR_MODELS:
        # Same process at the moment for TMR and Kimodo
        pass

    resolved_device = resolve_torch_device(device)
    text_encoder_url = get_env_var("TEXT_ENCODER_URL", DEFAULT_TEXT_ENCODER_URL)
    runtime_conf = OmegaConf.create(
        {
            "checkpoint_dir": str(model_path),
            "text_encoder": _select_text_encoder_conf(text_encoder_url, resolved_device),
        }
    )
    model_cfg = OmegaConf.to_container(OmegaConf.merge(model_conf, runtime_conf), resolve=True)
    model_cfg.pop("checkpoint_dir", None)

    if uses_nf4_export():
        model = _load_kimodo_nf4_text_encoder_first(model_cfg, resolved_device)
    else:
        model = instantiate_from_dict(model_cfg, overrides={"device": resolved_device})
    if eval_mode:
        model = model.eval()
    if return_resolved_name:
        return model, resolved_modelname
    return model
