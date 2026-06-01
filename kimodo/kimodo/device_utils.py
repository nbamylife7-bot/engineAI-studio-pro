from __future__ import annotations

import os
from typing import Optional, Union

import torch


def torch_cuda_available(torch_mod=torch) -> bool:
    cuda_mod = getattr(torch_mod, "cuda", None)
    is_available = getattr(cuda_mod, "is_available", None)
    return bool(is_available()) if callable(is_available) else False


def torch_mps_available(torch_mod=torch) -> bool:
    backends = getattr(torch_mod, "backends", None)
    mps_backend = getattr(backends, "mps", None)
    if mps_backend is None:
        return False
    is_built = getattr(mps_backend, "is_built", None)
    is_available = getattr(mps_backend, "is_available", None)
    built_ok = bool(is_built()) if callable(is_built) else True
    avail_ok = bool(is_available()) if callable(is_available) else False
    return built_ok and avail_ok


def resolve_torch_device(device: Optional[str] = None, torch_mod=torch) -> str:
    requested = str(device or "auto").strip().lower()
    if requested == "auto":
        if torch_cuda_available(torch_mod):
            return "cuda"
        if torch_mps_available(torch_mod):
            return "mps"
        return "cpu"
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda" if torch_cuda_available(torch_mod) else "cpu"
    if requested.startswith("cuda:"):
        return requested if torch_cuda_available(torch_mod) else "cpu"
    if requested == "mps":
        return "mps" if torch_mps_available(torch_mod) else "cpu"
    if requested.isdigit():
        return f"cuda:{requested}" if torch_cuda_available(torch_mod) else "cpu"
    if "," in requested:
        return resolve_torch_device(requested.split(",", 1)[0], torch_mod=torch_mod)
    return requested


def resolve_text_encoder_device(
    model_device: Optional[Union[str, torch.device]] = None,
    torch_mod=torch,
) -> str:
    """Pick where the local text encoder runs (TEXT_ENCODER_DEVICE env).

    Defaults to CPU when the Kimodo model uses MPS so large LLM weights stay off
    unified GPU memory. Set TEXT_ENCODER_DEVICE=auto to follow the model device.

    NF4 (``LLM2VEC_QUANTIZE``) is GPU-only: defaults to ``cuda:0`` when CUDA is available.
    """
    from kimodo.model.llm2vec_paths import uses_nf4_export

    override = (os.environ.get("TEXT_ENCODER_DEVICE") or "").strip().lower()
    model_resolved = resolve_torch_device(str(model_device) if model_device is not None else "auto", torch_mod=torch_mod)

    if uses_nf4_export():
        if override and override != "auto":
            if override == "cpu":
                raise RuntimeError(
                    "LLM2VEC_QUANTIZE=nf4 requires a CUDA device (bnb 4-bit cannot run on CPU). "
                    "Set TEXT_ENCODER_DEVICE=cuda:0"
                )
            return resolve_torch_device(override, torch_mod=torch_mod)
        if torch_cuda_available(torch_mod):
            return resolve_torch_device("cuda:0", torch_mod=torch_mod)
        raise RuntimeError("LLM2VEC_QUANTIZE=nf4 requires CUDA.")

    if override and override != "auto":
        if override == "cpu":
            return "cpu"
        return resolve_torch_device(override, torch_mod=torch_mod)

    if model_resolved == "mps":
        return "cpu"
    return model_resolved


def preferred_text_encoder_dtype(device: Optional[str], override: Optional[str] = None, torch_mod=torch) -> str:
    resolved_device = resolve_text_encoder_device(device, torch_mod=torch_mod)
    if override:
        dtype = override.lower()
    elif resolved_device == "mps":
        dtype = "float16"
    else:
        dtype = "bfloat16"
    if resolved_device == "mps" and dtype in ("bfloat16", "bf16"):
        return "float16"
    return dtype


def tensor_from_numpy_on_device(array, device: Union[str, torch.device]) -> torch.Tensor:
    """Convert NumPy motion arrays to tensors on ``device`` (float32 on MPS)."""
    import numpy as np

    tensor = torch.from_numpy(np.asarray(array))
    if str(device).lower().startswith("mps") and tensor.is_floating_point():
        tensor = tensor.float()
    return tensor.to(device)


def resolve_skin_compute_device(skeleton_device: Union[str, torch.device], torch_mod=torch) -> torch.device:
    """Device for mesh skinning (SMPL-X LBS). On Mac MPS, default CPU to spare GPU memory."""
    override = (os.environ.get("KIMODO_SKIN_DEVICE") or "auto").strip().lower()
    if override == "cpu":
        return torch.device("cpu")
    if override in ("mps", "cuda"):
        return torch.device(resolve_torch_device(override, torch_mod=torch_mod))
    if override == "auto" and torch_mps_available(torch_mod):
        return torch.device("cpu")
    return torch.device(skeleton_device)


_FAST_MATMUL_ENABLED = False


def enable_fast_matmul_precision(torch_mod=torch) -> None:
    """Enable TF32 / high matmul precision on NVIDIA GPUs (Ampere+ / Blackwell sm_120).

    RTX 50xx (Blackwell) has fast TF32 and BF16 tensor cores. ``set_float32_matmul_precision('high')``
    lets PyTorch use TF32 for fp32 matmuls and ``allow_tf32`` covers cuBLAS/cuDNN. Idempotent and a
    no-op off CUDA. Disable with ``KIMODO_FAST_MATMUL=0``.
    """
    global _FAST_MATMUL_ENABLED
    if _FAST_MATMUL_ENABLED:
        return
    if os.environ.get("KIMODO_FAST_MATMUL", "1").strip().lower() in ("0", "false", "no"):
        return
    if not torch_cuda_available(torch_mod):
        return
    try:
        torch_mod.set_float32_matmul_precision("high")
        torch_mod.backends.cuda.matmul.allow_tf32 = True
        torch_mod.backends.cudnn.allow_tf32 = True
        torch_mod.backends.cudnn.benchmark = True
    except Exception as exc:  # pragma: no cover - defensive on exotic builds
        print(f"[perf] Could not enable TF32 fast matmul: {exc}")
        return
    _FAST_MATMUL_ENABLED = True
    print("[perf] TF32 fast matmul enabled (float32_matmul_precision=high, allow_tf32).")


def release_device_memory(device: Optional[Union[str, torch.device]] = None) -> None:
    """Return cached accelerator memory after heavy inference or viz precompute."""
    import gc

    gc.collect()
    resolved = str(device or "auto").lower()
    if resolved == "auto":
        if torch_mps_available():
            torch.mps.empty_cache()
        elif torch_cuda_available():
            torch.cuda.empty_cache()
        return
    if resolved.startswith("mps") and torch_mps_available():
        torch.mps.empty_cache()
    elif resolved.startswith("cuda") and torch_cuda_available():
        torch.cuda.empty_cache()
