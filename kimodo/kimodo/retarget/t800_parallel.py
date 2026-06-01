"""Parallel T800 retarget across generation samples (CPU process pool).

The GMR IK loop is single-threaded CPU and ~3-4 s per 180-frame sample. Threads do NOT help
(mink/daqp hold the GIL and oversubscribe), so independent samples are spread over a ``spawn``
process pool. Only the CPU-only NPZ→qpos step runs in workers; the torch/GPU SMPL-X→AMASS export
and all viser handle creation stay in the main process.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

_POOL: Optional[ProcessPoolExecutor] = None
_POOL_WORKERS = 0


def _worker_init(counter=None, n_cores: int = 0) -> None:
    """Pin each worker to a distinct CPU core and warm GMR + SMPL-X.

    mink/daqp/BLAS spawn internal threads; left unpinned, N workers oversubscribe the cores and
    parallel retarget is no faster (or slower) than sequential. Pinning each worker to one core via
    ``sched_setaffinity`` confines its threads, giving a near-linear speedup. Pre-loading the GMR
    backend and the ~150 MB SMPL-X body model here means the first real task pays no cold-start.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    # Pin to a distinct core (Linux only); ignore where unsupported.
    if counter is not None and n_cores > 0 and hasattr(os, "sched_setaffinity"):
        try:
            with counter.get_lock():
                core = counter.value % n_cores
                counter.value += 1
            os.sched_setaffinity(0, {core})
        except Exception:
            pass

    try:
        import threadpoolctl

        threadpoolctl.threadpool_limits(1)
    except Exception:
        pass

    try:
        from kimodo.retarget.gmr_bootstrap import bootstrap_gmr

        bootstrap_gmr()
        from general_motion_retargeting.utils.smpl import (
            _get_cached_smplx_body_model,
            resolve_smplx_model_file,
        )
        from scripts.smplx_npz_to_robot import resolve_smplx_body_models_path

        try:
            body_models = resolve_smplx_body_models_path()
            model_file = resolve_smplx_model_file(str(body_models), "neutral")
            _get_cached_smplx_body_model(model_file, "neutral")
        except Exception:
            pass
    except Exception:
        pass


def _worker_retarget(args: dict):
    """Run in a worker process: AMASS NPZ -> smoothed T800 qpos frames."""
    from kimodo.retarget.t800 import retarget_npz_to_qpos

    qpos_frames, motion_fps = retarget_npz_to_qpos(
        args["npz_path"],
        args["fps"],
        flatten_feet=args["flatten_feet"],
        auto_ground=args["auto_ground"],
        ik_safety_break=args["ik_safety_break"],
        ik_stride=args["ik_stride"],
        smooth=args["smooth"],
        smooth_window=args["smooth_window"],
        output_pkl=args.get("output_pkl"),
    )
    # numpy arrays pickle cleanly back to the parent.
    return args["key"], qpos_frames, motion_fps


def resolve_worker_count(requested: int, num_tasks: int) -> int:
    """Worker count. 0/1 => sequential (default). Parallel is opt-in via an explicit value >= 2.

    Measured speedup is only ~1.3x (the retarget mixes multi-core torch SMPL-X FK with single-thread
    IK, so per-worker core pinning starves the FK), so processes are not enabled by default.
    """
    if int(requested) <= 1:
        return 1
    cpu = os.cpu_count() or 2
    return max(1, min(int(requested), num_tasks, cpu))


def get_retarget_pool(num_workers: int) -> ProcessPoolExecutor:
    """Lazily create (and reuse) a spawn-based retarget pool sized to ``num_workers``."""
    global _POOL, _POOL_WORKERS
    if _POOL is not None and _POOL_WORKERS >= num_workers:
        return _POOL
    if _POOL is not None:
        _POOL.shutdown(wait=False, cancel_futures=False)
        _POOL = None
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    counter = ctx.Value("i", 0)
    n_cores = os.cpu_count() or num_workers
    _POOL = ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(counter, n_cores),
    )
    _POOL_WORKERS = num_workers
    return _POOL


def shutdown_retarget_pool() -> None:
    global _POOL, _POOL_WORKERS
    if _POOL is not None:
        _POOL.shutdown(wait=False, cancel_futures=True)
        _POOL = None
        _POOL_WORKERS = 0
