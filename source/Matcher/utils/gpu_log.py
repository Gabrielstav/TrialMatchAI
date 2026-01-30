import torch
from Matcher.utils.logging_config import setup_logging

logger = setup_logging(__name__)


def log_gpu_memory_state(label: str = "") -> None:
    """Log GPU memory state and nvidia-smi output for debugging."""
    if not torch.cuda.is_available():
        return

    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()

    logger.info(
        f"[GPU {label}] free={free / 2**30:.2f}GB, total={total / 2**30:.2f}GB, "
        f"allocated={allocated / 2**30:.2f}GB, reserved={reserved / 2**30:.2f}GB"
    )

    # run nvidia-smi for process-level visibility
    try:
        import subprocess

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.stdout.strip():
            logger.info(f"[GPU processes]\n{result.stdout.strip()}")
    except Exception as e:
        logger.debug(f"nvidia-smi query failed: {e}")


def log_live_cuda_tensors(limit: int = 25) -> None:
    """Log live CUDA tensors to help diagnose memory not being freed.

    Helps distinguish between "live refs" vs "allocator pinned" scenarios.
    If this prints non-trivial tensors after cleanup, references still exist.
    """
    if not torch.cuda.is_available():
        return

    import gc

    items = []
    for o in gc.get_objects():
        try:
            if torch.is_tensor(o) and o.is_cuda:
                n = o.numel() * o.element_size()
                items.append((n, tuple(o.shape), str(o.dtype), type(o).__name__))
        except Exception: # noqa
            pass

    items.sort(reverse=True, key=lambda x: x[0])
    logger.info(f"[CUDA] live cuda tensors={len(items)}")
    for n, shape, dtype, typ in items[:limit]:
        logger.info(
            f"[CUDA] {n / 2**20:7.2f} MiB  {typ:<18s} shape={shape} dtype={dtype}"
        )


def cleanup_gpu_memory(*objs) -> None:
    """
    Break CUDA ties by moving models to CPU.

    NOTE: This function only breaks CUDA ties. The caller MUST set their
    references to None after calling this, then run gc.collect() etc.
    """
    import gc

    log_gpu_memory_state("before cleanup")

    for obj in objs:
        if obj is None:
            continue
        # handle wrapped models like LLMReranker with .model attr
        inner = getattr(obj, "model", None)
        for m in (inner, obj):
            if m is None:
                continue
            try:
                # prefer to_empty to avoid CPU mem spike, if available
                if hasattr(m, "to_empty"):
                    m.to_empty(device="cpu")
                elif hasattr(m, "to"):
                    m.to("cpu")
            except Exception: # noqa
                pass

    # run twice to handle cyclic references
    gc.collect()
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

    log_gpu_memory_state("after cleanup")
    log_live_cuda_tensors()