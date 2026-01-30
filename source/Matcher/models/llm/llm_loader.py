from typing import Tuple

import torch
from ....Matcher.utils.logging_config import setup_logging
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = setup_logging(__name__)


def load_model_and_tokenizer(
    model_config: dict, device
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a model and tokenizer with safe device handling and optional 4-bit."""
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available() and device == "mps"
    quant_config = None
    attn_impl = None

    # Select best dtype based on device
    compute_dtype = torch.float32
    if use_cuda and torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
    elif use_cuda:
        compute_dtype = torch.float16
    elif use_mps:
        compute_dtype = torch.float16  # MPS supports float16

    # Determine device string
    if use_mps:
        device_str = "mps"
        logger.info("Using MPS (Apple Silicon) device.")
    elif use_cuda:
        cuda_count = torch.cuda.device_count()
        idx = int(device) if isinstance(device, int) else 0
        if idx < 0 or idx >= cuda_count:
            logger.warning(
                f"Requested CUDA device {device} invalid; using 0 (num_gpus={cuda_count})."
            )
            idx = 0
        try:
            torch.cuda.set_device(idx)
        except Exception as e:
            logger.warning(
                f"torch.cuda.set_device({idx}) failed: {e}. Falling back to 0."
            )
            idx = 0
            torch.cuda.set_device(idx)
        device_str = f"cuda:{idx}"

        # Prefer FlashAttention-2 if available, else SDPA
        attn_impl = "sdpa"
        try:
            import flash_attn  # noqa

            major, minor = torch.cuda.get_device_capability(idx)
            if (major * 10 + minor) >= 75:
                attn_impl = "flash_attention_2"
                logger.info("Using FlashAttention-2.")
            else:
                logger.info("FlashAttention-2 unsupported on this GPU; using SDPA.")
        except Exception: # noqa
            logger.info("flash-attn not available; using SDPA.")
    else:
        device_str = "cpu"
        logger.warning("No GPU available; loading model on CPU.")

    # Setup quantization only on CUDA with valid config
    quant_cfg = model_config.get("quantization")
    if use_cuda and quant_cfg and quant_cfg.get("load_in_4bit"):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=bool(quant_cfg.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_quant_type=str(quant_cfg.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_compute_dtype=compute_dtype,
        )
        logger.info(f"Loading model on {device_str} with 4-bit quantization.")

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["base_model"],
        use_fast=True,
        padding_side="left",
        trust_remote_code=True,
    )
    # Always left-pad decoder-only models; keep most recent tokens if truncation occurs.
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    if attn_impl == "flash_attention_2":
        logger.info(
            "Using FlashAttention-2; keeping padding_side='left' for decoder-only models."
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_config["base_model"],
        trust_remote_code=True,
        torch_dtype=compute_dtype if (use_cuda or use_mps) else torch.float32,
        device_map=device_str,
        attn_implementation=attn_impl,
        quantization_config=quant_config,
        low_cpu_mem_usage=True,
    )
    # Ensure KV cache usage for faster generation
    try:
        model.config.use_cache = True
    except Exception:
        pass

    # Load PEFT adapter if specified
    adapter_path = model_config.get("cot_adapter_path")
    if adapter_path:
        model = PeftModel.from_pretrained(
            model, adapter_path, device_map=device_str
        )
        logger.info(f"Loaded PEFT adapter from {adapter_path}")
    else:
        logger.info("No PEFT adapter specified; using base model.")

    # Optional: compile for extra speed when supported
    if bool(model_config.get("compile", False)):
        try:
            model = torch.compile(model, mode="max-autotune", fullgraph=False)
            logger.info("Model compiled with torch.compile.")
        except Exception as e:
            logger.warning(f"torch.compile failed; continuing without it. Err: {e}")

    if isinstance(model, torch.nn.Module):
        model.eval()
    else:
        logger.warning("Model is not an instance of torch.nn.Module; skipping eval.")
    logger.info(f"Model loaded on {device_str}.")
    return model, tokenizer  # type: ignore[return-value]
