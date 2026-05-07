from __future__ import annotations

from typing import Optional, Tuple

import torch

fp8_dtype = torch.float8_e4m3fn
fp8_max = torch.finfo(fp8_dtype).max
fp8_min = -fp8_max


def _native_dynamic_per_tensor_quant_fp8(output, input, scale):
    """Native PyTorch fallback for dynamic per-tensor FP8 quantization when vLLM is unavailable."""
    eps = 1e-12
    absmax = input.abs().max()
    absmax = torch.clamp(absmax, min=eps)
    scale_val = absmax / fp8_max
    # Use copy_ instead of fill_ with .item() to avoid CPU-GPU sync
    scale.view(-1).copy_(scale_val.view(-1))
    # Quantize
    output_data = torch.clamp(input / scale_val, fp8_min, fp8_max).to(fp8_dtype)
    output.copy_(output_data)


def _native_dynamic_per_token_quant_fp8(output, input, scale):
    """Native PyTorch fallback for dynamic per-token FP8 quantization when vLLM is unavailable."""
    M, N = input.shape
    eps = 1e-12
    # Compute per-token scale
    absmax = input.abs().max(dim=1, keepdim=True).values
    absmax = torch.clamp(absmax, min=eps)
    scale_val = absmax / fp8_max
    scale.copy_(scale_val)
    # Quantize
    output_data = torch.clamp(input / scale_val, fp8_min, fp8_max).to(fp8_dtype)
    output.copy_(output_data)


def _native_static_quant_fp8(output, input, scale):
    """Native PyTorch fallback for static FP8 quantization when vLLM is unavailable."""
    # Use tensor directly instead of .item() to avoid CPU-GPU sync
    output_data = torch.clamp(input / scale, fp8_min, fp8_max).to(fp8_dtype)
    output.copy_(output_data)


def scaled_fp8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    num_token_padding: Optional[int] = None,
    use_per_token_if_dynamic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert input.ndim == 2, f"Expected 2D input tensor, got {input.ndim}D"
    shape = input.shape
    if num_token_padding:
        shape = (max(num_token_padding, input.shape[0]), shape[1])
    output = torch.empty(shape, device=input.device, dtype=fp8_dtype)

    if scale is None:
        # Dynamic scaling
        if use_per_token_if_dynamic:
            scale = torch.empty((shape[0], 1), device=input.device, dtype=torch.float32)
            _native_dynamic_per_token_quant_fp8(output, input, scale)
        else:
            scale = torch.zeros(1, device=input.device, dtype=torch.float32)
            _native_dynamic_per_tensor_quant_fp8(output, input, scale)
    else:
        # Static scaling
        assert scale.numel() == 1, f"Expected scalar scale, got numel={scale.numel()}"
        _native_static_quant_fp8(output, input, scale)

    return output, scale
