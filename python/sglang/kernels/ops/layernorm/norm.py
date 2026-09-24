from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module


logger = logging.getLogger(__name__)


@cache_once
def _jit_qknorm_module(head_dim: int, dtype: torch.dtype) -> Module:
    args = make_cpp_args(head_dim, is_arch_support_pdl(), dtype)
    return load_jit(
        "qknorm",
        *args,
        cuda_files=["elementwise/qknorm.cuh"],
        cuda_wrappers=[("qknorm", f"QKNormKernel<{args}>::run")],
    )


_RMSNORM_WARP_SIZES = frozenset({64, 128, 256})
_RMSNORM_MAX_HIDDEN_SIZE = 16384
_RMSNORM_HALF_BLOCK_MIN_SIZE = 2048


def _is_supported_rmsnorm_hidden_size(d: int) -> bool:
    return d in _RMSNORM_WARP_SIZES or (
        (d > 256 and d % 256 == 0 and d <= 8192)
        or (d >= 8192 and d % 512 == 0 and d <= 16384)
    )


def _rmsnorm_kernel_class(hidden_size: int) -> str:
    if hidden_size in _RMSNORM_WARP_SIZES:
        return "RMSNormWarpKernel"
    if hidden_size == 512:
        return "RMSNormHalfKernel"
    if hidden_size >= _RMSNORM_HALF_BLOCK_MIN_SIZE:
        if hidden_size % 512 == 0:
            return "RMSNormHalfKernel"
    return "RMSNormKernel"


@cache_once
def _jit_rmsnorm_module(hidden_size: int, dtype: torch.dtype) -> Module:
    args = make_cpp_args(hidden_size, is_arch_support_pdl(), dtype)
    kernel_class = f"{_rmsnorm_kernel_class(hidden_size)}<{args}>"
    return load_jit(
        "rmsnorm",
        *args,
        cuda_files=["elementwise/rmsnorm.cuh"],
        cuda_wrappers=[("rmsnorm", f"{kernel_class}::run")],
    )


def is_supported_jit_fused_add_rmsnorm_hidden_size(hidden_size: int) -> bool:
    return hidden_size > 0 and hidden_size % 16 == 0 and hidden_size <= 8192


@cache_once
def _jit_fused_add_rmsnorm_module(
    dtype: torch.dtype, cast_x_before_out_mul: bool
) -> Module:
    args = make_cpp_args(cast_x_before_out_mul, dtype)
    return load_jit(
        "fused_add_rmsnorm",
        *args,
        cuda_files=["elementwise/fused_add_rmsnorm.cuh"],
        cuda_wrappers=[("fused_add_rmsnorm", f"FusedAddRMSNormKernel<{args}>::run")],
    )


@cache_once
def _jit_qknorm_across_heads_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "qknorm_across_heads",
        *args,
        cuda_files=["elementwise/qknorm_across_heads.cuh"],
        cuda_wrappers=[
            ("qknorm_across_heads", f"QKNormAcrossHeadsKernel<{args}>::run")
        ],
    )


@torch.compiler.assume_constant_result
@cache_once
def can_use_fused_inplace_qknorm(head_dim: int, dtype: torch.dtype) -> bool:
    if head_dim not in [64, 128, 256, 512, 1024]:
        logger.warning(f"Unsupported head_dim={head_dim} for JIT QK-Norm kernel")
        return False
    try:
        _jit_qknorm_module(head_dim, dtype)
        return True
    except Exception as e:
        logger.warning(f"Failed to load JIT QK-Norm kernel: {e}")
        return False


@debug_kernel_api
def fused_inplace_qknorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    head_dim: int = 0,
) -> None:
    head_dim = head_dim or q.size(-1)
    module = _jit_qknorm_module(head_dim, q.dtype)
    module.qknorm(q, k, q_weight, k_weight, eps)


@debug_kernel_api
def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> None:
    out = out if out is not None else input
    hidden_size = input.size(-1)
    if not _is_supported_rmsnorm_hidden_size(hidden_size):
        raise RuntimeError(
            f"jit rmsnorm: unsupported hidden_size={hidden_size}. "
            f"Supported: {sorted(_RMSNORM_WARP_SIZES)}, and multiples of 256 in "
            f"(256, {_RMSNORM_MAX_HIDDEN_SIZE}]."
        )
    module = _jit_rmsnorm_module(hidden_size, input.dtype)
    module.rmsnorm(input, weight, out, eps)


@debug_kernel_api
def fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    cast_x_before_out_mul: bool = False,
) -> None:
    module = _jit_fused_add_rmsnorm_module(input.dtype, cast_x_before_out_mul)
    module.fused_add_rmsnorm(input, residual, weight, eps)


@debug_kernel_api
def fused_inplace_qknorm_across_heads(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float = 1e-6,
) -> None:
    """
    Fused inplace QK normalization across all heads.

    Args:
        q: Query tensor of shape [batch_size, num_heads * head_dim]
        k: Key tensor of shape [batch_size, num_heads * head_dim]
        q_weight: Query weight tensor of shape [num_heads * head_dim]
        k_weight: Key weight tensor of shape [num_heads * head_dim]
        eps: Epsilon for numerical stability
    """
    module = _jit_qknorm_across_heads_module(q.dtype)
    module.qknorm_across_heads(q, k, q_weight, k_weight, eps)


@triton.jit
def _gemma_norm_add_norm_kernel(
    x_ptr, res_ptr, w1_ptr, w2_ptr, n_cols, stride_x, stride_r, eps1, eps2, BLOCK_N: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w1 = tl.load(w1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd1 = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / n_cols + eps1)
    # Round h to the activation dtype, as the separate gemma_rmsnorm output would be.
    h = (x * rstd1 * (1.0 + w1)).to(x_ptr.dtype.element_ty).to(tl.float32)
    r = tl.load(res_ptr + row * stride_r + cols, mask=mask, other=0.0).to(tl.float32)
    s = h + r
    tl.store(res_ptr + row * stride_r + cols, s.to(res_ptr.dtype.element_ty), mask=mask)
    w2 = tl.load(w2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd2 = 1.0 / tl.sqrt(tl.sum(s * s, axis=0) / n_cols + eps2)
    tl.store(x_ptr + row * stride_x + cols, (s * rstd2 * (1.0 + w2)).to(x_ptr.dtype.element_ty), mask=mask)


def gemma_norm_add_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    eps1: float,
    eps2: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gemma RMSNorm(x, w1) -> add into residual -> Gemma RMSNorm(residual, w2), one kernel.

    Same result as gemma_rmsnorm followed by gemma_fused_add_rmsnorm. Writes the output
    into x's storage and updates residual in place; returns (x, residual).
    """
    n = x.shape[-1]
    assert x.stride(-1) == 1 and residual.stride(-1) == 1
    x2, r2 = x.view(-1, n), residual.view(-1, n)
    block_n = triton.next_power_of_2(n)
    _gemma_norm_add_norm_kernel[(x2.shape[0],)](
        x2, r2, weight1, weight2, n, x2.stride(0), r2.stride(0), eps1, eps2,
        BLOCK_N=block_n, num_warps=8 if block_n >= 1024 else 4,
    )
    return x, residual
