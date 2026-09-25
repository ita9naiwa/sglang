import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

softcap_out_autotune = triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config(kwargs={"BLOCK_SIZE": 128}, num_warps=8),
        triton.Config(kwargs={"BLOCK_SIZE": 128}, num_warps=16),
        triton.Config(kwargs={"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config(kwargs={"BLOCK_SIZE": 256}, num_warps=8),
        triton.Config(kwargs={"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config(kwargs={"BLOCK_SIZE": 512}, num_warps=8),
        triton.Config(kwargs={"BLOCK_SIZE": 512}, num_warps=16),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=16),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=32),
        triton.Config(kwargs={"BLOCK_SIZE": 2048}, num_warps=32),
        triton.Config(kwargs={"BLOCK_SIZE": 4096}, num_warps=32),
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=32),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=32),
        triton.Config(kwargs={"BLOCK_SIZE": 32768}, num_warps=32),
    ],
    key=["n_ele"],
)


@triton.jit
def softcap_out_kernel(
    output_ptr,
    input_ptr,
    n_ele,
    softcap_const: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_ele
    x = tl.load(input_ptr + offsets, mask=mask)
    fx = x.to(tl.float32)
    fxs = fx / softcap_const
    exped = tl.exp(2 * fxs)
    top = exped - 1
    bottom = exped + 1
    output = top / bottom * softcap_const
    tl.store(output_ptr + offsets, output, mask=mask)


softcap_out_kernel_autotuned = softcap_out_autotune(softcap_out_kernel)


def softcap_out(x, softcap_const, autotune=False):
    output = torch.empty_like(x, dtype=torch.float32)
    n_elements = output.numel()
    if autotune:

        def grid(meta):
            return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        softcap_out_kernel_autotuned[grid](output, x, n_elements, softcap_const)
    else:
        softcap_out_kernel[(triton.cdiv(n_elements, 128),)](
            output, x, n_elements, softcap_const, BLOCK_SIZE=128, num_warps=8
        )
    return output


@triton.jit
def softcap_inplace_logits_kernel(
    full_logits_ptr,
    softcapping_value,
    ncols,
    row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(1).to(tl.int64)
    pid = tl.program_id(0).to(tl.int64)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < ncols

    # Load values
    row_ptr = full_logits_ptr + row * row_stride
    x = tl.load(row_ptr + offsets, mask=mask)

    # Perform operations in-place
    x = x / softcapping_value
    x = libdevice.tanh(x)
    x = x * softcapping_value

    # Store result
    tl.store(row_ptr + offsets, x, mask=mask)


def softcap_inplace_logits(full_logits, final_logit_softcapping):
    if full_logits.is_contiguous():
        nrows, ncols = 1, full_logits.numel()
        row_stride = ncols
    else:
        assert full_logits.ndim == 2, "non-contiguous softcap requires 2D tensor"
        assert full_logits.stride(1) == 1, (
            "non-contiguous softcap requires contiguous columns"
        )
        nrows, ncols = full_logits.shape
        row_stride = full_logits.stride(0)

    BLOCK_SIZE = 1024
    grid = ((ncols + BLOCK_SIZE - 1) // BLOCK_SIZE, nrows)

    softcap_inplace_logits_kernel[grid](
        full_logits_ptr=full_logits,
        softcapping_value=final_logit_softcapping,
        ncols=ncols,
        row_stride=row_stride,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return full_logits


@triton.jit
def softcap_copy_to_fp32_kernel(
    dst_ptr,
    src_ptr,
    softcapping_value,
    ncols,
    dst_row_stride,
    src_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(1).to(tl.int64)
    pid = tl.program_id(0).to(tl.int64)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < ncols

    # Same fp32 op sequence as softcap_inplace_logits_kernel, so results match
    # an upcast followed by the in-place softcap bit for bit.
    x = tl.load(src_ptr + row * src_row_stride + offsets, mask=mask).to(tl.float32)
    x = x / softcapping_value
    x = libdevice.tanh(x)
    x = x * softcapping_value

    tl.store(dst_ptr + row * dst_row_stride + offsets, x, mask=mask)


def softcap_copy_to_fp32(dst, src, final_logit_softcapping):
    """dst = softcap(src.float()) in one pass; src rows may be strided."""
    assert dst.dtype == torch.float32 and dst.shape == src.shape
    if dst.is_contiguous() and src.is_contiguous():
        nrows, ncols = 1, src.numel()
        dst_row_stride = src_row_stride = ncols
    else:
        assert src.ndim == 2, "non-contiguous softcap copy requires 2D tensors"
        assert src.stride(1) == 1 and dst.stride(1) == 1, (
            "non-contiguous softcap copy requires contiguous columns"
        )
        nrows, ncols = src.shape
        dst_row_stride, src_row_stride = dst.stride(0), src.stride(0)

    BLOCK_SIZE = 1024
    grid = ((ncols + BLOCK_SIZE - 1) // BLOCK_SIZE, nrows)

    softcap_copy_to_fp32_kernel[grid](
        dst_ptr=dst,
        src_ptr=src,
        softcapping_value=final_logit_softcapping,
        ncols=ncols,
        dst_row_stride=dst_row_stride,
        src_row_stride=src_row_stride,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return dst
