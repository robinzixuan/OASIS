import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor, finfo, zeros, ones
from torch.nn import scaled_dot_product_attention
from typing import Optional
from math import sqrt
from torch.backends.cuda import sdp_kernel
from torch.nn.functional import pad, scaled_dot_product_attention
from collections import namedtuple
from torch.device import device_obj
from torch import bool as torch_bool
from torch.cuda import is_available, get_device_properties

import triton
import triton.language as tl



@triton.jit
def max_fn(x, y):
    return tl.math.max(x, y)


@triton.jit
def _fwd_kernel(
    tl.store(O_block_ptr, acc.to(tl.float16))


@triton.jit
def _bwd_preprocess(
    tl.store(Delta + off_m, delta)


@triton.jit
def _bwd_kernel(
        tl.store(dv_ptrs, dv)


empty = torch.empty(128, device="cuda")


def softmax_n_shifted_zeros(input: torch.Tensor, n: int, dim=-1) -> torch.Tensor:
    """
    $\text(softmax)_n(x_i) = exp(x_i) / (n + \sum_j exp(x_j))$

    Note: softmax_n, with fixed input, is _not_ shift-symmetric when n != 0
    """
    # compute the maxes along the last dimension
    input_maxes = input.max(dim=dim, keepdim=True).values
    # shift the input to prevent overflow (and underflow in the denominator)
    shifted_inputs = torch.subtract(input, input_maxes)
    # compute the numerator and softmax_0 denominator using the shifted input
    numerator = torch.exp(shifted_inputs)
    original_denominator = numerator.sum(dim=dim, keepdim=True)
    # we need to shift the zeros in the same way we shifted the inputs
    shifted_zeros = torch.multiply(input_maxes, -1)
    # and then add this contribution to the denominator
    denominator = torch.add(original_denominator, torch.multiply(torch.exp(shifted_zeros), n))
    return torch.divide(numerator, denominator)


def softmax_1(input: torch.Tensor, dim=-1) -> torch.Tensor:
    """
    $\text(softmax)_n(x_i) = exp(x_i) / (1 + \sum_j exp(x_j))$
    """
    return softmax_n_shifted_zeros(input, 1, dim=dim)
    
class Softmax_1(nn.Module):
    __constants__ = ["dim"]

    def __init__(self, dim=-1):
        """
        dim: The dimension we want to cast the operation over. Default -1
        """
        super(Softmax_1, self).__init__()
        self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "dim"):
            self.dim = None

    def forward(self, input):
        a = softmax_1(input, self.dim) 
        return a

    def extra_repr(self):
        return f"dim={self.dim}"


_EfficientAttentionConfig = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

def _flash_attn_config(query: Tensor) -> _EfficientAttentionConfig:
    """
    determine efficient attention configs for cuda and cpu
    """

    cpu_config = _EfficientAttentionConfig(True, True, True)
    cuda_config = None

    if is_available():
        device_properties = get_device_properties(device_obj('cuda'))

        if device_properties.major in {7, 8} and device_properties.minor == 0:
            # A100 (and A30) GPU == 8.0, V100 GPU = 7.0
            cuda_config = _EfficientAttentionConfig(True, False, False)
        else:
            # Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda
            cuda_config = _EfficientAttentionConfig(False, True, True)

    return cuda_config if query.is_cuda else cpu_config


def _create_causal_mask(i: int, j: int, device: device_obj) -> Tensor:
    return ones((i, j), device=device, dtype=torch_bool).triu(j - i + 1)



def flash_attention_n(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        softmax_n_param: Optional[int] = None,
        scale: Optional[float] = None,
        dropout_p: float = 0.,
        attn_mask: Optional[Tensor] = None,
        attn_bias: Optional[Tensor] = None,
        is_causal: bool = False
) -> Tensor:
    """
    CUDA implementation of Flash Attention with Softmax_n inspired by x-transformers
    :param query: Query tensor; shape (N, ..., L, E).
    :param key: Key tensor; shape (N, ..., S, E).
    :param value: Value tensor; shape (N, ..., S, Ev).
    :param softmax_n_param: Regularization parameter for the generalized softmax_n.
    :param scale: Scaling factor applied prior to softmax. If None, the default value is set to 1 / sqrt(E).
    :param dropout_p: Dropout probability; if greater than 0.0, dropout is applied
    :param attn_mask: Attention mask; shape (N, ..., L, S)
    :param attn_bias: ALiBi positional bias; shape(..., L, S)
    :param is_causal: If true, assumes causal attention masking.
    :return: Attention output; shape (N, ..., L, Ev).
    """
    if softmax_n_param is not None and softmax_n_param > 0:
        key, value = map(lambda t: pad(t, (0, 0, softmax_n_param, 0), value=0.), (key, value))

        if attn_mask is not None:
            attn_mask = pad(attn_mask, (softmax_n_param, 0), value=True)

        if attn_bias is not None:
            attn_bias = pad(attn_bias, (softmax_n_param, 0), value=0.)

    if key.ndim == 3:
        key = rearrange(key, 'b ... -> b 1 ...').expand_as(query)

    if value.ndim == 3:
        value = rearrange(value, 'b ... -> b 1 ...').expand_as(query)

    if scale is not None:
        default_scale = 1 / sqrt(query.shape[-1])
        query = query * (scale / default_scale)

    batch, heads, q_len, _, k_len, is_cuda, device, dtype = *query.shape, key.shape[-2], query.is_cuda, query.device, query.dtype

    if attn_mask is not None:
        assert attn_mask.ndim == 4
        attn_mask = attn_mask.expand(batch, heads, q_len, k_len)

        if is_causal:
            causal_mask = _create_causal_mask(q_len, k_len, device)
            attn_mask = attn_mask & ~causal_mask
            is_causal = False

    # the built-in argument `is_causal` of `scaled_dot_product_attention` appears to not work for $n > 0$, so ensure that a causal mask gets added to attn_mask or attn_bias
    if is_causal and attn_bias is None:
        attn_bias = zeros((heads, q_len, k_len), device=device, dtype=dtype)

    if attn_bias is not None:
        if attn_bias.ndim == 3:
            attn_bias = rearrange(attn_bias, 'h i j -> 1 h i j')
        attn_bias = attn_bias.expand(batch, heads, -1, -1)

        mask_value = -finfo(dtype).max

        if attn_mask is not None:
            attn_bias = attn_bias.masked_fill(~attn_mask, mask_value // 2)
        elif is_causal:
            causal_mask = _create_causal_mask(q_len, k_len, device=device)
            attn_bias = attn_bias.masked_fill(causal_mask, mask_value // 2)

        attn_mask = attn_bias

    config = _flash_attn_config(query)
    with sdp_kernel(**config._asdict()):
        return scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,  # causal mask, if requested, is included here
            dropout_p=dropout_p,
            is_causal=False  # see comment above
        )

class _FlashAttentionN(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                causal: bool = False,
                sm_scale: Optional[float] = None,
                sm_n: Optional[float] = None
                ) -> torch.Tensor:
        """
        Triton implementation of forward pass of Flash Attention with Softmax_1

        :param ctx: context
        :param q: Query tensor; shape (N, ..., L, E).
        :param k: Key tensor; shape (N, ..., S, E).
        :param v: Value tensor; shape (N, ..., S, Ev).
        :param causal: If true, assumes causal attention masking.
        :param sm_scale: Scaling factor applied prior to softmax. If None, the default value is set to 1 / sqrt(E).
        :return: Attention output; shape (N, ..., L, Ev).
        """
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        if sm_scale is None:
            sm_scale = 1 / math.sqrt(Lq)
        if sm_n is None:
            sm_n = 0.
        o = torch.empty_like(q)
        BLOCK_M = 128
        BLOCK_N = 64
        grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
        L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        P_SEQ = 0 if q.shape[-2] == k.shape[-2] else k.shape[-2] - q.shape[-2]
        num_warps = 4 if Lk <= 64 else 8
        _fwd_kernel[grid](
            q, k, v, sm_scale,
            L,
            o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1], q.shape[2], P_SEQ,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=Lk,
            IS_CAUSAL=causal,
            num_warps=num_warps,
            num_stages=4,
            SM_N=sm_n)

        ctx.save_for_backward(q, k, v, o, L)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        ctx.P_SEQ = P_SEQ
        return o

    @staticmethod
    def backward(ctx, do: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        """
        Triton implementation of backward pass of Flash Attention with Softmax_1
        :param ctx: context
        :param do: Output Gradient tensor; shape (N, ..., L, Ev). $\partial \phi / \partial \vec{o}$ where $\phi$ is the loss function
        :return: Gradients of the Query, Key, and Value tensors along with two null values.
        """
        BLOCK = 128
        q, k, v, o, L = ctx.saved_tensors
        do = do.contiguous()
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta = torch.empty_like(L)
        _bwd_preprocess[(ctx.grid[0] * ctx.grid[1], )](
            o, do,
            delta,
            BLOCK_M=BLOCK, D_HEAD=ctx.BLOCK_DMODEL,
        )
        _bwd_kernel[(ctx.grid[1],)](
            q, k, v, ctx.sm_scale,
            o, do,
            dq, dk, dv,
            L, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            q.shape[0], q.shape[1], q.shape[2], ctx.P_SEQ,
            ctx.grid[0], triton.cdiv(k.shape[2], BLOCK),
            BLOCK_M=BLOCK, BLOCK_N=BLOCK,
            BLOCK_DMODEL=ctx.BLOCK_DMODEL, num_warps=8,
            CAUSAL=ctx.causal,
            num_stages=1,
        )
        return dq, dk, dv, None, None, None


def flash_attention_n_triton(query: torch.Tensor,
                             key: torch.Tensor,
                             value: torch.Tensor,
                             is_causal: bool = False,
                             scale: Optional[float] = None,
                             softmax_n_param: Optional[float] = None
                             ) -> torch.Tensor:
    """
    Triton implementation of Flash Attention with Softmax_1

    :param query: Query tensor; shape (N, ..., L, E).
    :param key: Key tensor; shape (N, ..., S, E).
    :param value: Value tensor; shape (N, ..., S, Ev).
    :param is_causal: If true, assumes causal attention masking.
    :param scale: Scaling factor applied prior to softmax. If None, the default value is set to 1 / sqrt(E).
    :param softmax_n_param: Regularization parameter for the generalized softmax_n.
    :return: Attention output; shape (N, ..., L, Ev).
    """
    return _FlashAttentionN.apply(query, key, value, is_causal, scale, softmax_n_param)