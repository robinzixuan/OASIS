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
        Q, K, V, sm_scale,
        L,
        Out,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vk, stride_vn,
        stride_oz, stride_oh, stride_om, stride_on,
        Z, H, N_CTX, P_SEQ,
        BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        SM_N: tl.constexpr  # *** added by CWM ***
    ):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    q_offset = off_hz * stride_qh
    kv_offset = off_hz * stride_kh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(BLOCK_DMODEL, N_CTX + P_SEQ),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(N_CTX + P_SEQ, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    q = (q * qk_scale).to(tl.float16)
    # loop over k, v and update accumulator
    lo = 0
    hi = P_SEQ + (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX + P_SEQ
    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if IS_CAUSAL:
            qk = tl.where(P_SEQ + offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        qk += tl.dot(q, k)
        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(tl.float16), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    # write back l and m
    acc = acc / (SM_N * tl.exp(-m_i[:, None]) + l_i[:, None])  # *** modified by CWM ***
    l_ptrs = L + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, m_i + tl.math.log2(l_i))
    # write back O
    O_block_ptr = tl.make_block_ptr(
        base=Out + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(tl.float16))



@triton.jit
def _bwd_preprocess(
        Out, DO,
        Delta,
        BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr,
    ):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    # load
    o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    # compute
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_m, delta)


@triton.jit
def _bwd_kernel(
        Q, K, V, sm_scale, Out, DO,
        DQ, DK, DV,
        L,
        D,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vk, stride_vn,
        Z, H, N_CTX, P_SEQ,
        num_block_q, num_block_kv,
        BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,
        CAUSAL: tl.constexpr,
    ):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    qk_scale = sm_scale * 1.44269504
    # offset pointers for batch/head
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    DO += off_z * stride_qz + off_h * stride_qh
    DQ += off_z * stride_qz + off_h * stride_qh
    DK += off_z * stride_kz + off_h * stride_kh
    DV += off_z * stride_vz + off_h * stride_vh
    for start_n in range(0, num_block_kv):
        if CAUSAL:
            lo = tl.math.max(start_n * BLOCK_M - P_SEQ, 0)
        else:
            lo = 0
        # initialize row/col offsets
        offs_qm = lo + tl.arange(0, BLOCK_M)
        offs_n = start_n * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_DMODEL)
        # initialize pointers to value-like data
        q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        k_ptrs = K + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        v_ptrs = V + (offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        do_ptrs = DO + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        dq_ptrs = DQ + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        # pointer to row-wise quantities in value-like data
        D_ptrs = D + off_hz * N_CTX
        l_ptrs = L + off_hz * N_CTX
        # initialize dk amd dv
        dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        # k and v stay in SRAM throughout
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        # loop over rows
        for start_m in range(lo, num_block_q * BLOCK_M, BLOCK_M):
            offs_m_curr = start_m + offs_m
            # load q, k, v, do on-chip
            q = tl.load(q_ptrs)
            # recompute p = softmax(qk, dim=-1).T
            if CAUSAL:
                qk = tl.where(P_SEQ + offs_m_curr[:, None] >= (offs_n[None, :]), float(0.), float("-inf"))
            else:
                qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, tl.trans(k))
            qk *= qk_scale
            l_i = tl.load(l_ptrs + offs_m_curr)
            p = tl.math.exp2(qk - l_i[:, None])
            # compute dv
            do = tl.load(do_ptrs)
            dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)
            # compute dp = dot(v, do)
            Di = tl.load(D_ptrs + offs_m_curr)
            dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
            dp += tl.dot(do, tl.trans(v))
            # compute ds = p * (dp - delta[:, None])
            ds = p * dp * sm_scale
            # compute dk = dot(ds.T, q)
            dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q)
            # compute dq
            dq = tl.load(dq_ptrs)
            dq += tl.dot(ds.to(Q.dtype.element_ty), k)
            tl.store(dq_ptrs, dq)
            # increment pointers
            dq_ptrs += BLOCK_M * stride_qm
            q_ptrs += BLOCK_M * stride_qm
            do_ptrs += BLOCK_M * stride_qm
        # write-back
        dk_ptrs = DK + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        dv_ptrs = DV + (offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        tl.store(dk_ptrs, dk)
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