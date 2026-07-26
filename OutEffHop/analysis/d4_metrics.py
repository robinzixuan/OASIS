"""Core metrics for the empirical validation of Assumption D.4.

The functions in this module deliberately operate on already-normalized
probabilities.  They are independent of the data loader and checkpoint format,
which makes the mathematical part of the experiment easy to unit test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import torch


@dataclass
class TokenAttentionSummary:
    """Per-query summaries of a token-attention distribution.

    Every tensor has shape ``[batch, heads, query]``.
    """

    leakage: Dict[str, torch.Tensor]
    concentration: torch.Tensor
    neg_entropy: torch.Tensor


@dataclass
class JointAttentionSummary:
    """Metrics of the joint depth/token distribution ``q(i, j)=alpha_i p_i(j)``."""

    leakage: Dict[str, torch.Tensor]
    concentration: torch.Tensor
    neg_entropy: torch.Tensor


def negative_entropy(probabilities: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Return ``sum(p log p)`` with the convention ``0 log 0 = 0``."""

    probabilities = probabilities.float()
    if not torch.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite")
    if (probabilities < 0).any():
        raise ValueError("Probabilities cannot be negative")
    return torch.xlogy(probabilities, probabilities).sum(dim=dim)


def token_attention_summary(
    attention_probabilities: torch.Tensor,
    prefix_sizes: Iterable[int],
) -> TokenAttentionSummary:
    """Summarize token attention.

    Args:
        attention_probabilities: Tensor shaped ``[B, H, T_query, T_key]``.
        prefix_sizes: Prefix lengths used as operational definitions of the
            sink-prone set N_t.  For example, 1 means position 0 and 4 means
            positions 0 through 3.
    """

    if attention_probabilities.ndim != 4:
        raise ValueError(
            "attention_probabilities must have shape [B, H, T_query, T_key], "
            f"got {tuple(attention_probabilities.shape)}"
        )

    probabilities = attention_probabilities.float()
    probability_sums = probabilities.sum(dim=-1)
    if not torch.allclose(
        probability_sums,
        torch.ones_like(probability_sums),
        atol=1.0e-5,
        rtol=1.0e-5,
    ):
        maximum_error = (probability_sums - 1.0).abs().amax().item()
        raise ValueError(
            "Token attention is not a normalized probability distribution; "
            f"maximum absolute sum error is {maximum_error:.6g}"
        )
    key_length = probabilities.shape[-1]
    leakage: Dict[str, torch.Tensor] = {}
    for prefix_size in prefix_sizes:
        if prefix_size <= 0:
            raise ValueError(f"prefix sizes must be positive, got {prefix_size}")
        if prefix_size > key_length:
            raise ValueError(
                f"prefix size {prefix_size} exceeds key length {key_length}"
            )
        leakage[prefix_label(prefix_size)] = probabilities[..., :prefix_size].sum(dim=-1)

    return TokenAttentionSummary(
        leakage=leakage,
        concentration=probabilities.amax(dim=-1),
        neg_entropy=negative_entropy(probabilities, dim=-1),
    )


def prefix_label(prefix_size: int) -> str:
    """Return a stable label for a prefix-based sink definition."""

    return "first_token" if prefix_size == 1 else f"prefix_{prefix_size}"


def compose_joint_summary(
    depth_probabilities: torch.Tensor,
    branch_leakage: Dict[str, torch.Tensor],
    branch_concentration: torch.Tensor,
    branch_neg_entropy: torch.Tensor,
) -> JointAttentionSummary:
    """Compute D.4 metrics without materializing the full joint distribution.

    Args:
        depth_probabilities: ``alpha`` with shape ``[B, T, L]``.
        branch_leakage: One tensor per sink definition, each ``[B, H, T, L]``.
        branch_concentration: ``max_j p_i(j)`` with shape ``[B, H, T, L]``.
        branch_neg_entropy: ``sum_j p_i(j) log p_i(j)`` with shape
            ``[B, H, T, L]``.
    """

    if depth_probabilities.ndim != 3:
        raise ValueError(
            "depth_probabilities must have shape [B, T, L], "
            f"got {tuple(depth_probabilities.shape)}"
        )
    expected = (
        depth_probabilities.shape[0],
        branch_concentration.shape[1],
        depth_probabilities.shape[1],
        depth_probabilities.shape[2],
    )
    if tuple(branch_concentration.shape) != expected:
        raise ValueError(
            f"branch_concentration has shape {tuple(branch_concentration.shape)}; "
            f"expected {expected}"
        )
    if tuple(branch_neg_entropy.shape) != expected:
        raise ValueError(
            f"branch_neg_entropy has shape {tuple(branch_neg_entropy.shape)}; "
            f"expected {expected}"
        )

    alpha = depth_probabilities.float().unsqueeze(1)
    depth_neg_entropy = negative_entropy(depth_probabilities.float(), dim=-1).unsqueeze(1)

    joint_leakage: Dict[str, torch.Tensor] = {}
    for name, values in branch_leakage.items():
        if tuple(values.shape) != expected:
            raise ValueError(
                f"branch leakage '{name}' has shape {tuple(values.shape)}; "
                f"expected {expected}"
            )
        joint_leakage[name] = (alpha * values.float()).sum(dim=-1)

    # max_{i,j} alpha_i p_i(j) = max_i alpha_i max_j p_i(j)
    concentration = (alpha * branch_concentration.float()).amax(dim=-1)

    # sum_{i,j} alpha_i p_i(j) log(alpha_i p_i(j))
    # = sum_i alpha_i log alpha_i
    #   + sum_i alpha_i sum_j p_i(j) log p_i(j)
    neg_entropy = depth_neg_entropy + (
        alpha * branch_neg_entropy.float()
    ).sum(dim=-1)

    return JointAttentionSummary(
        leakage=joint_leakage,
        concentration=concentration,
        neg_entropy=neg_entropy,
    )


def compose_joint_summary_with_identity_outcome(
    depth_probabilities: torch.Tensor,
    attention_branch_leakage: Dict[str, torch.Tensor],
    attention_branch_concentration: torch.Tensor,
    attention_branch_neg_entropy: torch.Tensor,
    identity_leakage: Dict[str, torch.Tensor],
) -> JointAttentionSummary:
    """Compute D.4 metrics when depth branch 0 is a position-preserving route.

    The paper's depth router includes the initial residual state as branch 0,
    while the repository implements that branch as the input embedding at the
    same sequence position.  The natural routing completion is therefore

    ``p_0(j | t) = 1[j=t]``,

    with ``q(i,j)=alpha_i p_i(j)`` for every branch.  This is isomorphic to one
    standalone identity outcome for concentration and entropy, but it also
    assigns ``alpha_0`` leakage when the query position itself belongs to the
    declared sink set.

    The supplied attention-branch summaries exclude branch 0 and have shape
    ``[B, H, T, L-1]``.  ``identity_leakage`` supplies the deterministic
    ``1[t in N_t]`` term for each declared sink set, with shape ``[B, H, T]``.
    """

    if depth_probabilities.ndim != 3:
        raise ValueError(
            "depth_probabilities must have shape [B, T, L], "
            f"got {tuple(depth_probabilities.shape)}"
        )
    if depth_probabilities.shape[-1] < 2:
        raise ValueError(
            "identity-outcome composition requires at least one identity and "
            "one attention-bearing branch"
        )

    expected = (
        depth_probabilities.shape[0],
        attention_branch_concentration.shape[1],
        depth_probabilities.shape[1],
        depth_probabilities.shape[2] - 1,
    )
    if tuple(attention_branch_concentration.shape) != expected:
        raise ValueError(
            "attention_branch_concentration has shape "
            f"{tuple(attention_branch_concentration.shape)}; expected {expected}"
        )
    if tuple(attention_branch_neg_entropy.shape) != expected:
        raise ValueError(
            "attention_branch_neg_entropy has shape "
            f"{tuple(attention_branch_neg_entropy.shape)}; expected {expected}"
        )

    depth_probabilities = depth_probabilities.float()
    attention_alpha = depth_probabilities[..., 1:].unsqueeze(1)
    identity_alpha = depth_probabilities[..., 0].unsqueeze(1)

    joint_leakage: Dict[str, torch.Tensor] = {}
    for name, values in attention_branch_leakage.items():
        if tuple(values.shape) != expected:
            raise ValueError(
                f"attention branch leakage '{name}' has shape "
                f"{tuple(values.shape)}; expected {expected}"
            )
        if name not in identity_leakage:
            raise ValueError(
                f"Missing identity leakage for sink definition '{name}'"
            )
        identity_values = identity_leakage[name]
        expected_identity = expected[:-1]
        if tuple(identity_values.shape) != expected_identity:
            raise ValueError(
                f"identity leakage '{name}' has shape "
                f"{tuple(identity_values.shape)}; expected {expected_identity}"
            )
        joint_leakage[name] = (
            identity_alpha * identity_values.float()
            + (attention_alpha * values.float()).sum(dim=-1)
        )

    attention_concentration = (
        attention_alpha * attention_branch_concentration.float()
    ).amax(dim=-1)
    concentration = torch.maximum(identity_alpha, attention_concentration)

    neg_entropy = negative_entropy(depth_probabilities, dim=-1).unsqueeze(1)
    neg_entropy = neg_entropy + (
        attention_alpha * attention_branch_neg_entropy.float()
    ).sum(dim=-1)

    return JointAttentionSummary(
        leakage=joint_leakage,
        concentration=concentration,
        neg_entropy=neg_entropy,
    )


def condition_on_attention_branches(
    depth_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Return exact depth probabilities conditioned on a non-identity branch."""

    if depth_probabilities.ndim != 3:
        raise ValueError(
            "depth_probabilities must have shape [B, T, L], "
            f"got {tuple(depth_probabilities.shape)}"
        )
    if depth_probabilities.shape[-1] < 2:
        raise ValueError(
            "conditioning requires at least one identity and one "
            "attention-bearing branch"
        )
    attention_probabilities = depth_probabilities.float()[..., 1:]
    normalizer = attention_probabilities.sum(dim=-1, keepdim=True)
    if not torch.isfinite(normalizer).all() or (normalizer <= 0).any():
        raise ValueError(
            "Cannot condition on attention-bearing branches because their "
            "total probability is zero or non-finite"
        )
    return attention_probabilities / normalizer


def head_update_norm(
    pre_projection_attention_output: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """Compute the paper's ``||A_{t,h}||_2`` before the output projection."""

    if pre_projection_attention_output.ndim != 3:
        raise ValueError(
            "pre_projection_attention_output must have shape [B, T, H*d_h], "
            f"got {tuple(pre_projection_attention_output.shape)}"
        )
    hidden_size = pre_projection_attention_output.shape[-1]
    if hidden_size % num_heads != 0:
        raise ValueError(
            f"hidden size {hidden_size} is not divisible by num_heads={num_heads}"
        )
    head_dim = hidden_size // num_heads
    per_head = pre_projection_attention_output.float().reshape(
        *pre_projection_attention_output.shape[:2], num_heads, head_dim
    )
    return per_head.norm(dim=-1).permute(0, 2, 1)


def maximum_branch_update_norm(
    branch_update_norms: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Return ``max_i ||A^(i)_{t,h}||_2`` for matched D.1 filtering."""

    if not branch_update_norms:
        raise ValueError("At least one attention-bearing branch is required")
    expected_shape = tuple(branch_update_norms[0].shape)
    for index, values in enumerate(branch_update_norms):
        if tuple(values.shape) != expected_shape:
            raise ValueError(
                f"Branch {index} update norm has shape {tuple(values.shape)}; "
                f"expected {expected_shape}"
            )
    return torch.stack(list(branch_update_norms), dim=-1).amax(dim=-1)
