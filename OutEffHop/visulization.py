"""Visualize attention probabilities, value norms, activation outliers, and OASIS signals.

Produces:
  1. Per-layer per-head attention probability heatmaps
  2. Attention sink: prob mass on token 0 across layers/heads
  3. Activation outlier summary: kurtosis and inf-norm per layer
  4. Value state inf-norm per layer
  5. (OASIS) Null posterior per layer — how much mass is routed to null space
  6. (OASIS) Depth routing weights — how each layer aggregates across depth

Usage:
    # Non-OASIS (OutEffHop / vanilla / softmax1):
    python visulization.py \
        --model_name_or_path /scratch/hlv8980/residual/output/softmax1_llama3/ \
        --attn_softmax softmax1 --attn_res_softmax_fn vanilla \
        --block_size 128

    # OASIS:
    python visulization.py \
        --model_name_or_path /scratch/hlv8980/residual/output/oasis_llama3/ \
        --attn_softmax softmax1 --attn_res_softmax_fn softmax1 \
        --oasis --block_size 128
"""

import argparse
import os
import pickle
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import warnings
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")
from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers_language.models.softmax import SOFTMAX_MAPPING
from transformers_language.utils import kurtosis


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def extract_attention_and_values(model, input_ids, is_oasis=False):
    """Hook every self_attn and layer to capture attention probs, value states,
    hidden states, and (for OASIS) null posteriors and depth routing weights."""
    attn_maps = {}       # layer_idx -> (batch, heads, seq, seq)
    value_maps = {}      # layer_idx -> (batch, seq, v_dim)
    hidden_maps = {}     # layer_idx -> (batch, seq, hidden)
    null_posteriors = {} # layer_idx -> (batch, seq)  [OASIS only]
    depth_weights = {}   # layer_idx -> {sa: (batch, seq, L), mlp: (batch, seq, L)}  [OASIS only]

    def make_attn_hook(layer_idx):
        def hook_fn(module, args, output):
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                attn_maps[layer_idx] = output[1].detach().cpu().float()
            # OASIS: output[2] is branch_null (B, T)
            if is_oasis and isinstance(output, tuple) and len(output) >= 3:
                null_posteriors[layer_idx] = output[2].detach().cpu().float()
        return hook_fn

    def make_value_hook(layer_idx):
        def hook_fn(module, args, output):
            value_maps[layer_idx] = output.detach().cpu().float()
        return hook_fn

    def make_layer_hook(layer_idx):
        def hook_fn(module, args, output):
            out = output[0] if isinstance(output, tuple) else output
            hidden_maps[layer_idx] = out.detach().cpu().float()
        return hook_fn

    def make_depth_routing_hook(layer_idx, sublayer):
        """Hook on AttentionResidual to capture depth routing weights."""
        def hook_fn(module, args, output):
            # Recompute the routing weights from the inputs
            layer_outputs = args[0]
            null_posts = args[1] if len(args) > 1 else None
            num_layers = len(layer_outputs)
            if num_layers <= 1:
                return

            current = layer_outputs[-1]
            stacked = torch.stack(layer_outputs, dim=2)
            query = module.q_proj(current)
            keys = module.k_proj(stacked)
            scores = torch.einsum("btd,btld->btl", query, keys) * module.layer_attn_scaling

            device = scores.device
            recency = torch.arange(num_layers, device=device, dtype=scores.dtype)
            recency = recency / max(num_layers - 1, 1)
            scores = scores + module.recency_bias * recency.unsqueeze(0).unsqueeze(0)

            # OASIS coupling
            if null_posts is not None and len(null_posts) == num_layers:
                beta = F.softplus(module.oasis_beta_raw).clamp(max=10.0)
                psi = torch.stack(null_posts, dim=-1)
                delta_psi = psi - psi.mean(dim=-1, keepdim=True)
                scores = scores - beta * delta_psi

            scores = scores.clamp(min=-50.0, max=50.0)
            weights = module.attn_res_softmax_fn(scores.float(), dim=-1, dtype=torch.float32)

            if layer_idx not in depth_weights:
                depth_weights[layer_idx] = {}
            depth_weights[layer_idx][sublayer] = weights.detach().cpu().float()
        return hook_fn

    hooks = []
    for idx, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.register_forward_hook(make_attn_hook(idx)))
        hooks.append(layer.self_attn.v_proj.register_forward_hook(make_value_hook(idx)))
        hooks.append(layer.register_forward_hook(make_layer_hook(idx)))
        # OASIS: hook depth routing
        if is_oasis and hasattr(layer, 'attn_res_sa'):
            hooks.append(layer.attn_res_sa.register_forward_hook(make_depth_routing_hook(idx, 'sa')))
        if is_oasis and hasattr(layer, 'attn_res_mlp'):
            hooks.append(layer.attn_res_mlp.register_forward_hook(make_depth_routing_hook(idx, 'mlp')))

    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    result = {
        "attn_maps": attn_maps,
        "value_maps": value_maps,
        "hidden_maps": hidden_maps,
    }
    if is_oasis:
        result["null_posteriors"] = null_posteriors
        result["depth_weights"] = depth_weights
    return result


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def _bluebird_azurite_cmap():
    """Red-white-blue colormap (BluebirdAzurite style)."""
    from matplotlib.colors import LinearSegmentedColormap
    colors = ["#ff5b46", "#ffffff", "#7bc6fc"]
    return LinearSegmentedColormap.from_list("BluebirdAzurite", colors, N=256)


def plot_attention_heatmaps(attn_maps, tokens, save_dir, max_heads=4):
    """Per-layer, per-head attention probability heatmap (lower triangle only)."""
    os.makedirs(save_dir, exist_ok=True)
    seq_len = len(tokens)
    cmap = _bluebird_azurite_cmap()

    for layer_idx in sorted(attn_maps.keys()):
        attn = attn_maps[layer_idx][0]  # (heads, seq, seq)
        num_heads = attn.shape[0]
        n = min(num_heads, max_heads)

        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]

        for h in range(n):
            ax = axes[h]
            data = attn[h, :seq_len, :seq_len].numpy()
            mask = np.triu(np.ones_like(data, dtype=bool), k=1)
            masked_data = np.ma.array(data, mask=mask)
            im = ax.imshow(masked_data, cmap=cmap, vmin=0, aspect="auto")
            ax.set_title(f"Head {h}", fontsize=10)
            ax.set_xlabel("Key")
            ax.set_ylabel("Query")
            if seq_len <= 40:
                ax.set_xticks(range(seq_len))
                ax.set_xticklabels(tokens, rotation=90, fontsize=6)
                ax.set_yticks(range(seq_len))
                ax.set_yticklabels(tokens, fontsize=6)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(f"Layer {layer_idx} — Attention Probabilities", fontsize=13)
        plt.tight_layout()
        path = os.path.join(save_dir, f"layer_{layer_idx:02d}_attn.pdf")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  {path}")


def plot_attention_sink(attn_maps, save_path):
    """Attention sink: mean prob mass on token 0 across layers x heads."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sorted_keys = sorted(attn_maps.keys())
    num_layers = len(sorted_keys)
    num_heads = attn_maps[sorted_keys[0]].shape[1]

    sink_mass = np.zeros((num_layers, num_heads))
    for i, k in enumerate(sorted_keys):
        attn = attn_maps[k][0]
        sink_mass[i] = attn[:, :, 0].mean(dim=-1).numpy()

    fig, ax = plt.subplots(figsize=(max(num_heads * 0.6, 6), max(num_layers * 0.35, 4)))
    im = ax.imshow(sink_mass, cmap="Reds", aspect="auto", vmin=0)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title("Attention Sink: Mean Prob on Token 0")
    ax.set_xticks(range(num_heads))
    ax.set_yticks(range(num_layers))
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    npy_path = save_path.replace(".pdf", ".npy")
    np.save(npy_path, sink_mass)
    print(f"  {save_path}  (data: {npy_path})")


def plot_attn_entropy_and_max_prob(attn_maps, save_path):
    """Attention entropy and max prob per layer x head."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sorted_keys = sorted(attn_maps.keys())
    num_layers = len(sorted_keys)
    num_heads = attn_maps[sorted_keys[0]].shape[1]

    entropy_arr = np.zeros((num_layers, num_heads))
    maxprob_arr = np.zeros((num_layers, num_heads))

    for i, k in enumerate(sorted_keys):
        attn = attn_maps[k][0]
        p = attn.clamp(min=1e-12)
        ent = -(p * p.log()).sum(dim=-1).mean(dim=-1)
        entropy_arr[i] = ent.numpy()
        maxprob_arr[i] = attn.max(dim=-1).values.mean(dim=-1).numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(num_layers * 0.35, 4)))

    im1 = ax1.imshow(entropy_arr, cmap="coolwarm", aspect="auto")
    ax1.set_xlabel("Head"); ax1.set_ylabel("Layer")
    ax1.set_title("Attention Entropy")
    ax1.set_xticks(range(num_heads)); ax1.set_yticks(range(num_layers))
    fig.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(maxprob_arr, cmap="hot", aspect="auto", vmin=0, vmax=1)
    ax2.set_xlabel("Head"); ax2.set_ylabel("Layer")
    ax2.set_title("Max Attention Prob (outlier indicator)")
    ax2.set_xticks(range(num_heads)); ax2.set_yticks(range(num_layers))
    fig.colorbar(im2, ax=ax2)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    npy_path = save_path.replace(".pdf", ".npz")
    np.savez(npy_path, entropy=entropy_arr, max_prob=maxprob_arr)
    print(f"  {save_path}  (data: {npy_path})")


def plot_value_inf_norm(value_maps, save_path):
    """Value state inf-norm per layer."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sorted_keys = sorted(value_maps.keys())
    num_layers = len(sorted_keys)

    layer_indices = []
    inf_norms = []
    for k in sorted_keys:
        v = value_maps[k]
        v_flat = v.view(v.shape[0], v.shape[1], -1)
        norms = v_flat.norm(dim=-1, p=float("inf")).mean().item()
        layer_indices.append(k)
        inf_norms.append(norms)

    fig, ax = plt.subplots(figsize=(max(num_layers * 0.4, 6), 4))
    ax.bar(range(num_layers), inf_norms, color="steelblue")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Inf-Norm")
    ax.set_title("Value State Inf-Norm per Layer")
    ax.set_xticks(range(num_layers))
    ax.set_xticklabels(layer_indices)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    npy_path = save_path.replace(".pdf", ".npy")
    np.save(npy_path, np.array(inf_norms))
    print(f"  {save_path}  (data: {npy_path})")


def plot_hidden_outliers(hidden_maps, save_path):
    """Hidden state kurtosis and inf-norm per layer (OutEffHop outlier definition)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sorted_keys = sorted(hidden_maps.keys())
    num_layers = len(sorted_keys)

    kurt_vals = []
    inf_vals = []
    for k in sorted_keys:
        h = hidden_maps[k]
        h_flat = h.view(-1, h.shape[-1])
        kurt_vals.append(kurtosis(h_flat).mean().item())
        inf_vals.append(h_flat.norm(dim=1, p=float("inf")).mean().item())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(num_layers * 0.4, 8), 7), sharex=True)

    ax1.bar(range(num_layers), kurt_vals, color="#ff5b46")
    ax1.set_ylabel("Kurtosis")
    ax1.set_title("Hidden State Kurtosis per Layer (higher = more outlier)")

    ax2.bar(range(num_layers), inf_vals, color="#7bc6fc")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Inf-Norm")
    ax2.set_title("Hidden State Inf-Norm per Layer")
    ax2.set_xticks(range(num_layers))

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    npy_path = save_path.replace(".pdf", ".npz")
    np.savez(npy_path, kurtosis=np.array(kurt_vals), inf_norm=np.array(inf_vals))
    print(f"  {save_path}  (data: {npy_path})")


# ---------------------------------------------------------------------------
# OASIS-specific plots
# ---------------------------------------------------------------------------

def plot_null_posteriors(null_posteriors, save_path):
    """Null posterior (1 - sum(attn)) per layer, averaged over tokens.

    Shows how much probability mass Softmax_1 routes to the null space at each layer.
    Higher = more tokens are deemed redundant by that layer's attention.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sorted_keys = sorted(null_posteriors.keys())
    num_layers = len(sorted_keys)

    # Mean null posterior per layer (averaged over batch and tokens)
    mean_null = [null_posteriors[k].mean().item() for k in sorted_keys]

    fig, ax = plt.subplots(figsize=(max(num_layers * 0.4, 8), 4))
    ax.bar(range(num_layers), mean_null, color="#ff5b46", alpha=0.85)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Null Posterior")
    ax.set_title("OASIS: Null Posterior per Layer (mass routed to null space)")
    ax.set_xticks(range(num_layers))
    ax.axhline(y=0, color="gray", linewidth=0.5)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    npy_path = save_path.replace(".pdf", ".npy")
    np.save(npy_path, np.array(mean_null))
    print(f"  {save_path}  (data: {npy_path})")


def plot_null_posterior_per_token(null_posteriors, tokens, save_path):
    """Null posterior heatmap: layers x tokens.

    Each cell shows how much attention mass that token routes to null at that layer.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sorted_keys = sorted(null_posteriors.keys())
    num_layers = len(sorted_keys)
    seq_len = len(tokens)

    # (num_layers, seq_len)
    null_grid = np.zeros((num_layers, seq_len))
    for i, k in enumerate(sorted_keys):
        null_grid[i] = null_posteriors[k][0, :seq_len].numpy()  # first batch element

    cmap = _bluebird_azurite_cmap()
    fig, ax = plt.subplots(figsize=(max(seq_len * 0.4, 8), max(num_layers * 0.35, 4)))
    im = ax.imshow(null_grid, cmap=cmap, aspect="auto", vmin=0)
    ax.set_xlabel("Token")
    ax.set_ylabel("Layer")
    ax.set_title("OASIS: Null Posterior per Token per Layer")
    if seq_len <= 40:
        ax.set_xticks(range(seq_len))
        ax.set_xticklabels(tokens, rotation=90, fontsize=6)
    ax.set_yticks(range(num_layers))
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    npy_path = save_path.replace(".pdf", ".npy")
    np.save(npy_path, null_grid)
    print(f"  {save_path}  (data: {npy_path})")


def plot_depth_routing(depth_weights, save_path):
    """Depth routing weights: how each layer aggregates across previous layers.

    Shows a heatmap where row = current layer, col = source layer index.
    Mean over batch and tokens.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sorted_keys = sorted(depth_weights.keys())
    num_layers = len(sorted_keys)

    # Determine max depth (last layer has most sources)
    max_depth = max(
        depth_weights[k][sub].shape[-1]
        for k in sorted_keys
        for sub in depth_weights[k]
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, max(num_layers * 0.35, 4)))
    cmap = _bluebird_azurite_cmap()

    for ax, sublayer, title in zip(axes, ['sa', 'mlp'],
                                    ['Post Self-Attn', 'Post MLP']):
        grid = np.zeros((num_layers, max_depth))
        grid[:] = np.nan
        for i, k in enumerate(sorted_keys):
            if sublayer in depth_weights[k]:
                w = depth_weights[k][sublayer][0].mean(dim=0).numpy()  # mean over tokens
                grid[i, :len(w)] = w

        masked_grid = np.ma.array(grid, mask=np.isnan(grid))
        im = ax.imshow(masked_grid, cmap=cmap, aspect="auto", vmin=0, vmax=1)
        ax.set_xlabel("Source Layer Index")
        ax.set_ylabel("Current Layer")
        ax.set_title(f"Depth Routing Weights ({title})")
        ax.set_yticks(range(num_layers))
        fig.colorbar(im, ax=ax)

    fig.suptitle("OASIS: Depth Routing Weights", fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {save_path}")


def plot_oasis_beta(model, save_path):
    """Plot learned OASIS coupling strength beta per layer."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    betas_sa = []
    betas_mlp = []
    for layer in model.model.layers:
        if hasattr(layer, 'attn_res_sa') and hasattr(layer.attn_res_sa, 'oasis_beta_raw'):
            betas_sa.append(F.softplus(layer.attn_res_sa.oasis_beta_raw).item())
        if hasattr(layer, 'attn_res_mlp') and hasattr(layer.attn_res_mlp, 'oasis_beta_raw'):
            betas_mlp.append(F.softplus(layer.attn_res_mlp.oasis_beta_raw).item())

    if not betas_sa:
        return

    num_layers = len(betas_sa)
    x = np.arange(num_layers)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(num_layers * 0.5, 8), 4))
    ax.bar(x - width / 2, betas_sa, width, label="Post Self-Attn", color="#ff5b46")
    ax.bar(x + width / 2, betas_mlp, width, label="Post MLP", color="#7bc6fc")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Beta (coupling strength)")
    ax.set_title("OASIS: Learned Coupling Strength beta per Layer")
    ax.set_xticks(x)
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    npy_path = save_path.replace(".pdf", ".npz")
    np.savez(npy_path, beta_sa=np.array(betas_sa), beta_mlp=np.array(betas_mlp))
    print(f"  {save_path}  (data: {npy_path})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Attention & outlier visualization")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="llama")
    parser.add_argument("--attn_softmax", type=str, default="vanilla")
    parser.add_argument("--attn_res_softmax_fn", type=str, default="vanilla")
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="attn_vis")
    parser.add_argument("--max_heads", type=int, default=4)
    parser.add_argument("--oasis", action="store_true",
                        help="Enable OASIS mode: use run_clm_oasis module, capture null posteriors and depth routing")
    parser.add_argument("--load_pkl", type=str, default=None,
                        help="Load cached activations from .pkl instead of running the model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    pkl_path = os.path.join(save_dir, "activations.pkl")

    model = None  # keep reference for beta plot

    if args.load_pkl and os.path.exists(args.load_pkl):
        print(f"Loading cached activations from {args.load_pkl}")
        with open(args.load_pkl, "rb") as f:
            cache = pickle.load(f)
        attn_maps = cache["attn_maps"]
        value_maps = cache["value_maps"]
        hidden_maps = cache["hidden_maps"]
        tokens = cache["tokens"]
        null_posteriors = cache.get("null_posteriors", {})
        depth_weights = cache.get("depth_weights", {})
        is_oasis = bool(null_posteriors)
        print(f"Loaded {len(attn_maps)} layers, {len(tokens)} tokens")
        if is_oasis:
            print(f"OASIS data: {len(null_posteriors)} null posteriors, {len(depth_weights)} depth weight layers")
        print()
    else:
        # Select the right replace_attention_modules based on --oasis
        if args.oasis:
            from accelerate import PartialState
            PartialState()
            from run_clm_oasis import replace_attention_modules, patch_model_forward_for_oasis
        else:
            from run_clm_ddp import replace_attention_modules

        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

        # Step 1: Load vanilla model structure (OASIS weights loaded later in Step 4)
        import logging as _logging
        _hf_logger = _logging.getLogger("transformers.modeling_utils")
        _prev_level = _hf_logger.level
        _hf_logger.setLevel(_logging.ERROR)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
        _hf_logger.setLevel(_prev_level)

        # Step 2: Replace layers with custom attention modules (creates attn_res_sa, attn_res_mlp, etc.)
        replace_attention_modules(model, args)

        # Step 3: OASIS — patch forward to propagate history
        if args.oasis:
            patch_model_forward_for_oasis(model)

        # Step 4: Reload ALL checkpoint weights into the now-correct architecture
        from safetensors.torch import load_file as load_safetensors
        import glob, json
        st_single = os.path.join(args.model_name_or_path, "model.safetensors")
        st_index = os.path.join(args.model_name_or_path, "model.safetensors.index.json")
        bin_path = os.path.join(args.model_name_or_path, "pytorch_model.bin")

        state_dict = {}
        if os.path.exists(st_single):
            state_dict = load_safetensors(st_single)
        elif os.path.exists(st_index):
            # Sharded safetensors
            with open(st_index) as f:
                index = json.load(f)
            shard_files = set(index["weight_map"].values())
            for shard in shard_files:
                shard_path = os.path.join(args.model_name_or_path, shard)
                state_dict.update(load_safetensors(shard_path))
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")

        if state_dict:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"Missing keys (expected for new params): {len(missing)}")
            if unexpected:
                print(f"Unexpected keys: {len(unexpected)}")

        model.to(args.device)
        model.eval()

        # Tokenize
        text = args.text or "The quick brown fox jumps over the lazy dog and then it runs back to the forest"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.block_size)
        input_ids = inputs["input_ids"].to(args.device)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        print(f"Model: {args.model_name_or_path}")
        print(f"Softmax: {args.attn_softmax}")
        print(f"OASIS: {args.oasis}")
        print(f"Tokens ({len(tokens)}): {' '.join(tokens)}")

        # Extract
        result = extract_attention_and_values(model, input_ids, is_oasis=args.oasis)
        attn_maps = result["attn_maps"]
        value_maps = result["value_maps"]
        hidden_maps = result["hidden_maps"]
        null_posteriors = result.get("null_posteriors", {})
        depth_weights = result.get("depth_weights", {})
        is_oasis = args.oasis
        print(f"Captured {len(attn_maps)} layers")

        # Save activations to pkl
        with open(pkl_path, "wb") as f:
            pickle.dump({
                "attn_maps": attn_maps,
                "value_maps": value_maps,
                "hidden_maps": hidden_maps,
                "tokens": tokens,
                "null_posteriors": null_posteriors,
                "depth_weights": depth_weights,
            }, f)
        print(f"Cached activations to {pkl_path}\n")

    # ---- Common plots ----
    print("Generating plots:")
    plot_attention_heatmaps(attn_maps, tokens, save_dir=os.path.join(save_dir, "heatmaps"), max_heads=args.max_heads)
    plot_attention_sink(attn_maps, save_path=os.path.join(save_dir, "attention_sink.pdf"))
    plot_attn_entropy_and_max_prob(attn_maps, save_path=os.path.join(save_dir, "entropy_maxprob.pdf"))
    plot_value_inf_norm(value_maps, save_path=os.path.join(save_dir, "value_inf_norm.pdf"))
    plot_hidden_outliers(hidden_maps, save_path=os.path.join(save_dir, "hidden_outliers.pdf"))

    # ---- OASIS-specific plots ----
    if null_posteriors:
        print("\nOASIS plots:")
        plot_null_posteriors(null_posteriors, save_path=os.path.join(save_dir, "null_posterior.pdf"))
        plot_null_posterior_per_token(null_posteriors, tokens, save_path=os.path.join(save_dir, "null_posterior_tokens.pdf"))
    if depth_weights:
        plot_depth_routing(depth_weights, save_path=os.path.join(save_dir, "depth_routing.pdf"))
    if model is not None and is_oasis:
        plot_oasis_beta(model, save_path=os.path.join(save_dir, "oasis_beta.pdf"))

    print(f"\nAll plots saved to {save_dir}/")


if __name__ == "__main__":
    main()
