"""Checkpoint reconstruction for the D.4 experiment.

Custom AttentionResidual modules must be installed before their parameters are
loaded.  Loading a checkpoint produced by the repository's custom
``run_clm_oasis.py`` entry directly with ``AutoModelForCausalLM`` would
silently discard those router parameters as unexpected keys.  D.4 collection
reconstructs that entry with standard token/depth Softmax.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Set, Tuple

import torch
from transformers import AutoModelForCausalLM


LOGGER = logging.getLogger(__name__)


def _model_config_signature(model: torch.nn.Module) -> Dict[str, object]:
    """Return architecture fields that must match across the paired models."""

    config = model.config
    names = (
        "model_type",
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "hidden_act",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
        "partial_rotary_factor",
        "rope_theta",
        "rms_norm_eps",
        "attention_dropout",
        "resid_pdrop",
        "sliding_window",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
    )
    return {name: getattr(config, name, None) for name in names}


def _replacement_args(block_size: int) -> SimpleNamespace:
    """Construct the subset of training arguments needed by module replacement."""

    return SimpleNamespace(
        attn_softmax="vanilla",
        attn_res_softmax_fn="vanilla",
        block_size=block_size,
        alpha=None,
        skip_attn=False,
        attn_gate_type="none",
        attn_gate_init=0.0,
        attn_gate_mlp=False,
        attn_gate_mlp2=False,
        attn_gate_linear_all_features=False,
        fine_tuning=False,
    )


def _is_huggingface_model_directory(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").exists() and any(
        (path / name).exists()
        for name in (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
    )


def _weight_files(checkpoint: Path) -> List[Path]:
    if checkpoint.is_file():
        return [checkpoint]

    for index_name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        index_path = checkpoint / index_name
        if index_path.exists():
            with index_path.open("r", encoding="utf-8") as handle:
                index = json.load(handle)
            names = sorted(set(index["weight_map"].values()))
            return [checkpoint / name for name in names]

    candidates: List[Path] = []
    for pattern in (
        "model*.safetensors",
        "pytorch_model*.bin",
    ):
        candidates.extend(sorted(checkpoint.glob(pattern)))
    if not candidates:
        raise FileNotFoundError(
            f"No model weights found under checkpoint path: {checkpoint}"
        )
    return candidates


def _is_attention_residual_key(key: str) -> bool:
    return ".attn_res_sa." in key or ".attn_res_mlp." in key


def _load_weight_file(
    path: Path,
    *,
    only_attention_residual: bool,
) -> Dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise RuntimeError(
                "Loading safetensors checkpoints requires the safetensors package"
            ) from exc
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            if only_attention_residual:
                keys = [key for key in keys if _is_attention_residual_key(key)]
            return {key: handle.get_tensor(key) for key in keys}

    state = torch.load(str(path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint object in {path}: {type(state)}")
    if only_attention_residual:
        state = {
            key: value for key, value in state.items() if _is_attention_residual_key(key)
        }
    return state


def _normalize_state_dict_keys(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    prefixes = ("module.", "_orig_mod.")
    normalized = state_dict
    for prefix in prefixes:
        if normalized and all(key.startswith(prefix) for key in normalized):
            normalized = {
                key[len(prefix) :]: value for key, value in normalized.items()
            }
    return normalized


def _load_checkpoint_weights(
    model: torch.nn.Module,
    checkpoint: Path,
    *,
    only_attention_residual: bool,
) -> Tuple[Set[str], Set[str]]:
    loaded: Set[str] = set()
    unexpected: Set[str] = set()
    model_keys = set(model.state_dict().keys())

    for weight_file in _weight_files(checkpoint):
        state = _normalize_state_dict_keys(
            _load_weight_file(
                weight_file,
                only_attention_residual=only_attention_residual,
            )
        )
        if not state:
            continue
        result = model.load_state_dict(state, strict=False)
        loaded.update(key for key in state if key in model_keys)
        unexpected.update(result.unexpected_keys)
        del state

    return loaded, unexpected


def _validate_loaded_keys(
    model: torch.nn.Module,
    loaded_keys: Set[str],
    *,
    model_kind: str,
    full_checkpoint_load: bool,
) -> None:
    model_keys = set(model.state_dict().keys())
    if model_kind == "attn_residual":
        required = {
            key
            for key in model_keys
            if ".attn_res_sa." in key or ".attn_res_mlp." in key
        }
        missing = sorted(required - loaded_keys)
        if missing:
            preview = "\n  ".join(missing[:12])
            raise RuntimeError(
                "The checkpoint did not provide all AttentionResidual parameters. "
                "This usually means a vanilla or pre-replacement checkpoint was "
                f"selected. Missing examples:\n  {preview}"
            )

    if full_checkpoint_load:
        # Vanilla checkpoints can contain unused randomly initialized router
        # weights.  Core weights, however, must all be present.
        core = {
            key
            for key in model_keys
            if ".attn_res_sa." not in key and ".attn_res_mlp." not in key
        }
        missing_core = sorted(core - loaded_keys)
        if missing_core:
            preview = "\n  ".join(missing_core[:12])
            raise RuntimeError(
                "The accelerator checkpoint is incomplete for the reconstructed "
                f"model. Missing core keys include:\n  {preview}"
            )


def load_d4_model(
    *,
    base_model: str,
    checkpoint: str,
    model_kind: str,
    block_size: int,
    dtype: torch.dtype,
    device: torch.device,
    cache_dir: Optional[str] = None,
) -> Tuple[torch.nn.Module, Dict[str, object]]:
    """Reconstruct and load a Phi-4 Vanilla or AttentionResidual model."""

    if model_kind not in {"vanilla", "attn_residual"}:
        raise ValueError(f"Unsupported model_kind: {model_kind}")

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    is_hf_directory = _is_huggingface_model_directory(checkpoint_path)
    initial_source = str(checkpoint_path) if is_hf_directory else base_model
    LOGGER.info("Instantiating standard backbone from %s", initial_source)
    model = AutoModelForCausalLM.from_pretrained(
        initial_source,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        torch_dtype=dtype,
    )

    args = _replacement_args(block_size)
    if model_kind == "attn_residual":
        from run_clm_oasis import (
            patch_model_forward_for_oasis,
            replace_attention_modules,
        )

        decoder_info = replace_attention_modules(model, args)
        patch_model_forward_for_oasis(model)
    else:
        from run_clm_ddp import replace_attention_modules

        decoder_info = replace_attention_modules(model, args)

    if decoder_info["arch"] != "phi4":
        raise ValueError(
            "The initial D.4 implementation supports the paired Phi-4 training "
            f"entries only; detected architecture '{decoder_info['arch']}'."
        )

    # When the checkpoint is a complete HF directory, core parameters were
    # already loaded by from_pretrained.  Reload only the custom router keys.
    # Accelerator checkpoint folders have no config, so every model key is
    # loaded after reconstruction.
    if is_hf_directory and model_kind == "vanilla":
        # Core weights were already loaded by from_pretrained and the router
        # modules are inactive in the unpatched Vanilla forward.
        loaded_keys: Set[str] = set()
        unexpected: Set[str] = set()
    else:
        loaded_keys, unexpected = _load_checkpoint_weights(
            model,
            checkpoint_path,
            only_attention_residual=is_hf_directory,
        )
    _validate_loaded_keys(
        model,
        loaded_keys,
        model_kind=model_kind,
        full_checkpoint_load=not is_hf_directory,
    )
    if unexpected:
        LOGGER.warning(
            "Ignored %d unexpected checkpoint keys; examples: %s",
            len(unexpected),
            sorted(unexpected)[:8],
        )

    model.to(device=device, dtype=dtype)
    model.eval()
    metadata: Dict[str, object] = {
        "base_model": base_model,
        "checkpoint": str(checkpoint_path),
        "checkpoint_is_hf_directory": is_hf_directory,
        "model_kind": model_kind,
        "architecture": decoder_info["arch"],
        "model_config_signature": _model_config_signature(model),
        "num_layers": len(decoder_info["layers"]),
        "num_heads": model.config.num_attention_heads,
        "reconstruction_entrypoint": (
            "run_clm_oasis.py"
            if model_kind == "attn_residual"
            else "run_clm_ddp.py"
        ),
        "reconstructed_token_softmax": "vanilla",
        "reconstructed_depth_softmax": (
            "vanilla" if model_kind == "attn_residual" else None
        ),
        "loaded_custom_or_full_keys": len(loaded_keys),
        "unexpected_keys": sorted(unexpected),
    }
    return model, metadata
