#!/usr/bin/env python
"""Collect paired-ready D.4 metrics from one Phi-4 checkpoint.

Run this script once for the AttentionResidual checkpoint and once for the
Vanilla checkpoint using identical data arguments.  The report script performs
the actual pairing and near-no-op filtering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

import numpy as np
import torch
from transformers import AutoTokenizer

from analysis.checkpoint_loader import load_d4_model
from analysis.d4_metrics import (
    TokenAttentionSummary,
    compose_joint_summary_with_identity_outcome,
    condition_on_attention_branches,
    compose_joint_summary,
    head_update_norm,
    maximum_branch_update_norm,
    prefix_label,
    token_attention_summary,
)


LOGGER = logging.getLogger("d4_collect")

ANALYSIS_VARIANTS = (
    "identity_outcome",
    "attention_only_conditional",
)

METRIC_COLUMNS = [
    "model_kind",
    "analysis_variant",
    "sample_id",
    "layer",
    "head",
    "query_position",
    "input_token_id",
    "sink_definition",
    "leakage",
    "concentration",
    "neg_entropy",
    "update_norm",
]


@dataclass
class LayerRecord:
    layer: int
    analysis_variant: str
    head_indices: List[int]
    leakage: Dict[str, torch.Tensor]
    concentration: torch.Tensor
    neg_entropy: torch.Tensor
    update_norm: torch.Tensor


class MetricWriter:
    """Append metric batches to CSV or Parquet without retaining the full run."""

    def __init__(self, path: Path, *, overwrite: bool = False):
        self.path = path
        self.format = path.suffix.lower().lstrip(".")
        if self.format not in {"parquet", "csv"}:
            raise ValueError("Output must end in .parquet or .csv")
        if path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Output already exists: {path}. Pass --overwrite to replace it."
                )
            path.unlink()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._parquet_writer = None
        self._csv_header_written = False
        if self.format == "parquet":
            try:
                import pyarrow  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "Parquet output requires pyarrow. Install "
                    "`OutEffHop/analysis/requirements.txt` or use an output "
                    "ending in .csv."
                ) from exc

    def write(self, columns: Dict[str, np.ndarray]) -> None:
        if not columns or len(columns["sample_id"]) == 0:
            return
        if self.format == "csv":
            import pandas as pd

            frame = pd.DataFrame(columns, columns=METRIC_COLUMNS)
            frame.to_csv(
                self.path,
                mode="a",
                header=not self._csv_header_written,
                index=False,
            )
            self._csv_header_written = True
            return

        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table(columns)
        if self._parquet_writer is None:
            self._parquet_writer = pq.ParquetWriter(
                str(self.path),
                table.schema,
                compression="zstd",
                use_dictionary=[
                    "model_kind",
                    "analysis_variant",
                    "sink_definition",
                ],
            )
        self._parquet_writer.write_table(table)

    def close(self) -> None:
        if self._parquet_writer is not None:
            self._parquet_writer.close()


class D4ForwardCapture:
    """Hooks that reduce attention tensors to D.4 metrics during one forward."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        model_kind: str,
        prefix_sizes: Sequence[int],
        selected_layers: Sequence[int],
        selected_heads: Sequence[int],
        analysis_variants: Sequence[str],
    ):
        self.model = model
        self.model_kind = model_kind
        self.prefix_sizes = list(prefix_sizes)
        self.selected_layers = set(selected_layers)
        self.selected_heads = list(selected_heads)
        self.analysis_variants = list(analysis_variants)
        self.layers = model.model.layers
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._token_observer_previous: Dict[int, object] = {}
        self._depth_observer_previous: Dict[int, object] = {}
        self._head_updates: Dict[int, torch.Tensor] = {}
        self._token_summaries: Dict[int, TokenAttentionSummary] = {}
        self._records: List[LayerRecord] = []
        self._register()

    def _register(self) -> None:
        for layer_idx, layer in enumerate(self.layers):
            token_previous = getattr(
                layer.self_attn, "_d4_token_observer", None
            )
            if token_previous is not None:
                raise RuntimeError(
                    f"Layer {layer_idx} already has an active "
                    "_d4_token_observer"
                )
            self._token_observer_previous[layer_idx] = token_previous
            layer.self_attn._d4_token_observer = self._make_token_observer(
                layer_idx
            )
            self._handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(
                    self._make_o_proj_hook(layer_idx)
                )
            )
            self._handles.append(
                layer.self_attn.register_forward_hook(
                    self._make_self_attention_hook(layer_idx)
                )
            )
            if self.model_kind == "attn_residual":
                module = layer.attn_res_sa
                previous = getattr(module, "_d4_observer", None)
                if previous is not None:
                    raise RuntimeError(
                        f"Layer {layer_idx} already has an active _d4_observer"
                    )
                self._depth_observer_previous[layer_idx] = previous
                module._d4_observer = self._make_depth_observer(layer_idx)

    def _make_o_proj_hook(self, layer_idx: int):
        def hook(_module, inputs):
            if not inputs:
                raise RuntimeError("o_proj pre-hook received no input")
            value = inputs[0]
            self._head_updates[layer_idx] = head_update_norm(
                value, self.model.config.num_attention_heads
            )

        return hook

    def _make_self_attention_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            if not isinstance(output, tuple) or len(output) < 2:
                raise RuntimeError(
                    f"Layer {layer_idx} self-attention did not return attention weights"
                )
            if layer_idx not in self._token_summaries:
                raise RuntimeError(
                    f"Layer {layer_idx} did not expose float32 token "
                    "probabilities through _d4_token_observer"
                )
            summary = self._token_summaries[layer_idx]

            if (
                self.model_kind == "vanilla"
                and layer_idx in self.selected_layers
            ):
                update_norm = self._head_updates[layer_idx]
                heads = self.selected_heads
                for analysis_variant in self.analysis_variants:
                    self._records.append(
                        LayerRecord(
                            layer=layer_idx,
                            analysis_variant=analysis_variant,
                            head_indices=heads,
                            leakage={
                                name: value[:, heads].detach().float().cpu()
                                for name, value in summary.leakage.items()
                            },
                            concentration=summary.concentration[:, heads]
                            .detach()
                            .float()
                            .cpu(),
                            neg_entropy=summary.neg_entropy[:, heads]
                            .detach()
                            .float()
                            .cpu(),
                            update_norm=update_norm[:, heads]
                            .detach()
                            .float()
                            .cpu(),
                        )
                    )

        return hook

    def _make_token_observer(self, layer_idx: int):
        def observe(probabilities: torch.Tensor) -> None:
            if layer_idx in self._token_summaries:
                raise RuntimeError(
                    f"Layer {layer_idx} token observer ran more than once in "
                    "one no-cache forward"
                )
            self._token_summaries[layer_idx] = token_attention_summary(
                probabilities, self.prefix_sizes
            )

        return observe

    def _make_depth_observer(self, layer_idx: int):
        def observe(alpha: torch.Tensor) -> None:
            if alpha.ndim != 3:
                raise RuntimeError(
                    f"Layer {layer_idx} depth probabilities must have shape "
                    f"[B, T, L], got {tuple(alpha.shape)}"
                )
            if not torch.isfinite(alpha).all():
                raise RuntimeError(
                    f"Layer {layer_idx} produced non-finite depth probabilities"
                )
            probability_sums = alpha.float().sum(dim=-1)
            if not torch.allclose(
                probability_sums,
                torch.ones_like(probability_sums),
                atol=1.0e-5,
                rtol=1.0e-5,
            ):
                maximum_error = (probability_sums - 1.0).abs().amax().item()
                raise RuntimeError(
                    f"Layer {layer_idx} depth probabilities are not normalized; "
                    f"maximum absolute sum error is {maximum_error:.6g}"
                )

            # Branch 0 is the initial residual state.  The paper does not
            # define token attention for it.  Branch k+1 is associated with
            # the token attention from decoder layer k.
            attention_summaries = [
                self._token_summaries[index] for index in range(layer_idx + 1)
            ]
            branch_count = alpha.shape[-1]
            expected_branches = layer_idx + 2
            if branch_count != expected_branches:
                raise RuntimeError(
                    f"Layer {layer_idx} exposes {branch_count} depth branches; "
                    f"expected {expected_branches}"
                )

            attention_branch_leakage: Dict[str, torch.Tensor] = {}
            for prefix_size in self.prefix_sizes:
                name = prefix_label(prefix_size)
                attention_branch_leakage[name] = torch.stack(
                    [summary.leakage[name] for summary in attention_summaries],
                    dim=-1,
                )

            attention_branch_concentration = torch.stack(
                [summary.concentration for summary in attention_summaries],
                dim=-1,
            )
            attention_branch_neg_entropy = torch.stack(
                [summary.neg_entropy for summary in attention_summaries],
                dim=-1,
            )
            batch_size, num_heads, query_count, _ = (
                attention_branch_concentration.shape
            )
            query_positions = torch.arange(
                query_count, device=alpha.device
            ).view(1, 1, query_count)
            identity_leakage = {
                prefix_label(prefix_size): (
                    query_positions < prefix_size
                ).expand(batch_size, num_heads, query_count)
                for prefix_size in self.prefix_sizes
            }

            summaries = {}
            if "identity_outcome" in self.analysis_variants:
                summaries["identity_outcome"] = (
                    compose_joint_summary_with_identity_outcome(
                        alpha,
                        attention_branch_leakage,
                        attention_branch_concentration,
                        attention_branch_neg_entropy,
                        identity_leakage,
                    )
                )
            if "attention_only_conditional" in self.analysis_variants:
                attention_alpha = condition_on_attention_branches(alpha)
                summaries["attention_only_conditional"] = compose_joint_summary(
                    attention_alpha,
                    attention_branch_leakage,
                    attention_branch_concentration,
                    attention_branch_neg_entropy,
                )

            if layer_idx in self.selected_layers:
                heads = self.selected_heads
                branch_update_norm = maximum_branch_update_norm(
                    [
                        self._head_updates[index]
                        for index in range(layer_idx + 1)
                    ]
                )
                for analysis_variant, joint in summaries.items():
                    self._records.append(
                        LayerRecord(
                            layer=layer_idx,
                            analysis_variant=analysis_variant,
                            head_indices=heads,
                            leakage={
                                name: value[:, heads].detach().float().cpu()
                                for name, value in joint.leakage.items()
                            },
                            concentration=joint.concentration[:, heads]
                            .detach()
                            .float()
                            .cpu(),
                            neg_entropy=joint.neg_entropy[:, heads]
                            .detach()
                            .float()
                            .cpu(),
                            update_norm=branch_update_norm[:, heads]
                            .detach()
                            .float()
                            .cpu(),
                        )
                    )

        return observe

    def begin_batch(self) -> None:
        self._head_updates.clear()
        self._token_summaries.clear()
        self._records.clear()

    def pop_records(self) -> List[LayerRecord]:
        records = self._records
        self._records = []
        return records

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for layer_idx, previous in self._token_observer_previous.items():
            self.layers[layer_idx].self_attn._d4_token_observer = previous
        self._token_observer_previous.clear()
        for layer_idx, previous in self._depth_observer_previous.items():
            self.layers[layer_idx].attn_res_sa._d4_observer = previous
        self._depth_observer_previous.clear()


def _parse_index_spec(spec: str, upper_bound: int, name: str) -> List[int]:
    if spec.strip().lower() == "all":
        return list(range(upper_bound))
    selected = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending {name} range: {item}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(item))
    invalid = sorted(index for index in selected if index < 0 or index >= upper_bound)
    if invalid:
        raise ValueError(
            f"{name} indices outside [0, {upper_bound - 1}]: {invalid}"
        )
    if not selected:
        raise ValueError(f"No {name} indices selected")
    return sorted(selected)


def _iter_texts(args: argparse.Namespace) -> Iterator[str]:
    if args.text_file is not None:
        with Path(args.text_file).open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.rstrip("\n")
                if text:
                    yield text
        return

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Loading a Hugging Face dataset requires the datasets package. "
            "Alternatively pass --text-file."
        ) from exc

    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.dataset_split,
        cache_dir=args.data_cache_dir,
    )
    if args.text_column not in dataset.column_names:
        raise ValueError(
            f"Text column '{args.text_column}' not found; columns are "
            f"{dataset.column_names}"
        )
    for value in dataset[args.text_column]:
        if value:
            yield value


def _build_chunks(
    tokenizer,
    texts: Iterable[str],
    *,
    sequence_length: int,
    num_samples: int,
    prepend_bos: bool,
) -> np.ndarray:
    if prepend_bos and tokenizer.bos_token_id is None:
        raise ValueError("--prepend-bos was set, but the tokenizer has no BOS token")

    required_content = sequence_length - (1 if prepend_bos else 0)
    if required_content <= 0:
        raise ValueError(
            "sequence-length must leave room for at least one non-BOS token"
        )
    buffer: List[int] = []
    chunks: List[List[int]] = []
    for text in texts:
        buffer.extend(
            tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
        )
        while len(buffer) >= required_content and len(chunks) < num_samples:
            content = buffer[:required_content]
            del buffer[:required_content]
            if prepend_bos:
                content = [tokenizer.bos_token_id] + content
            chunks.append(content)
        if len(chunks) >= num_samples:
            break

    if len(chunks) < num_samples:
        raise RuntimeError(
            f"Only constructed {len(chunks)} full sequences, fewer than "
            f"--num-samples={num_samples}"
        )
    return np.asarray(chunks, dtype=np.int64)


def _metric_columns(
    record: LayerRecord,
    *,
    model_kind: str,
    sample_ids: np.ndarray,
    input_ids: np.ndarray,
    prefix_sizes: Sequence[int],
    min_query_position: int,
    query_stride: int,
    include_prefix_queries: bool,
) -> Iterator[Dict[str, np.ndarray]]:
    batch_size, num_heads, sequence_length = record.concentration.shape
    head_indices = np.asarray(record.head_indices, dtype=np.int16)

    for prefix_size in prefix_sizes:
        sink_name = prefix_label(prefix_size)
        query_start = min_query_position
        if not include_prefix_queries:
            query_start = max(query_start, prefix_size)
        query_positions = np.arange(
            query_start, sequence_length, query_stride, dtype=np.int32
        )
        if len(query_positions) == 0:
            continue

        row_count = batch_size * num_heads * len(query_positions)
        tokens = input_ids[:, query_positions]
        yield {
            "model_kind": np.full(row_count, model_kind, dtype=object),
            "analysis_variant": np.full(
                row_count, record.analysis_variant, dtype=object
            ),
            "sample_id": np.repeat(
                sample_ids.astype(np.int32), num_heads * len(query_positions)
            ),
            "layer": np.full(row_count, record.layer, dtype=np.int16),
            "head": np.tile(
                np.repeat(head_indices, len(query_positions)), batch_size
            ),
            "query_position": np.tile(
                query_positions, batch_size * num_heads
            ),
            "input_token_id": np.repeat(
                tokens[:, np.newaxis, :], num_heads, axis=1
            ).reshape(-1).astype(np.int32),
            "sink_definition": np.full(row_count, sink_name, dtype=object),
            "leakage": record.leakage[sink_name][
                :, :, query_positions
            ].numpy().reshape(-1),
            "concentration": record.concentration[
                :, :, query_positions
            ].numpy().reshape(-1),
            "neg_entropy": record.neg_entropy[
                :, :, query_positions
            ].numpy().reshape(-1),
            "update_norm": record.update_norm[
                :, :, query_positions
            ].numpy().reshape(-1),
        }


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


def _autocast_context(device: torch.device, dtype: torch.dtype):
    """Match the mixed-precision context used by the repository's train jobs.

    The custom depth router computes normalized weights in float32 while the
    routed hidden states can be FP16/BF16.  The original training entry points
    execute that mixed-dtype operation under Accelerate autocast.
    """

    enabled = device.type == "cuda" and dtype in {
        torch.float16,
        torch.bfloat16,
    }
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _metadata_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".metadata.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-kind", choices=["attn_residual", "vanilla"], required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--dataset-name", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--dataset-split", default="validation")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--data-cache-dir", default=None)
    parser.add_argument("--model-cache-dir", default=None)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    bos_group = parser.add_mutually_exclusive_group()
    bos_group.add_argument(
        "--prepend-bos",
        dest="prepend_bos",
        action="store_true",
        help="Prepend the tokenizer BOS token to every evaluation sequence.",
    )
    bos_group.add_argument(
        "--no-prepend-bos",
        dest="prepend_bos",
        action="store_false",
        help="Do not prepend BOS; position 0 is then only a generic first token.",
    )
    parser.set_defaults(prepend_bos=True)

    parser.add_argument(
        "--prefix-sizes",
        type=int,
        nargs="+",
        default=[1, 4],
        help="Prefix lengths used as sink-prone token sets.",
    )
    parser.add_argument("--layers", default="all", help="all or e.g. 0,8,16-20")
    parser.add_argument("--heads", default="all", help="all or e.g. 0,4,8-12")
    parser.add_argument("--min-query-position", type=int, default=1)
    parser.add_argument("--query-stride", type=int, default=1)
    parser.add_argument(
        "--include-prefix-queries",
        action="store_true",
        help="Include queries that themselves lie inside the sink prefix.",
    )
    parser.add_argument(
        "--analysis-variants",
        nargs="+",
        choices=ANALYSIS_VARIANTS,
        default=list(ANALYSIS_VARIANTS),
        help=(
            "Identity handling for AttnResidual. Vanilla metrics are duplicated "
            "under the same labels for exact pairing."
        ),
    )

    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument("--seed", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if args.num_samples <= 0 or args.sequence_length <= 0 or args.batch_size <= 0:
        raise ValueError("num-samples, sequence-length, and batch-size must be positive")
    if args.query_stride <= 0:
        raise ValueError("query-stride must be positive")
    if any(size <= 0 or size > args.sequence_length for size in args.prefix_sizes):
        raise ValueError("prefix sizes must lie in [1, sequence-length]")
    if len(set(args.prefix_sizes)) != len(args.prefix_sizes):
        raise ValueError("prefix-sizes cannot contain duplicates")
    if len(set(args.analysis_variants)) != len(args.analysis_variants):
        raise ValueError("analysis-variants cannot contain duplicates")
    if args.min_query_position < 0 or args.min_query_position >= args.sequence_length:
        raise ValueError("min-query-position must lie in [0, sequence-length)")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    dtype = _dtype_from_name(args.dtype)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"--device={args.device!r} requires CUDA, but "
            "torch.cuda.is_available() is false"
        )

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    tokenizer_source = args.tokenizer
    if tokenizer_source is None:
        tokenizer_source = (
            str(checkpoint_path)
            if checkpoint_path.is_dir()
            and (checkpoint_path / "tokenizer_config.json").exists()
            else args.base_model
        )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        cache_dir=args.model_cache_dir,
        use_fast=True,
    )
    chunks = _build_chunks(
        tokenizer,
        _iter_texts(args),
        sequence_length=args.sequence_length,
        num_samples=args.num_samples,
        prepend_bos=args.prepend_bos,
    )
    dataset_sha256 = hashlib.sha256(chunks.tobytes()).hexdigest()

    writer = MetricWriter(args.output, overwrite=args.overwrite)
    model, model_metadata = load_d4_model(
        base_model=args.base_model,
        checkpoint=args.checkpoint,
        model_kind=args.model_kind,
        block_size=args.sequence_length,
        dtype=dtype,
        device=device,
        cache_dir=args.model_cache_dir,
    )
    layers = _parse_index_spec(
        args.layers, len(model.model.layers), "layer"
    )
    heads = _parse_index_spec(
        args.heads, model.config.num_attention_heads, "head"
    )
    capture = D4ForwardCapture(
        model,
        model_kind=args.model_kind,
        prefix_sizes=args.prefix_sizes,
        selected_layers=layers,
        selected_heads=heads,
        analysis_variants=args.analysis_variants,
    )

    rows_written = 0
    try:
        for batch_start in range(0, len(chunks), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(chunks))
            batch_numpy = chunks[batch_start:batch_end]
            batch = torch.from_numpy(batch_numpy).to(device)
            sample_ids = np.arange(batch_start, batch_end, dtype=np.int32)
            capture.begin_batch()
            with torch.inference_mode(), _autocast_context(device, dtype):
                model.model(input_ids=batch, use_cache=False)
            records = capture.pop_records()
            expected_records = {
                (layer, analysis_variant)
                for layer in layers
                for analysis_variant in args.analysis_variants
            }
            actual_records = {
                (record.layer, record.analysis_variant) for record in records
            }
            if actual_records != expected_records or len(records) != len(
                expected_records
            ):
                raise RuntimeError(
                    "Forward capture did not produce exactly one record for "
                    "every selected layer and analysis variant. "
                    f"Expected {sorted(expected_records)}, "
                    f"received {sorted(actual_records)}."
                )
            for record in records:
                for columns in _metric_columns(
                    record,
                    model_kind=args.model_kind,
                    sample_ids=sample_ids,
                    input_ids=batch_numpy,
                    prefix_sizes=args.prefix_sizes,
                    min_query_position=args.min_query_position,
                    query_stride=args.query_stride,
                    include_prefix_queries=args.include_prefix_queries,
                ):
                    writer.write(columns)
                    rows_written += len(columns["sample_id"])
            LOGGER.info(
                "Collected samples %d-%d / %d (%d rows total)",
                batch_start,
                batch_end - 1,
                len(chunks),
                rows_written,
            )
    finally:
        capture.close()
        writer.close()

    if rows_written == 0:
        raise RuntimeError(
            "The collection produced no rows. Check query-position and layer/head "
            "selection arguments."
        )

    metadata = {
        **model_metadata,
        "output": str(args.output.resolve()),
        "dataset_name": args.dataset_name if args.text_file is None else None,
        "dataset_config": args.dataset_config if args.text_file is None else None,
        "dataset_split": args.dataset_split if args.text_file is None else None,
        "text_file": str(Path(args.text_file).resolve()) if args.text_file else None,
        "dataset_sha256": dataset_sha256,
        "tokenizer": tokenizer_source,
        "num_samples": args.num_samples,
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "prepend_bos": args.prepend_bos,
        "prefix_sizes": args.prefix_sizes,
        "selected_layers": layers,
        "selected_heads": heads,
        "min_query_position": args.min_query_position,
        "query_stride": args.query_stride,
        "include_prefix_queries": args.include_prefix_queries,
        "analysis_variants": args.analysis_variants,
        "attention_branch_mapping": (
            "depth_branch_k_plus_1_to_decoder_attention_k"
            if args.model_kind == "attn_residual"
            else None
        ),
        "depth_probability_source": (
            "exact_forward_observer"
            if args.model_kind == "attn_residual"
            else None
        ),
        "token_probability_source": (
            "exact_float32_forward_observer_before_model_dtype_cast"
        ),
        "near_noop_norm_definition": (
            "max_attention_branch_pre_o_proj_head_norm"
            if args.model_kind == "attn_residual"
            else "target_layer_pre_o_proj_head_norm"
        ),
        "dtype": args.dtype,
        "autocast_enabled": (
            device.type == "cuda"
            and dtype in {torch.float16, torch.bfloat16}
        ),
        "seed": args.seed,
        "rows_written": rows_written,
    }
    metadata_path = _metadata_path(args.output)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    LOGGER.info("Wrote %s and %s", args.output, metadata_path)


if __name__ == "__main__":
    main()
