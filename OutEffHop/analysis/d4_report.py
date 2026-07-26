#!/usr/bin/env python
"""Pair D.4 metric files, apply near-no-op filters, and build a report."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("d4_report")

KEY_COLUMNS = [
    "analysis_variant",
    "sample_id",
    "layer",
    "head",
    "query_position",
    "sink_definition",
]
VALUE_COLUMNS = [
    "input_token_id",
    "leakage",
    "concentration",
    "neg_entropy",
    "update_norm",
]
METRICS = ["leakage", "concentration", "neg_entropy"]
VARIANT_ORDER = [
    "identity_outcome",
    "attention_only_conditional",
]


def metadata_path(metric_path: Path) -> Path:
    return metric_path.with_suffix(metric_path.suffix + ".metadata.json")


def read_metrics(path: Path) -> pd.DataFrame:
    columns = ["model_kind", *KEY_COLUMNS, *VALUE_COLUMNS]
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path, columns=columns)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, usecols=columns)
    else:
        raise ValueError(f"Unsupported metric file extension: {path.suffix}")
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return frame


def read_metadata(metric_path: Path) -> Optional[Dict[str, object]]:
    path = metadata_path(metric_path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_metadata(
    attn_residual: Optional[Dict[str, object]],
    vanilla: Optional[Dict[str, object]],
    *,
    allow_missing: bool = False,
) -> None:
    if attn_residual is None or vanilla is None:
        if not allow_missing:
            raise ValueError(
                "Both metric metadata sidecars are required for a verified "
                "paired report.  Re-run collection or pass "
                "--allow-missing-metadata to accept an unverified pairing."
            )
        LOGGER.warning(
            "One or both metadata sidecars are missing; row keys will still be "
            "checked, but dataset identity cannot be verified up front."
        )
        return

    expected_metadata = (
        (
            "AttnResidual",
            attn_residual,
            {
                "model_kind": "attn_residual",
                "reconstruction_entrypoint": "run_clm_oasis.py",
                "reconstructed_token_softmax": "vanilla",
                "reconstructed_depth_softmax": "vanilla",
            },
        ),
        (
            "Vanilla",
            vanilla,
            {
                "model_kind": "vanilla",
                "reconstruction_entrypoint": "run_clm_ddp.py",
                "reconstructed_token_softmax": "vanilla",
            },
        ),
    )
    for label, metadata, expectations in expected_metadata:
        for key, expected in expectations.items():
            observed = metadata.get(key)
            if observed is not None and observed != expected:
                raise ValueError(
                    f"{label} metadata has {key}={observed!r}; "
                    f"expected {expected!r}"
                )

    comparable_keys = [
        "base_model",
        "architecture",
        "model_config_signature",
        "num_layers",
        "num_heads",
        "dtype",
        "autocast_enabled",
        "reconstructed_token_softmax",
        "dataset_sha256",
        "num_samples",
        "sequence_length",
        "prepend_bos",
        "prefix_sizes",
        "selected_layers",
        "selected_heads",
        "min_query_position",
        "query_stride",
        "include_prefix_queries",
        "analysis_variants",
    ]
    mismatches = {
        key: (attn_residual.get(key), vanilla.get(key))
        for key in comparable_keys
        if attn_residual.get(key) != vanilla.get(key)
    }
    if mismatches:
        formatted = "\n".join(
            f"  {key}: AttnResidual={values[0]!r}, Vanilla={values[1]!r}"
            for key, values in mismatches.items()
        )
        raise ValueError(
            "The metric files were not collected with an identical paired "
            f"protocol:\n{formatted}"
        )


def pair_metrics(
    attn_residual: pd.DataFrame,
    vanilla: pd.DataFrame,
) -> pd.DataFrame:
    ar_kinds = set(attn_residual["model_kind"].unique())
    vanilla_kinds = set(vanilla["model_kind"].unique())
    if ar_kinds != {"attn_residual"}:
        raise ValueError(
            f"Expected only model_kind='attn_residual', found {sorted(ar_kinds)}"
        )
    if vanilla_kinds != {"vanilla"}:
        raise ValueError(
            f"Expected only model_kind='vanilla', found {sorted(vanilla_kinds)}"
        )

    finite_columns = [*METRICS, "update_norm"]
    for label, frame in (
        ("AttnResidual", attn_residual),
        ("Vanilla", vanilla),
    ):
        values = frame[finite_columns].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{label} metric file contains non-finite values")

    if attn_residual.duplicated(KEY_COLUMNS).any():
        raise ValueError("AttnResidual metric file contains duplicate pairing keys")
    if vanilla.duplicated(KEY_COLUMNS).any():
        raise ValueError("Vanilla metric file contains duplicate pairing keys")

    paired = attn_residual.drop(columns=["model_kind"]).merge(
        vanilla.drop(columns=["model_kind"]),
        on=KEY_COLUMNS,
        how="inner",
        suffixes=("_ar", "_v"),
        validate="one_to_one",
    )
    if len(paired) != len(attn_residual) or len(paired) != len(vanilla):
        raise ValueError(
            "Metric files do not have identical pairing keys: "
            f"AttnResidual={len(attn_residual)}, Vanilla={len(vanilla)}, "
            f"paired={len(paired)}"
        )
    token_mismatch = paired["input_token_id_ar"] != paired["input_token_id_v"]
    if token_mismatch.any():
        example = paired.loc[token_mismatch, KEY_COLUMNS].iloc[0].to_dict()
        raise ValueError(
            "Input token IDs differ for a paired row; the two models did not "
            f"receive identical inputs. Example key: {example}"
        )
    return paired


def _cluster_bootstrap_interval(
    sample_ids: pd.Series,
    values: np.ndarray,
    *,
    iterations: int,
    rng: np.random.Generator,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    temporary = pd.DataFrame(
        {
            "sample_id": sample_ids.to_numpy(),
            "value": np.asarray(values, dtype=np.float64),
        }
    )
    grouped = temporary.groupby("sample_id", sort=False)["value"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(dtype=np.float64)
    counts = grouped["count"].to_numpy(dtype=np.float64)
    cluster_count = len(grouped)
    if cluster_count == 0:
        return float("nan"), float("nan")
    if iterations <= 0 or cluster_count < 2:
        return float("nan"), float("nan")

    estimates = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        draw = rng.integers(0, cluster_count, size=cluster_count)
        estimates[index] = sums[draw].sum() / counts[draw].sum()
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(estimates, [tail, 1.0 - tail])
    return float(lower), float(upper)


def summarize_paired_metrics(
    paired: pd.DataFrame,
    *,
    quantiles: Sequence[float],
    bootstrap_iterations: int,
    seed: int,
    dominance_atol: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    root_rng = np.random.default_rng(seed)

    for (analysis_variant, sink_definition), sink_frame in paired.groupby(
        ["analysis_variant", "sink_definition"], sort=True
    ):
        pooled_norms = np.concatenate(
            [
                sink_frame["update_norm_ar"].to_numpy(dtype=np.float64),
                sink_frame["update_norm_v"].to_numpy(dtype=np.float64),
            ]
        )
        for quantile in quantiles:
            delta = float(np.quantile(pooled_norms, quantile))
            selected = sink_frame.loc[
                (sink_frame["update_norm_ar"] <= delta)
                & (sink_frame["update_norm_v"] <= delta)
            ].copy()
            base = {
                "analysis_variant": analysis_variant,
                "sink_definition": sink_definition,
                "norm_quantile": quantile,
                "delta": delta,
                "near_noop_pairs": int(len(selected)),
                "near_noop_sequences": int(selected["sample_id"].nunique()),
                "near_noop_layers": int(selected["layer"].nunique()),
                "total_pairs": int(len(sink_frame)),
                "total_layers": int(sink_frame["layer"].nunique()),
                "selection_rate": float(len(selected) / len(sink_frame)),
            }

            if selected.empty:
                for metric in [*METRICS, "joint"]:
                    rows.append(
                        {
                            **base,
                            "metric": metric,
                            "attn_residual_mean": None,
                            "vanilla_mean": None,
                            "mean_difference": None,
                            "mean_difference_ci_low": None,
                            "mean_difference_ci_high": None,
                            "dominance_rate": None,
                            "dominance_ci_low": None,
                            "dominance_ci_high": None,
                        }
                    )
                continue

            joint_dominance = np.ones(len(selected), dtype=bool)
            for metric in METRICS:
                ar_values = selected[f"{metric}_ar"].to_numpy(dtype=np.float64)
                vanilla_values = selected[f"{metric}_v"].to_numpy(dtype=np.float64)
                differences = ar_values - vanilla_values
                dominance = ar_values + dominance_atol >= vanilla_values
                joint_dominance &= dominance
                child_seed = int(root_rng.integers(0, np.iinfo(np.int32).max))
                difference_ci = _cluster_bootstrap_interval(
                    selected["sample_id"],
                    differences,
                    iterations=bootstrap_iterations,
                    rng=np.random.default_rng(child_seed),
                )
                child_seed = int(root_rng.integers(0, np.iinfo(np.int32).max))
                dominance_ci = _cluster_bootstrap_interval(
                    selected["sample_id"],
                    dominance.astype(np.float64),
                    iterations=bootstrap_iterations,
                    rng=np.random.default_rng(child_seed),
                )
                rows.append(
                    {
                        **base,
                        "metric": metric,
                        "attn_residual_mean": float(ar_values.mean()),
                        "vanilla_mean": float(vanilla_values.mean()),
                        "mean_difference": float(differences.mean()),
                        "mean_difference_ci_low": difference_ci[0],
                        "mean_difference_ci_high": difference_ci[1],
                        "dominance_rate": float(dominance.mean()),
                        "dominance_ci_low": dominance_ci[0],
                        "dominance_ci_high": dominance_ci[1],
                    }
                )

            child_seed = int(root_rng.integers(0, np.iinfo(np.int32).max))
            joint_ci = _cluster_bootstrap_interval(
                selected["sample_id"],
                joint_dominance.astype(np.float64),
                iterations=bootstrap_iterations,
                rng=np.random.default_rng(child_seed),
            )
            rows.append(
                {
                    **base,
                    "metric": "joint",
                    "attn_residual_mean": None,
                    "vanilla_mean": None,
                    "mean_difference": None,
                    "mean_difference_ci_low": None,
                    "mean_difference_ci_high": None,
                    "dominance_rate": float(joint_dominance.mean()),
                    "dominance_ci_low": joint_ci[0],
                    "dominance_ci_high": joint_ci[1],
                }
            )

    return pd.DataFrame(rows)


def summarize_layer_coverage(
    paired: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Describe how the shared no-op filter is distributed over layers."""

    rows: List[Dict[str, object]] = []
    threshold_rows = summary.drop_duplicates(
        ["analysis_variant", "sink_definition", "norm_quantile"]
    )
    for _, threshold in threshold_rows.iterrows():
        analysis_variant = threshold["analysis_variant"]
        sink_definition = threshold["sink_definition"]
        delta = float(threshold["delta"])
        sink_frame = paired.loc[
            (paired["analysis_variant"] == analysis_variant)
            & (paired["sink_definition"] == sink_definition)
        ]
        for layer, layer_frame in sink_frame.groupby("layer", sort=True):
            selected = layer_frame.loc[
                (layer_frame["update_norm_ar"] <= delta)
                & (layer_frame["update_norm_v"] <= delta)
            ]
            rows.append(
                {
                    "analysis_variant": analysis_variant,
                    "sink_definition": sink_definition,
                    "norm_quantile": float(threshold["norm_quantile"]),
                    "delta": delta,
                    "layer": int(layer),
                    "near_noop_pairs": int(len(selected)),
                    "total_pairs": int(len(layer_frame)),
                    "selection_rate": float(len(selected) / len(layer_frame)),
                    "near_noop_sequences": int(
                        selected["sample_id"].nunique()
                    ),
                    "total_sequences": int(
                        layer_frame["sample_id"].nunique()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _write_markdown(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    attn_residual_path: Path,
    vanilla_path: Path,
    bootstrap_iterations: int,
    dominance_atol: float,
) -> None:
    lines = [
        "# Assumption D.4 empirical validation",
        "",
        f"- AttnResidual metrics: `{attn_residual_path}`",
        f"- Vanilla metrics: `{vanilla_path}`",
        "- Probability capture: token probabilities are observed after "
        "float32 Softmax normalization and depth probabilities are observed "
        "immediately before the real depth aggregation.",
        "- Near-no-op rule: both paired update norms are no larger than the "
        "reported pooled quantile threshold.  AttnResidual uses the maximum "
        "pre-o_proj per-head norm over every attention-bearing branch in the "
        "joint metric; Vanilla uses the target layer's corresponding norm.",
        f"- Confidence intervals: Monte Carlo sequence-cluster bootstrap "
        f"({bootstrap_iterations} iterations); this resampling step is the "
        "report's only deliberate approximation.",
        f"- Dominance comparison tolerance: `{dominance_atol:g}`.",
        "- Layer-retention diagnostics: `d4_layer_coverage.csv`.  These should "
        "be inspected before interpreting a pooled result, because a global "
        "near-no-op threshold can retain different fractions at different "
        "depths.",
        "",
    ]
    variant_descriptions = {
        "identity_outcome": (
            "Main analysis.  The initial residual route is the deterministic "
            "same-position route p_0(j | t) = 1[j=t], with probability alpha_0."
        ),
        "attention_only_conditional": (
            "Sensitivity analysis conditioned on selecting an "
            "attention-bearing route; non-identity alpha values are "
            "renormalized exactly."
        ),
    }
    present_variants = set(summary["analysis_variant"].unique())
    ordered_variants = [
        variant for variant in VARIANT_ORDER if variant in present_variants
    ]
    ordered_variants.extend(sorted(present_variants - set(ordered_variants)))
    for analysis_variant in ordered_variants:
        variant_frame = summary.loc[
            summary["analysis_variant"] == analysis_variant
        ]
        lines.extend(
            [
                f"## Analysis variant: {analysis_variant}",
                "",
                variant_descriptions.get(analysis_variant, ""),
                "",
            ]
        )
        for sink_definition, sink_frame in variant_frame.groupby(
            "sink_definition", sort=True
        ):
            lines.extend([f"### Sink set: {sink_definition}", ""])
            for quantile, threshold_frame in sink_frame.groupby(
                "norm_quantile", sort=True
            ):
                first = threshold_frame.iloc[0]
                lines.extend(
                    [
                        f"#### Pooled update-norm quantile {quantile:.0%}",
                        "",
                        f"`delta={first['delta']:.6g}`; "
                        f"{int(first['near_noop_pairs'])} paired cases from "
                        f"{int(first['near_noop_sequences'])} sequences and "
                        f"{int(first['near_noop_layers'])}/"
                        f"{int(first['total_layers'])} selected layers "
                        f"({first['selection_rate']:.3%} of candidate pairs).",
                        "",
                        "| Metric | AR mean | Vanilla mean | Mean difference (95% CI) "
                        "| Dominance rate (95% CI) |",
                        "|---|---:|---:|---:|---:|",
                    ]
                )
                for _, row in threshold_frame.iterrows():
                    metric = str(row["metric"])
                    if metric == "joint":
                        mean_ar = mean_v = difference = "—"
                    else:
                        mean_ar = (
                            "—"
                            if pd.isna(row["attn_residual_mean"])
                            else f"{row['attn_residual_mean']:.6g}"
                        )
                        mean_v = (
                            "—"
                            if pd.isna(row["vanilla_mean"])
                            else f"{row['vanilla_mean']:.6g}"
                        )
                        difference = (
                            "—"
                            if pd.isna(row["mean_difference"])
                            else (
                                f"{row['mean_difference']:.6g} "
                                "[CI unavailable]"
                                if pd.isna(row["mean_difference_ci_low"])
                                or pd.isna(row["mean_difference_ci_high"])
                                else f"{row['mean_difference']:.6g} "
                                f"[{row['mean_difference_ci_low']:.6g}, "
                                f"{row['mean_difference_ci_high']:.6g}]"
                            )
                        )
                    dominance = (
                        "—"
                        if pd.isna(row["dominance_rate"])
                        else (
                            f"{row['dominance_rate']:.3%} [CI unavailable]"
                            if pd.isna(row["dominance_ci_low"])
                            or pd.isna(row["dominance_ci_high"])
                            else f"{row['dominance_rate']:.3%} "
                            f"[{row['dominance_ci_low']:.3%}, "
                            f"{row['dominance_ci_high']:.3%}]"
                        )
                    )
                    lines.append(
                        f"| {metric} | {mean_ar} | {mean_v} | {difference} | "
                        f"{dominance} |"
                    )
                lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _json_records(frame: pd.DataFrame) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for record in frame.to_dict(orient="records"):
        records.append(
            {
                key: (
                    None
                    if value is None
                    or (isinstance(value, (float, np.floating)) and not np.isfinite(value))
                    else value.item()
                    if isinstance(value, np.generic)
                    else value
                )
                for key, value in record.items()
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attn-residual", type=Path, required=True)
    parser.add_argument("--vanilla", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.10, 0.20],
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--allow-missing-metadata",
        action="store_true",
        help=(
            "Allow pairing metric tables without both collector metadata "
            "sidecars.  This disables up-front architecture/protocol checks."
        ),
    )
    parser.add_argument(
        "--dominance-atol",
        type=float,
        default=0.0,
        help=(
            "Optional absolute tolerance for AR >= Vanilla comparisons. "
            "The default performs the literal comparison in D.4."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if any(value <= 0.0 or value >= 1.0 for value in args.quantiles):
        raise ValueError("Every quantile must lie strictly between 0 and 1")
    if args.bootstrap_iterations <= 0:
        raise ValueError("bootstrap-iterations must be positive")
    if len(set(args.quantiles)) != len(args.quantiles):
        raise ValueError("quantiles cannot contain duplicates")
    if args.dominance_atol < 0:
        raise ValueError("dominance-atol cannot be negative")

    outputs = [
        args.output_dir / "d4_summary.csv",
        args.output_dir / "d4_layer_coverage.csv",
        args.output_dir / "d4_summary.json",
        args.output_dir / "d4_report.md",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Report output already exists; pass --overwrite to replace: "
            + ", ".join(str(path) for path in existing)
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ar_metadata = read_metadata(args.attn_residual)
    vanilla_metadata = read_metadata(args.vanilla)
    validate_metadata(
        ar_metadata,
        vanilla_metadata,
        allow_missing=args.allow_missing_metadata,
    )

    LOGGER.info("Reading metric tables")
    ar_frame = read_metrics(args.attn_residual)
    vanilla_frame = read_metrics(args.vanilla)
    paired = pair_metrics(ar_frame, vanilla_frame)
    LOGGER.info("Paired %d metric rows", len(paired))

    summary = summarize_paired_metrics(
        paired,
        quantiles=args.quantiles,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
        dominance_atol=args.dominance_atol,
    )
    layer_coverage = summarize_layer_coverage(paired, summary)
    summary.to_csv(outputs[0], index=False)
    layer_coverage.to_csv(outputs[1], index=False)
    with outputs[2].open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "protocol": {
                    "attn_residual": str(args.attn_residual.resolve()),
                    "vanilla": str(args.vanilla.resolve()),
                    "quantiles": args.quantiles,
                    "bootstrap_iterations": args.bootstrap_iterations,
                    "seed": args.seed,
                    "dominance_atol": args.dominance_atol,
                    "allow_missing_metadata": args.allow_missing_metadata,
                    "paired_rows": len(paired),
                },
                "attn_residual_metadata": ar_metadata,
                "vanilla_metadata": vanilla_metadata,
                "results": _json_records(summary),
                "layer_coverage": _json_records(layer_coverage),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    _write_markdown(
        summary,
        outputs[3],
        attn_residual_path=args.attn_residual,
        vanilla_path=args.vanilla,
        bootstrap_iterations=args.bootstrap_iterations,
        dominance_atol=args.dominance_atol,
    )
    LOGGER.info("Wrote report to %s", args.output_dir)


if __name__ == "__main__":
    main()
