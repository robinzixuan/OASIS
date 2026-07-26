# Empirical validation of Assumption D.4

This directory implements a paired experiment comparing the repository's
Phi-4 AttentionResidual (`run_clm_oasis.py`) and Vanilla
(`run_clm_ddp.py`) checkpoints.

The analysis is inference-only.  It calls `model.eval()` under
`torch.inference_mode()`, does not construct an optimizer, and never calls
`backward()` or updates either checkpoint.

Here, `AttentionResidual` specifically means the repository's
`run_clm_oasis.py` implementation reconstructed with
`--attn_softmax vanilla --attn_res_softmax_fn vanilla`.  This is the
standard dual-Softmax model to which Assumption D.4 applies.  Do not pass an
OASIS/Softmax1 checkpoint: normalization functions are not encoded in the
checkpoint weights, so using such a checkpoint would reconstruct a different
forward from the one that was trained.

## Required checkpoint provenance

A successfully generated report is not, by itself, a controlled comparison.
To attribute a difference to AttentionResidual, the two checkpoints must come
from a matched training pair:

- the same initial Phi-4 backbone and tokenizer;
- the same training and validation data, data percentages, preprocessing,
  sequence length, example order, and number of examples/tokens seen;
- the same optimizer, learning-rate schedule, warmup, batch size, gradient
  accumulation, weight decay, clipping, precision, and training-step budget;
- the same seed for a single paired run, or a predeclared matched multi-seed
  design; and
- the same checkpoint step, with the intended normalization flags used during
  training.

The intended pair is therefore:

```text
one common Phi-4 initialization
  -> run_clm_oasis.py, token Softmax=vanilla, depth Softmax=vanilla
  -> run_clm_ddp.py,   token Softmax=vanilla
```

with all non-architectural training choices matched.  A trained
AttentionResidual checkpoint versus an untouched base Phi-4 checkpoint is not
a valid controlled pair.  Neither is an AttentionResidual run trained on 100%
of a dataset versus a Vanilla run trained on 5%.

These training facts and the normalization used during training are **not
recoverable from the model weights**.  The loader records how it reconstructed
the models for inference, but that does not prove how the checkpoints were
trained.  Treat checkpoint provenance as user-asserted: retain the two original
training commands/configs, job IDs, checkpoint steps, and dataset/version
identifiers with the report, and manually verify the checklist above before
citing the comparison.

## What is measured

For every selected `(sample, layer, head, query position)` the Vanilla model
uses its token-attention distribution `p`.  The AttentionResidual model uses
the D.4 depth/token surrogate

```text
q(i, j) = alpha_i * p_i(j).
```

The three D.4 quantities are:

```text
L = probability mass assigned to the sink-prone token set
C = maximum probability assigned to one depth/token route
E = sum q log q  (negative entropy; larger means more concentrated)
```

The collector obtains `alpha` through an opt-in observer inside
`AttentionResidual.forward()`.  These are the exact probabilities used by the
model's subsequent `einsum`; the collector does not recompute the router.
Token probabilities are observed immediately after the float32 Softmax and
normalization, before they are cast back to FP16/BF16 for value aggregation.
This avoids treating finite-precision sum drift as if it were a normalized
distribution.  Consequently, the reported token distribution is the float32,
pre-cast distribution, while the actual low-precision value aggregation
consumes its cast version; the two follow the same forward path but are not
claimed to be bitwise identical.

`q` is never materialized.  The collector uses the algebraically exact closed
forms:

```text
L(q) = sum_i alpha_i L(p_i)
C(q) = max_i alpha_i C(p_i)
E(q) = sum_i alpha_i log(alpha_i) + sum_i alpha_i E(p_i)
```

For the main `identity_outcome` analysis, no division, sampling, or
attention-rollout approximation is used in these formulas.  The separate
`attention_only_conditional` sensitivity analysis does divide by
`1-alpha_0`, as its definition explicitly conditions on choosing an
attention-bearing route.

FP16/BF16 collection runs under CUDA autocast, matching the mixed-precision
execution context used by the repository's Accelerate training scripts.  This
is required because the router normalizes in float32 while the routed hidden
states may use a lower precision.  The dtype and whether autocast was enabled
are recorded in each metadata sidecar and must match across the paired runs.

The update norm used for near-no-op matching is the raw per-head L2 norm taken
before `o_proj`, exactly matching the vector `sum_j p_j v_j`:

```text
r(t,h,l) = ||A(t,h,l)||_2.
```

For an AttnResidual target layer `l`, D.1 is required for every
attention-bearing branch included in the D.4 metric:

```text
r_AR(t,h,l) = max_{i=0,...,l} ||A_i(t,h)||_2.
```

For Vanilla, `r_V(t,h,l)=||A_l(t,h)||_2`.  The report computes a pooled
update-norm threshold `delta` at each requested quantile and retains a pair
only if both models satisfy `r_AR <= delta` and `r_V <= delta`.

## Operational definitions

The main sink set is position 0 (`first_token`).  `prefix_4` is included as a
robustness check.  Queries inside the chosen prefix are excluded by default, so
the query token itself is not labeled as an irrelevant prefix token.

The paper includes the initial residual state in depth routing but does not
write down token probabilities for it.  The repository makes the missing
semantics clear: branch 0 is the input embedding at the same sequence position.
We therefore complete the routing definition as
`p_0(j | t) = 1[j=t]` and emit two predeclared variants:

- `identity_outcome` (main): the initial residual route is a deterministic
  self-token route, `q(0,j)=alpha_0 1[j=t]`.  For attention-bearing routes,
  `q(i,j)=alpha_i p_i(j)`.  This gives
  `C=max(alpha_0, max alpha_i p_i(j))` and includes
  `alpha_0 log(alpha_0)` in negative entropy.  Its leakage contribution is
  `alpha_0` exactly when the query itself belongs to the declared sink set;
  default collection excludes those queries.
- `attention_only_conditional` (sensitivity): condition on selecting an
  attention-bearing route and exactly renormalize
  `alpha_i/(1-alpha_0)`.  This is a different conditional sample space and is
  not presented as the main D.4 result.

At self-attention router `l`, attention-bearing depth branch `k+1` is associated
with token attention from decoder layer `k`, following the paper's
`alpha_i p_i` surrogate.  The implementation computes the quantities exactly
**under this predeclared operationalization**.  "Exact" here describes the
arithmetic after the definitions below are fixed; it does not mean that the
paper uniquely specifies these implementation choices.  The metric also does
**not** claim to recover causal token contributions to the routed hidden state:
historical hidden states contain residual and MLP computation, so causal
contribution would require intervention or attribution experiments.

Three protocol choices remain because the paper leaves them unspecified:

1. `N_t` is instantiated as BOS/first-token leakage, with prefix-4 as
   robustness.
2. The initial residual route is handled by the two variants above.
3. "Matched no-op tolerance" is instantiated as the same raw-norm threshold
   with every included AttnResidual attention branch required to satisfy D.1.

These are explicit operational definitions, not numerical approximations.
The point estimates and D.4 comparisons use literal formulas with zero
comparison tolerance.  The only deliberate approximation in the report is the
Monte Carlo sequence-cluster bootstrap used for confidence intervals; its
iteration count and seed are recorded, and the default is 10,000 iterations.

## Checkpoint loading

The loader reconstructs custom decoder layers before loading
`attn_res_sa`/`attn_res_mlp`.  It supports:

- final Hugging Face output directories containing `config.json` and model
  weights;
- Accelerate checkpoint directories containing model weights, provided
  `--base-model` identifies the original Phi-4 backbone.

For an AttentionResidual run, collection fails if any router parameter is
missing.  This guards against accidentally measuring randomly initialized
depth routing.  The paired report also verifies the backbone architecture
signature, number of layers and heads, dtype, autocast mode, tokenized input
hash, and sampling protocol.  These checks validate inference compatibility
and pairing only.  They do not verify the matched-training checklist or prove
the normalization provenance described above.

## Run

Run the analysis in the same `outlier` conda environment used by the supplied
Phi-4 Slurm training scripts, then install the one analysis-only dependency:

```bash
pip install -r analysis/requirements.txt
```

Do not use the repository's legacy root `requirement.txt` as evidence of the
active Phi-4 Transformers version: it pins `transformers==4.31.0`, while the
checked-in Phi-4 modules import newer Transformers APIs.  The validation job
must use the already working Phi-4 training environment; the Slurm script does
this by activating `outlier`.

From `OutEffHop/`, collect the two models with identical data and selection
arguments.  The following `--num-samples 32` setting is a pilot configuration,
not a default claim of confirmatory sample adequacy:

```bash
python analysis/d4_collect.py \
  --model-kind attn_residual \
  --base-model microsoft/Phi-4-mini-instruct \
  --checkpoint /path/to/attn_residual_checkpoint \
  --output /path/to/d4/attn_residual.parquet \
  --num-samples 32 \
  --sequence-length 256 \
  --layers all \
  --heads all \
  --analysis-variants identity_outcome attention_only_conditional

python analysis/d4_collect.py \
  --model-kind vanilla \
  --base-model microsoft/Phi-4-mini-instruct \
  --checkpoint /path/to/vanilla_checkpoint \
  --output /path/to/d4/vanilla.parquet \
  --num-samples 32 \
  --sequence-length 256 \
  --layers all \
  --heads all \
  --analysis-variants identity_outcome attention_only_conditional
```

Then build the paired report:

```bash
python analysis/d4_report.py \
  --attn-residual /path/to/d4/attn_residual.parquet \
  --vanilla /path/to/d4/vanilla.parquet \
  --output-dir /path/to/d4/report
```

The report directory contains:

- `d4_summary.csv`: one row per sink set, norm quantile, and metric;
- `d4_layer_coverage.csv`: retained pair/sequence counts and selection rates
  for every layer, so a pooled result dominated by a small depth range is
  visible;
- `d4_summary.json`: the same results plus complete run metadata;
- `d4_report.md`: a human-readable table.

Each collection also writes `<metrics>.metadata.json`.  The report refuses to
pair runs whose input token hash, selected layers/heads, or sampling protocol
differs.  Both metadata sidecars are required by default; the explicit
`--allow-missing-metadata` escape hatch should only be used for legacy tables
whose pairing has been verified independently.

For a quick smoke test without `pyarrow`, use output filenames ending in
`.csv`.  CSV is much larger and is not recommended for the full experiment.

## Recommended first run

Before the full job, use:

```text
--num-samples 2 --sequence-length 64 --layers 0,8,16 --heads 0,1
```

After this smoke test succeeds, use all layers and heads.  If storage or report
memory is limiting, keep all layers but use a preregistered head subset; do not
select heads after inspecting the result.

The script and submission wrapper default to 32 sequences so that an initial
end-to-end pilot is affordable.  Ten thousand bootstrap iterations do not turn
32 sequence clusters into a large confirmatory sample.  Before the formal run,
choose and record the sequence count without inspecting the D.4 result,
increase it as compute permits or justify a smaller pilot explicitly, and
report both the requested and retained sequence counts.  Statistical
uncertainty is driven by the number and diversity of independent sequence
clusters, not by the number of token/head/layer rows or bootstrap resamples.

The default confidence intervals use 10,000 sequence-cluster bootstrap
iterations.  Tokens, heads, and layers from the same sequence are therefore not
treated as independent samples.  At least two retained sequences are required
for a confidence interval; with fewer than two, the point estimate is retained
but the report marks the interval unavailable.

Always inspect `d4_layer_coverage.csv` before citing a pooled result.  The
AttnResidual no-op statistic is a maximum over an increasing number of
historical branches, so a single global raw-norm threshold can retain different
fractions at shallow and deep layers.  If coverage is concentrated in a small
layer subset, report that subset explicitly or run a preregistered
layer-stratified analysis rather than describing the result as network-wide.

## Release validation status

For this packaged release, Python syntax, Slurm syntax, report/metric unit
tests, end-to-end synthetic report generation, and archive integrity were
checked.  A real forward smoke test with the target Phi-4 checkpoints was not
available in the packaging environment because it lacked the required
PyTorch/Transformers/GPU/checkpoint runtime.

Before launching the full job, the cluster operator must therefore run the
two-sequence smoke test above with the actual matched checkpoints and verify:

- both collectors finish without missing observer or dtype errors;
- both metadata sidecars are written and the report accepts the pair;
- the row counts and selected layer/head sets are as requested;
- `d4_layer_coverage.csv` contains retained cases beyond only a narrow shallow
  layer range; and
- the recorded checkpoint paths and manually retained training provenance
  refer to the intended matched pair.

Only after this real-checkpoint smoke test succeeds should the full run be
treated as operationally validated.
