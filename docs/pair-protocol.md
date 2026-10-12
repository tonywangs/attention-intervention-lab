# Exhaustive paired head interventions

This follow-up uses the three existing checkpoints, with no retraining or
checkpoint selection. The historical protocol, model source, input files,
checkpoints, report and results are retained byte-for-byte. The new protocol is
`experiments/interactions/protocol.json`, SHA-256
`23c45240eb2f2823b0bdd5952f90f710d8d5d5f71cb4ec8ba5b9c798d2aef3d8`.
Its adjacent freeze record was written before any new checkpoint evaluation.
The baseline file records the starting file hashes. It is a local preregistration,
not an independently timestamped public registration.

## Question and fixed design

Do joint removals have output effects that are not the sum of the individual
removals? All eight heads and all 28 unordered pairs are eligible, within and
across the two layers. Every saved model receives the same 512 new examples
under 38 conditions: intact, no-op, eight singles and 28 pairs. No pair is selected
or excluded using the new outcomes. Tables use the frozen lexicographic order.

Data use seed 8675309, the existing held-out dictionary hash partition, 128
examples per answer position, and canonical-dictionary exclusion against all
saved training, development and earlier evaluation inputs. Reorderings or
different queries on the same dictionary are also excluded. Generation derives
the label from its selected slot; a dictionary-lookup oracle checks it
independently. Saved inputs and SHA-256 hashes precede evaluation.

An intervention zeros each selected head's `z = softmax(QKᵀ / √d)V` at every
position immediately before the shared output projection. The implementation
uses a temporary pre-hook on that projection's flattened `[batch, token, H*D]`
input. It masks contiguous head slices and preserves the projection bias.
Cross-layer pairs run in one forward pass; downstream activations recompute.
The original model code is unchanged. Simultaneous does not mean freezing the
second layer to its intact activation. A clone-only hook is the no-op control.
Hooks always detach on exit; concurrent use of one hooked model is unsupported.

## Estimands and uncertainty

For each example and metric `m`, define effect `E(A) = m(A) − m(intact)` and
interaction `I(A,B) = m(A,B) − m(A) − m(B) + m(intact)`.
Metrics are cross-entropy loss, correct-answer probability and binary accuracy.
Positive loss effects are harmful; negative probability/accuracy effects are
harmful. No sign is renamed “synergy” or “redundancy.” Zero is metric-specific:
loss curvature, softmax and an accuracy threshold can create different results.

The protocol freezes 10,000 paired stratified bootstrap replicates (NumPy PCG64,
seed 20261012). Each replicate draws 128 examples with replacement separately
within each of the four answer-position strata. Identical indices apply to all
conditions, pairs and metrics. This conditions on the balanced stratum counts.
Report pointwise 95% percentile intervals for effects and interactions.

For each model, the multiple-comparison family comprises all 84 pair × metric
interactions. Calculate each interaction's bootstrap standard error (sample SD,
ddof=1). In each replicate take the maximum absolute centered bootstrap mean
divided by its standard error, omitting SE ≤ 1e-12. Its 95th percentile `q`
gives simultaneous intervals `observed mean ± q * SE`. Quantiles use linear
interpolation. An interaction is marked resolved only if nondegenerate and its
family interval clears zero by more than 2e-6. This is an approximate bootstrap
procedure, not exact familywise coverage; sparse binary changes can be poorly
estimated. Degenerate contrasts remain descriptive even with collapsed intervals.
Unadjusted single/joint intervals must not be used for multiplicity-adjusted
claims. No pooled estimate, head alignment, or inference across training seeds
is supported, and the three seed families do not provide a global guarantee.

The descriptive “hidden accuracy” flag requires both individual accuracies to
equal intact accuracy exactly and joint accuracy to be lower. It does not assert
population equality of the individual effects, a statistically resolved pair,
or a compensating computational role.

## Reproduction and bounds

Evaluation has a maximum two-hour wall-clock budget and one Torch CPU thread,
with 128-example batches. The reproduction wrapper also sets BLAS/OMP threads
to one. Original saved predictions are replayed before the new experiment;
logit absolute/relative tolerances are 2e-6. No-op equivalence is bitwise.
Independent aggregation tolerance is 1e-10. New artifacts preserve float32 logits
and float64 per-example outcomes, condition order, inputs, protocol hashes,
source hashes, environment, elapsed time, peak process RSS and file sizes.

Output collisions fail without overwriting. SIGINT/SIGTERM and ordinary failures
remove staged output. A hard kill can leave an empty reservation or staging
directory; there is no automatic resume. Interrupted runs are not successful
results and should be reported separately. The report embeds only the fixed
first 16 examples per seed and all aggregate pairs; the raw arrays retain all
512 examples. Publication limits remain 10 MiB/file, 32 MiB/tree and 1,000 files.

## Prior work and scope

[Michel, Levy and Neubig (2019)](https://arxiv.org/abs/1905.10650)
study removal of attention heads and demonstrate substantial pruning tolerance.
Their [released implementation](https://github.com/pmichel31415/are-16-heads-really-better-than-1)
is relevant prior art for head-importance and pruning experiments. This project
does not claim head ablation or exhaustive pair comparison is novel.

[Wang et al. (2022), §3.4 and §4](https://arxiv.org/html/2211.00593v1)
describe backup name movers and evaluate faithfulness, completeness and
minimality. Their circuit analysis uses more evidence than a pairwise output
contrast. They also discuss the problems of zero ablation and use mean ablation.
Our zero-ablation continuation preserves the original intervention definition;
it does not reproduce their task, circuit or mean-ablation design.

[McGrath et al. (2023)](https://arxiv.org/html/2307.15771v1)
study compensating downstream computations after attention-layer ablation in
language models. This motivates looking beyond isolated interventions. A
non-additive pair here would not by itself establish the Hydra effect: direct
activation/path evidence and alternative interventions would be needed.

The [pinned TransformerLens attention implementation](https://github.com/TransformerLensOrg/TransformerLens/blob/b5a16f849649a237cc02cc2c272ae4dc2085abe4/transformer_lens/components/abstract_attention.py)
exposes `hook_z` and subsequently flattens head dimensions before applying
`W_O` plus `b_O`. We inspected that implementation and use the equivalent
intervention location in this much smaller model. TransformerLens is not a
runtime dependency. These sources were reviewed before evaluation on 2026-10-12.

This experiment measures interventions on a fixed synthetic task. It cannot
establish a complete mechanistic explanation, natural-language generalization,
or what would happen under mean ablation, activation patching or retraining.
