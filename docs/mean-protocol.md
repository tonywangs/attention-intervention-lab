# Position-specific mean ablation

This follow-up asks whether exhaustive paired-head non-additivity depends on
using zero as the replacement. It makes no novelty claim. All eight heads and
28 unordered pairs in each of three saved models are included; none are chosen
using new outcomes. Historical artifacts remain immutable.

## Prior work and choice of replacement

[Wang et al., §2.1](https://arxiv.org/html/2211.00593v1) replace head outputs
with averages over a reference distribution and group by template to preserve
position-related information. Their ABC distribution differs from IOI; this
experiment instead uses the same synthetic retrieval distribution for calibration
and evaluation. Consequently its means may retain task-specific constant information.

The [ARENA implementation](https://github.com/callummcdougall/ARENA_3.0/blob/main/chapter1_transformer_interp/exercises/part41_indirect_object_identification/solutions.py)
was inspected on 2026-10-18 (source file SHA-256
`7c26c39bd667f6ae46a0d8e465950631f2c5bbea403c2c536cf0138c4ccce499`):
`compute_means_by_template` caches intact `z`,
reduces over the example dimension within templates, and `hook_fn_mask_z`
substitutes those means. This project independently implements that general
operation using PyTorch pre-projection hooks; it does not copy that code or
require TransformerLens. All sequences here share one fixed ten-token template.

[Zhang and Nanda](https://arxiv.org/abs/2309.16042) show that methodological
choices can change interpretability findings. That motivates reporting three
metrics and a paired difference between replacement methods, rather than
interpreting changed significance flags as a method difference.

## Frozen design

The machine-readable specification is `experiments/mean/protocol.json`;
`freeze.json` records its SHA-256 and the catalog tree before new input generation
or inference. `baseline.json` hashes the pre-change tracked files. Every old
file under experiments and results must remain byte-identical.

Generate 512 calibration and 512 evaluation examples with separate frozen seeds,
balanced across four query-value positions, in the historical eval hash bucket.
Reject all canonical dictionaries from training, development, both historical
evaluation sets, and previously accepted examples. Calibration and evaluation
are mutually dictionary-disjoint. The positional generator is checked with a
separate dictionary-lookup oracle, and the audit independently checks labels.

For each saved checkpoint independently, capture intact z immediately before
shared W_O. Sum across calibration examples in float64, divide by 512, cast once
to float32, preserving dimensions layer × absolute position × head × channel
(2 × 10 × 4 × 16). Do not group by label or query slot. These same fixed means
replace each selected head at **all** positions for every single and pair.
Shared projection bias and weights remain unchanged. Two selected heads are
intervened upon in one forward pass; downstream activations recompute normally.
No downstream means are recalibrated after an upstream intervention.

Run intact and clone-only no-op plus eight singles and 28 pairs for each method
on identical examples (74 unique conditions; intact/no-op are repeated for each
method as checks). Use one CPU thread, batches of 128 and a two-hour wall-clock
evaluation cap, including calibration and summaries. No retraining.

## Estimands and uncertainty

For loss, correct-answer probability and accuracy, report condition − intact.
For each example and pair, interaction is joint − single A − single B + intact.
The method difference is mean interaction − zero interaction. Positive loss or
negative probability/accuracy interactions indicate excess harm. Loss curvature,
softmax, normalization, downstream computation and decision thresholds can all
produce non-additivity without backup behavior.

Use 10,000 paired bootstrap draws, sampling 128 examples with replacement in
each position stratum. Identical draws serve every method, condition, pair and
metric. Report percentile pointwise intervals and centered standardized maximum
intervals over **252 contrasts per model** (28 × 3 × zero/mean/difference).
Bootstrap SE uses ddof=1; quantiles use linear interpolation. A contrast resolves
only if nondegenerate and its simultaneous interval lies beyond ±2e-6.
SE ≤1e-12 is degenerate: retain the estimate and interval, but never declare it
resolved. Single/joint effect intervals are descriptive and unadjusted.

These intervals are approximate, conditional on the fixed calibration means,
the four stratum counts and this checkpoint. They do not include calibration
sampling or training variability, and provide no joint inference across models.
Sparse accuracy contrasts can yield poor bootstrap coverage. Null, disappearing,
and inconsistent findings are valid outcomes. Historical adjusted intervals use
a smaller family and a different sample; their resolution flags are contextual.

No-op logits must match bitwise. Independent numerical reference checks use
2e-6 absolute/relative tolerance for float32 logits and outcomes; summary audits
use 1e-10 absolute tolerance. Pinned-environment replay additionally requires
byte-identical deterministic files. Other architectures may differ numerically.
