# Mean-ablation comparison results

Several paired effects survive position-specific mean ablation, while others
weaken. The result depends on both the pair and the trained model. This is not
evidence that all observed interactions represent backup heads.

Open [the offline comparison](../results/mean/report.html). All 28 pairs, all
three metrics and every model are available with full-precision numeric evidence.
The [frozen protocol](mean-protocol.md) defines the replacement distribution,
location, comparisons and uncertainty before new evaluation.

## The two previously highlighted pairs

Seed 29 previously had two adjusted accuracy interactions with unchanged single
removal accuracies. On these **fresh** examples, both individual accuracies remain
100% under both methods for these pairs. The following values are accuracy
percentage points; intervals belong to the new 252-contrast within-model family.

| Seed 29 pair | Zero interaction [95% family interval] | Mean interaction [95% family interval] | Mean − zero [95% family interval] |
|---|---:|---:|---:|
| L0H0 + L0H2 | −4.883 [−8.650, −1.116] | −2.344 [−5.015, 0.327] | 2.539 [−1.316, 6.394] |
| L1H0 + L1H3 | −5.078 [−8.847, −1.309] | −5.078 [−8.850, −1.306] | 0.000 [−1.538, 1.538] |

The first mean interaction is unresolved after adjustment. Its paired difference
is also unresolved: loss of resolution is **not** evidence that the effect
changed. The second retains a resolved negative mean-ablation interaction, with
no resolved method difference. Equal aggregate means do not imply identical
per-example outcomes. These pairs are contextual examples; all pairs were tested
and reported, without a special post-selection family.

## Exhaustive results

All intact models achieved 512/512 correct; clone-only no-ops matched logits
bitwise. Counts below are resolved contrasts out of 28 for each metric, using
one approximate family over zero, mean and difference contrasts jointly.

| Seed | Method/contrast | Loss | Correct-answer probability | Accuracy |
|---|---|---:|---:|---:|
| 11 | Zero | 14 | 16 | 7 |
| 11 | Mean | 9 | 15 | 6 |
| 11 | Mean − zero | 7 | 8 | 3 |
| 29 | Zero | 12 | 16 | 10 |
| 29 | Mean | 10 | 13 | 6 |
| 29 | Mean − zero | 4 | 5 | 1 |
| 47 | Zero | 10 | 13 | 5 |
| 47 | Mean | 7 | 12 | 3 |
| 47 | Mean − zero | 5 | 7 | 3 |

These counts summarize this sample, not the number of causal mechanisms. Heads
are not aligned across seeds. Zero results use a new sample and a larger family
than the historical paired report, so resolution flags need not match it.

## Reproduction and evidence

After the README setup, run:

```sh
.venv/bin/python scripts/reproduce_mean.py --output /tmp/attention-mean-reproduction
```

Choose a new output directory. The command regenerates both disjoint input sets,
re-estimates fixed calibration means from intact checkpoints, evaluates both
methods, and demands byte-identical deterministic artifacts in the pinned CPU
environment. No training, API access, or model download occurs.

`results/mean/` contains full float32 logits and float64 outcomes for 38 conditions
per method (including repeated intact/no-op controls), all examples, three models,
and the position-specific calibration means. `summary.json` gives all estimates,
pointwise and simultaneous intervals, and degenerate/resolved flags. Inputs,
protocol, file SHA-256 manifests, source hashes, environment, measured wall time,
peak process RSS and artifact sizes accompany the results.

The initial evaluation took **38.990 seconds**, with **367,932 KiB peak process
RSS** (Linux process high-water mark, not incremental experiment memory).
Reproduction includes extra validation and may take longer. The CPU evaluation
cap is 7,200 seconds. Historical protocols, checkpoints, inputs and results
remain byte-identical.

Validation includes 224 seeded float64 placement/order/weight/arithmetic cases,
an independent intact-calibration calculation, direct independent attention
replay of every prediction, scalar outcome calculations, and an independent
index-gather bootstrap recomputation of every summary. A projection-space
float32 reference initially exceeded the frozen tolerance because its algebraic
rearrangement changes accumulation. The tolerance was not relaxed: full replay
uses direct activation replacement with independent head indexing, while the
projection-space method remains the float64 placement reference.

Report input validation was hardened after evaluation. The original renderer is
preserved under `experiments/mean/frozen-renderer/`, and the rendering amendment
records both source hashes. A mobile browser check subsequently found 39-pixel selectors; their minimum
height was increased to 44 pixels and the report regenerated. Both report hashes
are retained in the amendment. Protocol, calibration, predictions, summaries and
numerical tolerances did not change.

## Limits and next experiment

These synthetic models have one sequence template and only eight heads. Mean
replacement can still create unusual activations, retain constant task
information, and induce downstream compensation. Interactions can arise from
nonlinear metrics, normalization and decision thresholds, without identifying
redundancy or a complete circuit.

Intervals condition on one fixed calibration set and the four query-position
counts. They omit calibration uncertainty and training variability. Sparse
binary contrasts and degenerate intervals require caution; unresolved estimates
are retained and do not establish absence. No pooled claim across seeds is made.
The browser checks cover Chromium, not all browsers or assistive technologies.

A useful follow-up would freeze multiple independent calibration samples and
compare the variability of the same fixed contrasts, or use controlled activation
patching to test a specific information-flow explanation. That would be a new
experiment, not a reinterpretation of these intervals.
