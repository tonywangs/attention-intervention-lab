# Observations from the frozen experiment

The three final seeds were trained and evaluated on CPU on 2026-10-11. No
architecture or training hyperparameter was adjusted using evaluation results.
Raw values, intervals and individual predictions are in
`results/evaluation/summary.json` and `seed-{11,29,47}.json`. The readable report
is `results/evaluation/report.html`.

## Learning and baselines

All selected models passed the 90% development gate: 512/512 development
examples correct per seed. All selected steps were 1000. Held-out accuracy was
also 512/512 for each seed. This is a successfully learned, intentionally simple
retrieval task; there is no failed final training seed to diagnose.

| Seed | Held-out cross-entropy | Mean P(correct answer) | Intact − no-op logits |
|---|---:|---:|---:|
| 11 | 0.0014519851 | 0.9985491998 | exactly 0 |
| 29 | 0.0013323390 | 0.9986686359 | exactly 0 |
| 47 | 0.0012972191 | 0.9987036962 | exactly 0 |

Uniform vocabulary predictions have expected accuracy 6.25%, cross-entropy
ln(16) ≈ 2.773, and P(answer) 0.0625. Uniform predictions over the four context
values have expected accuracy 25%, loss ln(4) ≈ 1.386, and P(answer) 0.25. Always
choosing the first context value obtains exactly 25% accuracy because positions
are balanced; its deterministic distribution gives infinite loss on wrong
answers (represented as null with a note). The dictionary oracle obtains 100%.

## Attention/effect correspondence and negative findings

Across eight heads, Spearman correlations between mean answer-position
attention and mean answer-probability drop were 0.642857 (seed 11), 0.666667
(seed 29), and 0.309524 (seed 47). There were no exact ties in these two rankings
and no undefined correlations in the observed runs. The implementation and tests
also cover tied ranks and undefined constant-rank correlations. These eight-head
correlations are descriptive, not significance tests or a pooled seed estimate.

The head with largest mean probability drop differed by seed:

| Seed | Head | Its attention rank | Mean P(answer) drop [paired 95% CI] | Accuracy drop [paired 95% CI], percentage points |
|---|---|---:|---|---|
| 11 | L0H2 | 5 | 0.086215 [0.065900, 0.107487] | 7.031 [4.883, 9.375] |
| 29 | L1H2 | 1 | 0.050196 [0.036867, 0.064436] | 3.125 [1.758, 4.688] |
| 47 | L0H0 | 5 | 0.020876 [0.012054, 0.031339] | 1.758 [0.781, 2.930] |

In seed 11, L1H0 has the highest mean relevant-value attention (0.944690), yet
zeroing L0H2, whose score is 0.187011, causes the largest observed drop. Seed 47
also has a first-layer head as the strongest intervention while the largest
attention score is in layer 1. Seed 29 is the contrasting case: L1H2 tops both
rankings. The results therefore do **not** support an exact universal ordering
of head usefulness by this attention score.

Many removals have no measured accuracy cost despite small answer-probability
decreases. No tested single head is indispensable for most examples: even the
most damaging removal retains 92.97% accuracy in seed 11. That does not mean
the model can lose many heads simultaneously or that the small-effect heads
have no function. Pair/group interventions and compensating pathways were not
measured. All observed mean probability drops were positive; no general claim
is made that head removal cannot help on another distribution or model.

First-layer heads may prepare value representations using earlier key tokens,
which the chosen final-query/value attention score would miss. This is a
**possible explanation**, not an experimentally established mechanism. Testing
it would require a separately specified patching or interaction experiment.
No such interpretation is inferred solely from a heatmap.

## Runtime and resource observations

Measured on Linux x86_64, CPU model string `DO-Regular`, one PyTorch thread,
CPython 3.12.3, PyTorch 2.6.0+cpu and NumPy 2.2.6:

| Phase | Elapsed seconds | Peak process RSS (KiB) |
|---|---:|---:|
| Development-only pilot, 1500 steps | 78.000 | 314648 |
| Three final seeds, 1000 steps each, including data export | 133.708 | 314816 |
| Final evaluation, summary and HTML export | 4.047 | 271752 |

These are actual measurements, not projected GPU or other-hardware timings.
Peak RSS is a process high-water mark including library imports; final training
RSS is shared across sequential seeds rather than independently reset for each.
Validation histories retain their own step timing. All runtime budgets were met.
The model contains 70,864 parameters, below the 250,000 parameter cap. Artifact
byte counts and SHA-256 hashes are preserved in each phase's manifest; the full
project is checked against 10 MiB/file and 32 MiB/1,000-file limits.

## Validation corrections and reproducibility

The independent audit detected a small float32 accumulation difference in mean
attention mass. Aggregation now promotes stored float32 observations to float64;
the corrected rerun retained byte-identical per-example predictions and attention
matrices. This correction did not change the architecture, training, inputs,
metric definitions or observed rank ordering. A second implementation change
validates checkpoint array headers before allocation, including maliciously
oversized declared shapes. `experiments/amendments.json` preserves both changes;
AST comparison proves the frozen forward-pass code is unchanged.

The report is deterministic for fixed evaluation artifacts. Timings, dependency
versions and RSS are deliberately in separate provenance JSON. The offline
installed replay requires all predictions to match and logits/probabilities to
stay within the frozen 2e-6 tolerances; the pinned environment additionally
requires byte-identical deterministic JSON and HTML. See `results/tests.log`
for the actual latest verification output.

These results concern a bounded synthetic distribution, not natural-language
reasoning. Perfect accuracy on 512 examples per model is not a population
accuracy guarantee. Bootstrap intervals condition on the trained model and
constructed examples, do not measure uncertainty across only three seeds, and
are not multiple-comparison-adjusted. Zeroing outputs is an intervention with
potentially out-of-distribution effects; it is not a complete causal explanation.
