# Attention Intervention Lab

An offline CPU workbench for comparing **where attention goes** with **what
changes when a head's output is removed**. It includes a finished synthetic
key-value retrieval experiment, three trained checkpoints, per-example results,
and self-contained interactive reports. No API key, GPU, model download or
server is needed to explore the saved report.

**Latest: [zero versus mean-ablation comparison](results/mean/report.html).**
The exhaustive follow-up uses 512 separate calibration examples and 512 fresh
evaluation examples, without retraining. Of seed 29's two previously highlighted
accuracy interactions, L1H0 + L1H3 remains resolved under mean ablation;
L0H0 + L0H2 is smaller and unresolved. Neither pair has a resolved paired method
difference. All 28 pairs and all three models are reported, including null results.
See [results and limitations](docs/mean-results.md) and [the frozen design](docs/mean-protocol.md).

```sh
.venv/bin/python scripts/reproduce_mean.py --output /tmp/attention-mean-reproduction
.venv/bin/python scripts/verify_mean.py --wheelhouse /tmp/attention-wheelhouse --browsers /tmp/attention-browsers
```

The first command reproduces this experiment offline into a new directory. The
second retains all historical checks and adds independent calibration, replacement,
statistics, malformed-artifact, cancellation, Chromium and isolated installed-CLI
checks. Setup is below. The separate `attention-mean` CLI provides `prepare`,
`evaluate`, `replay` and `report`; historical `attention-lab` behavior is preserved.

**Previous: open [the paired-intervention report](results/interactions/report.html).**
It evaluates all 28 head pairs on 512 fresh, dictionary-disjoint examples in
each saved model. Pair interactions expose effects that individual removals
miss, with substantial differences between models. In seed 29, two pairs
reduced accuracy by about five percentage points while both single removals
left accuracy unchanged; their interaction intervals remained below zero after
the preregistered within-model comparison adjustment. Seed 47 had four such
descriptive pairs but none resolved after adjustment; seed 11 had none.
See [the full paired results and limits](docs/pair-results.md) and
[the frozen design and prior work](docs/pair-protocol.md). This does not establish
backup-head roles or a complete mechanism.

Reproduce the paired experiment offline, after setup, with one command:

```sh
.venv/bin/python scripts/reproduce_pairs.py --output /tmp/attention-pair-reproduction
```

The output path must not already exist. The command regenerates the frozen
new inputs, replays historical predictions, evaluates all singles and pairs,
and compares saved logits, outcomes, statistics and HTML. No model is retrained.

**Open [results/evaluation/report.html](results/evaluation/report.html) locally
in a browser.** It offers seed/example/head selection, attention heatmaps and
numeric tables, intact-versus-ablated predictions, all output probabilities,
and paired uncertainty. The first 16 examples per seed were fixed before
final evaluation; aggregate results use all 512 held-out examples.

## Observed result

All three 70,864-parameter models achieved 100% development and held-out accuracy.
The association between final-query attention on the correct value and the
probability drop from removing a head was positive but variable:

| Training seed | Held-out accuracy | Attention/effect Spearman ρ | Largest probability-drop head | Its attention rank |
|---|---:|---:|---|---:|
| 11 | 512/512 | 0.643 | L0H2 | 5 of 8 |
| 29 | 512/512 | 0.667 | L1H2 | 1 of 8 |
| 47 | 512/512 | 0.310 | L0H0 | 5 of 8 |

In two seeds, the most consequential single head by this metric was **not** the
head with highest answer-position attention. Many individual removals left
accuracy unchanged while reducing answer probability. These are descriptive
results for a simple synthetic task, not evidence of a complete retrieval
mechanism or a general rule about language models. See [results](docs/results.md),
[protocol](docs/protocol.md) and [related work](docs/related-work.md).

## Setup

Tested on Linux x86_64, CPython 3.12.3, PyTorch 2.6.0+cpu, NumPy 2.2.6 and
Chromium 134.0.6998.35. Install dependencies once with network access:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
PLAYWRIGHT_BROWSERS_PATH=/tmp/attention-browsers .venv/bin/python -m playwright install chromium
.venv/bin/python -m pip download --only-binary=:all: --dest /tmp/attention-wheelhouse -r requirements-lock.txt
```

A Python installation with `venv`/`ensurepip` support is needed for this setup.
Chromium also needs its usual operating-system shared libraries. The browser
and wheel cache are not project artifacts and are not committed. The cache is
specific to the OS/Python architecture; prepare a matching cache for another
machine. Dependencies use roughly 1 GiB installed disk space; the tracked
experiments are about 27 MiB. The CPU wheel index avoids CUDA dependencies.

## One verification command

After setup, this command runs completely offline:

```sh
.venv/bin/python scripts/verify_pairs.py --wheelhouse /tmp/attention-wheelhouse --browsers /tmp/attention-browsers
```

It runs 240 seeded scalar-reference attention checks plus full-model head/layer
address checks; malformed-artifact, budget, cancellation and collision tests;
Chromium navigation/keyboard/hostile-text/mobile/bounded-rendering tests with
networking blocked; independent recomputation of all reported statistics; and
an installed-CLI replay. The latter builds a wheel, installs it with local
wheels into a fresh isolated environment, runs outside the checkout with Python
network connections blocked, and compares all three checkpoint evaluations.
It requires identical predictions and, on the pinned environment, byte-identical
result JSON and HTML. The paired suite additionally runs 224 seeded simultaneous
placement cases and 224 scalar interaction-arithmetic cases, audits every pair
statistic independently, exercises all seed/pair/metric table values in Chromium,
and replays the installed pair CLI. New verification prints to stdout so the
historical `results/tests.log` stays unchanged.

The verification never retrains the three models and uses temporary directories
for replay. It normally takes a few minutes, primarily installing the isolated
CPU environment and exercising Chromium. A failed required check exits nonzero.

## CLI workflow

The package installs `attention-lab`. Choose **new** output paths; every command
refuses to replace an existing directory.

```sh
# Re-evaluate saved checkpoints and compare against the preserved results.
.venv/bin/attention-lab replay --training experiments/final --reference results/evaluation --output /tmp/attention-replay

# Evaluate without a reference comparison, or regenerate just the report.
.venv/bin/attention-lab evaluate --training experiments/final --output /tmp/attention-evaluation
.venv/bin/attention-lab report --input results/evaluation --output /tmp/attention-report

# Optional: reproduce training using the already-frozen protocol.
.venv/bin/attention-lab train --protocol experiments/protocol.json --output /tmp/attention-training
.venv/bin/attention-lab evaluate --training /tmp/attention-training --output /tmp/attention-new-results
```

For the fully isolated installed workflow alone:

```sh
.venv/bin/python scripts/installed_cli.py --wheelhouse /tmp/attention-wheelhouse
```

`--max-seconds` on `evaluate` and `replay` can lower the 600-second evaluation
budget. Training has a shared three-hour cap; the retained development-only
pilot had a 30-minute cap. SIGINT/SIGTERM clean up uncommitted output. A hard
kill or power failure can leave an empty reserved directory or hidden staging
directory; remove only that incomplete output before retrying. There is no
scheduler, cloud service, website deployment or automatic downloading at runtime.

## What's preserved

- `experiments/protocol.json`: frozen architecture, data rules/seeds, training
  seeds, checkpoint selection, metrics, intervention, subset and budgets.
- `experiments/pilot/`: development-only pilot checkpoint and all validation
  observations; `experiments/final/`: three selected checkpoints, training/dev
  input shards, development histories, versions, runtimes and RSS.
- `results/evaluation/`: held-out inputs; all 15,360 condition/example results
  with 16-way logits; attention scores for all examples and full matrices for
  the fixed subset; summaries, HTML, provenance and file manifests.
- `experiments/amendments.json`: transparent post-freeze implementation fixes
  (bounded checkpoint headers and float64 attention-score aggregation), with
  the original model source preserved. No scientific protocol or training
  settings changed. Verification checks forward-pass identity.
- `scripts/audit_results.py`: independent scalar-math probability/loss/rank
  calculations and bootstrap multiplicity-weight recomputation.

Checkpoint loading uses numeric NPZ arrays with prevalidated shapes, types and
sizes, never pickle. The dataset oracle uses dictionary lookup, independent of
the transformer. SHA-256 manifests detect accidental changes, but do not
establish authenticity if both data and manifest are replaced.

## Limits

Only four key-value pairs, 16 keys/values, one fixed sequence length and three
training seeds are tested. Heads are not aligned across seeds. Zero and mean ablation
can create unusual activations; patching and natural-language generalization
remain untested. The original experiment's bootstrap intervals are paired
within one model, conditional on this constructed test distribution, and not
adjusted for multiple comparisons. A perfect finite test score does not imply
perfect population accuracy. The report was exercised in Chromium, not every
browser or assistive technology. Exact replay is verified in the pinned CPU
environment; other platforms may differ within the frozen numerical tolerances.

The historical paired follow-up tests zero-ablation interactions only. The new
mean-ablation comparison is separate; activation patching remains untested. Its family intervals account for all
84 pair-by-metric interactions **within** one model, using an approximate paired
stratified bootstrap. They provide no inference across training seeds. Constant
or sparse binary contrasts need particular caution. The original single-head
experiment and its unadjusted intervals remain unchanged.

This project makes no novelty claim. [AttentionViz](https://arxiv.org/abs/2305.03210),
[Michel et al.'s head-ablation work](https://arxiv.org/abs/1905.10650v3) and the
[pinned TransformerLens implementation](https://github.com/TransformerLensOrg/TransformerLens/blob/b5a16f849649a237cc02cc2c272ae4dc2085abe4/transformer_lens/components/abstract_attention.py)
provide relevant prior work. See the related-work note for the precise scope.
