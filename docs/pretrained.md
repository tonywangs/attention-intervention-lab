# Pretrained attention interventions on public text

This example answers a bounded question: **for each head of a pinned Pythia-14M
checkpoint, how does replacing its output change next-token loss on a short
public-text excerpt, and what attention patterns accompany those effects?**
The saved report links the same head, example and query position to measured
losses; it is not a synthetic stand-in or a claim to explain a complete circuit.

## Launch the saved report

Open [the self-contained HTML](../results/pretrained/report.html) in a browser,
or from the repository root run:

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory results/pretrained
```

Visit <http://127.0.0.1:8765/report.html>. No inference, download, API key, GPU or
internet access is needed to inspect saved results. Select a head, block,
replacement and query position. The heatmap and numeric attention table show
intact attention; the adjacent loss display shows the actual next-token target
and measured intervention effect. Clicking a row in the all-head table selects
that head. The first two evaluation blocks are visualized; aggregates use all 16.

## Reproduce on CPU

From the repository root, with Python 3.12 on Linux:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-pretrained-lock.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m attention_lab.pretrained fetch
.venv/bin/python -m attention_lab.pretrained run --output /tmp/attention-pretrained-replay
.venv/bin/python scripts/verify_pretrained.py --replay /tmp/attention-pretrained-replay
.venv/bin/python -m pytest -q tests/test_pretrained.py
```

Choose an output directory that does not exist. The first command needing model
access is `fetch`: it downloads six public, revision-pinned files with standard
HTTPS, checks each SHA-256, and stores them in ignored `.cache/pretrained`.
It does not search for hub credentials. Weights are safetensors; no pickle or
remote model code is loaded. Model loading explicitly uses local files only.
After dependency installation and `fetch`, the experiment and numerical checks
run offline. Model weights and tokenizer together need about 30 MB of cache;
the environment and inference memory need additional space. The tracked tree
stores neither the large weights nor a second copy of the tokenizer.

The observed run used CPython 3.12.3, Torch 2.6.0+cpu, Transformers 4.51.3,
NumPy 2.2.6, float32, eager attention, two CPU threads, batch size two, seed 1729,
deterministic Torch algorithms, evaluation mode and no KV cache. The initial run took
85.69 seconds including local loading, calibration and all intervention passes,
excluding downloads and report rendering. Timing is machine-specific, not a
performance guarantee. Full dependency versions are pinned in the additional
lock file. CPU numerical differences on other builds may exceed the replay
check's `atol=2e-6, rtol=1e-5`; this is not a cross-platform bitwise guarantee.

Optional browser checks (install Chromium once):

```sh
PLAYWRIGHT_BROWSERS_PATH=/tmp/attention-browsers .venv/bin/python -m playwright install chromium
PLAYWRIGHT_BROWSERS_PATH=/tmp/attention-browsers .venv/bin/python -m pytest -q tests/test_pretrained_browser.py
```

## Model and input provenance

Model: [EleutherAI/pythia-14m](https://huggingface.co/EleutherAI/pythia-14m),
revision `cf967c0a9a04383db6f7b1108d86b2962634b4ac`, Apache-2.0.
The [pinned model card](https://huggingface.co/EleutherAI/pythia-14m/blob/cf967c0a9a04383db6f7b1108d86b2962634b4ac/README.md)
describes a six-layer, four-head model, hidden width 128, pretrained on the Pile.
This is the standard-Pile model currently at that revision; it is not the older
deduplicated model previously hosted under the same name. Pinning is material.
The checkpoint is used unchanged, with its stored weights cast to float32.

Calibration: Lewis Carroll, *Alice’s Adventures in Wonderland*,
[Project Gutenberg #11](https://www.gutenberg.org/ebooks/11).
Evaluation: Arthur Conan Doyle, *The Adventures of Sherlock Holmes*,
[Project Gutenberg #1661](https://www.gutenberg.org/ebooks/1661).
These works are public domain in the USA; the source pages state the conditions
for other jurisdictions. The saved plain-text excerpts contain the literary
text, without Gutenberg headers, branding or license boilerplate.

`experiments/pretrained/sources.json` records original download URLs, retrieval
times, full raw-download hashes, normalization, character offsets and excerpt
hashes. Each saved 12,000-character excerpt begins at the specified prose marker.
Original bytes were decoded as UTF-8 and CRLF changed to LF; no other
normalization was applied. Tokenize each saved excerpt with the pinned tokenizer,
without special tokens. Take its first 1,040 tokens; split sequentially into 16
non-overlapping 65-token blocks. The run verifies this against `inputs.json`.
Positions 0–63 are inputs; positions 1–64 are targets. All 64 losses count,
including first-position predictions with one-token context. No BOS, padding,
random sampling or cross-block context is used. Prediction across block
boundaries is omitted. Reproduction uses preserved excerpts, not a mutable
live download. Calibration and evaluation are different books, but **neither
is known to be absent from pretraining**.

`protocol.json` fixed the workload, selection, measures and limitations before
outcomes were computed. Its SHA-256 is
`b9973e5a246f4f166d8b6c0ed20481ea6c84c87e34f710c0b5595d44b2f7f29b`.
This is a locally recorded design, not an externally timestamped preregistration.

## Intervention definition and controls

For each of 24 heads independently, replace its attention-weighted value vector
at all 64 positions, just before the shared attention output projection:

- **Zero:** set the selected 32-component head slice to zero.
- **Mean:** replace it with that head's mean intact activation over all 1,024
  calibration positions. Accumulate in float64, then store the mean in float32.
  This is one vector per head, pooled over positions, including position zero.
- **No-op:** clone the same projection input and reassign the original head slice.
  Every per-token loss must exactly match the intact forward pass.

All other heads, the projection bias, residual paths and MLP remain intact at
the intervention site. Later computations can change. Calibration runs the
intact model on a separate book; evaluation never updates means. Effects are
intervention minus intact cross-entropy in natural-log units, averaged over
1,024 target tokens. The mean-minus-zero contrast uses the same evaluation
inputs. Negative deltas mean improved next-token prediction under replacement.

The hook location follows the inspected Transformers v4.51.3
[GPT-NeoX implementation](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/gpt_neox/modeling_gpt_neox.py):
attention output is transposed to batch/query/head/dimension, flattened across
head/dimension, and passed to `attention.dense`. Tests independently compare
zero replacement to independently zeroing the selected value-projection vector
before attention weighting, and mean replacement to
adding `W_head (mean − intact_head)` *after* projection, preserving bias
(in float64 for the algebraic check). The pinned library's public `head_mask` path converts masks to additive values
but multiplies them into attention probabilities, producing NaNs; it is not
used by this workbench or as a valid numerical reference. A
separate one-block-at-a-time calibration check tests the pooled means, and
manual log-softmax/gather checks next-token alignment. Hook removal is checked
on exceptions. Saved-result auditing independently recomputes every summary
from per-token arrays and verifies the HTML contains those same measurements.

## Results and interpretation

Intact mean loss: **4.615252 nats/token**. All 24 no-op controls had **exactly zero**
change in every token loss. Largest degradation for both methods was L0H1:
**+0.889919** nats/token with zero and **+0.755246** with mean replacement.

| Head | Zero Δ nats/token | Mean Δ nats/token | Descriptive observation |
|---|---:|---:|---|
| L0H1 | +0.889919 | +0.755246 | Largest loss increase for both methods |
| L2H3 | +0.064313 | −0.022846 | Sign reverses under mean replacement |
| L3H3 | +0.018804 | −0.000203 | Sign reverses, mean effect near zero |
| L3H1 | −0.000546 | −0.003811 | Both replacements slightly improve loss |
| L5H3 | +0.001203 | +0.000494 | Both effects near zero |

All 24 means produced smaller loss increases (or greater decreases) than zero
replacement. Zero replacement improved the aggregate for one head; mean
replacement improved it for three. Two heads had opposite aggregate signs.
These are descriptive observations, with no significance threshold. The full
table and per-block deltas are in the report and `summary.json`, not just the
highlighted rows above. There is no population confidence interval: adjacent
blocks from a single book are dependent, and calibration uncertainty is not
estimated. Near-zero deltas must not be interpreted as proof of no function.

The workbench demonstrates a reproducible link from **actual pretrained
attention** to **actual intervention measurements**. It also shows why a head's
consequence depends on the replacement definition. This does not show that
attention patterns fully explain the model, that the observed functions persist
on other text, or that a head whose replacement helps is universally harmful.
A different calibration corpus can change mean-ablation effects. More models,
texts, calibration samples and interaction experiments would be needed to
support broader mechanistic claims.

## Artifacts and prior work

- `experiments/pretrained`: model hashes, immutable local design, exact token
  blocks, source metadata and the two preserved excerpts.
- `results/pretrained/measurements.npz`: intact losses `[16,64]`; zero/mean/no-op
  losses `[6,4,16,64]`; calibration means `[6,4,32]`; intact attention
  `[2,6,4,64,64]`, all float32. Load with `numpy.load(..., allow_pickle=False)`.
- `summary.json`, `examples.json`, `provenance.json`, `manifest.json`: readable
  metrics, visualized text/token IDs, actual environment and integrity records.
- `report.html`: self-contained interactive view, attention rounded to six
  decimals for display; full precision remains in NPZ.

Sources inspected on 2026-10-19 include the model card, Gutenberg source pages,
the pinned Transformers implementation, and
[BertViz](https://github.com/jessevig/bertviz), which provides established
interactive attention views. [Earlier related-work notes](related-work.md)
discuss AttentionViz, head ablation and TransformerLens's pre-projection hooks.
This is a small integration and reproducibility contribution with **no novelty
claim**. Historical synthetic experiments remain available and unchanged.
