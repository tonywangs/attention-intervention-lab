# Experimental protocol

The machine-readable frozen specification is `experiments/protocol.json`.
The development-only pilot, including unsuccessful checkpoints if any, is in
`experiments/pilot/`. The freeze record includes source and pilot hashes, so the
training and evaluation artifacts can be linked to the decision made before
final evaluation. Data-validation unit tests may inspect held-out synthetic
inputs and labels; no held-out model performance is used for model selection.

## Task and splits

An input has four distinct keys (from 16), four distinct values (from 16),
interleaved as `K V K V K V K V QUERY K`. The label is the value assigned to the
last key. An independent dictionary lookup validates all examples. Labels are
16-way value classes, not next-token prediction over the 33 input-token types.
Matching value positions cycle through 1, 3, 5, 7. The model receives only the
token sequence, never the answer or matching position.

A SHA-256 bucket of the sorted dictionary pairs assigns each mapping to train
(0–79), development (80–89), or evaluation (90–99). Reordering a context or
querying a different key cannot move a dictionary between splits. Rejection
sampling also removes duplicate dictionaries within each dataset. Sampling
uses Python's seeded `random.Random`; replay uses the saved inputs and hashes.
There are 16,384 training, 512 development, and 512 evaluation examples. All
three seeds use identical splits. Evaluation examples are not used for gradient
updates, stopping, hyperparameters, checkpoint selection or export selection.

Baselines: a uniform distribution over 16 values; a uniform distribution over
the four context values; always predict the first context value; and the
symbolic dictionary oracle. Uniform baselines report expected accuracy and
proper loss/probability, not sampled predictions. The latter oracle has direct
access to the generative rule and is a correctness ceiling, not a fair learned
model comparison.

## Selection and training

A single two-layer, four-head, width-64 pre-LN causal transformer with learned
positions, GELU feed-forward width 128 and no dropout is piloted for at most
30 minutes. It uses seed 7, only training/development data, AdamW (learning rate
0.001, weight decay 0.01), batch 128, global gradient clipping at norm 1 and
validation every 100 steps. The final run uses seeds 11, 29 and 47. The number
of steps is frozen after the pilot; it cannot be changed from evaluation
results. Checkpoint selection maximizes development accuracy, breaks ties by
lower development cross-entropy, then keeps the earlier step. Only selected
checkpoints are retained; every validation point is recorded. Training uses
one CPU thread and deterministic PyTorch operations, with a shared three-hour
wall-clock training budget. No GPU or paid inference is used.

Each model must independently reach 90% development accuracy before any
interpretation of a learned retrieval mechanism. If a seed fails, its held-out
results are retained and labeled **learning-floor diagnosis**. No test-driven
retuning is permitted. Failure diagnosis compares the initial/selected dev
scores, learning curves, held-out performance, position slices and baselines;
it does not invent a mechanism from failed learning.

## Interventions and measurements

For each selected checkpoint, evaluate intact execution, a no-op clone of z in
all layers, and each `(layer, head)` zeroed individually at **every position**.
The intervention sets the head's weighted-value output to zero immediately
before head concatenation and the output projection. The shared output bias,
all other heads, residual stream, MLPs and model weights stay unchanged.
Downstream attention may change; descriptive attention always comes from the
intact run. All models are in evaluation mode. Argmax ties choose the lowest
value class. Evaluate in fixed batches of 128.

Save per-example logits, probabilities, prediction, correctness and negative
log probability of the answer for each condition. Aggregates are accuracy,
mean answer probability, and mean cross-entropy. Signed paired changes are
intact minus ablated accuracy, ablated minus intact loss, and intact minus
ablated correct-answer probability. Positive values consistently indicate
harm from ablation. The no-op changes must be exactly zero. Float32 checkpoint
replay must preserve all predictions and logits/probabilities within frozen
absolute/relative tolerances of 2e-6; same-environment replay is also tested for
byte-identical deterministic result JSON.

Within each seed, use 2,000 paired example bootstrap resamples (PCG64 seed
20261011), shared across conditions. Report percentile 95% intervals of the
mean paired change (linear quantiles). These describe example-sampling
uncertainty conditional on one model and the constructed balanced test set,
not training-seed uncertainty, causal certainty or multiple-comparison-adjusted
significance. They do not pool the three seeds. Correlated comparisons and only
eight heads make rank correlations exploratory.

The attention score for a head is its intact final-query attention mass on the
matching value position, averaged over all 512 examples. Compute descending
average ranks (exact numeric ties), with tied head groups explicitly recorded.
Spearman correlation compares those ranks to mean paired answer-probability
drops. A constant rank vector yields JSON `null` and an explanation, never
NaN or an invented zero. Report all seeds separately, plus a descriptive range;
head indices are not matched across seeds.

## Export, resource limits and provenance

Only the first 16 evaluation examples in generator order export full intact
attention matrices. This fixed subset covers all four retrieval positions and
is never selected by confidence, correctness or intervention effect. All 512
examples retain compact attention scores and all condition predictions. The
report renders a single 10×10 matrix at a time, with a numeric table alternative.

The evaluation budget is 600 seconds; all artifacts are bounded to 10 MiB each,
32 MiB total and at most 1,000 files. Store numeric NPZ checkpoints (no pickle),
JSON dataset shards, deterministic result JSON and HTML, manifests with hashes
and sizes, dependency versions, timings and peak process RSS. Timing/environment
metadata is separate from deterministic results. Peak RSS is the Linux process
high-water mark and includes imports; it is not isolated model memory or
machine-wide memory. Checksummed artifacts are not a cryptographic trust chain
against a party replacing both an artifact and its manifest.

Runtime budgets are checked between bounded batches/steps. SIGINT and SIGTERM
remove uncommitted outputs; SIGKILL/power loss cannot be cleaned up by Python.
Output names are reserved exclusively; an existing destination is never
replaced. A staging directory is atomically committed after successful work.
