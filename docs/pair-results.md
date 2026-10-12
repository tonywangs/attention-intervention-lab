# Paired removals expose non-additive effects, with differences between models

The frozen exhaustive evaluation completed on all three existing models and
512 new examples per model, with no retraining. Intact and clone-only no-op
execution achieved 512/512 accuracy in every model and identical logits.
All eight individual removals and all 28 unordered joint removals were evaluated.
The inputs, checkpoint hashes and model weights remained unchanged.

Open [the offline report](../results/interactions/report.html) to compare every
pair, metric and seed, or use the adjacent arrays and JSON for independent work.
The [protocol](pair-protocol.md) was frozen before new evaluation; no pair was
selected from these outcomes. This is evidence of output-level non-additivity
under zero ablation, not a complete mechanistic explanation.

## Results per trained model

“Resolved” below means the approximate 95% simultaneous interval, covering the
84 pair × metric interaction family **within that model**, excludes zero by
more than the frozen numerical floor. Counts are descriptive summaries of that
fixed family, not independent discoveries or a pooled test across models.

| Training seed | Resolved loss interactions / 28 | Resolved probability interactions / 28 | Resolved accuracy interactions / 28 | Constant accuracy contrasts / 28 | Hidden-accuracy pairs / 28 | Hidden-accuracy pairs resolved on accuracy |
|---|---:|---:|---:|---:|---:|---:|
| 11 | 12 | 15 | 8 | 2 | 0 | 0 |
| 29 | 14 | 19 | 10 | 3 | 7 | 2 |
| 47 | 12 | 17 | 4 | 6 | 4 | 0 |

A hidden-accuracy pair has both individual accuracies equal to intact accuracy,
but lower joint accuracy. This flag is descriptive; equality on 512 examples
does not establish zero population effect. In seed 29, two such pairs also
have resolved accuracy interactions:

| Seed 29 pair | Intact | Single A | Single B | Joint | Accuracy interaction | 95% family interval |
|---|---:|---:|---:|---:|---:|---:|
| L0H0 + L0H2 | 512/512 | 512/512 | 512/512 | 487/512 | −0.048828 | [−0.082687, −0.014969] |
| L1H0 + L1H3 | 512/512 | 512/512 | 512/512 | 486/512 | −0.050781 | [−0.085257, −0.016306] |

Thus individual accuracy can conceal joint damage in this experiment. That
observation is not shared uniformly: seed 47 has four descriptive hidden pairs,
but their simultaneous accuracy intervals all include zero; seed 11 has none.
Head indices denote locations, not matched computational roles across models.

For context, the largest absolute accuracy interaction in each model is shown
below. These are explicitly **post-evaluation descriptive maxima**, drawn from
the complete retained pair family; they were not used to choose interventions.

| Seed | Pair | Accuracy interaction | 95% family interval |
|---|---|---:|---:|
| 11 | L0H1 + L0H2 | −0.330078 | [−0.403337, −0.256819] |
| 29 | L0H1 + L0H2 | −0.152344 | [−0.207988, −0.096700] |
| 47 | L0H0 + L0H1 | −0.185547 | [−0.247465, −0.123629] |

The null and small effects are retained. Accuracy interactions are exactly zero
on every sampled example for 2, 3 and 6 pairs in seeds 11, 29 and 47 respectively.
Those contrasts have degenerate empirical bootstrap distributions and are never
marked resolved. Several tiny interactions have signs opposite to the largest
effects: for example, seed 11's accuracy range reaches +0.001953. The complete
report shows these as well as unresolved estimates. A collapsed or zero-crossing
interval does not prove population additivity.

## Evidence, runtime and reproducibility

The saved evaluation took **20.523 seconds** on one CPU thread, including replay
of the original predictions, and used **352,460 KiB peak process RSS** (about
344.2 MiB). The maximum authorized evaluation budget was 7,200 seconds. The
environment was CPython 3.12.3, PyTorch 2.6.0+cpu and NumPy 2.2.6 on Linux x86_64.
`results/interactions/provenance.json` preserves exact values and source hashes;
the manifest gives SHA-256 and bytes for every result artifact.

The experiment preserves 58,368 condition/example outcomes across the three
models. For each seed, `logits-SEED.npy` has shape `[38, 512, 16]` and dtype
float32; `outcomes-SEED.npy` has shape `[38, 512, 3]` and dtype float64 with last
axis `[loss, p_correct, accuracy]`. Condition order comes from `protocol.json`;
example order comes from `examples.json`. Load with `numpy.load(...,
allow_pickle=False)`. The result directory is about 5.5 MB; arrays are numeric
and bounded, with no pickled objects or external model downloads.

An independent audit uses scalar softmax/loss computations and direct gathers
of bootstrap indices, rather than the evaluator's resampling-count matrix.
It checks every outcome, mean, uncertainty interval, multiplicity flag, data
label, split and hash. Independent parameter-space removal checks the intervention
against zeroing the corresponding output-projection columns in disposable model
copies, covering all 28 addresses over 224 seeded small cases. Another 224
cases check scalar metric and interaction arithmetic.

The report initially overflowed on narrow screens with long text. A rendering-only
amendment added wrapping and width constraints, and report validation now checks
uncertainty and flag consistency. The original renderer is preserved under
`experiments/interactions/frozen-renderer/`, and the original report is
reconstructible from it. `rendering-amendment.json` records both report hashes.
The original evaluation provenance still describes that first evaluation;
no numeric result, protocol, checkpoint, input or summary was changed by this
amendment. The source hashes for the updated renderer are recorded separately.

There were no interrupted scientific evaluation runs. Deliberately cancelled
SIGINT/SIGTERM test runs verify cleanup and are not included in the scientific
results. Initial verification failures were a missing browser installation,
the mobile layout defect and two test-expectation issues (early size validation
and decimal tie rounding). The corrected checks are rerun by the command below.

After the setup in the [README](../README.md), reproduce with:

```sh
.venv/bin/python scripts/reproduce_pairs.py --output /tmp/attention-pair-reproduction
```

For all original regressions, independent audits, Chromium checks with external
networking blocked, and a fresh installed-CLI replay with networking blocked:

```sh
.venv/bin/python scripts/verify_pairs.py --wheelhouse /tmp/attention-wheelhouse --browsers /tmp/attention-browsers
```

The latter command uses temporary output and does not rewrite historical test
logs. Exact summary and HTML replay is required in the pinned environment;
logits/outcomes additionally use the frozen tolerance checks. Cross-platform
byte identity is not promised.

## Interpretation limits

The three models solve a fixed synthetic retrieval task, with four pairs and
16 possible answers. Bootstrap intervals condition on each fitted model and
the balanced constructed test distribution. Sparse binary errors and near-zero
contrasts can make bootstrap coverage unreliable. Family adjustment is only
within a model; there is no inference over a population of training seeds.

Zero ablation can create unusual activations. Non-additivity could arise from
softmax, loss curvature, LayerNorm, downstream computation or thresholded
accuracy, without proving that two heads implement compensating roles.
Activation/path measurements and alternative replacements would be needed to
distinguish those explanations. No natural-language, mean-ablation, patching,
hardware-specific or broad model-generalization claim follows from this result.
