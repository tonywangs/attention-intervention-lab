# Related work and scope

This is a small engineering and reproducibility exercise, with no novelty claim.
Sources were reviewed on 2026-10-11.

- **AttentionViz** (Yeh et al., 2023): [paper](https://arxiv.org/abs/2305.03210),
  [technical description](https://catherinesyeh.github.io/attn-docs/),
  [implementation](https://github.com/catherinesyeh/attention-viz).
  Its global views project joint query/key representations across many inputs;
  its sentence/image views inspect individual inputs. This workbench instead
  renders small, exact attention matrices for synthetic examples beside
  independently measured intervention outcomes. It does not implement the joint
  embedding projection, reproduce the paper's experiments, or make claims about
  BERT, GPT-2, or vision transformers.
- **Are Sixteen Heads Really Better than One?** (Michel, Levy, Neubig, NeurIPS
  2019): [paper](https://arxiv.org/abs/1905.10650v3),
  [authors' implementation](https://github.com/pmichel31415/are-16-heads-really-better-than-1).
  The study evaluates removing heads and shows that many can be removed with
  little task-performance change in its evaluated models. Our measurement is
  one head removed at a time, without pruning, retraining, greedy search or
  claims of speedup. Lack of effect can reflect redundancy or the test task;
  it does not establish that a head has no function.
- **TransformerLens**, pinned tag **v2.15.4**, commit
  `b5a16f849649a237cc02cc2c272ae4dc2085abe4`:
  [reviewed attention implementation](https://github.com/TransformerLensOrg/TransformerLens/blob/b5a16f849649a237cc02cc2c272ae4dc2085abe4/transformer_lens/components/abstract_attention.py).
  The inspected source file has SHA-256
  `a5eb085e3fb78484db4ce6cf01996c6029c48d5a14dc6302d1ba26736c1a244c`.
  Its causal mask precedes softmax, `hook_pattern` exposes attention probabilities,
  and `hook_z` exposes the attention-weighted values before multiplication by
  the output projection. Our independent model follows that intervention
  location, using layout B,H,T,D internally (rather than B,T,H,D).
  TransformerLens is not a runtime dependency; no equivalence with all of its
  architectures or numerical options is claimed. Numeric tests use a separate
  scalar-loop NumPy reference, not the library or a copied implementation.

## What the displays mean

Attention is a normalized routing weight, not the transported value vector or
its downstream effect. The descriptive score here is the mean attention weight
from the last query token to its matching **value** position. A layer that
prepares values using earlier keys may be useful without attending to that value
from the final query. This score is intentionally narrow.

A zero-head intervention measures the change produced by a specific operation
on this trained model and this input distribution. It leaves the shared output
bias and residual/MLP paths intact. Zeroing can create out-of-distribution hidden
states. Individual removals omit interactions, redundancy, compensating paths,
and alternatives such as mean ablation or activation patching. Neither a high
attention weight nor a large zero-ablation effect alone is a complete mechanistic
explanation. Head indices across seeds do not identify corresponding learned
functions.
