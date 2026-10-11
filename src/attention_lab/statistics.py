"""Per-seed paired uncertainty and explicit rank ties, with no cross-seed pooling."""
import numpy as np


def from_logits(logits, answers):
    logits = np.asarray(logits, dtype=np.float64)
    centered = logits - logits.max(axis=1, keepdims=True)
    log_probs = centered - np.log(np.exp(centered).sum(axis=1, keepdims=True))
    loss = -log_probs[np.arange(len(answers)), answers]
    prediction = logits.argmax(axis=1)
    return {"prediction": prediction, "correct": prediction == answers,
            "loss": loss, "p_correct": np.exp(-loss)}


def metrics(values):
    return {"accuracy": float(np.mean(values["correct"])),
            "loss": float(np.mean(values["loss"])),
            "p_correct": float(np.mean(values["p_correct"]))}


def bootstrap_indices(n, replicates=2000, seed=20261011):
    return np.random.Generator(np.random.PCG64(seed)).integers(0, n, (replicates, n))


def paired_changes(intact, condition, indices):
    result = {}
    for field, name, direction in [("correct", "accuracy_drop", 1),
                                    ("loss", "loss_increase", -1),
                                    ("p_correct", "probability_drop", 1)]:
        delta = direction * (np.asarray(intact[field], dtype=np.float64) - condition[field])
        means = delta[indices].mean(axis=1)
        interval = np.quantile(means, [0.025, 0.975], method="linear")
        result[name] = {"mean": float(delta.mean()), "ci95": interval.tolist()}
    return result


def ranks_descending(values):
    values = np.asarray(values, dtype=float)
    ranks = np.empty(len(values), dtype=float)
    ties = []
    for value in sorted(set(values), reverse=True):
        same = np.flatnonzero(values == value)
        ranks[same] = 1 + np.sum(values > value) + (len(same)-1)/2
        if len(same) > 1:
            ties.append(same.tolist())
    return ranks, ties


def rank_comparison(attention, effects, names):
    a, at = ranks_descending(attention)
    b, bt = ranks_descending(effects)
    constant = np.ptp(a) == 0 or np.ptp(b) == 0
    return {"attention_mean": list(map(float, attention)),
            "probability_drop_mean": list(map(float, effects)),
            "attention_rank": a.tolist(), "effect_rank": b.tolist(),
            "attention_ties": [[names[i] for i in group] for group in at],
            "effect_ties": [[names[i] for i in group] for group in bt],
            "spearman": None if constant else float(np.corrcoef(a, b)[0, 1]),
            "undefined_reason": "constant rank vector" if constant else None}
