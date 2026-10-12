"""Paired effects; fixed-stratum resampling and within-model simultaneous intervals."""
import numpy as np

METRICS = ('loss', 'p_correct', 'accuracy')


def outcomes(logits, answers):
    a = np.asarray(logits, dtype=np.float64)
    shifted = a - a.max(axis=-1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    correct_logp = np.take_along_axis(logp, np.asarray(answers)[None, :, None], axis=-1)[..., 0]
    return np.stack((-correct_logp, np.exp(correct_logp),
                     (a.argmax(axis=-1) == np.asarray(answers)).astype(float)), axis=-1)


def interaction_values(values, conditions, pairs):
    base = values[conditions.index('intact')]
    return np.stack([values[conditions.index('+'.join(pair))]
                     - values[conditions.index(pair[0])]
                     - values[conditions.index(pair[1])] + base for pair in pairs])


def resample_weights(rows, settings):
    rng = np.random.default_rng(settings['seed'])
    n = len(rows)
    counts = np.zeros((settings['replicates'], n), dtype=np.float64)
    # Draw indices explicitly, so an independent implementation can use direct gathers.
    for pos in (1, 3, 5, 7):
        group = np.array([i for i, r in enumerate(rows) if r['value_position'] == pos])
        indices = rng.integers(0, len(group), size=(settings['replicates'], len(group)))
        for b, idx in enumerate(indices):
            counts[b, group] = np.bincount(idx, minlength=len(group))
    return counts / n


def summarize(values, rows, protocol, budget):
    names, pairs = protocol['conditions'], protocol['pairs']
    cfg = protocol['uncertainty']
    weights = resample_weights(rows, cfg)
    budget.check()
    base = values[0]
    effects = values - base
    # Matrix operations retain all paired example covariances.
    boot_effects = (weights @ effects.transpose(1, 0, 2).reshape(len(rows), -1)).reshape(-1, len(names), 3)
    inter = interaction_values(values, names, pairs)
    means = inter.mean(axis=1)
    boot = (weights @ inter.transpose(1, 0, 2).reshape(len(rows), -1)).reshape(-1, len(pairs), 3)
    se = boot.std(axis=0, ddof=1)
    active = se > cfg['degenerate_threshold']
    standardized = np.zeros_like(boot)
    np.divide(np.abs(boot - means), se, out=standardized, where=active)
    q = float(np.quantile(standardized.max(axis=(1, 2)), cfg['level'], method='linear'))
    low, high = np.quantile(boot, [0.025, 0.975], axis=0, method='linear')
    elow, ehigh = np.quantile(boot_effects, [0.025, 0.975], axis=0, method='linear')
    conditions = []
    for c, name in enumerate(names):
        conditions.append({'name': name, 'metrics': {
            metric: {'mean': float(values[c, :, m].mean()),
                     'effect': float(effects[c, :, m].mean()),
                     'effect_ci95': [float(elow[c, m]), float(ehigh[c, m])]}
            for m, metric in enumerate(METRICS)}})
    results = []
    floor = protocol['tolerances']['resolved_interaction_floor']
    for i, pair in enumerate(pairs):
        metrics = {}
        for m, metric in enumerate(METRICS):
            lo, hi = means[i, m] - q*se[i, m], means[i, m] + q*se[i, m]
            metrics[metric] = {'mean': float(means[i, m]), 'ci95_pointwise': [float(low[i, m]), float(high[i, m])],
                              'ci95_family': [float(lo), float(hi)], 'bootstrap_se': float(se[i, m]),
                              'degenerate': bool(not active[i, m]),
                              'resolved': bool(active[i, m] and (lo > floor or hi < -floor))}
        a, b, joint = [names.index(s) for s in (*pair, '+'.join(pair))]
        hidden = (np.array_equal(values[a, :, 2], base[:, 2])
                  and np.array_equal(values[b, :, 2], base[:, 2])
                  and values[joint, :, 2].mean() < base[:, 2].mean())
        # The frozen flag compares aggregate accuracy. At a perfect intact score
        # equal accuracy and equal per-example correctness coincide.
        hidden = (values[a, :, 2].mean() == base[:, 2].mean()
                  and values[b, :, 2].mean() == base[:, 2].mean()
                  and values[joint, :, 2].mean() < base[:, 2].mean())
        results.append({'pair': pair, 'metrics': metrics, 'hidden_accuracy': bool(hidden)})
    budget.check()
    return {'conditions': conditions, 'pairs': results, 'family_critical_value': q,
            'family_size': 84, 'nondegenerate_family_size': int(active.sum())}
