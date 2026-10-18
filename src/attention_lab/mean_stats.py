"""Paired method comparison with one 252-contrast family per model."""
import numpy as np
from .interaction_stats import METRICS, interaction_values, resample_weights


def summarize(values, rows, protocol, budget):
    # values: method, condition (intact/noop/eight singles/28 pairs), example, metric
    names, pairs = protocol['conditions'], protocol['pairs']
    cfg = protocol['uncertainty']
    weights = resample_weights(rows, cfg)
    inter = np.stack([interaction_values(v, names, pairs) for v in values])
    contrasts = np.concatenate((inter, (inter[1]-inter[0])[None]), axis=0)
    means = contrasts.mean(axis=2)
    boot = (weights @ contrasts.transpose(2, 0, 1, 3).reshape(len(rows), -1)).reshape(-1, 3, 28, 3)
    se = boot.std(axis=0, ddof=1)
    active = se > cfg['degenerate_threshold']
    standardized = np.zeros_like(boot)
    np.divide(np.abs(boot-means), se, out=standardized, where=active)
    q = float(np.quantile(standardized.max(axis=(1, 2, 3)), cfg['level'], method='linear'))
    low, high = np.quantile(boot, [.025, .975], axis=0, method='linear')
    floor = protocol['tolerances']['resolved_interaction_floor']
    result = {'family_size': 252, 'nondegenerate_family_size': int(active.sum()),
              'family_critical_value': q, 'comparisons': {}, 'conditions': {}}
    for k, method in enumerate(('zero', 'mean', 'difference')):
        entries = []
        for j, pair in enumerate(pairs):
            metrics = {}
            for m, metric in enumerate(METRICS):
                lo, hi = means[k,j,m] + np.array([-1,1])*q*se[k,j,m]
                metrics[metric] = {'mean': float(means[k,j,m]), 'ci95_pointwise': [float(low[k,j,m]),float(high[k,j,m])],
                    'ci95_family': [float(lo),float(hi)], 'bootstrap_se': float(se[k,j,m]),
                    'degenerate': bool(not active[k,j,m]),
                    'resolved': bool(active[k,j,m] and (lo > floor or hi < -floor))}
            entries.append({'pair': pair, 'metrics': metrics})
        result['comparisons'][method] = entries
        if k < 2:
            effects = values[k]-values[k,0]
            boots = (weights @ effects.transpose(1,0,2).reshape(len(rows),-1)).reshape(-1,38,3)
            bounds = np.quantile(boots,[.025,.975],axis=0,method='linear')
            result['conditions'][method] = [{'name': name, 'metrics': {metric: {
                'mean': float(values[k,c,:,m].mean()), 'effect': float(effects[c,:,m].mean()),
                'effect_ci95': [float(bounds[0,c,m]),float(bounds[1,c,m])]}
                for m,metric in enumerate(METRICS)}} for c,name in enumerate(names)]
        budget.check()
    return result
