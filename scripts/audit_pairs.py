#!/usr/bin/env python3
"""Independent scalar metrics and direct-index bootstrap audit of every saved pair.

Does not import production evaluation, statistics, data generation or reporting.
Uses standard JSON/hashlib and NumPy only; production count-matrix aggregation
is replaced by directly gathering the resampled contrast rows.
"""
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    np.testing.assert_allclose(a, b, atol=1e-10, rtol=1e-10)


def audit(directory=ROOT/'results/interactions'):
    directory = Path(directory)
    p = read(directory/'protocol.json')
    assert digest(directory/'protocol.json') == '23c45240eb2f2823b0bdd5952f90f710d8d5d5f71cb4ec8ba5b9c798d2aef3d8'
    freeze = read(ROOT/'experiments/interactions/freeze.json')
    assert freeze['protocol_sha256'] == digest(directory/'protocol.json')
    assert freeze['new_checkpoint_evaluations_started'] is False
    baseline = read(ROOT/'experiments/interactions/baseline.json')
    assert digest(ROOT/'experiments/interactions/baseline.json') == p['baseline_sha256']
    # Original scientific record is immutable, including docs and frozen source.
    for name, h in baseline.items():
        if name.startswith(('experiments/', 'results/', 'docs/')) or name == 'src/attention_lab/model.py':
            assert digest(ROOT/name) == h, name
    manifest = read(directory/'manifest.json')
    assert {f.name for f in directory.iterdir()} == set(manifest) | {'manifest.json'}
    for name, item in manifest.items():
        assert item == {'sha256': digest(directory/name), 'bytes': (directory/name).stat().st_size}
    provenance = read(directory/'provenance.json')
    assert 0 < provenance['elapsed_seconds'] <= provenance['evaluation_budget_seconds'] <= 7200
    assert provenance['environment']['threads'] == 1
    assert provenance['peak_process_rss_kib'] > 0
    rows = read(directory/'examples.json')
    assert len(rows) == 512
    historic = sorted((ROOT/'experiments/final/data').glob('*.json')) + [ROOT/'results/evaluation/eval.json']
    contexts = set()
    for path in historic:
        for row in read(path):
            contexts.add(tuple(sorted(zip(row['tokens'][:8:2], row['tokens'][1:8:2]))))
    for i, row in enumerate(rows):
        tokens = row['tokens']
        mapping = {tokens[k]: tokens[k+1]-16 for k in (0, 2, 4, 6)}
        assert row['answer'] == mapping[tokens[-1]]
        assert row['value_position'] == next(k+1 for k in (0, 2, 4, 6) if tokens[k] == tokens[-1]) == 2*(i % 4)+1
        key = tuple(sorted(zip(tokens[:8:2], tokens[1:8:2])))
        assert key not in contexts
        contexts.add(key)
        raw = json.dumps(key, separators=(',', ':')).encode()
        assert int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') % 100 >= 90
    assert digest(directory/'examples.json') == digest(ROOT/'experiments/interactions/data/examples.json')
    heads = [f'L{l}H{h}' for l in range(2) for h in range(4)]
    pairs = list(itertools.combinations(heads, 2))
    conditions = ['intact', 'noop'] + heads + ['+'.join(pair) for pair in pairs]
    assert p['pairs'] == [list(v) for v in pairs] and p['conditions'] == conditions
    summary = read(directory/'summary.json')
    assert summary['examples_sha256'] == digest(directory/'examples.json')
    assert [s['seed'] for s in summary['seeds']] == [11, 29, 47]
    # Same PRNG protocol, independent direct-index application. Indices generated
    # before blocking; changing audit block size cannot change the resamples.
    rng = np.random.default_rng(20261012)
    indices = [rng.integers(0, 128, size=(10000, 128)) for _ in range(4)]
    groups = [np.array([i for i, r in enumerate(rows) if r['value_position'] == pos]) for pos in (1, 3, 5, 7)]
    for saved in summary['seeds']:
        seed = saved['seed']
        assert saved['checkpoint_sha256'] == digest(ROOT/f'experiments/final/seed-{seed}.npz')
        logits = np.load(directory/f'logits-{seed}.npy', allow_pickle=False)
        assert logits.shape == (38, 512, 16) and logits.dtype == np.float32
        assert np.isfinite(logits).all() and np.array_equal(logits[0], logits[1])
        expected = np.zeros((38, 512, 3))
        for c in range(38):
            for i, row in enumerate(rows):
                scores = [float(v) for v in logits[c, i]]
                peak = max(scores)
                total = math.fsum(math.exp(v-peak) for v in scores)
                logp = scores[row['answer']]-peak-math.log(total)
                expected[c, i] = [-logp, math.exp(logp), float(scores.index(peak) == row['answer'])]
        close(np.load(directory/f'outcomes-{seed}.npy', allow_pickle=False), expected)
        effects = expected-expected[0]
        interactions = np.stack([expected[conditions.index('+'.join(pair))]
                                  + expected[0] - expected[conditions.index(pair[0])]
                                  - expected[conditions.index(pair[1])] for pair in pairs])
        columns = np.concatenate((effects.transpose(1, 0, 2).reshape(512, -1),
                                  interactions.transpose(1, 0, 2).reshape(512, -1)), axis=1)
        boot = np.empty((10000, columns.shape[1]))
        for start in range(0, 10000, 50):
            block = np.zeros((50, columns.shape[1]))
            for group, idx in zip(groups, indices):
                block += columns[group[idx[start:start+50]]].sum(axis=1) / 512
            boot[start:start+50] = block
        effect_ci = np.quantile(boot[:, :114].reshape(10000, 38, 3), [0.025, 0.975], axis=0)
        iboot = boot[:, 114:].reshape(10000, 28, 3)
        means = interactions.mean(axis=1)
        se = np.std(iboot, axis=0, ddof=1)
        active = se > 1e-12
        max_samples = np.max(np.abs(iboot[:, active]-means[active])/se[active], axis=1) if active.any() else np.zeros(10000)
        critical = float(np.quantile(max_samples, 0.95))
        ci = np.quantile(iboot, [0.025, 0.975], axis=0)
        close(saved['family_critical_value'], critical)
        assert saved['nondegenerate_family_size'] == int(active.sum()) and saved['family_size'] == 84
        for c, condition in enumerate(saved['conditions']):
            assert condition['name'] == conditions[c]
            for m, metric in enumerate(('loss', 'p_correct', 'accuracy')):
                v = condition['metrics'][metric]
                close(v['mean'], math.fsum(expected[c, :, m])/512)
                close(v['effect'], math.fsum(effects[c, :, m])/512)
                close(v['effect_ci95'], effect_ci[:, c, m])
        for k, pair in enumerate(saved['pairs']):
            assert pair['pair'] == list(pairs[k])
            a, b, j = [conditions.index(n) for n in (*pairs[k], '+'.join(pairs[k]))]
            hidden = expected[a, :, 2].mean() == expected[0, :, 2].mean() == expected[b, :, 2].mean() and expected[j, :, 2].mean() < expected[0, :, 2].mean()
            assert pair['hidden_accuracy'] == hidden
            for m, metric in enumerate(('loss', 'p_correct', 'accuracy')):
                v = pair['metrics'][metric]
                lo, hi = means[k, m] - critical*se[k, m], means[k, m] + critical*se[k, m]
                close(v['mean'], math.fsum(interactions[k, :, m])/512)
                close(v['ci95_pointwise'], ci[:, k, m])
                close(v['bootstrap_se'], se[k, m])
                close(v['ci95_family'], [lo, hi])
                assert v['degenerate'] == (not active[k, m])
                assert v['resolved'] == bool(active[k, m] and (lo > 2e-6 or hi < -2e-6))
        print(f'Independent pair audit: seed {seed}, 19,456 condition/example outcomes, all 84 interactions, both interval types and all flags passed.')
    print('Historical scientific files unchanged; fresh data disjoint; labels, artifact hashes, controls, runtime and numerical arrays verified.')


if __name__ == '__main__':
    audit(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/'results/interactions')
