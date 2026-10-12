"""Bounded, self-contained pair report with text-only DOM insertion."""
from pathlib import Path
import math
import numpy as np
from .artifacts import canonical, read_json
from .interaction_stats import METRICS
from .interactions import load_array, load_protocol


def finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('report requires finite numbers')
    return value


def interval(value):
    if not isinstance(value, list) or len(value) != 2 or finite(value[0]) > finite(value[1]):
        raise ValueError('invalid report interval')


def payload(directory):
    d = Path(directory)
    p = load_protocol(d/'protocol.json')
    s = read_json(d/'summary.json', 512*1024)
    rows = read_json(d/'examples.json', 256*1024)
    if (not isinstance(s, dict) or s.get('schema') != 1 or s.get('protocol_sha256') is None
            or s.get('n') != 512 or not isinstance(s.get('seeds'), list) or len(s['seeds']) != 3
            or not isinstance(rows, list) or len(rows) != 512):
        raise ValueError('invalid pair report shape')
    from .interactions import PROTOCOL_SHA256, validate_rows
    from .artifacts import sha256
    if s['protocol_sha256'] != PROTOCOL_SHA256 or s.get('examples_sha256') != sha256(d/'examples.json'):
        raise ValueError('report input hash mismatch')
    validate_rows(rows, set())
    limits = s.get('limits')
    if (not isinstance(limits, list) or not 1 <= len(limits) <= 8
            or any(not isinstance(t, str) or len(t) > 1000 for t in limits)):
        raise ValueError('invalid report text bounds')
    for seed, expected in zip(s['seeds'], p['training_seeds']):
        if (not isinstance(seed, dict) or seed.get('seed') != expected
                or not isinstance(seed.get('conditions'), list) or len(seed['conditions']) != 38
                or not isinstance(seed.get('pairs'), list) or len(seed['pairs']) != 28):
            raise ValueError('invalid pair report seed')
        values = load_array(d/f'outcomes-{expected}.npy', (38, 512, 3), np.float64)
        for i, (condition, name) in enumerate(zip(seed['conditions'], p['conditions'])):
            if condition.get('name') != name or set(condition.get('metrics', {})) != set(METRICS):
                raise ValueError('invalid report condition')
            for m, metric in enumerate(METRICS):
                v = condition['metrics'][metric]
                finite(v['mean']); finite(v['effect']); interval(v['effect_ci95'])
                if (abs(v['mean']-values[i, :, m].mean()) > 1e-10
                        or abs(v['effect']-(values[i, :, m]-values[0, :, m]).mean()) > 1e-10):
                    raise ValueError('report condition disagrees with outcomes')
        for pair, expected_pair in zip(seed['pairs'], p['pairs']):
            if pair.get('pair') != expected_pair or set(pair.get('metrics', {})) != set(METRICS):
                raise ValueError('invalid report pair')
            a, b, joint = [p['conditions'].index(v) for v in (*expected_pair, '+'.join(expected_pair))]
            for m, metric in enumerate(METRICS):
                v = pair['metrics'][metric]
                finite(v['mean']); finite(v['bootstrap_se'])
                interval(v['ci95_pointwise']); interval(v['ci95_family'])
                if type(v['resolved']) is not bool or type(v['degenerate']) is not bool:
                    raise ValueError('invalid report flags')
                if abs(v['mean']-(values[joint, :, m]-values[a, :, m]-values[b, :, m]+values[0, :, m]).mean()) > 1e-10:
                    raise ValueError('report interaction disagrees with outcomes')
        seed['examples'] = values[:, :16, :].tolist()
    return {'protocol': p, 'summary': s, 'rows': rows[:16]}


def render(data):
    raw = canonical(data).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    if len(raw) > 1024*1024:
        raise ValueError('report payload budget exceeded')
    template = Path(__file__).with_name('pair_report.html').read_text()
    return template.replace('__PAYLOAD__', raw)


def render_directory(directory, destination):
    Path(destination).write_text(render(payload(directory)))
