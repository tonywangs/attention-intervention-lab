"""Frozen, exhaustive two-head evaluation of the three saved retrieval models."""
from pathlib import Path
import hashlib
import itertools
import math
import random
import resource
import shutil
import time
import numpy as np
import torch
from .artifacts import (Budget, canonical, manifest, read_json, sha256, transaction,
                        verify_manifest, write_json)
from .data import belongs, oracle
from .evaluate import state_hash
from .interaction_stats import METRICS, outcomes, summarize
from .model import Config, Transformer, load_weights
from .pair_intervention import zero_heads
from .train import configure, environment, tensors

PROTOCOL_SHA256 = '23c45240eb2f2823b0bdd5952f90f710d8d5d5f71cb4ec8ba5b9c798d2aef3d8'


def load_protocol(path):
    p = read_json(path, 65536)
    if sha256(path) != PROTOCOL_SHA256:
        raise ValueError('interaction protocol differs from frozen specification')
    return p


def dictionary_key(tokens):
    return tuple(sorted(zip(tokens[:8:2], tokens[1:8:2])))


def historical_inputs(training, previous):
    paths = sorted((Path(training)/'data').glob('train-*.json')) + [Path(training)/'data/dev.json', Path(previous)/'eval.json']
    if len(paths) != 10:
        raise ValueError('expected eight training shards, development and previous evaluation')
    forbidden = set()
    hashes = {}
    for i, path in enumerate(paths):
        rows = read_json(path)
        if not isinstance(rows, list) or not 1 <= len(rows) <= 4096:
            raise ValueError('invalid historical data')
        for row in rows:
            if oracle(row['tokens']) != (row['answer'], row['value_position']):
                raise ValueError('historical oracle mismatch')
            forbidden.add(dictionary_key(row['tokens']))
        hashes[f'{i:02d}-{path.name}'] = sha256(path)
    return forbidden, hashes


def generate_rows(protocol, forbidden):
    rng = random.Random(protocol['data']['seed'])
    seen = set(forbidden)
    rows = []
    while len(rows) < protocol['data']['n']:
        keys = rng.sample(range(16), 4)
        values = rng.sample(range(16, 32), 4)
        slot = len(rows) % 4
        tokens = [t for pair in zip(keys, values) for t in pair] + [32, keys[slot]]
        key = dictionary_key(tokens)
        if key in seen or not belongs(tokens, 'eval'):
            continue
        seen.add(key)
        # Generation uses positional construction; validation uses dictionary lookup.
        rows.append({'id': f'pair-{len(rows):05d}', 'tokens': tokens,
                     'answer': values[slot]-16, 'value_position': 2*slot+1})
    validate_rows(rows, forbidden)
    return rows


def validate_rows(rows, forbidden):
    if not isinstance(rows, list) or len(rows) != 512:
        raise ValueError('expected 512 new examples')
    seen = set(forbidden)
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {'id', 'tokens', 'answer', 'value_position'}:
            raise ValueError('invalid interaction example')
        if (type(row['answer']) is not int or type(row['value_position']) is not int
                or row['id'] != f'pair-{i:05d}' or row['value_position'] != 2*(i % 4)+1
                or oracle(row['tokens']) != (row['answer'], row['value_position'])
                or not belongs(row['tokens'], 'eval')):
            raise ValueError('invalid interaction label or split')
        key = dictionary_key(row['tokens'])
        if key in seen:
            raise ValueError('interaction data overlap')
        seen.add(key)


def prepare(protocol_path, training, previous, destination):
    p = load_protocol(protocol_path)
    verify_manifest(training)
    verify_manifest(previous)
    forbidden, hashes = historical_inputs(training, previous)
    rows = generate_rows(p, forbidden)
    with transaction(destination) as out:
        write_json(out/'examples.json', rows)
        write_json(out/'inputs.json', {'protocol_sha256': PROTOCOL_SHA256,
                   'examples_sha256': sha256(out/'examples.json'), 'historical_input_hashes': hashes,
                   'excluded_dictionaries': len(forbidden), 'oracle_checked': True})
        write_json(out/'manifest.json', manifest(out))


def load_array(path, shape, dtype):
    """Bounded numeric .npy reader; validate the header before numpy allocates."""
    path = Path(path)
    dtype = np.dtype(dtype)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 10*1024*1024:
        raise ValueError('invalid numeric artifact')
    with path.open('rb') as stream:
        if np.lib.format.read_magic(stream) != (1, 0):
            raise ValueError('unsupported numeric artifact version')
        actual, fortran, kind = np.lib.format.read_array_header_1_0(stream)
        if (actual != shape or fortran or kind != dtype
                or path.stat().st_size-stream.tell() != math.prod(shape)*dtype.itemsize):
            raise ValueError('numeric artifact shape or type mismatch')
    array = np.load(path, allow_pickle=False)
    if not np.isfinite(array).all():
        raise ValueError('nonfinite numeric artifact')
    return array


def check_sources(training, previous, protocol_path):
    training, previous = Path(training), Path(previous)
    baseline = read_json(Path(protocol_path).parent/'baseline.json', 65536)
    p = load_protocol(protocol_path)
    if sha256(Path(protocol_path).parent/'baseline.json') != p['baseline_sha256']:
        raise ValueError('baseline hash mismatch')
    verify_manifest(training)
    verify_manifest(previous)
    for directory, prefix in ((training, 'experiments/final/'), (previous, 'results/evaluation/')):
        for name, digest in baseline.items():
            if name.startswith(prefix) and sha256(directory/name[len(prefix):]) != digest:
                raise ValueError('historical artifact differs from catalog baseline')
    return p


@torch.no_grad()
def infer(model, x, addresses, noop, budget):
    batches = []
    with zero_heads(model, addresses, noop=noop):
        for batch in x.split(128):
            budget.check()
            batches.append(model(batch)[0].numpy())
    return np.concatenate(batches)


def evaluate_pairs(training, previous, protocol_path, data, destination, max_seconds=7200):
    if not 0 < max_seconds <= 7200:
        raise ValueError('pair evaluation budget must be in (0, 7200] seconds')
    budget = Budget(max_seconds)
    configure()
    training, previous, data = Path(training), Path(previous), Path(data)
    p = check_sources(training, previous, protocol_path)
    verify_manifest(data)
    forbidden, hashes = historical_inputs(training, previous)
    rows = read_json(data/'examples.json')
    validate_rows(rows, forbidden)
    if rows != generate_rows(p, forbidden):
        raise ValueError('examples differ from frozen generator')
    inputs = read_json(data/'inputs.json')
    if (inputs['protocol_sha256'] != PROTOCOL_SHA256 or inputs['historical_input_hashes'] != hashes
            or inputs['examples_sha256'] != sha256(data/'examples.json')):
        raise ValueError('input provenance mismatch')
    x, y = tensors(rows)
    addresses = {f'L{l}H{h}': (l, h) for l in range(2) for h in range(4)}
    summary = {'schema': 1, 'protocol_sha256': PROTOCOL_SHA256,
               'examples_sha256': sha256(data/'examples.json'), 'n': 512, 'seeds': [],
               'uncertainty': p['uncertainty'], 'limits': p['limits']}
    checkpoint_hashes = {str(s): sha256(training/f'seed-{s}.npz') for s in p['training_seeds']}
    with transaction(destination) as out:
        budget.check()
        shutil.copyfile(protocol_path, out/'protocol.json')
        shutil.copyfile(data/'examples.json', out/'examples.json')
        shutil.copyfile(data/'inputs.json', out/'inputs.json')
        # Reproduce every historical saved prediction before running the new inputs.
        from .evaluate import replay
        replay(training, previous, out/'historical-replay', min(600, max_seconds))
        shutil.copyfile(out/'historical-replay/replay.json', out/'historical-replay.json')
        shutil.rmtree(out/'historical-replay')
        budget.check()
        for seed in p['training_seeds']:
            started = time.monotonic()
            model = Transformer(Config()).eval()
            load_weights(model, training/f'seed-{seed}.npz')
            before = state_hash(model)
            logits = []
            for name in p['conditions']:
                selected = [] if name in ('intact', 'noop') else [addresses[h] for h in name.split('+')]
                logits.append(infer(model, x, selected, name == 'noop', budget))
            logits = np.stack(logits)
            if not np.array_equal(logits[0], logits[1]):
                raise ValueError('pair no-op differs')
            if before != state_hash(model):
                raise ValueError('pair interventions changed weights')
            values = outcomes(logits, y.numpy())
            seed_summary = summarize(values, rows, p, budget)
            seed_summary.update({'seed': seed, 'checkpoint_sha256': checkpoint_hashes[str(seed)],
                                 'weights_unchanged': True, 'noop_max_logit_difference': 0.0})
            np.save(out/f'logits-{seed}.npy', logits, allow_pickle=False)
            np.save(out/f'outcomes-{seed}.npy', values, allow_pickle=False)
            summary['seeds'].append(seed_summary)
            print(f'Seed {seed}: 38 conditions, 512 examples; {time.monotonic()-started:.3f} seconds', flush=True)
        write_json(out/'summary.json', summary)
        from .interaction_report import render_directory
        render_directory(out, out/'report.html')
        if historical_inputs(training, previous)[1] != hashes:
            raise ValueError('input files changed during evaluation')
        if any(sha256(training/f'seed-{s}.npz') != checkpoint_hashes[str(s)] for s in p['training_seeds']):
            raise ValueError('checkpoint files changed during evaluation')
        source_hashes = {f.name: sha256(f) for f in sorted(Path(__file__).parent.glob('*.py'))}
        source_hashes['pair_report.html'] = sha256(Path(__file__).with_name('pair_report.html'))
        budget.check()
        write_json(out/'provenance.json', {'environment': environment(), 'source_sha256': source_hashes,
                   'elapsed_seconds': time.monotonic()-budget.start,
                   'peak_process_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   'protocol_sha256': PROTOCOL_SHA256, 'evaluation_budget_seconds': max_seconds,
                   'checkpoint_sha256': checkpoint_hashes, 'historical_input_hashes': hashes,
                   'data_manifest_sha256': sha256(data/'manifest.json'),
                   'completion': 'complete', 'artifact_bytes_before_provenance': sum(v['bytes'] for v in manifest(out).values())})
        write_json(out/'manifest.json', manifest(out))
    return summary


def replay_pairs(training, previous, protocol_path, data, reference, destination, max_seconds=7200):
    reference = Path(reference)
    verify_manifest(reference)
    p = load_protocol(protocol_path)
    with transaction(destination) as out:
        evaluate_pairs(training, previous, protocol_path, data, out/'evaluation', max_seconds)
        generated = out/'evaluation'
        checks = []
        for name in ('protocol.json', 'examples.json', 'inputs.json', 'summary.json', 'report.html', 'historical-replay.json'):
            identical = sha256(reference/name) == sha256(generated/name)
            checks.append({'file': name, 'byte_identical': identical})
            if not identical:
                raise ValueError(f'pair replay differs: {name}')
        for seed in p['training_seeds']:
            for kind, shape, dtype in (('logits', (38, 512, 16), np.float32), ('outcomes', (38, 512, 3), np.float64)):
                name = f'{kind}-{seed}.npy'
                old, new = (load_array(d/name, shape, dtype) for d in (reference, generated))
                if not np.allclose(old, new, atol=2e-6, rtol=2e-6):
                    raise ValueError('pair replay numeric mismatch')
                if kind == 'logits' and not np.array_equal(old.argmax(-1), new.argmax(-1)):
                    raise ValueError('pair replay prediction mismatch')
                checks.append({'file': name, 'byte_identical': sha256(reference/name) == sha256(generated/name)})
        write_json(out/'replay.json', {'within_frozen_tolerances': True,
                   'all_deterministic_files_identical': all(c['byte_identical'] for c in checks), 'checks': checks})
        write_json(out/'manifest.json', manifest(out))
