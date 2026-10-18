"""Frozen mean-ablation comparison; no retraining or runtime network access."""
from pathlib import Path
import shutil
import resource
import time
import numpy as np
import torch
from .artifacts import Budget, read_json, write_json, sha256, manifest, verify_manifest, transaction
from .interactions import historical_inputs, dictionary_key, generate_rows, load_array
from .interaction_stats import outcomes
from .mean_stats import summarize
from .mean_intervention import calibrate, mean_heads
from .pair_intervention import zero_heads
from .model import Transformer, load_weights
from .evaluate import state_hash
from .train import configure, tensors, environment

PROTOCOL_SHA256 = 'd6335d674b92abe5dfc4dbf56ac463c9a4e66d39aaf44e631ab433c4d791638d'


def load_protocol(path):
    p = read_json(path, 65536)
    if sha256(path) != PROTOCOL_SHA256:
        raise ValueError('mean protocol differs from frozen specification')
    return p


def sources(root, protocol_path):
    root = Path(root)
    p = load_protocol(protocol_path)
    baseline_path = Path(protocol_path).parent/'baseline.json'
    if sha256(baseline_path) != p['baseline_sha256']:
        raise ValueError('mean baseline mismatch')
    baseline = read_json(baseline_path, 65536)
    for name, digest in baseline.items():
        # Preserve every historical protocol, checkpoint, input and result.
        if name.startswith(('experiments/', 'results/')) and sha256(root/name) != digest:
            raise ValueError(f'historical artifact changed: {name}')
    for folder in ('experiments/final','results/evaluation','results/interactions'):
        verify_manifest(root/folder)
    forbidden, hashes = historical_inputs(root/'experiments/final', root/'results/evaluation')
    for row in read_json(root/'results/interactions/examples.json'):
        forbidden.add(dictionary_key(row['tokens']))
    hashes['previous_pairs'] = sha256(root/'results/interactions/examples.json')
    return p, forbidden, hashes


def make_data(p, forbidden):
    cp = {**p, 'data': {**p['data'], 'seed': p['data']['calibration_seed']}}
    calibration = generate_rows(cp, forbidden)
    excluded = forbidden | {dictionary_key(r['tokens']) for r in calibration}
    evaluation = generate_rows(p, excluded)
    return calibration, evaluation


def prepare(root, protocol_path, destination):
    p, forbidden, hashes = sources(root, protocol_path)
    calibration, evaluation = make_data(p, forbidden)
    with transaction(destination) as out:
        write_json(out/'calibration.json', calibration)
        write_json(out/'evaluation.json', evaluation)
        write_json(out/'inputs.json', {'protocol_sha256': PROTOCOL_SHA256, 'historical_inputs': hashes,
                   'calibration_sha256': sha256(out/'calibration.json'), 'evaluation_sha256': sha256(out/'evaluation.json'),
                   'excluded_dictionaries': len(forbidden), 'mutually_dictionary_disjoint': True})
        write_json(out/'manifest.json', manifest(out))


@torch.no_grad()
def infer(model, x, context, budget):
    parts = []
    with context:
        for batch in x.split(128):
            budget.check()
            parts.append(model(batch)[0].numpy())
    return np.concatenate(parts)


def evaluate(root, protocol_path, data, destination, max_seconds=7200):
    if not 0 < max_seconds <= 7200:
        raise ValueError('mean evaluation budget must be in (0, 7200] seconds')
    budget = Budget(max_seconds)
    configure()
    root, data = Path(root), Path(data)
    p, forbidden, hashes = sources(root, protocol_path)
    verify_manifest(data)
    calibration, rows = make_data(p, forbidden)
    if calibration != read_json(data/'calibration.json') or rows != read_json(data/'evaluation.json'):
        raise ValueError('mean inputs differ from frozen generator')
    expected_inputs = {'protocol_sha256': PROTOCOL_SHA256, 'historical_inputs': hashes,
                      'calibration_sha256': sha256(data/'calibration.json'), 'evaluation_sha256': sha256(data/'evaluation.json'),
                      'excluded_dictionaries': len(forbidden), 'mutually_dictionary_disjoint': True}
    if read_json(data/'inputs.json') != expected_inputs:
        raise ValueError('mean input provenance mismatch')
    cx, _ = tensors(calibration)
    x, y = tensors(rows)
    addresses = {f'L{l}H{h}': (l,h) for l in range(2) for h in range(4)}
    summary = {'schema': 1, 'protocol_sha256': PROTOCOL_SHA256, 'n': 512,
               'calibration_n': 512, 'limits': p['limits'], 'seeds': []}
    with transaction(destination) as out:
        budget.check()
        for name in ('calibration.json','evaluation.json','inputs.json'):
            shutil.copyfile(data/name, out/name)
        shutil.copyfile(protocol_path, out/'protocol.json')
        timings = {}
        for seed in p['training_seeds']:
            started = time.monotonic()
            model = Transformer().eval()
            load_weights(model, root/f'experiments/final/seed-{seed}.npz')
            before = state_hash(model)
            means = calibrate(model, cx, budget)
            saved_means = means.clone()
            logits = []
            for method in p['methods']:
                predictions = []
                for name in p['conditions']:
                    heads = [] if name in ('intact','noop') else [addresses[h] for h in name.split('+')]
                    context = (zero_heads(model, heads, noop=name=='noop') if method=='zero'
                               else mean_heads(model, means, heads, noop=name=='noop'))
                    predictions.append(infer(model,x,context,budget))
                logits.append(np.stack(predictions))
            logits = np.stack(logits)
            if not all(np.array_equal(logits[0,0], logits[m,c]) for m in range(2) for c in (0,1)):
                raise ValueError('mean no-op differs')
            if before != state_hash(model) or not torch.equal(means,saved_means):
                raise ValueError('weights or fixed means changed')
            values = np.stack([outcomes(a,y.numpy()) for a in logits])
            s = summarize(values,rows,p,budget)
            s.update(seed=seed,checkpoint_sha256=sha256(root/f'experiments/final/seed-{seed}.npz'),
                     weights_unchanged=True, means_unchanged=True, noop_max_logit_difference=0.0)
            summary['seeds'].append(s)
            np.save(out/f'means-{seed}.npy',means.numpy(),allow_pickle=False)
            # Separate method files keep every artifact well below 10 MiB.
            for m, method in enumerate(p['methods']):
                np.save(out/f'{method}-logits-{seed}.npy',logits[m],allow_pickle=False)
                np.save(out/f'{method}-outcomes-{seed}.npy',values[m],allow_pickle=False)
            timings[str(seed)] = time.monotonic()-started
            print(f'Seed {seed}: both methods, all singles/pairs, {timings[str(seed)]:.3f}s',flush=True)
        write_json(out/'summary.json',summary)
        from .mean_report import render_directory
        render_directory(out,out/'report.html')
        sources(root,protocol_path)
        budget.check()
        write_json(out/'provenance.json',{'protocol_sha256':PROTOCOL_SHA256,'environment':environment(),
                   'elapsed_seconds':time.monotonic()-budget.start,'per_seed_seconds':timings,
                   'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   'evaluation_budget_seconds':max_seconds,'data_manifest_sha256':sha256(data/'manifest.json'),
                   'source_sha256':{f.name:sha256(f) for f in Path(__file__).parent.glob('*.py')},
                   'artifact_bytes_before_provenance':sum(v['bytes'] for v in manifest(out).values()),'completion':'complete'})
        write_json(out/'manifest.json',manifest(out))
    return summary


def replay(root, protocol_path, data, reference, destination, max_seconds=7200):
    reference = Path(reference)
    verify_manifest(reference)
    with transaction(destination) as out:
        evaluate(root,protocol_path,data,out/'evaluation',max_seconds)
        generated = out/'evaluation'
        checks = []
        for path in sorted(reference.iterdir()):
            if path.name in ('provenance.json','manifest.json'):
                continue
            identical = sha256(path)==sha256(generated/path.name)
            checks.append({'file':path.name,'byte_identical':identical})
            if not identical:
                raise ValueError(f'mean deterministic replay differs: {path.name}')
        write_json(out/'replay.json',{'all_deterministic_files_identical':True,'checks':checks})
        write_json(out/'manifest.json',manifest(out))
