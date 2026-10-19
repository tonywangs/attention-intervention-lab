"""Pinned Pythia CPU experiment. Downloads are explicit; inference is offline."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import platform
import time
import urllib.request
import numpy as np
import torch
from torch.nn import functional as F


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def verify_manifest(directory):
    directory = Path(directory)
    for name, expected in read(directory / 'manifest.json').items():
        if digest(directory / name) != expected:
            raise ValueError(f'Artifact hash mismatch: {directory / name}')


def fetch(spec_dir, cache):
    """Public HTTPS only; no hub login, token discovery or executable model code."""
    spec_dir, cache = Path(spec_dir), Path(cache)
    verify_manifest(spec_dir)
    spec = read(spec_dir / 'model.json')
    cache.mkdir(parents=True, exist_ok=True)
    for name, expected in spec['files'].items():
        target = cache / name
        if target.exists() and digest(target) == expected:
            continue
        url = f"https://huggingface.co/{spec['model_id']}/resolve/{spec['revision']}/{name}"
        temporary = target.with_suffix(target.suffix + '.part')
        try:
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as f:
                while chunk := response.read(1024 * 1024):
                    f.write(chunk)
            if digest(temporary) != expected:
                raise ValueError(f'Download hash mismatch: {name}')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f'Verified {name}', flush=True)


def load(spec_dir, cache):
    from transformers import AutoTokenizer, GPTNeoXForCausalLM
    verify_manifest(spec_dir)
    spec = read(Path(spec_dir) / 'model.json')
    for name, expected in spec['files'].items():
        if digest(Path(cache) / name) != expected:
            raise ValueError(f'Model hash mismatch: {name}')
    tokenizer = AutoTokenizer.from_pretrained(cache, local_files_only=True, trust_remote_code=False)
    model = GPTNeoXForCausalLM.from_pretrained(
        cache, local_files_only=True, use_safetensors=True, trust_remote_code=False,
        torch_dtype=torch.float32, attn_implementation='eager').cpu().eval()
    model.requires_grad_(False)
    if (model.config.num_hidden_layers, model.config.num_attention_heads, model.config.hidden_size) != (6, 4, 128):
        raise ValueError('Unsupported architecture')
    return model, tokenizer


@contextmanager
def intervention(model, layer, head, method, means=None):
    """Modify exactly one contiguous head slice before the shared output projection."""
    if method not in ('noop', 'zero', 'mean'):
        raise ValueError(method)
    if not 0 <= layer < 6 or not 0 <= head < 4:
        raise ValueError('Head index out of bounds')
    if method == 'mean' and (means is None or tuple(means.shape) != (6, 4, 32)):
        raise ValueError('Expected calibration means [6,4,32]')
    def hook(module, args):
        original = args[0]
        output = original.clone()
        section = slice(head * 32, (head + 1) * 32)
        if method == 'noop':
            output[..., section] = original[..., section]
        elif method == 'zero':
            output[..., section] = 0
        else:
            output[..., section] = means[layer, head].to(output)
        return (output,) + args[1:]
    handle = model.gpt_neox.layers[layer].attention.dense.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def batches(blocks, batch_size=2):
    for start in range(0, len(blocks), batch_size):
        yield torch.tensor(blocks[start:start + batch_size], dtype=torch.long)


@torch.inference_mode()
def calibrate(model, blocks):
    sums = torch.zeros(6, 4, 32, dtype=torch.float64)
    counts = [0] * 6
    handles = []
    def capture(layer):
        def hook(module, args):
            x = args[0].reshape(-1, 4, 32)
            sums[layer].add_(x.double().sum(0))
            counts[layer] += x.shape[0]
        return hook
    try:
        for layer in range(6):
            handles.append(model.gpt_neox.layers[layer].attention.dense.register_forward_pre_hook(capture(layer)))
        for block in batches(blocks):
            model(block[:, :-1], use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    assert counts == [len(blocks) * 64] * 6
    return (sums / counts[0]).float()


@torch.inference_mode()
def losses(model, blocks, capture=False):
    values, attention = [], []
    for i, block in enumerate(batches(blocks)):
        output = model(block[:, :-1], use_cache=False, output_attentions=capture and i == 0)
        values.append(F.cross_entropy(output.logits.flatten(0, 1), block[:, 1:].reshape(-1), reduction='none').reshape(-1, 64).numpy())
        if capture and i == 0:
            attention = torch.stack(output.attentions, dim=1).numpy()  # example, layer, head, query, key
    return np.concatenate(values), attention


def summarize(base, zero, mean, noop):
    rows = []
    for layer in range(6):
        for head in range(4):
            row = {'layer': layer, 'head': head}
            for method, array in [('zero', zero), ('mean', mean)]:
                delta = array[layer, head] - base
                blocks = delta.mean(-1)
                row[method] = {'delta_nats': float(delta.mean()), 'loss_nats': float(array[layer, head].mean()),
                               'block_deltas': blocks.tolist(), 'min_block_delta': float(blocks.min()),
                               'max_block_delta': float(blocks.max()), 'harmed_blocks': int((blocks > 0).sum()),
                               'improved_blocks': int((blocks < 0).sum())}
            row['mean_minus_zero_nats'] = float((mean[layer, head] - zero[layer, head]).mean())
            rows.append(row)
    return {'intact_loss_nats': float(base.mean()), 'noop_max_abs_token_loss_delta': float(np.abs(noop-base).max()),
            'evaluation_blocks': len(base), 'scored_tokens': int(base.size), 'heads': rows}


def run(spec_dir, cache, output):
    import transformers
    spec_dir, output = Path(spec_dir), Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}; use a new output directory')
    torch.set_num_threads(2)
    torch.manual_seed(1729)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    model, tokenizer = load(spec_dir, cache)
    inputs = read(spec_dir / 'inputs.json')
    for split, blocks in inputs.items():
        ids = tokenizer.encode((spec_dir / (split + '.txt')).read_text(), add_special_tokens=False)
        assert blocks == [ids[i*65:(i+1)*65] for i in range(16)], 'Tokenization differs from preserved inputs'
    means = calibrate(model, inputs['calibration'])
    base, attention = losses(model, inputs['evaluation'], capture=True)
    arrays = {method: np.empty((6, 4, 16, 64), dtype=np.float32) for method in ('noop', 'zero', 'mean')}
    for layer in range(6):
        for head in range(4):
            for method in arrays:
                with intervention(model, layer, head, method, means):
                    arrays[method][layer, head], _ = losses(model, inputs['evaluation'])
            print(f'L{layer}H{head} evaluated', flush=True)
    assert np.array_equal(arrays['noop'], np.broadcast_to(base, arrays['noop'].shape)), 'No-op control failed'
    output.mkdir(parents=True)
    np.savez_compressed(output / 'measurements.npz', intact=base, means=means.numpy(), attention=attention, **arrays)
    summary = summarize(base, arrays['zero'], arrays['mean'], arrays['noop'])
    write(output / 'summary.json', summary)
    write(output / 'examples.json', [{'index': i, 'token_ids': block, 'tokens': [tokenizer.decode([t]) for t in block],
                                    'text': tokenizer.decode(block)} for i, block in enumerate(inputs['evaluation'][:2])])
    write(output / 'provenance.json', {'model': read(spec_dir / 'model.json'), 'protocol_sha256': digest(spec_dir / 'protocol.json'),
          'inputs_sha256': digest(spec_dir / 'inputs.json'), 'spec_manifest_sha256': digest(spec_dir / 'manifest.json'),
          'implementation_sha256': digest(__file__),
          'renderer_sha256': digest(Path(__file__).with_name('pretrained_report.py')), 'python': platform.python_version(), 'platform': platform.platform(),
          'torch': torch.__version__, 'numpy': np.__version__, 'transformers': transformers.__version__,
          'elapsed_seconds': time.perf_counter()-started, 'device': 'cpu', 'threads': 2, 'dtype': 'float32'})
    from .pretrained_report import render
    render(output, spec_dir)
    write(output / 'manifest.json', {p.name: digest(p) for p in sorted(output.iterdir())})
    print(json.dumps({'intact_loss_nats': summary['intact_loss_nats'], 'output': str(output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['fetch', 'run', 'report', 'verify'])
    parser.add_argument('--spec', type=Path, default=Path('experiments/pretrained'))
    parser.add_argument('--cache', type=Path, default=Path('.cache/pretrained'))
    parser.add_argument('--output', type=Path, default=Path('results/pretrained'))
    args = parser.parse_args()
    if args.command == 'fetch':
        fetch(args.spec, args.cache)
    elif args.command == 'run':
        run(args.spec, args.cache, args.output)
    elif args.command == 'report':
        from .pretrained_report import render
        verify_manifest(args.output)
        verify_manifest(args.spec)
        provenance = read(args.output / 'provenance.json')
        provenance['renderer_sha256'] = digest(Path(__file__).with_name('pretrained_report.py'))
        write(args.output / 'provenance.json', provenance)
        render(args.output, args.spec)
        write(args.output / 'manifest.json', {p.name: digest(p) for p in sorted(args.output.iterdir()) if p.name != 'manifest.json'})
    else:
        verify_manifest(args.spec)
        verify_manifest(args.output)
        print('Pretrained input and result hashes verified')


if __name__ == '__main__':
    main()
