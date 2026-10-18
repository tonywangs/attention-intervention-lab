"""Position-specific calibration and simultaneous fixed-mean z replacement."""
from contextlib import contextmanager
import torch


def validate_means(model, means):
    cfg = model.cfg
    expected = (cfg.layers, cfg.length, cfg.heads, cfg.width // cfg.heads)
    parameter = next(model.parameters())
    if (not isinstance(means, torch.Tensor) or tuple(means.shape) != expected
            or means.dtype != parameter.dtype or means.device != parameter.device
            or means.requires_grad or not torch.isfinite(means).all()):
        raise ValueError('invalid calibration means')


@torch.no_grad()
def calibrate(model, tokens, budget):
    if (model.training or tokens.ndim != 2 or tokens.shape[1] != model.cfg.length
            or not 1 <= len(tokens) <= 512):
        raise ValueError('calibration requires bounded full-length inputs and eval mode')
    if any(b.attn.out._forward_pre_hooks for b in model.blocks):
        raise ValueError('calibration requires intact model without intervention hooks')
    cfg = model.cfg
    sums = torch.zeros((cfg.layers, cfg.length, cfg.width), dtype=torch.float64)
    counts = [0] * cfg.layers
    handles = []
    try:
        for layer, block in enumerate(model.blocks):
            def collect(module, args, layer=layer):
                sums[layer].add_(args[0].detach().double().sum(dim=0))
                counts[layer] += len(args[0])
            handles.append(block.attn.out.register_forward_pre_hook(collect))
        for batch in tokens.split(128):
            budget.check()
            model(batch)
    finally:
        for handle in handles:
            handle.remove()
    if counts != [len(tokens)] * cfg.layers:
        raise ValueError('incomplete calibration')
    return (sums / len(tokens)).reshape(cfg.layers, cfg.length, cfg.heads, -1).to(next(model.parameters()).dtype)


@contextmanager
def mean_heads(model, means, heads=(), noop=False):
    validate_means(model, means)
    if not isinstance(heads, (list, tuple)) or len(heads) > 2:
        raise ValueError('expected at most two head addresses')
    addresses = []
    for address in heads:
        if (not isinstance(address, (list, tuple)) or len(address) != 2
                or any(type(n) is not int for n in address)
                or not 0 <= address[0] < model.cfg.layers
                or not 0 <= address[1] < model.cfg.heads):
            raise ValueError('invalid head address')
        addresses.append(tuple(address))
    if len(set(addresses)) != len(addresses):
        raise ValueError('duplicate head address')
    if noop and addresses:
        raise ValueError('no-op cannot select heads')
    handles = []
    dim = model.cfg.width // model.cfg.heads
    try:
        for layer, block in enumerate(model.blocks):
            selected = tuple(sorted(h for l, h in addresses if l == layer))
            if not selected and not noop:
                continue
            def replace(module, args, selected=selected, layer=layer):
                z = args[0].clone()
                if z.shape[1] != model.cfg.length:
                    raise ValueError('mean replacement requires full sequence length')
                for head in selected:
                    z[..., head*dim:(head+1)*dim] = means[layer, :, head, :]
                return (z,)
            handles.append(block.attn.out.register_forward_pre_hook(replace))
        yield
    finally:
        for handle in handles:
            handle.remove()
