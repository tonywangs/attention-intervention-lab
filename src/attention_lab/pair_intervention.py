"""Simultaneous z interventions without altering the historical model implementation."""
from contextlib import contextmanager


@contextmanager
def zero_heads(model, heads=(), noop=False):
    """Mask the flattened B,T,H*D input to W_O, preserving its bias.

    Hooks are private to this synchronous context and removed even on failure.
    Do not share a model concurrently while an intervention is active.
    """
    if not isinstance(heads, (list, tuple)) or len(heads) > 2:
        raise ValueError("expected at most two head addresses")
    addresses = []
    for address in heads:
        if (not isinstance(address, (list, tuple)) or len(address) != 2
                or any(type(n) is not int for n in address)):
            raise ValueError("invalid head address")
        layer, head = address
        if not 0 <= layer < model.cfg.layers or not 0 <= head < model.cfg.heads:
            raise ValueError("head address out of range")
        addresses.append((layer, head))
    if len(set(addresses)) != len(addresses):
        raise ValueError("duplicate head address")
    handles = []
    dim = model.cfg.width // model.cfg.heads
    try:
        for layer, block in enumerate(model.blocks):
            selected = tuple(sorted(h for l, h in addresses if l == layer))
            if not selected and not noop:
                continue
            def mask(module, args, selected=selected):
                z = args[0].clone()
                for head in selected:
                    z[..., head*dim:(head+1)*dim] = 0
                return (z,)
            handles.append(block.attn.out.register_forward_pre_hook(mask))
        yield
    finally:
        for handle in handles:
            handle.remove()
