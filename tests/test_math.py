"""Independent scalar-loop NumPy oracle; no production attention helpers."""
import math
import numpy as np
import pytest
import torch
from attention_lab.model import Attention, Config, Transformer, save_weights, load_weights

torch.set_num_threads(1)


def reference(module, x, ablated=None):
    w = module.qkv.weight.detach().numpy()
    bias = module.qkv.bias.detach().numpy()
    wo = module.out.weight.detach().numpy()
    bo = module.out.bias.detach().numpy()
    b, t, width = x.shape
    heads, dim = module.heads, module.d
    projected = x @ w.T + bias
    pattern = np.zeros((b, heads, t, t))
    z = np.zeros((b, t, width))
    for batch in range(b):
        for head in range(heads):
            for query in range(t):
                q = projected[batch, query, head*dim:(head+1)*dim]
                scores = []
                for key in range(query+1):
                    k = projected[batch, key, width+head*dim:width+(head+1)*dim]
                    scores.append(sum(float(a)*float(c) for a, c in zip(q, k))/math.sqrt(dim))
                probs = np.exp(np.asarray(scores) - max(scores))
                probs /= sum(probs)
                pattern[batch, head, query, :query+1] = probs
                if head != ablated:
                    for key, p in enumerate(probs):
                        for feature in range(dim):
                            z[batch, query, head*dim+feature] += p * projected[batch, key, 2*width+head*dim+feature]
    return z @ wo.T + bo, pattern


@pytest.mark.parametrize("seed", range(240))
def test_independent_attention(seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    heads = int(rng.integers(1, 5))
    cfg = Config(width=heads*int(rng.integers(4, 9)), heads=heads)
    module = Attention(cfg).double()
    x = torch.tensor(rng.normal(size=(int(rng.integers(1, 3)), int(rng.integers(1, 8)), cfg.width)))
    saved = {k: v.clone() for k, v in module.state_dict().items()}
    normal, pattern = module(x)
    expected, ep = reference(module, x.numpy())
    np.testing.assert_allclose(normal.detach(), expected, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(pattern.detach(), ep, atol=1e-11, rtol=1e-11)
    assert torch.count_nonzero(torch.triu(pattern, 1)) == 0
    assert torch.equal(normal, module(x, noop=True)[0])
    # Every head, not just head zero. Compare after a nonzero output bias.
    for head in range(heads):
        ablated, ap = module(x, head=head)
        expected, _ = reference(module, x.numpy(), head)
        np.testing.assert_allclose(ablated.detach(), expected, atol=1e-11, rtol=1e-11)
        assert torch.equal(ap, pattern)
    # Perturb all future inputs: earlier outputs must not change.
    if x.shape[1] > 1:
        modified = x.clone()
        modified[:, 1:] += 100
        assert torch.equal(module(modified)[0][:, 0], normal[:, 0])
    assert all(torch.equal(saved[k], v) for k, v in module.state_dict().items())


def test_model_noop_weights_and_checkpoint(tmp_path):
    torch.manual_seed(121)
    m = Transformer()
    x = torch.randint(0, 33, (12, 10))
    baseline = m(x)[0]
    assert torch.equal(baseline, m(x, noop=True)[0])
    saved = {k: v.clone() for k, v in m.state_dict().items()}
    for layer in range(2):
        for head in range(4):
            assert not torch.equal(baseline, m(x, (layer, head))[0])
    assert all(torch.equal(saved[k], v) for k, v in m.state_dict().items())
    save_weights(m, tmp_path / "weights.npz")
    restored = Transformer()
    load_weights(restored, tmp_path / "weights.npz")
    assert torch.equal(baseline, restored(x)[0])


@pytest.mark.parametrize('seed',range(24))
def test_layer_and_head_address_matches_independent_projection_removal(seed):
    """Alternative intervention: remove W_O columns; never call head-ablation code."""
    import copy
    torch.manual_seed(2000+seed)
    heads=1+seed%4
    cfg=Config(width=4*heads,heads=heads,layers=2,ff=16)
    model=Transformer(cfg).double().eval()
    tokens=torch.randint(0,33,(2,10))
    for layer in range(2):
        for head in range(heads):
            reference=copy.deepcopy(model)
            dim=cfg.width//heads
            with torch.no_grad():
                reference.blocks[layer].attn.out.weight[:,head*dim:(head+1)*dim]=0
            expected=reference(tokens)[0]
            actual=model(tokens,(layer,head))[0]
            torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    with pytest.raises(ValueError):model(tokens,(2,0))
    with pytest.raises(ValueError):model(tokens,(0,heads))
