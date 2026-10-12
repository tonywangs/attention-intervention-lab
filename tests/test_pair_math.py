"""Independent W_O-column removal, activation placement and scalar arithmetic."""
import copy
import itertools
import math
import numpy as np
import pytest
import torch
from attention_lab.model import Config, Transformer
from attention_lab.pair_intervention import zero_heads
from attention_lab.interaction_stats import outcomes, interaction_values

torch.set_num_threads(1)
ADDRESSES = list(itertools.product(range(2), range(4)))
PAIRS = list(itertools.combinations(ADDRESSES, 2))


@pytest.mark.parametrize('seed', range(224))
def test_simultaneous_intervention_independent_reference(seed):
    torch.manual_seed(8000+seed)
    cfg = Config(width=16, heads=4, layers=2, ff=16)
    model = Transformer(cfg).double().eval()
    tokens = torch.randint(0, 33, (1+seed % 2, 1+seed % 10))
    pair = PAIRS[seed % 28]
    saved = {k: v.clone() for k, v in model.state_dict().items()}
    reference = copy.deepcopy(model)
    # Independent intervention in PARAMETER space, used only on this test copy.
    # Never calls the production hook or single-head masking code.
    with torch.no_grad():
        for layer, head in pair:
            reference.blocks[layer].attn.out.weight[:, head*4:(head+1)*4] = 0
    expected = reference(tokens)[0]
    raw, masked, spies = {}, {}, []
    for layer, block in enumerate(model.blocks):
        spies.append(block.attn.out.register_forward_pre_hook(
            lambda module, args, layer=layer: raw.__setitem__(layer, args[0].detach().clone())))
    try:
        with zero_heads(model, pair):
            for layer, block in enumerate(model.blocks):
                spies.append(block.attn.out.register_forward_pre_hook(
                    lambda module, args, layer=layer: masked.__setitem__(layer, args[0].detach().clone())))
            actual = model(tokens)[0]
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        for layer in range(2):
            for head in range(4):
                selected = masked[layer][..., head*4:(head+1)*4]
                if (layer, head) in pair:
                    assert torch.count_nonzero(selected) == 0
                else:
                    assert torch.equal(selected, raw[layer][..., head*4:(head+1)*4])
    finally:
        for handle in spies:
            handle.remove()
    with zero_heads(model, tuple(reversed(pair))):
        assert torch.equal(model(tokens)[0], actual)
    intact = model(tokens)[0]
    with zero_heads(model, noop=True):
        assert torch.equal(intact, model(tokens)[0])
    with zero_heads(model, [pair[0]]):
        single = model(tokens)[0]
    assert torch.equal(single, model(tokens, intervention=pair[0])[0])
    assert all(torch.equal(saved[k], v) for k, v in model.state_dict().items())
    assert all(not b.attn.out._forward_pre_hooks for b in model.blocks)


@pytest.mark.parametrize('seed', range(224))
def test_scalar_outcomes_and_interaction_arithmetic(seed):
    rng = np.random.default_rng(seed+40000)
    n = 1+seed % 12
    logits = rng.normal(size=(4, n, 16))*5
    answers = rng.integers(0, 16, n)
    names = ['intact', 'L0H0', 'L1H3', 'L0H0+L1H3']
    actual = outcomes(logits, answers)
    expected = np.empty((4, n, 3))
    for c in range(4):
        for i in range(n):
            row = logits[c, i].tolist()
            denominator = math.fsum(math.exp(v-max(row)) for v in row)
            p = math.exp(row[answers[i]]-max(row))/denominator
            expected[c, i] = (-math.log(p), p, float(row.index(max(row)) == answers[i]))
    np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=1e-13)
    interaction = interaction_values(actual, names, [['L0H0', 'L1H3']])[0]
    reference = np.array([[math.fsum([expected[3, i, m], -expected[1, i, m],
                                      -expected[2, i, m], expected[0, i, m]])
                           for m in range(3)] for i in range(n)])
    np.testing.assert_allclose(interaction, reference, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize('addresses', [[(0, 0), (0, 0)], [(2, 0)], [(0, 4)], [(True, 1)],
                                      [(0, 1, 2)], [(0, 0), (0, 1), (0, 2)], 'L0H0'])
def test_invalid_interventions_leave_no_hooks(addresses):
    model = Transformer()
    with pytest.raises(ValueError):
        with zero_heads(model, addresses):
            pass
    assert all(not b.attn.out._forward_pre_hooks for b in model.blocks)


def test_hook_cleanup_on_failure():
    model = Transformer()
    with pytest.raises(RuntimeError):
        with zero_heads(model, [(0, 1), (1, 2)]):
            raise RuntimeError('simulated interrupted forward')
    assert all(not b.attn.out._forward_pre_hooks for b in model.blocks)
