"""Numerical controls for real pretrained interventions, using independent paths."""
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.nn import functional as F
pytest.importorskip('transformers')
from attention_lab.pretrained import load, intervention, calibrate, losses, read

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def model():
    if not (ROOT/'.cache/pretrained/model.safetensors').exists():
        pytest.skip('Run python -m attention_lab.pretrained fetch for real-model checks')
    torch.set_num_threads(2)
    return load(ROOT/'experiments/pretrained',ROOT/'.cache/pretrained')[0]

@pytest.fixture(scope='module')
def blocks():
    return read(ROOT/'experiments/pretrained/inputs.json')

@pytest.mark.parametrize('layer,head',[(0,0),(2,3),(5,1)])
@torch.inference_mode()
def test_zero_matches_independent_value_mask(model,blocks,layer,head):
    x=torch.tensor(blocks['evaluation'][:1])[:,:-1]
    # Zero V before attention weighting, independently of the production z hook.
    # The pinned library's public head_mask path is inconsistent (additive then
    # multiplicative masking), so it is not a valid reference.
    def zero_value(module,args,output):
        result=output.clone()
        result[...,head*96+64:head*96+96]=0
        return result
    handle=model.gpt_neox.layers[layer].attention.query_key_value.register_forward_hook(zero_value)
    try:expected=model(x,use_cache=False).logits
    finally:handle.remove()
    with intervention(model,layer,head,'zero'):
        actual=model(x,use_cache=False).logits
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)

@pytest.mark.parametrize('layer,head',[(0,2),(3,1),(5,3)])
@torch.inference_mode()
def test_mean_matches_post_projection_delta(model,blocks,layer,head):
    x=torch.tensor(blocks['evaluation'][:1])[:,:-1]
    # Float64 separates algebraic correctness from float32 cancellation amplified
    # by subsequent layers under this deliberately atypical replacement.
    import copy
    model=copy.deepcopy(model).double()
    means=torch.linspace(-.5,.5,6*4*32,dtype=torch.float64).reshape(6,4,32)
    dense=model.gpt_neox.layers[layer].attention.dense
    def reference(module,args,output):
        section=slice(head*32,(head+1)*32)
        change=means[layer,head]-args[0][...,section]
        return output+F.linear(change,module.weight[:,section],bias=None)
    handle=dense.register_forward_hook(reference)
    try:expected=model(x,use_cache=False).logits
    finally:handle.remove()
    with intervention(model,layer,head,'mean',means):
        actual=model(x,use_cache=False).logits
    torch.testing.assert_close(actual,expected,rtol=1e-8,atol=1e-8)

@torch.inference_mode()
def test_means_use_separate_intact_calibration(model,blocks):
    actual=calibrate(model,blocks['calibration'])
    captured=[[] for _ in range(6)];handles=[]
    def capture(layer):
        def hook(module,args):captured[layer].append(args[0].reshape(-1,4,32).clone())
        return hook
    try:
        for layer in range(6):handles.append(model.gpt_neox.layers[layer].attention.dense.register_forward_pre_hook(capture(layer)))
        # Different batching from the production estimator.
        for block in blocks['calibration']:
            model(torch.tensor([block[:-1]]),use_cache=False)
    finally:
        for handle in handles:handle.remove()
    expected=torch.stack([torch.cat(layer).double().mean(0).float() for layer in captured])
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    saved=np.load(ROOT/'results/pretrained/measurements.npz')
    np.testing.assert_allclose(actual.numpy(),saved['means'],rtol=1e-5,atol=1e-6)

@torch.inference_mode()
def test_noop_and_exception_cleanup(model,blocks):
    x=torch.tensor(blocks['evaluation'][:1])[:,:-1]
    before=model(x,use_cache=False).logits
    with intervention(model,0,0,'noop'):
        torch.testing.assert_close(model(x,use_cache=False).logits,before,rtol=0,atol=0)
    with pytest.raises(RuntimeError):
        with intervention(model,0,0,'zero'):raise RuntimeError('interrupted')
    torch.testing.assert_close(model(x,use_cache=False).logits,before,rtol=0,atol=0)
    assert not model.gpt_neox.layers[0].attention.dense._forward_pre_hooks

@torch.inference_mode()
def test_loss_alignment_against_manual_logsoftmax(model,blocks):
    block=torch.tensor(blocks['evaluation'][:2])
    logits=model(block[:,:-1],use_cache=False).logits
    expected=-logits.log_softmax(-1).gather(-1,block[:,1:,None]).squeeze(-1)
    actual,_=losses(model,blocks['evaluation'][:2])
    np.testing.assert_allclose(actual,expected.numpy(),atol=1e-6)


def test_saved_result_independent_audit():
    import subprocess,sys
    subprocess.run([sys.executable,str(ROOT/'scripts/verify_pretrained.py')],check=True)
