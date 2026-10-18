"""Independent projected-contribution reference; 224 seeded simultaneous cases."""
import copy
import itertools
import math
import numpy as np
import pytest
import torch
from attention_lab.model import Config, Transformer
from attention_lab.mean_intervention import mean_heads, calibrate
from attention_lab.artifacts import Budget
from attention_lab.interaction_stats import interaction_values

PAIRS=list(itertools.combinations(itertools.product(range(2),range(4)),2))
torch.set_num_threads(1)


def projected_reference(model, means, heads):
    ref=copy.deepcopy(model)
    with torch.no_grad():
        for layer in range(model.cfg.layers):
            out=ref.blocks[layer].attn.out
            addition=torch.zeros((model.cfg.length,model.cfg.width),dtype=out.weight.dtype)
            dim=model.cfg.width//model.cfg.heads
            for l,h in heads:
                if l==layer:
                    addition += means[l,:,h] @ out.weight[:,h*dim:(h+1)*dim].T
                    out.weight[:,h*dim:(h+1)*dim]=0
            out.register_forward_hook(lambda module,args,result,addition=addition: result+addition)
    return ref


@pytest.mark.parametrize('seed',range(224))
def test_mean_placement_order_weights_and_arithmetic(seed):
    torch.manual_seed(71000+seed)
    model=Transformer(Config(width=16,ff=16)).double().eval()
    tokens=torch.randint(0,33,(1+seed%2,10))
    means=torch.randn(2,10,4,4,dtype=torch.float64)
    pair=PAIRS[seed%28]
    saved={k:v.clone() for k,v in model.state_dict().items()}
    original=means.clone()
    results=[]
    for heads in ([],[pair[0]],[pair[1]],pair):
        expected=projected_reference(model,means,heads)(tokens)[0]
        with mean_heads(model,means,heads):
            actual=model(tokens)[0]
        torch.testing.assert_close(actual,expected,atol=2e-12,rtol=2e-12)
        results.append(actual.detach().numpy())
    with mean_heads(model,means,tuple(reversed(pair))):
        np.testing.assert_array_equal(model(tokens)[0].detach().numpy(),results[3])
    with mean_heads(model,means,noop=True):
        np.testing.assert_array_equal(model(tokens)[0].detach().numpy(),results[0])
    assert all(torch.equal(saved[k],v) for k,v in model.state_dict().items())
    assert torch.equal(means,original)
    assert all(not b.attn.out._forward_pre_hooks for b in model.blocks)
    # Arbitrary paired metrics, independent scalar fsum interaction and difference.
    rng=np.random.default_rng(seed)
    a=rng.normal(size=(2,4,7,3))
    actual=[interaction_values(v,['intact','a','b','a+b'],[['a','b']])[0] for v in a]
    for i in range(7):
        for m in range(3):
            expected=[math.fsum([v[3,i,m],-v[1,i,m],-v[2,i,m],v[0,i,m]]) for v in a]
            np.testing.assert_allclose([actual[0][i,m],actual[1][i,m]],expected,atol=1e-14)
            assert abs((actual[1][i,m]-actual[0][i,m])-math.fsum([expected[1],-expected[0]]))<1e-14


def test_calibration_intact_position_reduction_independent_qkv():
    torch.manual_seed(999)
    model=Transformer(Config(width=16,ff=16)).eval()
    tokens=torch.randint(0,33,(257,10))
    collected=[[],[]];handles=[]
    for layer,block in enumerate(model.blocks):
        def capture(module,args,layer=layer):
            x=args[0];b,t,w=x.shape
            q,k,v=module.qkv(x).reshape(b,t,3,4,4).permute(2,0,3,1,4)
            # Independent row-wise causal softmax instead of the production mask.
            z=torch.empty_like(v)
            for pos in range(t):
                weights=(q[:,:,pos:pos+1]@k[:,:,:pos+1].transpose(-1,-2)/2).softmax(-1)
                z[:,:,pos:pos+1]=weights@v[:,:,:pos+1]
            collected[layer].append(z.transpose(1,2).detach())
        handles.append(block.attn.register_forward_pre_hook(capture))
    with torch.no_grad():
        for batch in tokens.split(128):model(batch)
    for h in handles:h.remove()
    expected=torch.stack([torch.cat(v).double().mean(0).float() for v in collected])
    actual=calibrate(model,tokens,Budget(30))
    torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-6)
    assert not torch.equal(actual[:,0],actual[:,9])


@pytest.mark.parametrize('bad',[[(0,0),(0,0)],[(2,0)],[(0,4)],[(True,1)],'x',[(0,0)]*3])
def test_invalid_addresses_cleanup(bad):
    model=Transformer().eval();means=torch.zeros(2,10,4,16)
    with pytest.raises(ValueError):
        with mean_heads(model,means,bad):pass
    assert all(not b.attn.out._forward_pre_hooks for b in model.blocks)


@pytest.mark.parametrize('kind',['shape','dtype','nan','grad'])
def test_invalid_means(kind):
    model=Transformer().eval();means=torch.zeros(2,10,4,16)
    if kind=='shape':means=means[:1]
    if kind=='dtype':means=means.double()
    if kind=='nan':means[0,0,0,0]=float('nan')
    if kind=='grad':means.requires_grad_()
    with pytest.raises(ValueError):
        with mean_heads(model,means,[(0,0)]):pass


def test_cleanup_and_intact_calibration_guard():
    model=Transformer().eval();means=torch.zeros(2,10,4,16)
    with pytest.raises(RuntimeError):
        with mean_heads(model,means,[(0,0),(1,3)]):raise RuntimeError('interrupt')
    assert all(not b.attn.out._forward_pre_hooks for b in model.blocks)
    with mean_heads(model,means,[(0,0)]):
        with pytest.raises(ValueError):calibrate(model,torch.zeros(1,10,dtype=torch.long),Budget(1))
