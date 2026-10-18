#!/usr/bin/env python3
"""Independent data, calibration, intervention, metric and bootstrap recomputation.

Does not call production calibration, intervention, outcomes, interaction or
summary routines. Recomputes attention with direct B,H,T,D replacement; the
seeded float64 tests separately use projection-column removal plus offsets.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import types
import numpy as np
import torch
from attention_lab.artifacts import verify_manifest
from attention_lab.model import Transformer, load_weights

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'results/mean'

def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def close(a,b):np.testing.assert_allclose(a,b,atol=1e-10,rtol=1e-10)


def key(row):
    t=row['tokens']
    assert len(t)==10 and t[8]==32
    assert len(set(t[:8:2]))==4 and len(set(t[1:8:2]))==4
    assert all(type(x) is int and 0<=x<16 for x in t[:8:2])
    assert all(type(x) is int and 16<=x<32 for x in t[1:8:2])
    match=[i for i in range(0,8,2) if t[i]==t[9]]
    assert len(match)==1
    assert row['answer']==t[match[0]+1]-16 and row['value_position']==match[0]+1
    return tuple(sorted((t[i],t[i+1]) for i in range(0,8,2)))


@torch.no_grad()
def independent_means(model,rows):
    captured=[[],[]];handles=[]
    for layer,block in enumerate(model.blocks):
        def capture(module,args,layer=layer):
            x=args[0];b,t,w=x.shape
            q,k,v=module.qkv(x).reshape(b,t,3,4,16).permute(2,0,3,1,4)
            z=torch.empty_like(v)
            for pos in range(t):
                weights=(q[:,:,pos:pos+1]@k[:,:,:pos+1].transpose(-1,-2)/4).softmax(-1)
                z[:,:,pos:pos+1]=weights@v[:,:,:pos+1]
            captured[layer].append(z.transpose(1,2).double())
        handles.append(block.attn.register_forward_pre_hook(capture))
    try:
        for start in range(0,512,128):model(torch.tensor([r['tokens'] for r in rows[start:start+128]]))
    finally:
        for h in handles:h.remove()
    return torch.stack([torch.cat(v).mean(0).float() for v in captured])


@torch.no_grad()
def independent_logits(model,means,rows,conditions,method):
    result=[]
    for name in conditions:
        ref=copy.deepcopy(model)
        selected=[] if name in ('intact','noop') else [(int(h[1]),int(h[3])) for h in name.split('+')]
        for layer,block in enumerate(ref.blocks):
            selected_heads=[h for l,h in selected if l==layer]
            def attention(module,x,head=None,noop=False,layer=layer,selected_heads=selected_heads):
                b,t,w=x.shape
                q,k,v=module.qkv(x).reshape(b,t,3,4,16).permute(2,0,3,1,4)
                scores=q@k.transpose(-1,-2)/4
                causal=torch.arange(t)[None,:]>torch.arange(t)[:,None]
                pattern=scores.masked_fill(causal,-torch.inf).softmax(-1)
                z=pattern@v
                for h in selected_heads:
                    z[:,h,:,:]=means[layer,:,h,:] if method=='mean' else 0
                return module.out(z.permute(0,2,1,3).reshape(b,t,w)),pattern
            block.attn.forward=types.MethodType(attention,block.attn)
        chunks=[]
        for start in range(0,512,128):
            chunks.append(ref(torch.tensor([r['tokens'] for r in rows[start:start+128]]))[0].numpy())
        result.append(np.concatenate(chunks))
    return np.stack(result)


def scalar_metrics(logits,rows):
    result=np.empty((38,512,3))
    for c in range(38):
        for i,row in enumerate(rows):
            a=logits[c,i].astype(float).tolist();maximum=max(a)
            logp=a[row['answer']]-maximum-math.log(math.fsum(math.exp(v-maximum) for v in a))
            result[c,i]=[-logp,math.exp(logp),float(a.index(maximum)==row['answer'])]
    return result


def main():
    torch.set_num_threads(1)
    p=read(ROOT/'experiments/mean/protocol.json');freeze=read(ROOT/'experiments/mean/freeze.json')
    assert digest(ROOT/'experiments/mean/protocol.json')==freeze['protocol_sha256']==digest(D/'protocol.json')
    assert digest(ROOT/'experiments/mean/baseline.json')==p['baseline_sha256']
    for name,h in read(ROOT/'experiments/mean/baseline.json').items():
        if name.startswith(('experiments/','results/')):assert digest(ROOT/name)==h,name
    for folder in [D,ROOT/'experiments/mean/data']:verify_manifest(folder)
    forbidden=set()
    paths=list((ROOT/'experiments/final/data').glob('*.json'))+[ROOT/'results/evaluation/eval.json',ROOT/'results/interactions/examples.json']
    for path in paths:
        for row in read(path):forbidden.add(key(row))
    calibration,rows=[read(D/f'{kind}.json') for kind in ('calibration','evaluation')]
    for kind,rr in [('calibration',calibration),('evaluation',rows)]:
        assert len(rr)==512
        for i,r in enumerate(rr):
            k=key(r);assert k not in forbidden;forbidden.add(k)
            assert r['value_position']==2*(i%4)+1
            raw=json.dumps(k,separators=(',',':')).encode()
            assert int.from_bytes(hashlib.sha256(raw).digest()[:8],'big')%100>=90
        assert digest(D/f'{kind}.json')==digest(ROOT/f'experiments/mean/data/{kind}.json')
    summary=read(D/'summary.json');cfg=p['uncertainty'];names=p['conditions'];metrics=p['metrics']
    # Explicit index draws, independent of production bincount-weight matrices.
    rng=np.random.default_rng(cfg['seed'])
    groups=[np.array([i for i,r in enumerate(rows) if r['value_position']==pos]) for pos in (1,3,5,7)]
    draws=[g[rng.integers(0,len(g),size=(cfg['replicates'],len(g)))] for g in groups]
    for seed,s in zip(p['training_seeds'],summary['seeds']):
        assert seed==s['seed'] and s['family_size']==252
        model=Transformer().eval();load_weights(model,ROOT/f'experiments/final/seed-{seed}.npz')
        means=np.load(D/f'means-{seed}.npy',allow_pickle=False)
        np.testing.assert_allclose(independent_means(model,calibration).numpy(),means,atol=2e-7,rtol=2e-6)
        values=[]
        for method in ('zero','mean'):
            a=np.load(D/f'{method}-logits-{seed}.npy',allow_pickle=False)
            assert a.shape==(38,512,16) and a.dtype==np.float32 and np.isfinite(a).all()
            predicted=independent_logits(model,torch.from_numpy(means),rows,names,method)
            np.testing.assert_allclose(predicted,a,atol=2e-6,rtol=2e-6)
            np.testing.assert_array_equal(predicted.argmax(-1),a.argmax(-1))
            np.testing.assert_array_equal(a[0],a[1])
            v=scalar_metrics(a,rows)
            close(v,np.load(D/f'{method}-outcomes-{seed}.npy',allow_pickle=False));values.append(v)
        values=np.stack(values)
        inter=np.empty((3,28,512,3))
        for j,(a,b) in enumerate(p['pairs']):
            for m in range(2):
                inter[m,j]=values[m,names.index(a+'+'+b)]+values[m,0]-values[m,names.index(a)]-values[m,names.index(b)]
        inter[2]=inter[1]-inter[0]
        boot=np.zeros((cfg['replicates'],3,28,3))
        # Direct gathers in bounded chunks, with identical indices for all contrasts.
        for start in range(0,cfg['replicates'],100):
            stop=min(start+100,cfg['replicates'])
            for idx in draws:
                boot[start:stop]+=inter[:,:,idx[start:stop],:].sum(axis=3).transpose(2,0,1,3)/512
        avg=inter.mean(2);se=boot.std(0,ddof=1);active=se>cfg['degenerate_threshold']
        standardized=np.zeros_like(boot);np.divide(abs(boot-avg),se,out=standardized,where=active)
        q=np.quantile(standardized.reshape(cfg['replicates'],-1).max(1),.95)
        close(q,s['family_critical_value']);assert int(active.sum())==s['nondegenerate_family_size']
        point=np.quantile(boot,[.025,.975],axis=0)
        for k,method in enumerate(('zero','mean','difference')):
            for j,pair in enumerate(p['pairs']):
                entry=s['comparisons'][method][j];assert entry['pair']==pair
                for m,metric in enumerate(metrics):
                    got=entry['metrics'][metric];lo,hi=avg[k,j,m]+np.array([-1,1])*q*se[k,j,m]
                    close(got['mean'],avg[k,j,m]);close(got['bootstrap_se'],se[k,j,m]);close(got['ci95_family'],[lo,hi]);close(got['ci95_pointwise'],point[:,k,j,m])
                    assert got['degenerate']==bool(not active[k,j,m])
                    assert got['resolved']==bool(active[k,j,m] and (lo>2e-6 or hi < -2e-6))
            if k<2:
                effects=values[k]-values[k,0]
                eb=np.zeros((cfg['replicates'],38,3))
                for start in range(0,cfg['replicates'],100):
                    stop=min(start+100,cfg['replicates'])
                    for idx in draws:eb[start:stop]+=effects[:,idx[start:stop]].sum(2).transpose(1,0,2)/512
                bounds=np.quantile(eb,[.025,.975],axis=0)
                for c,name in enumerate(names):
                    entry=s['conditions'][method][c];assert entry['name']==name
                    for m,metric in enumerate(metrics):
                        got=entry['metrics'][metric];close(got['mean'],values[k,c,:,m].mean());close(got['effect'],effects[c,:,m].mean());close(got['effect_ci95'],bounds[:,c,m])
        assert s['checkpoint_sha256']==digest(ROOT/f'experiments/final/seed-{seed}.npz')
        assert s['weights_unchanged'] and s['means_unchanged'] and s['noop_max_logit_difference']==0
        print(f'Independent mean audit seed {seed}: calibration, all predictions/outcomes, 252 contrasts, all condition summaries and intervals passed.',flush=True)
    provenance=read(D/'provenance.json')
    assert 0<provenance['elapsed_seconds']<=provenance['evaluation_budget_seconds']<=7200
    assert provenance['peak_process_rss_kib']>0
    print('Historical hashes unchanged; independently verified dictionary disjointness, labels, calibrated replacements and summary arithmetic.')

if __name__=='__main__':main()
