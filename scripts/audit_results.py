#!/usr/bin/env python3
"""Independent result audit: reconstruct metrics from logits, never import statistics.py.

Uses scalar math for probability/loss/ranks and bootstrap multiplicity weights
rather than the production gather-and-mean implementation.
"""
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def close(actual, expected, tolerance=2e-12):
    if abs(actual-expected) > tolerance:
        raise AssertionError(f"numeric mismatch: {actual} vs {expected}")


def average(xs):
    return math.fsum(xs)/len(xs)


def quantile(xs, q):
    ordered = sorted(xs)
    index = (len(ordered)-1)*q
    lower = math.floor(index)
    alpha = index-lower
    return ordered[lower]*(1-alpha)+ordered[min(lower+1,len(xs)-1)]*alpha


def independent_rank(values):
    return [1+sum(other>v for other in values)+(sum(other==v for other in values)-1)/2 for v in values]


def correlation(x, y):
    a, b = average(x), average(y)
    xx = math.fsum((v-a)**2 for v in x)
    yy = math.fsum((v-b)**2 for v in y)
    return None if not xx or not yy else math.fsum((v-a)*(w-b) for v,w in zip(x,y))/math.sqrt(xx*yy)


def audit(root=ROOT):
    root = Path(root)
    directory = root/'results/evaluation'
    summary = read(directory/'summary.json')
    protocol = read(root/'experiments/protocol.json')
    protocol_hash = hashlib.sha256((root/'experiments/protocol.json').read_bytes()).hexdigest()
    assert protocol_hash == summary['protocol_sha256']
    assert (root/'experiments/final/protocol.json').read_bytes() == (root/'experiments/protocol.json').read_bytes()
    amendments=read(root/'experiments/amendments.json')
    assert amendments['protocol_sha256']==protocol_hash
    assert amendments['scientific_protocol_changed'] is False
    source_amendments={a['file']:a for a in amendments['changes'] if 'frozen_sha256' in a}
    for name, expected in protocol['source_hashes'].items():
        actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
        if name in source_amendments:
            amendment=source_amendments[name]
            assert amendment['frozen_sha256']==expected
            assert hashlib.sha256((root/amendment['frozen_copy']).read_bytes()).hexdigest()==expected
            assert actual==amendment['current_sha256']
        else:
            assert actual==expected, name
    assert hashlib.sha256((root/'experiments/pilot/pilot.json').read_bytes()).hexdigest() == protocol['pilot_sha256']
    assert hashlib.sha256((directory/'eval.json').read_bytes()).hexdigest() == summary['eval_sha256']
    inputs = read(directory/'eval.json')
    assert len(inputs) == summary['n'] == 512
    assert len(summary['seeds']) == 3
    # Recompute context split assignment independently from canonical dictionaries.
    context_sets = []
    for split in ('train','dev','eval'):
        if split == 'train':
            rows = sum([read(p) for p in sorted((root/'experiments/final/data').glob('train-*.json'))],[])
        elif split == 'dev':
            rows = read(root/'experiments/final/data/dev.json')
        else:
            rows = inputs
        assert len(rows) == (16384 if split=='train' else 512)
        contexts=set()
        for i,r in enumerate(rows):
            t=r['tokens']; d=dict(zip(t[:8:2],t[1:8:2]))
            assert len(d)==4 and len(set(d.values()))==4 and d[t[9]]-16==r['answer']
            assert r['value_position']==2*(i%4)+1 and t[r['value_position']]-16==r['answer']
            raw=json.dumps(sorted(d.items()),separators=(',',':')).encode()
            bucket=int.from_bytes(hashlib.sha256(raw).digest()[:8],'big')%100
            assert (bucket<80 if split=='train' else 80<=bucket<90 if split=='dev' else bucket>=90)
            contexts.add(tuple(sorted(d.items())))
        assert len(contexts)==len(rows)
        context_sets.append(contexts)
    assert all(not context_sets[a]&context_sets[b] for a,b in [(0,1),(0,2),(1,2)])
    # Independent resampling representation: each row counts draw multiplicities.
    random = np.random.Generator(np.random.PCG64(20261011))
    counts = np.stack([np.bincount(random.integers(0,512,512),minlength=512) for _ in range(2000)]) / 512
    for seed_summary in summary['seeds']:
        seed=seed_summary['seed']
        rows=read(directory/f'seed-{seed}.json')['examples']
        assert len(rows)==512
        values={}
        for condition in seed_summary['conditions']:
            name=condition['name']
            acc,loss,prob=[],[],[]
            for row,input_row in zip(rows,inputs):
                assert all(row[k]==input_row[k] for k in input_row)
                c=row['conditions'][name]
                assert len(c['logits'])==16
                maximum=max(c['logits'])
                denom=math.fsum(math.exp(v-maximum) for v in c['logits'])
                lp=c['logits'][row['answer']]-maximum-math.log(denom)
                prediction=max(range(16),key=lambda i:c['logits'][i])
                correct=prediction==row['answer']
                assert c['prediction']==prediction and c['correct']==correct
                close(c['loss'],-lp);close(c['p_correct'],math.exp(lp))
                acc.append(int(correct));loss.append(-lp);prob.append(math.exp(lp))
            values[name]=(acc,loss,prob)
            for field,array in zip(('accuracy','loss','p_correct'),values[name]):
                close(condition[field],average(array))
                for pos in (1,3,5,7):
                    close(condition['by_value_position'][str(pos)][field], average([v for r,v in zip(rows,array) if r['value_position']==pos]))
            for column,(field,direction) in enumerate([('accuracy_drop',1),('loss_increase',-1),('probability_drop',1)]):
                delta=[direction*(a-b) for a,b in zip(values['intact'][column],values[name][column])]
                close(condition['paired'][field]['mean'],average(delta))
                samples=counts @ np.array(delta)
                interval=[quantile(samples.tolist(),q) for q in (0.025,0.975)]
                for a,b in zip(condition['paired'][field]['ci95'],interval):close(a,b)
        assert values['intact']==values['noop']
        for row in rows:assert row['conditions']['intact']==row['conditions']['noop']
        attention=[average([r['attention_mass'][h] for r in rows]) for h in range(8)]
        effects=[average([a-b for a,b in zip(values['intact'][2],values[h][2])]) for h in summary['heads']]
        rank=seed_summary['ranking']
        for a,b in zip(rank['attention_mean'],attention):close(a,b)
        for a,b in zip(rank['probability_drop_mean'],effects):close(a,b)
        ar,er=independent_rank(attention),independent_rank(effects)
        assert ar==rank['attention_rank'] and er==rank['effect_rank']
        rho=correlation(ar,er)
        if rho is None:assert rank['spearman'] is None
        else:close(rank['spearman'],rho)
        for field,scores in [('attention_ties',attention),('effect_ties',effects)]:
            expected=[]
            for value in sorted(set(scores),reverse=True):
                group=[summary['heads'][i] for i,v in enumerate(scores) if v==value]
                if len(group)>1:expected.append(group)
            assert rank[field]==expected
        matrices=read(directory/f'attention-{seed}.json')['examples']
        assert [m['id'] for m in matrices]==[r['id'] for r in rows[:16]]
        for matrix,row in zip(matrices,rows):
            for layer in range(2):
                for head in range(4):
                    pattern=matrix['pattern'][layer][head]
                    for q,weights in enumerate(pattern):
                        close(sum(weights),1,1e-6)
                        assert all(v==0 for v in weights[q+1:])
                    close(pattern[-1][row['value_position']],row['attention_mass'][layer*4+head])
    close(summary['baselines']['first_context_value']['accuracy'],average([r['tokens'][1]-16==r['answer'] for r in inputs]))
    for name in ('pilot','final'):
        metadata=read(root/'experiments'/name/('pilot.json' if name=='pilot' else 'training.json'))
        assert metadata['eval_accessed'] is False
        assert metadata['elapsed_seconds'] <= (1800 if name=='pilot' else 10800)
    print('Independent audit passed: data splits/oracle, frozen hashes, 15,360 condition predictions, all aggregates, position slices, 95% intervals, ranks, ties, attention subset and no-op controls.')


if __name__=='__main__':
    audit(Path(sys.argv[1]) if len(sys.argv)>1 else ROOT)
