"""Bounded self-contained method comparison with accessible numeric tables."""
from pathlib import Path
import json
import math
import numpy as np
from .artifacts import read_json, sha256
from .mean_experiment import load_protocol, PROTOCOL_SHA256
from .interactions import load_array
from .interaction_stats import METRICS


def payload(directory):
    d = Path(directory)
    p = load_protocol(d/'protocol.json')
    s = read_json(d/'summary.json', 2*1024*1024)
    if (s['schema'] != 1 or s['protocol_sha256'] != PROTOCOL_SHA256 or s['n'] != 512
            or len(s['seeds']) != 3 or [v['seed'] for v in s['seeds']] != [11,29,47]):
        raise ValueError('invalid mean summary')
    from .mean_stats import summarize
    from .interaction_stats import outcomes
    from .artifacts import Budget
    rows = read_json(d/'evaluation.json')
    if not isinstance(rows, list) or len(rows) != 512:
        raise ValueError('invalid report examples')
    answers = [r['answer'] for r in rows]
    for seed in s['seeds']:
        if seed['family_size'] != 252:
            raise ValueError('invalid family')
        for method in ('zero','mean','difference'):
            entries = seed['comparisons'][method]
            if len(entries) != 28 or [v['pair'] for v in entries] != p['pairs']:
                raise ValueError('invalid comparison rows')
            for entry in entries:
                for metric in METRICS:
                    x = entry['metrics'][metric]
                    if (any(type(x[k]) is not bool for k in ('resolved','degenerate'))
                            or any(not isinstance(x[k],(int,float)) or not math.isfinite(x[k]) for k in ('mean','bootstrap_se'))
                            or any(len(x[k]) != 2 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in x[k]) for k in ('ci95_family','ci95_pointwise'))):
                        raise ValueError('invalid comparison metric')
        for method in ('zero','mean'):
            conditions = seed['conditions'][method]
            if len(conditions)!=38 or [v['name'] for v in conditions]!=p['conditions']:
                raise ValueError('invalid conditions')
            for row in conditions:
                for metric in METRICS:
                    x=row['metrics'][metric]
                    if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in [x['mean'],x['effect'],*x['effect_ci95']]) or len(x['effect_ci95'])!=2:
                        raise ValueError('invalid condition metric')
        arrays = [load_array(d/f'{method}-outcomes-{seed["seed"]}.npy', (38,512,3), np.float64) for method in ('zero','mean')]
        for method, array in zip(('zero','mean'), arrays):
            logits = load_array(d/f'{method}-logits-{seed["seed"]}.npy', (38,512,16), np.float32)
            if not np.allclose(outcomes(logits,answers),array,atol=1e-10,rtol=1e-10):
                raise ValueError('report outcomes inconsistent with logits')
        load_array(d/f'means-{seed["seed"]}.npy',(2,10,4,16),np.float32)
        recomputed = summarize(np.stack(arrays),rows,p,Budget(120))
        if any(seed[k] != v for k,v in recomputed.items()):
            raise ValueError('report summary inconsistent with outcomes')
        seed['examples'] = {m:load_array(d/f'{m}-outcomes-{seed["seed"]}.npy',(38,512,3),np.float64)[:,:16].tolist() for m in ('zero','mean')}
    if not isinstance(s['limits'],list) or len(s['limits'])>16 or any(not isinstance(v,str) or len(v)>2000 for v in s['limits']):
        raise ValueError('invalid report text')
    s['pairs'],s['conditions'],s['metrics'] = p['pairs'],p['conditions'],list(METRICS)
    return s


def render_directory(directory, destination):
    data = json.dumps(payload(directory),ensure_ascii=True,allow_nan=False,separators=(',',':')).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    html = TEMPLATE.replace('__DATA__',data)
    if len(html.encode()) > 2*1024*1024:
        raise ValueError('report exceeds rendering budget')
    Path(destination).write_text(html)


TEMPLATE = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'none'; img-src 'none'">
<title>Zero versus mean ablation • Attention Intervention Lab</title>
<style>body{font:16px system-ui;line-height:1.5;max-width:1100px;margin:auto;padding:24px;color:#20303d;background:#fafbfd}h1{line-height:1.15}label{display:inline-block;margin:8px 16px 8px 0}select{font:inherit;min-height:44px;padding:8px;max-width:100%}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:8px;border-bottom:1px solid #cdd5dd;text-align:right}th:first-child,td:first-child{text-align:left}caption{text-align:left;font-weight:600;padding:16px 0}.scroll{overflow:auto}summary{cursor:pointer}code{overflow-wrap:anywhere}#limits{border-left:4px solid #60809a;padding-left:20px}a{color:#174e87}</style>
<h1>Do paired-head effects survive mean ablation?</h1>
<p>Three saved models · 512 evaluation examples · 512 separate calibration examples · all eight heads and 28 pairs · no retraining.</p>
<p>Mean replacement uses intact calibration z, averaged across examples separately at each position, layer and head, before W_O. All pairs reuse the same fixed means.</p>
<p>Interaction = joint − single A − single B + intact. Difference = mean interaction − zero interaction. Positive loss and negative probability/accuracy interactions indicate excess harm. Values are in metric units, not percentages.</p>
<label>Model <select id="seed"></select></label><label>Metric <select id="metric"></select></label><label>Pair <select id="pair"></select></label>
<p id="finding" role="status"></p>
<div class="scroll"><table id="pairs"><caption>All pairs: interaction and simultaneous 95% interval</caption><thead><tr><th>Pair</th><th>Zero</th><th>Mean</th><th>Mean − zero</th></tr></thead><tbody></tbody></table></div>
<p>Each cell gives estimate [lower, upper], and “resolved” only when the adjusted interval excludes the numerical floor. One approximate bootstrap family covers all 252 contrasts within this model. Constant contrasts are descriptive only. Unresolved is not proof of no effect. Comparing resolved flags is not a test of method difference.</p>
<div class="scroll"><table id="conditions"><caption>Selected pair: metric means and effects versus intact (unadjusted 95% effect intervals)</caption><thead><tr><th>Condition</th><th>Zero: mean; effect [interval]</th><th>Mean: mean; effect [interval]</th></tr></thead><tbody></tbody></table></div>
<details><summary>First 16 examples: full-precision paired numeric evidence</summary><div class="scroll"><table id="examples"><caption>Fixed example subset; aggregate estimates use all 512</caption><thead><tr><th>Example index</th><th>Zero interaction</th><th>Mean interaction</th><th>Difference</th></tr></thead><tbody></tbody></table></div></details>
<details><summary>Full-precision aggregate evidence for selected pair</summary><pre id="exact" style="white-space:pre-wrap;overflow-wrap:anywhere"></pre></details>
<h2>Interpretation limits</h2><ul id="limits"></ul><p>Intervals condition on fixed calibration means and fixed position counts. They exclude uncertainty from calibration and training. No head alignment or pooled inference across models. This is an offline synthetic experiment, not a circuit proof.</p>
<p>Protocol SHA-256: <code id="hash"></code>. Complete per-example arrays, means and provenance accompany this HTML.</p>
<script id="data" type="application/json">__DATA__</script><script>
'use strict';const D=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);
function options(id,items){items.forEach((x,i)=>{const o=document.createElement('option');o.value=i;o.textContent=x;$(id).append(o)})}
options('seed',D.seeds.map(s=>s.seed));options('metric',D.metrics);options('pair',D.pairs.map(p=>p.join(' + ')));
function row(body,cells){const tr=document.createElement('tr');cells.forEach((v,i)=>{const c=document.createElement(i?'td':'th');if(!i)c.scope='row';c.textContent=v;tr.append(c)});body.append(tr)}
const fmt=x=>Number(x).toPrecision(6), cell=x=>`${fmt(x.mean)} [${x.ci95_family.map(fmt).join(', ')}] ${x.degenerate?'constant':x.resolved?'resolved':'unresolved'}`;
function draw(){const s=D.seeds[+$('seed').value],m=D.metrics[+$('metric').value],j=+$('pair').value,methods=['zero','mean','difference'];const body=$('pairs').tBodies[0];body.replaceChildren();D.pairs.forEach((p,i)=>row(body,[p.join(' + '),...methods.map(k=>cell(s.comparisons[k][i].metrics[m]))]));
$('finding').textContent=`Model ${s.seed}: resolved ${m} interactions: zero ${s.comparisons.zero.filter(x=>x.metrics[m].resolved).length}/28; mean ${s.comparisons.mean.filter(x=>x.metrics[m].resolved).length}/28; paired method differences ${s.comparisons.difference.filter(x=>x.metrics[m].resolved).length}/28.`;
const pair=D.pairs[j],names=['intact',...pair,pair.join('+')],cb=$('conditions').tBodies[0];cb.replaceChildren();names.forEach(n=>row(cb,[n,...['zero','mean'].map(k=>{const x=s.conditions[k][D.conditions.indexOf(n)].metrics[m];return `${fmt(x.mean)}; ${fmt(x.effect)} [${x.effect_ci95.map(fmt).join(', ')}]`})]));
$('exact').textContent=JSON.stringify(Object.fromEntries(methods.map(k=>[k,s.comparisons[k][j]])),null,2);
const eb=$('examples').tBodies[0];eb.replaceChildren();for(let i=0;i<16;i++){let vals=['zero','mean'].map(k=>{const a=s.examples[k],mi=+$('metric').value;return a[D.conditions.indexOf(names[3])][i][mi]-a[D.conditions.indexOf(names[1])][i][mi]-a[D.conditions.indexOf(names[2])][i][mi]+a[0][i][mi]});row(eb,[i,...vals,vals[1]-vals[0]])}}
D.limits.forEach(x=>{let li=document.createElement('li');li.textContent=x;$('limits').append(li)});$('hash').textContent=D.protocol_sha256;['seed','metric','pair'].forEach(id=>$(id).addEventListener('change',draw));draw();
</script></html>'''
