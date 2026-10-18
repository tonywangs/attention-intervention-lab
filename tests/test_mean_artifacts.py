from pathlib import Path
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import numpy as np
import pytest
from attention_lab.artifacts import read_json,write_json,manifest
from attention_lab.mean_experiment import prepare,evaluate,load_protocol
from attention_lab.mean_report import payload,render_directory

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'experiments/mean/protocol.json'
DATA=ROOT/'experiments/mean/data'
RESULTS=ROOT/'results/mean'


def test_inputs_reproduce(tmp_path):
    prepare(ROOT,P,tmp_path/'data')
    for path in DATA.iterdir():assert path.read_bytes()==(tmp_path/'data'/path.name).read_bytes()


@pytest.mark.parametrize('seconds',[0,-1,7201,float('nan'),float('inf'),1e-12])
def test_budget_cleanup(tmp_path,seconds):
    with pytest.raises((ValueError,TimeoutError)):evaluate(ROOT,P,DATA,tmp_path/'out',seconds)
    assert not (tmp_path/'out').exists() and not list(tmp_path.glob('.out-*'))


def test_collision(tmp_path):
    out=tmp_path/'out';out.mkdir();(out/'keep').write_text('keep')
    with pytest.raises(FileExistsError):evaluate(ROOT,P,DATA,out)
    assert (out/'keep').read_text()=='keep' and len(list(out.iterdir()))==1


@pytest.mark.parametrize('kind',['label','overlap','metadata','protocol'])
def test_mutated_inputs_even_with_updated_manifest(tmp_path,kind):
    data=tmp_path/'data';shutil.copytree(DATA,data)
    protocol=P
    if kind=='protocol':
        protocol=tmp_path/'protocol.json';p=read_json(P);p['data']['seed']+=1;write_json(protocol,p)
    elif kind=='metadata':
        p=read_json(data/'inputs.json');p['excluded_dictionaries']+=1;write_json(data/'inputs.json',p)
    else:
        rows=read_json(data/'evaluation.json')
        if kind=='label':rows[0]['answer']=(rows[0]['answer']+1)%16
        else:rows[0]=read_json(data/'calibration.json')[0]
        write_json(data/'evaluation.json',rows)
    write_json(data/'manifest.json',manifest(data))
    with pytest.raises(ValueError):evaluate(ROOT,protocol,data,tmp_path/'out')
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('kind',['count','pair','interval','mean','flag','critical','text','nan','header'])
def test_report_rejects_malformed(tmp_path,kind):
    d=tmp_path/'data';shutil.copytree(RESULTS,d)
    s=read_json(d/'summary.json');seed=s['seeds'][0];x=seed['comparisons']['mean'][0]['metrics']['loss']
    if kind=='count':seed['comparisons']['mean']*=100
    elif kind=='pair':seed['comparisons']['mean'][0]['pair']=['L9H0','L0H0']
    elif kind=='interval':x['ci95_family']=[1,-1]
    elif kind=='mean':x['mean']+=1
    elif kind=='flag':x['resolved']=not x['resolved']
    elif kind=='critical':seed['family_critical_value']=-1
    elif kind=='text':s['limits']=['x'*2001]
    elif kind=='nan':x['mean']=float('nan')
    elif kind=='header':
        with (d/'means-11.npy').open('wb') as f:
            np.lib.format.write_array_header_1_0(f,{'shape':(2,10,4,1000000000),'fortran_order':False,'descr':'<f4'})
    (d/'summary.json').write_text(json.dumps(s))
    with pytest.raises(ValueError):payload(d)


@pytest.mark.parametrize('sig',[signal.SIGINT,signal.SIGTERM])
def test_cancel_cleanup(tmp_path,sig):
    out=tmp_path/'cancel'
    args=[sys.executable,'-m','attention_lab.mean_cli','evaluate','--root',str(ROOT),'--protocol',str(P),'--data',str(DATA),'--output',str(out)]
    proc=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
    try:
        deadline=time.monotonic()+30
        while not list(tmp_path.glob('.cancel-*/protocol.json')):
            assert proc.poll() is None,proc.communicate()
            assert time.monotonic()<deadline
            time.sleep(.01)
        proc.send_signal(sig);stdout,stderr=proc.communicate(timeout=30)
        assert proc.returncode==2,(stdout,stderr)
        assert b'cancelled by signal' in stderr and b'Traceback' not in stderr
        assert not out.exists() and not list(tmp_path.glob('.cancel-*'))
    finally:
        if proc.poll() is None:proc.kill();proc.wait()


def test_saved_report_reproduces(tmp_path):
    render_directory(RESULTS,tmp_path/'report.html')
    assert (tmp_path/'report.html').read_bytes()==(RESULTS/'report.html').read_bytes()


def test_evaluation_sources_preserved_and_renderer_amendment(tmp_path):
    import importlib.util
    from attention_lab.artifacts import sha256
    provenance=read_json(RESULTS/'provenance.json')
    frozen=ROOT/'experiments/mean/frozen-renderer/mean_report.py'
    for name,digest in provenance['source_sha256'].items():
        path=frozen if name=='mean_report.py' else ROOT/'src/attention_lab'/name
        assert sha256(path)==digest,name
    amendment=read_json(ROOT/'experiments/mean/rendering-amendment.json')
    assert sha256(frozen)==amendment['original_renderer_sha256']
    assert sha256(ROOT/'src/attention_lab/mean_report.py')==amendment['updated_renderer_sha256']
    spec=importlib.util.spec_from_file_location('attention_lab._frozen_mean_report',frozen)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.render_directory(RESULTS,tmp_path/'frozen.html')
    assert sha256(tmp_path/'frozen.html')==amendment['evaluation_report_sha256']
    assert sha256(RESULTS/'report.html')==amendment['updated_report_sha256']
