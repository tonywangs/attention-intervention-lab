#!/usr/bin/env python3
"""Build/install into a fresh venv and run the mean CLI outside checkout, offline."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from installed_cli import command

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--wheelhouse',required=True);args=parser.parse_args()
    wheelhouse=Path(args.wheelhouse).resolve()
    if not wheelhouse.is_dir():raise ValueError('offline wheelhouse missing')
    started=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='mean-installed-') as temp:
        work=Path(temp);venv=work/'venv';wheels=work/'wheels';wheels.mkdir()
        env=os.environ.copy();env.pop('PYTHONPATH',None)
        env.update(PIP_NO_INDEX='1',PIP_DISABLE_PIP_VERSION_CHECK='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        command([sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation','--no-index','--wheel-dir',wheels,ROOT],work,env)
        command([sys.executable,'-m','venv','--without-pip',venv],work,env)
        python=venv/'bin/python';wheel=next(wheels.glob('attention_intervention_lab-*.whl'))
        command([sys.executable,'-m','pip','--python',python,'install','--no-index','--find-links',wheelhouse,wheel],work,env)
        guard=work/'network-guard';guard.mkdir()
        (guard/'sitecustomize.py').write_text('''import socket

def blocked(*args, **kwargs):
    raise RuntimeError("network access forbidden during offline mean replay")

socket.socket.connect=blocked
socket.socket.connect_ex=blocked
socket.create_connection=blocked
socket.getaddrinfo=blocked
''')
        env['PYTHONPATH']=str(guard)
        location=command([python,'-c','import attention_lab; print(attention_lab.__file__)'],work,env).strip()
        assert location.startswith(str(venv)) and not location.startswith(str(ROOT))
        command([python,'-c',"import socket\ntry: socket.create_connection(('example.invalid',443))\nexcept RuntimeError: print('network blocked')\nelse: raise AssertionError('network guard missing')"],work,env)
        cli=venv/'bin/attention-mean';common=['--root',ROOT,'--protocol',ROOT/'experiments/mean/protocol.json']
        data=work/'data';command([cli,'prepare',*common,'--output',data],work,env)
        for path in (ROOT/'experiments/mean/data').iterdir():assert path.read_bytes()==(data/path.name).read_bytes()
        out=work/'replay';replay=[cli,'replay',*common,'--data',data,'--reference',ROOT/'results/mean','--output',out]
        command(replay,work,env,timeout=240)
        assert json.loads((out/'replay.json').read_text())['all_deterministic_files_identical']
        before=(out/'replay.json').read_bytes();command(replay,work,env,expected=2)
        assert before==(out/'replay.json').read_bytes()
        failed=work/'timeout';command([cli,'evaluate',*common,'--data',data,'--output',failed,'--max-seconds','0.00000001'],work,env,expected=2)
        assert not failed.exists() and not list(work.glob('.timeout-*'))
        bad=work/'bad';bad.mkdir();(bad/'manifest.json').write_text('{"../escape":{}}')
        failure=command([cli,'evaluate',*common,'--data',bad,'--output',work/'bad-out'],work,env,expected=2)
        assert 'Traceback' not in failure and not (work/'bad-out').exists()
        report=work/'report';command([cli,'report','--input',ROOT/'results/mean','--output',report],work,env)
        assert (report/'report.html').read_bytes()==(ROOT/'results/mean/report.html').read_bytes()
    print(f'Installed mean CLI passed in {time.monotonic()-started:.3f}s: isolated wheel, blocked network, regenerated data, byte-identical means/logits/outcomes/summary/HTML, collisions, malformed data, timeout cleanup.')

if __name__=='__main__':main()
