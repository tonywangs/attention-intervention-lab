#!/usr/bin/env python3
"""Build a wheel locally, install to a fresh venv, and replay with network blocked.

Dependency wheels must have been downloaded beforehand. This script never uses
an index, system site-packages, the source checkout as a PYTHONPATH, or ensurepip.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]


def command(args, cwd, env, expected=0, timeout=180):
    result=subprocess.run([str(x) for x in args],cwd=cwd,env=env,text=True,
                          stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
    if result.returncode!=expected:
        raise AssertionError(f'{args[0]} returned {result.returncode}; expected {expected}\n{result.stdout}')
    return result.stdout


def run(wheelhouse, pairs=False):
    wheelhouse=Path(wheelhouse).resolve()
    if not wheelhouse.is_dir() or not list(wheelhouse.glob('torch-*.whl')):
        raise ValueError('Prepare the offline wheelhouse first; see README.')
    started=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='attention-installed-') as temp:
        work=Path(temp);venv=work/'venv';wheels=work/'wheels';wheels.mkdir()
        env=os.environ.copy();env.pop('PYTHONPATH',None)
        env.update({'PIP_NO_INDEX':'1','PIP_DISABLE_PIP_VERSION_CHECK':'1','OPENBLAS_NUM_THREADS':'1'})
        command([sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation','--no-index','--wheel-dir',wheels,ROOT],work,env)
        wheel=next(wheels.glob('attention_intervention_lab-*.whl'))
        command([sys.executable,'-m','venv','--without-pip',venv],work,env)
        python=venv/'bin/python'
        command([sys.executable,'-m','pip','--python',python,'install','--no-index','--find-links',wheelhouse,wheel],work,env)
        guard=work/'network-guard';guard.mkdir()
        (guard/'sitecustomize.py').write_text('''import socket

def blocked(*args, **kwargs):
    raise RuntimeError("network access forbidden during offline replay")

socket.socket.connect = blocked
socket.socket.connect_ex = blocked
socket.create_connection = blocked
socket.getaddrinfo = blocked
''')
        env['PYTHONPATH']=str(guard)
        location=command([python,'-c','import attention_lab; print(attention_lab.__file__)'],work,env).strip()
        assert location.startswith(str(venv)) and not location.startswith(str(ROOT))
        command([python,'-c',"import socket\ntry: socket.create_connection(('example.invalid',443))\nexcept RuntimeError: print('network blocked')\nelse: raise AssertionError('network guard missing')"],work,env)
        cli=venv/'bin/attention-lab'
        output=work/'replay'
        args=[cli,'replay','--training',ROOT/'experiments/final','--reference',ROOT/'results/evaluation','--output',output]
        command(args,work,env)
        result=json.loads((output/'replay.json').read_text())
        assert result['within_frozen_tolerances'] and result['predictions_match']
        assert result['all_deterministic_files_identical'],result
        command([python,'-c','from attention_lab.artifacts import verify_manifest; import sys; verify_manifest(sys.argv[1])',output],work,env)
        # A second execution must fail without overwriting the existing output.
        before=(output/'replay.json').read_bytes()
        command(args,work,env,expected=2)
        assert (output/'replay.json').read_bytes()==before
        failed=work/'timeout'
        command([cli,'evaluate','--training',ROOT/'experiments/final','--output',failed,'--max-seconds','0.0000001'],work,env,expected=2)
        assert not failed.exists() and not list(work.glob('.timeout-*'))
        report_out=work/'report'
        command([cli,'report','--input',ROOT/'results/evaluation','--output',report_out],work,env)
        assert (report_out/'report.html').read_bytes()==(ROOT/'results/evaluation/report.html').read_bytes()
        # Malformed, incomplete input fails cleanly and does not allocate an output.
        bad=work/'bad';bad.mkdir();(bad/'manifest.json').write_text('{"../escape":{}}')
        failure=command([cli,'evaluate','--training',bad,'--output',work/'bad-result'],work,env,expected=2)
        assert 'Traceback' not in failure and not (work/'bad-result').exists()
        print('Installed CLI passed: fresh isolated venv, local wheel, blocked network, three-seed checkpoint replay, byte-identical JSON/HTML, collision, timeout cleanup and malformed-input rejection.')
        if pairs:
            common = ['--training', ROOT/'experiments/final', '--previous', ROOT/'results/evaluation',
                      '--protocol', ROOT/'experiments/interactions/protocol.json']
            data = work/'pair-data'
            command([cli, 'pair-prepare', *common, '--output', data], work, env)
            for name in ('examples.json', 'inputs.json', 'manifest.json'):
                assert (data/name).read_bytes() == (ROOT/'experiments/interactions/data'/name).read_bytes()
            pair_out = work/'pair-replay'
            pair_args = [cli, 'pair-replay', *common, '--data', data,
                         '--reference', ROOT/'results/interactions', '--output', pair_out]
            command(pair_args, work, env)
            result = json.loads((pair_out/'replay.json').read_text())
            assert result['within_frozen_tolerances'] and result['all_deterministic_files_identical']
            before = (pair_out/'replay.json').read_bytes()
            command(pair_args, work, env, expected=2)
            assert (pair_out/'replay.json').read_bytes() == before
            failed = work/'pair-timeout'
            command([cli, 'pairs', *common, '--data', data, '--output', failed,
                     '--max-seconds', '0.0000001'], work, env, expected=2)
            assert not failed.exists() and not list(work.glob('.pair-timeout-*'))
            rendered = work/'pair-report'
            command([cli, 'pair-report', '--input', ROOT/'results/interactions', '--output', rendered], work, env)
            assert (rendered/'report.html').read_bytes() == (ROOT/'results/interactions/report.html').read_bytes()
            print('Installed pair CLI passed: regenerated disjoint inputs; all 114 condition/model combinations; byte-identical logits, outcomes, summary and HTML; collision protection; timeout cleanup; packaged template; blocked network.')
        print(f'Installed CLI verification elapsed seconds: {time.monotonic()-started:.3f}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheelhouse',required=True)
    parser.add_argument('--pairs', action='store_true', help='also replay exhaustive paired interventions')
    args=parser.parse_args()
    run(args.wheelhouse, args.pairs)
