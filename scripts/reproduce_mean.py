#!/usr/bin/env python3
"""Regenerate disjoint inputs and replay the frozen mean-ablation experiment offline."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output).absolute()
    if out.exists() or out.is_symlink():p.error('output already exists')
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    cli=[sys.executable,'-m','attention_lab.mean_cli']
    common=['--root',str(ROOT),'--protocol',str(ROOT/'experiments/mean/protocol.json')]
    with tempfile.TemporaryDirectory(prefix='mean-inputs-') as temp:
        data=str(Path(temp)/'data')
        subprocess.run([*cli,'prepare',*common,'--output',data],env=env,check=True,timeout=120)
        subprocess.run([*cli,'replay',*common,'--data',data,'--reference',str(ROOT/'results/mean'),'--output',str(out)],env=env,check=True,timeout=7200)
    print(f'Reproduction passed: regenerated inputs, fixed means, logits, outcomes, summaries and HTML. Evidence: {out / "replay.json"}')

if __name__=='__main__':main()
