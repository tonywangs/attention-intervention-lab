#!/usr/bin/env python3
"""All historical and mean checks, including independent audits and offline CLI installs."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--wheelhouse',required=True);p.add_argument('--browsers',required=True);args=p.parse_args()
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PLAYWRIGHT_BROWSERS_PATH':str(Path(args.browsers).resolve())}
    commands=[[sys.executable,'-m','pytest','-q'],
              [sys.executable,'scripts/audit_results.py'],[sys.executable,'scripts/audit_pairs.py'],[sys.executable,'scripts/audit_mean.py'],
              [sys.executable,'scripts/installed_cli.py','--wheelhouse',str(Path(args.wheelhouse).resolve()),'--pairs'],
              [sys.executable,'scripts/installed_mean_cli.py','--wheelhouse',str(Path(args.wheelhouse).resolve())],
              [sys.executable,'scripts/check_tree.py']]
    started=time.monotonic()
    for command in commands:
        print(' '.join(command),flush=True)
        result=subprocess.run(command,cwd=ROOT,env=env,timeout=900)
        if result.returncode:return result.returncode
    print(f'All mean and historical verification passed in {time.monotonic()-started:.3f}s.',flush=True)
    return 0

if __name__=='__main__':sys.exit(main())
