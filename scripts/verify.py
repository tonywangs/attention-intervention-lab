#!/usr/bin/env python3
"""One command for math, artifacts, Chromium, independent statistics and installed replay."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheelhouse',required=True,help='pre-downloaded offline dependency wheels')
    parser.add_argument('--browsers',help='pre-installed Playwright browser directory')
    args=parser.parse_args()
    env=os.environ.copy()
    env['OPENBLAS_NUM_THREADS']='1'
    if args.browsers:env['PLAYWRIGHT_BROWSERS_PATH']=str(Path(args.browsers).resolve())
    commands=[
        [sys.executable,'-m','pytest','-q'],
        [sys.executable,'scripts/audit_results.py'],
        [sys.executable,'scripts/installed_cli.py','--wheelhouse',str(Path(args.wheelhouse).resolve())],
        [sys.executable,'scripts/check_tree.py'],
    ]
    started=time.monotonic()
    log_path=ROOT/'results/tests.log'
    with log_path.open('w') as log:
        for command in commands:
            label=' '.join(command)
            print(label,flush=True);log.write(label+'\n');log.flush()
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,text=True,timeout=300)
            print(result.stdout,end='',flush=True);log.write(result.stdout);log.flush()
            if result.returncode:
                print(f'FAILED: exit {result.returncode}',flush=True)
                return result.returncode
        message=f'All verification commands passed in {time.monotonic()-started:.3f} seconds.\n'
        print(message,end='');log.write(message)
    return 0


if __name__=='__main__':
    sys.exit(main())
