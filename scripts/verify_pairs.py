#!/usr/bin/env python3
"""Full original and paired-experiment verification without rewriting historical logs."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheelhouse', required=True)
    parser.add_argument('--browsers', required=True)
    args = parser.parse_args()
    env = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'PLAYWRIGHT_BROWSERS_PATH': str(Path(args.browsers).resolve())}
    commands = [[sys.executable, '-m', 'pytest', '-q'],
                [sys.executable, 'scripts/audit_results.py'],
                [sys.executable, 'scripts/audit_pairs.py'],
                [sys.executable, 'scripts/installed_cli.py', '--wheelhouse', str(Path(args.wheelhouse).resolve()), '--pairs'],
                [sys.executable, 'scripts/check_tree.py']]
    started = time.monotonic()
    for command in commands:
        print(' '.join(command), flush=True)
        completed = subprocess.run(command, cwd=ROOT, env=env, timeout=600)
        if completed.returncode:
            return completed.returncode
    print(f'Original and paired verification passed in {time.monotonic()-started:.3f} seconds.', flush=True)
    return 0


if __name__ == '__main__': sys.exit(main())
