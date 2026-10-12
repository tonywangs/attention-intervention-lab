#!/usr/bin/env python3
"""One offline command: regenerate frozen inputs and replay the complete pair experiment."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='new output directory; never overwrite')
    args = parser.parse_args()
    output = Path(args.output).absolute()
    if output.exists() or output.is_symlink():
        parser.error('output already exists')
    env = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    # All commands are local. The isolated installed-CLI check separately blocks networking.
    cli = [sys.executable, '-m', 'attention_lab.cli']
    common = ['--training', str(ROOT/'experiments/final'), '--previous', str(ROOT/'results/evaluation'),
              '--protocol', str(ROOT/'experiments/interactions/protocol.json')]
    with tempfile.TemporaryDirectory(prefix='pair-inputs-') as temp:
        data = str(Path(temp)/'data')
        subprocess.run([*cli, 'pair-prepare', *common, '--output', data], env=env, check=True)
        subprocess.run([*cli, 'pair-replay', *common, '--data', data,
                        '--reference', str(ROOT/'results/interactions'), '--output', str(output)], env=env, check=True)
    print(f'Reproduction passed; predictions, statistics and report compared. Evidence: {output / "replay.json"}')


if __name__ == '__main__': main()
