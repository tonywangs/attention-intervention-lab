#!/usr/bin/env python3
"""Check actual publishable paths, manifests and frozen forward-pass identity."""
import ast
from pathlib import Path
import subprocess
from attention_lab.artifacts import MAX_FILE,MAX_TREE,verify_manifest

ROOT=Path(__file__).resolve().parents[1]


def main():
    # Include useful untracked files in a pre-commit snapshot, honoring .gitignore.
    raw=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT)
    paths=sorted(set(p for p in raw.decode().split('\0') if p))
    sizes=[]
    for name in paths:
        path=ROOT/name
        if not path.exists():continue
        assert not path.is_symlink(),name
        assert name=='results/tests.log' or not name.endswith('.log'),name
        assert '.venv' not in path.parts and '.git' not in path.parts,name
        size=path.stat().st_size
        assert size<=MAX_FILE,(name,size)
        sizes.append((size,name))
    assert len(sizes)<=1000 and sum(n for n,_ in sizes)<=MAX_TREE
    for directory in ('experiments/pilot','experiments/final','results/evaluation'):
        verify_manifest(ROOT/directory)
    old=ast.parse((ROOT/'experiments/frozen/model.py').read_text())
    new=ast.parse((ROOT/'src/attention_lab/model.py').read_text())
    # Security-only loader amendment cannot change the frozen model or save routine.
    keep=lambda tree:[ast.dump(n,include_attributes=False) for n in tree.body
                      if not isinstance(n,ast.FunctionDef) or n.name!='load_weights']
    assert keep(old)==keep(new),'frozen forward pass changed'
    print(f'Artifact bounds passed: {len(sizes)} files, {sum(n for n,_ in sizes):,} bytes; largest {max(sizes)[0]:,} bytes ({max(sizes)[1]}).')
    print('All three artifact manifests valid; frozen architecture/forward pass unchanged.')


if __name__=='__main__':main()
