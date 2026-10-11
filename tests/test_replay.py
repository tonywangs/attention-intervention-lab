"""Replay must reject coherent-manifest corruption of numeric evidence, not just logits."""
import shutil
from pathlib import Path
import pytest
from attention_lab.artifacts import manifest,read_json,write_json
from attention_lab.evaluate import replay

ROOT=Path(__file__).resolve().parents[1]


def test_replay_rejects_altered_summary_and_cleans_up(tmp_path):
    reference=tmp_path/'reference'
    shutil.copytree(ROOT/'results/evaluation',reference)
    summary=read_json(reference/'summary.json')
    summary['seeds'][0]['conditions'][0]['loss']+=.1
    write_json(reference/'summary.json',summary)
    write_json(reference/'manifest.json',manifest(reference))
    out=tmp_path/'replayed'
    with pytest.raises(ValueError,match='numeric evidence'):
        replay(ROOT/'experiments/final',reference,out)
    assert not out.exists() and not list(tmp_path.glob('.replayed-*'))
