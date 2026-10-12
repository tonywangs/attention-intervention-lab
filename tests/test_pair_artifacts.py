from pathlib import Path
import copy
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import numpy as np
import pytest
from attention_lab.artifacts import read_json, sha256, write_json
from attention_lab.interactions import (load_array, load_protocol, generate_rows,
    historical_inputs, validate_rows, prepare, evaluate_pairs, replay_pairs)
from attention_lab.interaction_report import payload

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT/'experiments/final'
OLD = ROOT/'results/evaluation'
PROTOCOL = ROOT/'experiments/interactions/protocol.json'
DATA = ROOT/'experiments/interactions/data'
RESULTS = ROOT/'results/interactions'


def test_new_inputs_reproduce_disjoint_and_oracle(tmp_path):
    p = load_protocol(PROTOCOL)
    forbidden, hashes = historical_inputs(TRAIN, OLD)
    rows = generate_rows(p, forbidden)
    assert rows == read_json(DATA/'examples.json')
    prepare(PROTOCOL, TRAIN, OLD, tmp_path/'prepared')
    for name in ('examples.json', 'inputs.json', 'manifest.json'):
        assert (tmp_path/'prepared'/name).read_bytes() == (DATA/name).read_bytes()
    assert len(hashes) == 10
    altered = copy.deepcopy(rows); altered[0]['answer'] = (altered[0]['answer']+1) % 16
    with pytest.raises(ValueError): validate_rows(altered, forbidden)
    altered = copy.deepcopy(rows); altered[4] = {**altered[0], 'id': 'pair-00004'}
    with pytest.raises(ValueError): validate_rows(altered, forbidden)
    with pytest.raises(ValueError): validate_rows(rows[:-1], forbidden)


def test_changed_protocol_rejected(tmp_path):
    p = read_json(PROTOCOL); p['pairs'].pop()
    write_json(tmp_path/'bad.json', p)
    with pytest.raises(ValueError, match='frozen'): load_protocol(tmp_path/'bad.json')


@pytest.mark.parametrize('seconds', [0, -1, 7201, float('nan'), float('inf'), 1e-12])
def test_resource_limit_and_cleanup(tmp_path, seconds):
    out = tmp_path/'output'
    with pytest.raises((ValueError, TimeoutError)):
        evaluate_pairs(TRAIN, OLD, PROTOCOL, DATA, out, seconds)
    assert not out.exists() and not list(tmp_path.glob('.output-*'))


def test_collision_preserves_output(tmp_path):
    out = tmp_path/'result'; out.mkdir(); (out/'keep').write_text('keep')
    with pytest.raises(FileExistsError): evaluate_pairs(TRAIN, OLD, PROTOCOL, DATA, out)
    assert (out/'keep').read_text() == 'keep'
    assert sorted(p.name for p in out.iterdir()) == ['keep']


def test_changed_input_manifest_rejected(tmp_path):
    copied = tmp_path/'data'; shutil.copytree(DATA, copied)
    rows = read_json(copied/'examples.json'); rows[0]['answer'] = 99
    write_json(copied/'examples.json', rows)
    with pytest.raises(ValueError, match='artifact (size|hash)'): evaluate_pairs(TRAIN, OLD, PROTOCOL, copied, tmp_path/'out')
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('dtype,shape', [('O', (38, 512, 16)), ('f4', (38, 512, 1000000000)), ('f8', (38, 512, 16))])
def test_hostile_numeric_headers_bounded_before_allocation(tmp_path, dtype, shape):
    path = tmp_path/'bad.npy'
    with path.open('wb') as stream:
        np.lib.format.write_array_header_1_0(stream, {'shape': shape, 'fortran_order': False, 'descr': np.dtype(dtype).str})
        stream.write(b'\0'*16)
    with pytest.raises(ValueError): load_array(path, (38, 512, 16), np.float32)


def test_numeric_nonfinite_and_symlink_rejected(tmp_path):
    a = np.zeros((2, 3), dtype=np.float32); a[0, 0] = np.nan
    np.save(tmp_path/'bad.npy', a)
    with pytest.raises(ValueError): load_array(tmp_path/'bad.npy', (2, 3), np.float32)
    (tmp_path/'link.npy').symlink_to(tmp_path/'bad.npy')
    with pytest.raises(ValueError): load_array(tmp_path/'link.npy', (2, 3), np.float32)


@pytest.mark.parametrize('change', ['count', 'mean', 'pair', 'text', 'interval', 'nan', 'flag', 'hidden', 'critical'])
def test_report_rejects_malformed_or_inconsistent_summary(tmp_path, change):
    d = tmp_path/'bad'; shutil.copytree(RESULTS, d)
    s = read_json(d/'summary.json')
    if change == 'count': s['seeds'][0]['pairs'] *= 100
    elif change == 'mean': s['seeds'][0]['pairs'][0]['metrics']['loss']['mean'] += 1
    elif change == 'pair': s['seeds'][0]['pairs'][0]['pair'] = ['L0H4', 'L1H0']
    elif change == 'text': s['limits'] = ['x'*1001]
    elif change == 'interval': s['seeds'][0]['pairs'][0]['metrics']['loss']['ci95_family'] = [1, -1]
    elif change == 'nan': s['seeds'][0]['conditions'][0]['metrics']['loss']['mean'] = float('nan')
    elif change == 'flag': s['seeds'][0]['pairs'][0]['metrics']['loss']['resolved'] = not s['seeds'][0]['pairs'][0]['metrics']['loss']['resolved']
    elif change == 'hidden': s['seeds'][0]['pairs'][0]['hidden_accuracy'] = not s['seeds'][0]['pairs'][0]['hidden_accuracy']
    elif change == 'critical': s['seeds'][0]['family_critical_value'] = -1
    (d/'summary.json').write_text(json.dumps(s))
    with pytest.raises(ValueError): payload(d)


def test_original_renderer_and_provenance_remain_reconstructible(tmp_path):
    import importlib.util
    frozen = ROOT/'experiments/interactions/frozen-renderer'
    spec = importlib.util.spec_from_file_location('attention_lab._original_pair_renderer', frozen/'interaction_report.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.render_directory(RESULTS, tmp_path/'original.html')
    amendment = read_json(ROOT/'experiments/interactions/rendering-amendment.json')
    assert sha256(tmp_path/'original.html') == amendment['evaluation_report_sha256']
    assert sha256(RESULTS/'report.html') == amendment['updated_report_sha256']
    provenance = read_json(RESULTS/'provenance.json')
    for name, digest in provenance['source_sha256'].items():
        source = frozen/name if name in ('pair_report.html', 'interaction_report.py') else ROOT/'src/attention_lab'/name
        assert sha256(source) == digest, name
    for name, digest in amendment['updated_renderer_sha256'].items():
        assert sha256(ROOT/'src/attention_lab'/name) == digest


@pytest.mark.parametrize('sig', [signal.SIGINT, signal.SIGTERM])
def test_cli_signal_cleans_reserved_and_nested_staging(tmp_path, sig):
    out = tmp_path/'cancelled'
    args = [sys.executable, '-m', 'attention_lab.cli', 'pairs', '--training', str(TRAIN),
            '--previous', str(OLD), '--protocol', str(PROTOCOL), '--data', str(DATA), '--output', str(out)]
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'})
    try:
        deadline = time.monotonic()+30
        # Wait for real work inside the transaction, not a sleep in application code.
        while not list(tmp_path.glob('.cancelled-*/historical-replay')):
            assert process.poll() is None, process.communicate()
            if time.monotonic() > deadline: pytest.fail('evaluation did not enter its transaction')
            time.sleep(0.01)
        process.send_signal(sig)
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 2, (stdout, stderr)
        assert b'cancelled by signal' in stderr and b'Traceback' not in stderr
        assert not out.exists() and not list(tmp_path.glob('.cancelled-*'))
    finally:
        if process.poll() is None:
            process.kill(); process.wait()
