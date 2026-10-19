#!/usr/bin/env python3
"""Independent saved-result audit; optional numerical replay comparison."""
import argparse
import json
from pathlib import Path
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def audit(directory):
    directory = Path(directory)
    for name, expected in json.loads((directory / 'manifest.json').read_text()).items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected, name
    summary = json.loads((directory / 'summary.json').read_text())
    with np.load(directory / 'measurements.npz', allow_pickle=False) as z:
        assert set(z.files) == {'intact', 'noop', 'zero', 'mean', 'means', 'attention'}
        for name in z.files:
            assert np.isfinite(z[name]).all(), name
        assert z['intact'].shape == (16, 64)
        assert z['means'].shape == (6, 4, 32)
        for method in ('noop','zero','mean'):
            assert z[method].shape == (6, 4, 16, 64)
        assert z['attention'].shape == (2, 6, 4, 64, 64)
        assert np.all(z['attention'] >= 0)
        np.testing.assert_allclose(z['attention'].sum(-1), 1, atol=2e-6)
        assert np.count_nonzero(np.triu(z['attention'], 1)) == 0
        np.testing.assert_array_equal(z['noop'], np.broadcast_to(z['intact'], z['noop'].shape))
        assert summary['noop_max_abs_token_loss_delta'] == 0
        assert summary['evaluation_blocks'] == 16 and summary['scored_tokens'] == 1024
        np.testing.assert_allclose(summary['intact_loss_nats'], z['intact'].astype('float64').mean(), atol=1e-6)
        assert len(summary['heads']) == 24
        for index, row in enumerate(summary['heads']):
            layer, head = divmod(index, 4)
            assert (row['layer'], row['head']) == (layer, head)
            for method in ('zero','mean'):
                delta = z[method][layer,head].astype('float64') - z['intact']
                blocks = delta.mean(1)
                saved = row[method]
                np.testing.assert_allclose(saved['delta_nats'], delta.mean(), atol=1e-6)
                np.testing.assert_allclose(saved['loss_nats'], z[method][layer,head].astype('float64').mean(), atol=1e-6)
                np.testing.assert_allclose(saved['block_deltas'], blocks, atol=1e-6)
                np.testing.assert_allclose([saved['min_block_delta'],saved['max_block_delta']], [blocks.min(),blocks.max()], atol=1e-6)
                assert saved['harmed_blocks'] == int((blocks>0).sum())
                assert saved['improved_blocks'] == int((blocks<0).sum())
            np.testing.assert_allclose(row['mean_minus_zero_nats'], (z['mean'][layer,head].astype('float64')-z['zero'][layer,head]).mean(), atol=1e-6)
        # The browser's data must be linked to these exact measurements, not stand-in data.
        html = (directory / 'report.html').read_text()
        payload = json.loads(html.split('<script id="data" type="application/json">')[1].split('</script>')[0])
        assert payload['summary'] == summary
        for method in ('zero','mean'):
            np.testing.assert_array_equal(payload[method], z[method][:,:,:2])
        np.testing.assert_array_equal(payload['intact'], z['intact'][:2])
        np.testing.assert_allclose(payload['attention'], z['attention'], atol=5.01e-7, rtol=0)
    return payload


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=ROOT/'results/pretrained')
    group=p.add_mutually_exclusive_group()
    group.add_argument('--replay', type=Path)
    group.add_argument('--reproduce', action='store_true', help='Run into a temporary directory with Python network connections blocked')
    p.add_argument('--cache', type=Path, default=ROOT/'.cache/pretrained')
    args=p.parse_args()
    temporary=None
    if args.reproduce:
        import tempfile
        from unittest.mock import patch
        from attention_lab.pretrained import run
        temporary=tempfile.TemporaryDirectory(prefix='attention-pretrained-')
        args.replay=Path(temporary.name)/'replay'
        with patch('socket.socket.connect', side_effect=AssertionError('Network access forbidden during replay')):
            run(ROOT/'experiments/pretrained',args.cache,args.replay)
    payload=audit(args.results)
    spec=ROOT/'experiments/pretrained'
    for name,expected in json.loads((spec/'manifest.json').read_text()).items():
        assert hashlib.sha256((spec/name).read_bytes()).hexdigest()==expected,name
    inputs=json.loads((spec/'inputs.json').read_text())
    assert len(inputs['calibration'])==len(inputs['evaluation'])==16
    assert all(len(block)==65 for blocks in inputs.values() for block in blocks)
    assert not set(map(tuple,inputs['calibration'])) & set(map(tuple,inputs['evaluation']))
    for e,example in enumerate(payload['examples']):
        assert example['index']==e and example['token_ids']==inputs['evaluation'][e]
        assert len(example['tokens'])==65
    provenance=json.loads((args.results/'provenance.json').read_text())
    for key,file in [('inputs_sha256','inputs.json'),('protocol_sha256','protocol.json'),('spec_manifest_sha256','manifest.json')]:
        assert provenance[key]==hashlib.sha256((spec/file).read_bytes()).hexdigest()
    assert payload['provenance']==provenance
    assert provenance['model']==json.loads((spec/'model.json').read_text())
    assert payload['sources']==json.loads((spec/'sources.json').read_text())
    assert payload['protocol']==json.loads((spec/'protocol.json').read_text())
    if args.replay:
        audit(args.replay)
        with np.load(args.results/'measurements.npz') as expected,np.load(args.replay/'measurements.npz') as actual:
            for name in expected.files:
                np.testing.assert_allclose(actual[name],expected[name],atol=2e-6,rtol=1e-5,err_msg=name)
        assert json.loads((args.results/'examples.json').read_text())==json.loads((args.replay/'examples.json').read_text())
        print('All replay losses, means and attention match (atol=2e-6, rtol=1e-5).')
    if temporary is not None:temporary.cleanup()
    print('Pretrained artifacts, independent metrics, causal attention and browser-data links verified.')

if __name__=='__main__':main()
