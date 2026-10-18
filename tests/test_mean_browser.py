"""Exercise all comparison cells and controls in network-blocked Chromium."""
import json
import numpy as np
from pathlib import Path
import pytest
from playwright.sync_api import sync_playwright
from attention_lab.mean_report import TEMPLATE

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def browser():
    with sync_playwright() as p:
        b=p.chromium.launch(headless=True);yield b;b.close()


def opened(browser,path,width=1360):
    c=browser.new_context(offline=True,viewport={'width':width,'height':900})
    requests=[];errors=[]
    def route(r):
        if r.request.url.startswith(('http:','https:')):requests.append(r.request.url);r.abort()
        else:r.continue_()
    c.route('**/*',route);page=c.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(path.resolve().as_uri());page.wait_for_function("document.querySelector('#finding').textContent.includes('Model')")
    return c,page,requests,errors


def test_all_cells_controls_and_precision(browser):
    summary=json.loads((ROOT/'results/mean/summary.json').read_text())
    c,page,requests,errors=opened(browser,ROOT/'results/mean/report.html')
    try:
        protocol=json.loads((ROOT/'results/mean/protocol.json').read_text())
        for si,s in enumerate(summary['seeds']):
            arrays={k:np.load(ROOT/f'results/mean/{k}-outcomes-{s["seed"]}.npy') for k in ('zero','mean')}
            page.select_option('#seed',str(si))
            for mi,m in enumerate(('loss','p_correct','accuracy')):
                page.select_option('#metric',str(mi))
                cells=page.locator('#pairs tbody tr').evaluate_all('(rs)=>rs.map(r=>Array.from(r.cells).map(c=>c.textContent))')
                assert len(cells)==28
                for pi in range(28):
                    for ki,k in enumerate(('zero','mean','difference')):
                        x=s['comparisons'][k][pi]['metrics'][m]
                        estimate,rest=cells[pi][ki+1].split(' [');interval,flag=rest.split('] ')
                        for actual,expected in zip([float(estimate),*map(float,interval.split(', '))],[x['mean'],*x['ci95_family']]):
                            assert abs(actual-expected)<=max(1e-10,abs(expected)*5e-6)
                        assert flag==('constant' if x['degenerate'] else 'resolved' if x['resolved'] else 'unresolved')
                    page.select_option('#pair',str(pi))
                    exact=json.loads(page.locator('#exact').text_content())
                    assert exact=={k:s['comparisons'][k][pi] for k in ('zero','mean','difference')}
                    rendered=page.locator('#examples tbody tr').evaluate_all('(rs)=>rs.map(r=>Array.from(r.cells).slice(1).map(c=>Number(c.textContent)))')
                    a,b=protocol['pairs'][pi];names=protocol['conditions']
                    expected=[]
                    for i in range(16):
                        v=[arr[names.index(a+'+'+b),i,mi]-arr[names.index(a),i,mi]-arr[names.index(b),i,mi]+arr[0,i,mi] for arr in arrays.values()]
                        expected.append([*v,v[1]-v[0]])
                    np.testing.assert_array_equal(rendered,expected)
                    condition_rows=page.locator('#conditions tbody tr').evaluate_all('(rs)=>rs.map(r=>Array.from(r.cells).map(c=>c.textContent))')
                    for ci,name in enumerate(['intact',a,b,a+'+'+b]):
                        assert condition_rows[ci][0]==name
                        for ki,k in enumerate(('zero','mean')):
                            got=float(condition_rows[ci][ki+1].split(';')[0]);want=s['conditions'][k][names.index(name)]['metrics'][m]['mean']
                            assert abs(got-want)<=max(1e-10,abs(want)*5e-6)
                    assert page.locator('#examples tbody tr').count()==16
                    assert page.locator('#conditions tbody tr').count()==4
                assert page.locator('*').count()<600
        page.locator('#pair').focus();page.keyboard.press('Home');page.keyboard.press('ArrowDown')
        assert page.locator('#pair').input_value()=='1'
        assert not requests and not errors
    finally:c.close()


def test_hostile_text_mobile_and_bounded_updates(browser,tmp_path):
    html=(ROOT/'results/mean/report.html').read_text()
    marker='<script id="data" type="application/json">'
    raw=html.split(marker)[1].split('</script>')[0];data=json.loads(raw)
    hostile='</script><img src="https://example.invalid/x" onerror="window.pwned=1"><script>window.pwned=2</script>&'
    data['limits']=[hostile]
    encoded=json.dumps(data).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    path=tmp_path/'hostile.html';path.write_text(TEMPLATE.replace('__DATA__',encoded))
    c,page,requests,errors=opened(browser,path,390)
    try:
        assert page.locator('#limits li').inner_text()==hostile
        assert page.evaluate('window.pwned') is None and page.locator('img').count()==0
        assert page.evaluate('document.documentElement.scrollWidth')<=390
        for selector in ('#seed','#metric','#pair'):
            box=page.locator(selector).bounding_box();assert box['x']>=0 and box['x']+box['width']<=390
            assert box['height']>=44
        for i in range(100):page.select_option('#pair',str(i%28))
        assert page.locator('*').count()<600
        assert not requests and not errors
    finally:c.close()
