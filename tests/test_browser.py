"""Real Chromium against file://; every HTTP request blocked and counted."""
from pathlib import Path
import pytest
from playwright.sync_api import sync_playwright
from attention_lab.report import make_payload, render

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def browser():
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        yield browser
        browser.close()


def open_page(browser,path,width=1440):
    context=browser.new_context(viewport={'width':width,'height':1000},offline=True)
    requests=[]
    def intercept(route):
        if route.request.url.startswith(('http:','https:')):
            requests.append(route.request.url)
            route.abort()
        else:route.continue_()
    context.route('**/*',intercept)
    page=context.new_page()
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(path.resolve().as_uri())
    page.wait_for_function("document.querySelector('#status').textContent.includes('Seed')")
    return context,page,requests,errors


def test_navigation_keyboard_numeric_tables_and_dom_bounds(browser):
    context,page,requests,errors=open_page(browser,ROOT/'results/evaluation/report.html')
    try:
        assert page.title()=='Attention / Intervention — Retrieval Lab'
        assert page.locator('#prev').is_disabled()
        page.locator('#next').focus();page.keyboard.press('Enter')
        assert page.locator('#example').input_value()=='1'
        page.locator('#example').focus();page.keyboard.press('ArrowDown')
        assert page.locator('#example').input_value()=='2'
        page.locator('#head').focus();page.keyboard.press('ArrowDown')
        assert page.locator('#head').input_value()=='1'
        page.locator('#condition').select_option('noop')
        assert 'probability drop: 0.000000' in page.locator('#delta').inner_text()
        assert 'loss increase: 0.000000' in page.locator('#delta').inner_text()
        page.locator('summary').filter(has_text='Full-precision attention table').focus()
        page.keyboard.press('Enter')
        assert page.locator('#attention-numeric').is_visible()
        assert page.locator('#attention-numeric tbody tr').count()==10
        assert page.locator('#attention-numeric tbody td').count()==100
        page.locator('summary').filter(has_text='All 16 output probabilities').click()
        assert page.locator('#probabilities tbody tr').count()==16
        payload=make_payload(ROOT/'results/evaluation')
        for seed in (11,29,47):
            page.locator(f'#seed-{seed}').click()
            assert page.locator(f'#seed-{seed}').get_attribute('aria-pressed')=='true'
            for example in (0,7,15):
                page.locator('#example').select_option(str(example))
                for head in range(8):
                    page.locator('#head').select_option(str(head))
                    page.locator('#condition').select_option('ablate')
                    assert page.locator('#heatmap tbody td').count()==100
                    assert page.locator('#heatmap tbody tr').count()==10
                    assert page.locator('#effects tbody tr').count()==8
                    assert page.locator('#all-metrics tbody tr').count()==10
                    r=payload['examples'][str(seed)][example]
                    expected=payload['attention'][str(seed)][example]['pattern'][head//4][head%4][9][r['value_position']]
                    assert f'{expected:.6f}' in page.locator('#mass').inner_text()
                    assert page.locator('*').count()<1200
            assert page.locator('#next').is_disabled()
        page.locator('button[aria-label="Inspect L0H2"]').click()
        assert page.locator('#head').input_value()=='2'
        assert page.locator('#head').evaluate('(el)=>el===document.activeElement')
        assert 'head=2' in page.url
        page.reload()
        assert page.locator('#head').input_value()=='2'
        assert page.locator('#example').input_value()=='15'
        assert 'Seed 47' in page.locator('#status').inner_text()
        assert not errors and not requests
    finally:context.close()


def test_hostile_text_and_invalid_url_bounds(browser,tmp_path):
    payload=make_payload(ROOT/'results/evaluation')
    attack='</script><img src="https://invalid.example/pwn" onerror="window.pwned=1">'
    payload['examples']['11'][0]['labels'][0]=attack
    payload['summary']['limitations']=attack
    path=tmp_path/'hostile.html';path.write_text(render(payload))
    context,page,requests,errors=open_page(browser,path)
    try:
        assert attack in page.locator('#tokens').inner_text()
        assert page.locator('#limitations').inner_text()==attack
        assert page.locator('img').count()==0
        assert page.evaluate('window.pwned') is None
        page.evaluate("location.hash='seed=999&example=1e99&head=-1&condition=malicious'")
        page.wait_for_function("document.querySelector('#head').value==='0'")
        assert page.locator('#example').input_value()=='0'
        assert page.locator('#condition').input_value()=='ablate'
        assert page.locator('#heatmap tbody td').count()==100
        assert not errors and not requests
    finally:context.close()


def test_mobile_controls_remain_usable(browser):
    context,page,requests,errors=open_page(browser,ROOT/'results/evaluation/report.html',390)
    try:
        assert page.evaluate('document.documentElement.scrollWidth')<=390
        page.locator('#next').click()
        assert page.locator('#example').input_value()=='1'
        page.locator('#head').select_option('7')
        assert page.locator('#head-tag').inner_text()=='L1H3'
        assert page.locator('#heatmap').is_visible()
        assert not errors and not requests
    finally:context.close()
