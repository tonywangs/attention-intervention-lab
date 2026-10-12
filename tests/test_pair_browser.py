"""Chromium checks every seed/pair/metric value with external requests blocked."""
import copy
from pathlib import Path
import pytest
from playwright.sync_api import sync_playwright
from attention_lab.interaction_report import payload, render

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def browser():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        yield browser
        browser.close()


def opened(browser, path, width=1360):
    context = browser.new_context(viewport={'width': width, 'height': 1000}, offline=True)
    requests, errors = [], []
    def route(r):
        if r.request.url.startswith(('http:', 'https:')):
            requests.append(r.request.url); r.abort()
        else: r.continue_()
    context.route('**/*', route)
    page = context.new_page(); page.on('pageerror', lambda e: errors.append(str(e)))
    page.goto(path.resolve().as_uri())
    page.wait_for_function("document.querySelector('#status').textContent.includes('Seed')")
    return context, page, requests, errors


def test_all_aggregate_values_and_bounded_rendering(browser):
    data = payload(ROOT/'results/interactions')
    context, page, requests, errors = opened(browser, ROOT/'results/interactions/report.html')
    try:
        assert page.title() == 'Paired Head Interventions — Retrieval Lab'
        for index, seed in enumerate(data['summary']['seeds']):
            page.select_option('#seed', str(index))
            for metric in ('loss', 'p_correct', 'accuracy'):
                page.select_option('#metric', metric)
                rendered = page.locator('#all-pairs tbody tr').evaluate_all('(rows)=>rows.map(r=>Array.from(r.cells).map(c=>c.textContent))')
                assert len(rendered) == 28
                conditions = {c['name']: c['metrics'][metric] for c in seed['conditions']}
                for k, pair in enumerate(seed['pairs']):
                    expected = [conditions[pair['pair'][0]]['effect'], conditions[pair['pair'][1]]['effect'],
                                conditions['+'.join(pair['pair'])]['effect'], pair['metrics'][metric]['mean']]
                    assert rendered[k][0] == ' + '.join(pair['pair'])
                    # JavaScript rounds binary-exact half-decimals away from zero;
                    # Python format uses ties-to-even. Check displayed precision.
                    assert all(abs(float(a)-b) <= 0.000000500001 for a, b in zip(rendered[k][1:5], expected))
                    lo, hi = pair['metrics'][metric]['ci95_family']
                    actual_ci = [float(x) for x in rendered[k][5].strip('[]').split(',')]
                    assert all(abs(a-b) <= 0.000000500001 for a, b in zip(actual_ci, [lo, hi]))
                    page.select_option('#pair', str(k))
                    assert abs(float(page.locator('#interaction').inner_text())-expected[3]) <= 0.000000500001
                    assert page.locator('#all-pairs tbody tr').count() == 28
                assert page.locator('*').count() < 1200
        assert not requests and not errors
    finally: context.close()


def test_keyboard_examples_and_full_precision(browser):
    data = payload(ROOT/'results/interactions')
    context, page, requests, errors = opened(browser, ROOT/'results/interactions/report.html')
    try:
        page.locator('#pair').focus(); page.keyboard.press('ArrowDown')
        assert page.locator('#pair').input_value() == '1'
        page.locator('#next').focus(); page.keyboard.press('Enter')
        assert page.locator('#example').input_value() == '1'
        page.locator('#prev').focus(); page.keyboard.press('Enter')
        assert page.locator('#example').input_value() == '0'
        assert page.locator('#prev').is_disabled()
        for si, seed in enumerate(data['summary']['seeds']):
            page.select_option('#seed', str(si))
            for pair_index in (0, 11, 27):
                page.select_option('#pair', str(pair_index))
                pair = data['protocol']['pairs'][pair_index]
                for example in (0, 7, 15):
                    page.select_option('#example', str(example))
                    rendered = page.locator('#example-values tbody tr').evaluate_all('(rows)=>rows.map(r=>Array.from(r.cells).slice(1).map(c=>Number(c.textContent)))')
                    expected = [seed['examples'][data['protocol']['conditions'].index(name)][example]
                                for name in ['intact', *pair, '+'.join(pair)]]
                    assert rendered == expected
        assert page.locator('#next').is_disabled()
        page.locator('summary').focus(); page.keyboard.press('Enter')
        assert page.locator('#precision').is_visible()
        page.locator('#all-pairs button').first.focus(); page.keyboard.press('Enter')
        assert page.locator('#pair').input_value() == '0'
        assert page.locator('#pair').evaluate('(e)=>document.activeElement===e')
        assert not requests and not errors
    finally: context.close()


def test_hostile_text_and_mobile_bounds(browser, tmp_path):
    data = payload(ROOT/'results/interactions')
    hostile = '</script><img src="https://example.invalid/x" onerror="window.pwned=1"><script>window.pwned=2</script>&'
    data['summary']['limits'] = [hostile]
    path = tmp_path/'hostile.html'; path.write_text(render(data))
    context, page, requests, errors = opened(browser, path, 390)
    try:
        assert page.locator('#limits li').inner_text() == hostile
        assert page.evaluate('window.pwned') is None
        assert page.locator('img').count() == 0
        assert page.evaluate('document.documentElement.scrollWidth') <= 390
        for selector in ('#seed', '#pair', '#metric', '#example', '#next'):
            page.locator(selector).scroll_into_view_if_needed()
            box = page.locator(selector).bounding_box()
            assert box['x'] >= 0 and box['x']+box['width'] <= 390
            assert box['height'] >= 44
        for i in range(75):
            page.select_option('#pair', str(i % 28))
        assert page.locator('*').count() < 1200
        assert not requests and not errors
    finally: context.close()
