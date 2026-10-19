"""Exercise the saved report's head/example/method/query links in Chromium."""
import json
import os
from pathlib import Path
import numpy as np
import pytest
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]


def test_report_links_and_mobile_layout(tmp_path):
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        page=browser.new_page(viewport={'width':1280,'height':900})
        errors=[];requests=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.on('request',lambda request:requests.append(request.url))
        page.goto((ROOT/'results/pretrained/report.html').as_uri())
        assert page.locator('#head option').count()==24
        assert page.locator('#weights tr').count()==64
        assert page.locator('#allheads tr').count()==24
        assert page.locator('#baseline').inner_text()=='4.615252'
        with np.load(ROOT/'results/pretrained/measurements.npz') as data:
            for i in range(24):
                page.select_option('#head',str(i))
                layer,head=divmod(i,4)
                for example in (0,1):
                    page.select_option('#example',str(example))
                    page.select_option('#query','63')
                    for method in ('zero','mean'):
                        page.select_option('#method',method)
                        base=data['intact'][example,63];value=data[method][layer,head,example,63]
                        text=page.locator('#effect').inner_text()
                        assert f'{float(base):.6f}' in text
                        assert f'{float(value):.6f}' in text
                        assert f'L{layer}H{head}' in page.locator('#selection').inner_text()
                        weights=page.locator('#weights tr td:nth-child(2)').all_text_contents()
                        np.testing.assert_allclose(list(map(float,weights)),data['attention'][example,layer,head,63],atol=5.1e-7,rtol=0)
        page.get_by_role('button',name='L2H3',exact=True).click()
        assert page.locator('#head').input_value()=='11'
        page.locator('#heatmap').click(position={'x':30,'y':30})
        assert page.locator('#query').input_value()=='3'
        page.select_option('#query','0')
        assert all(float(w)==0 for w in page.locator('#weights tr td:nth-child(2)').all_text_contents()[1:])
        page.screenshot(path=str(tmp_path/'pretrained-desktop.png'),full_page=True)
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        page.select_option('#head','0');page.select_option('#method','mean')
        page.screenshot(path=str(tmp_path/'pretrained-mobile.png'),full_page=True)
        assert not errors,errors
        assert not any(url.startswith(('http:','https:')) for url in requests),requests
        browser.close()
