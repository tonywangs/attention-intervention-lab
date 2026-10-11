from copy import deepcopy
from pathlib import Path
import pytest
from attention_lab.report import make_payload, render

ROOT=Path(__file__).resolve().parents[1]


def test_report_is_deterministic_and_self_contained():
    payload=make_payload(ROOT/'results/evaluation')
    html=render(payload)
    assert html==render(payload)==(ROOT/'results/evaluation/report.html').read_text()
    assert 'connect-src \'none\'' in html
    assert '<script src=' not in html and '<link ' not in html


@pytest.mark.parametrize('damage',['too_many_examples','oversized_label','huge_matrix','nonfinite','uncausal','unnormalized','too_many_logits'])
def test_report_rejects_unbounded_or_malformed_data(damage):
    payload=make_payload(ROOT/'results/evaluation')
    row=payload['examples']['11'][0]
    pattern=payload['attention']['11'][0]['pattern'][0][0]
    if damage=='too_many_examples':payload['examples']['11'].append(deepcopy(row))
    elif damage=='oversized_label':row['labels'][0]='x'*129
    elif damage=='huge_matrix':pattern.append([0]*10)
    elif damage=='nonfinite':pattern[0][0]=float('nan')
    elif damage=='uncausal':pattern[0][0]=.5;pattern[0][1]=.5
    elif damage=='unnormalized':pattern[0][0]=.2
    elif damage=='too_many_logits':row['conditions']['intact']['logits']*=100
    with pytest.raises(ValueError):render(payload)


def test_hostile_text_escaped_in_inert_json():
    payload=make_payload(ROOT/'results/evaluation')
    attack='</script><img src="https://invalid.example/pwn" onerror="window.pwned=1">'
    payload['examples']['11'][0]['labels'][0]=attack
    payload['summary']['limitations']=attack
    html=render(payload)
    assert attack not in html
    assert '\\u003c/script\\u003e' in html
