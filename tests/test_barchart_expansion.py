import copy

import pytest

from moexutils.barchart_expansion import inspect_captures


def inputs():
    manifest = {'campaign': 'test', 'contracts': [
        {'symbol': 'KCN25', 'asset': 'COFFEE', 'moex_first': '2025-06-17',
         'moex_expiration': '2025-06-20', 'moex_secid': 'KCM5'},
        {'symbol': 'OJN25', 'asset': 'ORANGE', 'moex_first': '2025-06-17',
         'moex_expiration': '2025-07-01', 'moex_secid': 'OJN5'}]}
    capture = {'schema_version': 2, 'campaign': 'test', 'responses': [
        {'symbol': 'KCN25', 'http_status': 200,
         'url': 'https://www.barchart.com/proxies/timeseries/historical/queryeod.ashx?symbol=KCN25&data=daily&volume=contract&backadjust=false&maxrecords=10000',
         'body': 'KCN25,2025-06-17,10,11,9,10,2,3\nKCN25,2025-06-18,11,12,10,11,0,3'}]}
    return capture, manifest


def test_partial_and_identical_resume_are_explicit():
    c, m = inputs()
    c['stopped'] = {'index': 1}
    f, cov = inspect_captures([c, c], m)
    assert f.height == 2
    assert cov[0]['zero_volume_rows'] == 1
    assert cov[1]['status'] == 'missing'


def test_conflicting_resume_is_rejected():
    c, m = inputs()
    changed = copy.deepcopy(c)
    changed['responses'][0]['body'] = changed['responses'][0]['body'].replace(',2,3', ',5,3')
    with pytest.raises(ValueError, match='Conflicting'):
        inspect_captures([c, changed], m)


def test_cap_and_short_coverage_flagged():
    c, m = inputs()
    c['responses'][0]['url'] = c['responses'][0]['url'].replace('10000', '2')
    m['contracts'][0]['moex_first'] = '2025-01-01'
    _, cov = inspect_captures([c], m)
    assert set(cov[0]['flags']) == {'possible_record_cap', 'starts_after_moex'}


def test_wrong_campaign_rejected():
    c, m = inputs()
    c['campaign'] = 'different'
    with pytest.raises(ValueError, match='campaign'):
        inspect_captures([c], m)
