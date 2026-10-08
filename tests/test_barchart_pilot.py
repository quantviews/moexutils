import copy
from urllib.parse import urlencode

import pytest

from moexutils.barchart_pilot import SYMBOLS, parse_capture, import_capture, read_contracts


def capture():
    return {'schema_version': 1, 'responses': [
        {'symbol': s, 'http_status': 200,
         'url': 'https://www.barchart.com/proxies/timeseries/historical/queryeod.ashx?' + urlencode(
             dict(symbol=s, data='daily', volume='contract', backadjust='false')),
         'body': f'{s},2026-01-05,10,12,9,11,0,50\n{s},2026-01-06,11,13,10,12,20,60'}
        for s in SYMBOLS]}


def test_parse_preserves_zero_volume_and_close_semantics():
    frame = parse_capture(capture())
    assert frame.height == 12
    assert (frame['volume'] == 0).sum() == 6
    assert frame['price_kind'].unique().to_list() == ['provider_close_not_verified_settlement']
    assert not frame['execution_model_ready'].any()


@pytest.mark.parametrize('problem', ['duplicate', 'nearest', 'wrong_symbol', 'nonfinite', 'partial', 'html'])
def test_rejects_invalid_capture(problem):
    data = copy.deepcopy(capture())
    r = data['responses'][0]
    if problem == 'duplicate':
        r['body'] += '\n' + r['body'].splitlines()[0]
    elif problem == 'nearest':
        r['url'] = r['url'].replace('data=daily', 'data=dailynearest')
    elif problem == 'wrong_symbol':
        r['body'] = r['body'].replace('KCH26', 'KCK26')
    elif problem == 'nonfinite':
        r['body'] = r['body'].replace(',11,0,', ',nan,0,')
    elif problem == 'partial':
        data['responses'].pop()
    else:
        r['body'] = '<html>login</html>'
    with pytest.raises(ValueError):
        parse_capture(data)


def test_import_checksum_and_no_overwrite(tmp_path):
    import json
    source, output = tmp_path / 'source.json', tmp_path / 'package'
    source.write_text(json.dumps(capture()), encoding='utf-8')
    import_capture(source, output)
    assert read_contracts(output).height == 12
    with pytest.raises(FileExistsError):
        import_capture(source, output)
    with (output / 'contracts.parquet').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        read_contracts(output)
