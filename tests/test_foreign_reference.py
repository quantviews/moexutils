import datetime as dt

import pytest

from moexutils.foreign_reference import parse_chart


def payload():
    return {'chart': {'error': None, 'result': [{'meta': {'symbol': 'KC=F', 'currency': 'USX',
        'exchangeTimezoneName': 'America/New_York'},
        'timestamp': [int(dt.datetime(2026, 10, d, 4, tzinfo=dt.timezone.utc).timestamp()) for d in (6, 7, 8)],
        'indicators': {'quote': [{'open': [1, 2, 3], 'high': [1, 2, 3], 'low': [1, 2, 3],
                                  'close': [None, 2, 3], 'volume': [0, 2, 3]}]}}]}}


def test_daily_range_excludes_current_bar_preserves_null_and_units():
    frame = parse_chart(payload(), 'COFFEE', dt.date(2026, 10, 6), dt.date(2026, 10, 7))
    assert frame.height == 2 and frame['close'][0] is None
    assert frame['source_currency'].unique().to_list() == ['USX']
    assert not frame['publication_time_verified'].any()


@pytest.mark.parametrize('fault', ['currency', 'split', 'length', 'duplicates'])
def test_bad_or_ambiguous_sources_fail(fault):
    data = payload()
    result = data['chart']['result'][0]
    if fault == 'currency':
        result['meta']['currency'] = 'USD'
    elif fault == 'split':
        result['events'] = {'splits': {'1': {'splitRatio': '10:1'}}}
    elif fault == 'length':
        result['indicators']['quote'][0]['close'] = [1]
    else:
        result['timestamp'][1] = result['timestamp'][0]
    with pytest.raises(ValueError):
        parse_chart(data, 'COFFEE', dt.date(2026, 10, 1), dt.date(2026, 10, 7))
