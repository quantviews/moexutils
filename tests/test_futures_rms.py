import datetime as dt
import json

import pytest

from moexutils import futures_rms as rms


def test_date_is_not_inferred_from_request():
    env = {'observed_at': '2026-10-06T20:00:00+00:00', 'source_url': 'https://iss.moex.com/test',
           'payload': {'staticparams': {'columns': ['tradedate', 'assetcode', 'updatetime'],
                                       'data': [['2026-10-06', 'Si', '2026-10-06 12:00:00']]}}}
    with pytest.raises(ValueError, match='wrong date'):
        rms.parse(env, 'staticparams', dt.date(2020, 1, 1))


def test_distinct_risk_categories_not_collapsed():
    env = {'observed_at': '2026-10-06T20:00:00+00:00', 'source_url': 'https://iss.moex.com/test',
           'payload': {'rclimits': {'columns': ['tradedate', 'assetcode', 'updatetime', 'level', 'mr1'],
                                   'data': [['2026-10-06', 'Si', '2026-10-05 12:00:00', 'L', .4],
                                            ['2026-10-06', 'Si', '2026-10-05 12:00:00', 'M', .3]]}}}
    frame = rms.parse(env, 'rclimits', dt.date(2026, 10, 6))
    assert frame.height == frame['row_hash'].n_unique() == 2
    assert rms.parse(env, 'rclimits', dt.date(2026, 10, 6))['row_hash'].to_list() == frame['row_hash'].to_list()


def test_no_arbitrary_sql_identifiers():
    with pytest.raises(ValueError, match='Unknown'):
        rms.read('bad', '2020-01-01', '2026-01-01')


@pytest.mark.parametrize('truncated', [False, True])
def test_archive_pagination_and_cache(monkeypatch, tmp_path, truncated):
    day = dt.date(2026, 10, 6)
    calls = []

    class Response:
        def __init__(self, offset):
            self.offset = offset

        def raise_for_status(self):
            pass

        def json(self):
            return {
                'staticparams': {
                    'columns': ['tradedate', 'assetcode', 'updatetime'],
                    'data': [] if truncated and self.offset else [
                        [str(day), ['Si', 'BR'][self.offset], '2026-10-06 12:00:00']]},
                'staticparams.cursor': {'columns': ['INDEX', 'TOTAL'],
                                        'data': [[self.offset, 2]]}}

    class Session:
        def get(self, url, params):
            calls.append(params['start'])
            return Response(params['start'])

    monkeypatch.setattr(rms.fp, '_session', Session)
    monkeypatch.setattr(rms.lake, 'DATA_ROOT', tmp_path)
    path = tmp_path / 'raw' / 'futures_rms' / 'staticparams' / f'{day}.json'
    if truncated:
        with pytest.raises(ValueError, match='1/2'):
            rms._day(('staticparams', day, False))
        assert not path.exists()
    else:
        frame = rms._day(('staticparams', day, False))
        assert frame['assetcode'].to_list() == ['Si', 'BR']
        assert len(json.loads(path.read_text(encoding='utf-8'))['pages']) == 2
        assert rms._day(('staticparams', day, False)).equals(frame)
    assert calls == [0, 1]
