import datetime as dt

import polars as pl
import pytest

from moexutils import futures_params as fp
from moexutils import lake


def envelope(block, columns, rows):
    return {'observed_at': '2026-10-06T20:00:00+00:00', 'source_url': 'https://iss.moex.com/test',
            'payload': {block: {'columns': columns, 'data': rows}}}


def risk(day='2020-01-10', total=1):
    value = envelope('limits', ['tradedate', 'assetcode', 'mr1', 'mr2', 'mr3', 'updatetime'],
                     [[day, 'Si', 0.1, 0.2, 0.3, day + ' 09:00:00']])
    value['payload']['limits.cursor'] = {'columns': ['INDEX', 'TOTAL'], 'data': [[0, total]]}
    return value


def test_historical_risk_retains_retrieval_time():
    frame = fp.risk_frame(risk(), dt.date(2020, 1, 10))
    assert frame['date'][0] == dt.date(2020, 1, 10)
    assert frame['observed_at'][0].year == 2026
    assert frame['availability'][0] == 'observed'


def test_ignored_date_and_truncated_response_rejected():
    with pytest.raises(ValueError, match='different date'):
        fp.risk_frame(risk('2026-10-06'), dt.date(2020, 1, 10))
    with pytest.raises(ValueError, match='incomplete'):
        fp.risk_frame(risk(total=2000), dt.date(2020, 1, 10))


def test_current_parameters_do_not_backdate():
    value = envelope('securities', ['SECID', 'BOARDID', 'MINSTEP', 'STEPPRICE', 'INITIALMARGIN', 'IMTIME'],
                     [['SiZ6', 'RFUD', 1, 1, 10000, '2026-10-05 19:00:00']])
    frame = fp.current_frame(value)
    assert frame['effective_from'][0] is None and frame['effective_to'][0] is None
    assert frame['IMTIME'][0] == '2026-10-05 19:00:00'
    value['payload']['securities']['data'] *= 2
    with pytest.raises(ValueError, match='duplicate'):
        fp.current_frame(value)


def test_multiple_source_update_times_preserved():
    value = risk(total=2)
    value['payload']['limits']['data'].append(['2020-01-10', 'Si', 0.15, 0.2, 0.3, '2020-01-10 19:00:00'])
    assert fp.risk_frame(value, dt.date(2020, 1, 10)).height == 2
    value['payload']['limits']['data'][1][-1] = '2020-01-10 09:00:00'
    with pytest.raises(ValueError, match='conflicting'):
        fp.risk_frame(value, dt.date(2020, 1, 10))


def test_reused_identifier_is_not_guessed():
    value = envelope('description', ['name', 'value'], [['SECID', 'SiZ5']])
    with pytest.raises(ValueError, match='mismatched'):
        fp.description_frame(value, 'SiZ5_2015')


def test_missing_fields_remain_unknown_and_raw_is_preserved():
    value = envelope('description', ['name', 'value'], [['SECID', 'RGF5_2015'], ['TYPE', 'futures']])
    frame = fp.description_frame(value, 'RGF5_2015')
    assert frame.schema['ASSETCODE'] == pl.String
    assert frame['ASSETCODE'][0] is None
    assert 'RGF5_2015' in frame['raw_json'][0]


def test_old_card_identifier_can_be_verified_by_boards():
    value = envelope('description', ['name', 'value'], [['TYPE', 'futures']])
    value['payload']['boards'] = {'columns': ['secid'], 'data': [['GAZR-17.12']]}
    assert fp.description_frame(value, 'GAZR-17.12')['identity_status'][0] == 'boards_secid'
    value['payload']['description']['data'] = []
    value['payload']['boards']['data'] = []
    frame = fp.description_frame(value, 'UNKNOWN')
    assert frame['identity_status'][0] == 'unverified'
    assert frame['availability'][0] == 'unavailable'


def test_cached_backfill_requires_no_network(tmp_path, monkeypatch):
    monkeypatch.setattr(fp, '_root', lambda: tmp_path)
    day = dt.date(2020, 1, 10)
    fp._save(tmp_path / 'risk_limits' / f'{day}.json', risk())
    monkeypatch.setattr(fp, '_get', lambda *a, **kw: pytest.fail('unexpected network'))
    assert fp._risk_day((day, False)).height == 1


def test_lake_versions_are_idempotent_and_read_preserves_source_updates(tmp_path, monkeypatch):
    monkeypatch.setenv('MOEX_LAKE_CATALOG', 'ducklake:' + (tmp_path / 'catalog.ducklake').as_posix())
    monkeypatch.setattr(lake, 'LAKE_DATA_PATH', str(tmp_path / 'data'))
    value = risk(total=2)
    value['payload']['limits']['data'].append(['2020-01-10', 'Si', 0.15, 0.2, 0.3, '2020-01-10 19:00:00'])
    frame = fp.risk_frame(value, dt.date(2020, 1, 10))
    lake.write('futures_risk_limits', frame)
    lake.write('futures_risk_limits', frame)
    assert lake.query('SELECT count(*) FROM lake.futures_risk_limits').item() == 2
    assert fp.read_risk_limits('2020-01-10', '2020-01-10', 'Si').height == 2
    value['observed_at'] = '2026-10-07T20:00:00+00:00'
    value['payload']['limits']['data'] = [['2020-01-10', 'RTS', 0.1, 0.2, 0.3, '2020-01-10 19:00:00']]
    value['payload']['limits.cursor']['data'][0][1] = 1
    lake.write('futures_risk_limits', fp.risk_frame(value, dt.date(2020, 1, 10)))
    assert fp.read_risk_limits('2020-01-10', '2020-01-10', 'Si').is_empty()
    assert lake.query('SELECT count(*) FROM lake.futures_risk_limits').item() == 3
