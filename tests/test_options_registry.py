import datetime as dt

import polars as pl
import pytest

from moexutils import lake, options


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv('MOEX_LAKE_CATALOG', 'ducklake:' + (tmp_path / 'catalog.ducklake').as_posix())
    monkeypatch.setattr(lake, 'LAKE_DATA_PATH', str(tmp_path / 'lake'))


def series_frame():
    return pl.DataFrame({'name': ['OLD', 'NEW'],
                         'start_date': [dt.date(2024, 1, 1), dt.date(2025, 1, 1)],
                         'expiration_date': [dt.date(2025, 3, 1), dt.date(2026, 3, 1)],
                         'asset_code': ['A', 'A'], 'underlying_asset': ['F1', 'F2'],
                         'is_traded': [0, 0]})


def result(row):
    frame = pl.DataFrame({'secid': ['REUSED'], 'shortname': [row['name']], 'option_type': ['C'],
                          'strike': [100.], 'history_from': [row['start_date']],
                          'history_till': [row['expiration_date']], 'is_traded': [0],
                          'strike_source': ['ISS'],
                          'series_name': [row['name']]})
    return frame, {**row, 'lot_size': 1., 'lot_size_source_secid': 'REUSED',
                   'unit': 'RUB', 'faceunit': 'RUB'}


def test_archive_preserves_reused_codes_and_resumes(env, monkeypatch):
    monkeypatch.setattr(options, 'fetch_series', series_frame)
    calls = []

    def fetch(row):
        calls.append(row['name'])
        return result(row)

    monkeypatch.setattr(options, 'fetch_series_contracts', fetch)
    out = options.update_registry('2025-01-01', '2026-09-30', workers=2, flush_every=1)
    assert out['contract_rows_written'] == 2
    read = options.read_contracts(secids='REUSED')
    assert read.height == 2
    assert read['underlying_asset'].to_list() == ['F1', 'F2']
    assert options.read_contracts(secids=[]).is_empty()
    assert options.update_registry('2025-01-01', '2026-09-30')['series_processed'] == 0
    assert sorted(calls) == ['NEW', 'OLD']
    assert options.read_contracts()['lot_size'].to_list() == [1., 1.]


def test_failed_series_remains_pending_without_losing_success(env, monkeypatch):
    monkeypatch.setattr(options, 'fetch_series', series_frame)

    def fetch(row):
        if row['name'] == 'NEW':
            raise ConnectionError('ISS failed')
        return result(row)

    monkeypatch.setattr(options, 'fetch_series_contracts', fetch)
    with pytest.raises(ExceptionGroup):
        options.update_registry('2025-01-01', '2026-09-30', workers=1)
    assert options.read_contracts()['series_name'].to_list() == ['OLD']
    state = lake.query('SELECT name FROM lake.load_state')['name'].to_list()
    assert 'options_registry' not in state
    assert options.STATE_PREFIX + 'NEW' not in state
    monkeypatch.setattr(options, 'fetch_series_contracts', result)
    assert options.update_registry('2025-01-01', '2026-09-30')['series_processed'] == 1
    assert options.read_contracts().height == 2


class Response:
    def raise_for_status(self):
        pass

    def json(self):
        return {'securities': {'columns': ['secid', 'shortname', 'option_type', 'strike',
                                           'history_from', 'history_till'],
                               'data': [['S', 'OLD', 'P', -10., '2025-01-01', '2025-03-01']]}}


class Session:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def get(self, *args, **kwargs):
        return Response()


def test_lot_size_not_taken_from_relisted_contract(monkeypatch):
    monkeypatch.setattr(options.iss, 'make_session', Session)
    monkeypatch.setattr(options.iss, 'security_description',
                        lambda *args: {'SERIES_NAME': 'DIFFERENT', 'LOTSIZE': '999'})
    contracts, metadata = options.fetch_series_contracts(series_frame().to_dicts()[0])
    assert contracts['strike'][0] == -10.
    assert metadata['lot_size'] is None
    assert metadata['lot_size_source_secid'] is None


def test_verified_series_lot_size_and_typed_dates(monkeypatch):
    monkeypatch.setattr(options.iss, 'make_session', Session)
    monkeypatch.setattr(options.iss, 'security_description',
                        lambda *args: {'SERIES_NAME': 'OLD', 'LOTSIZE': '1', 'UNIT': 'RUB'})
    contracts, metadata = options.fetch_series_contracts(series_frame().to_dicts()[0])
    assert contracts.schema['history_from'] == pl.Date
    assert metadata['lot_size'] == 1.
    assert metadata['lot_size_source_secid'] == 'S'


def test_missing_strike_uses_exact_full_series_code(monkeypatch):
    class MissingStrike(Response):
        def json(self):
            payload = super().json()
            payload['securities']['data'][0][1:4] = ['SPBEP110326PE310', 'P', None]
            return payload

    monkeypatch.setattr(Session, 'get', lambda *args, **kwargs: MissingStrike())
    monkeypatch.setattr(options.iss, 'make_session', Session)
    monkeypatch.setattr(options.iss, 'security_description', lambda *args: {})
    row = {**series_frame().to_dicts()[0], 'name': 'SPBEP110326XE'}
    contracts, _ = options.fetch_series_contracts(row)
    assert contracts['strike'][0] == 310.
    assert contracts['strike_source'][0] == 'full_code'
    with pytest.raises(ValueError, match='cannot recover'):
        options.fetch_series_contracts({**row, 'name': 'OTHERXE'})


def test_missing_archive_series_recovered_from_card(env, monkeypatch):
    monkeypatch.setattr(options, 'fetch_series', series_frame)
    monkeypatch.setattr(options, 'fetch_series_contracts', result)
    lake.write('options', pl.DataFrame({'SECID': ['ABSENT'], 'BOARDID': ['ROPD'],
                                       'date': [dt.date(2025, 2, 1)]}))
    monkeypatch.setattr(options.iss, 'make_session', Session)
    desc = {'SERIES_NAME': 'ARCHIVE', 'STRIKE': '0', 'OPTIONTYPE': 'P',
            'FRSTTRADE': '2025-01-01', 'LSTDELDATE': '2025-03-01',
            'ASSETCODE': 'A', 'UNDERLYINGASSET': 'F', 'SHORTNAME': 'ARCHIVEP0',
            'LOTSIZE': '1'}
    monkeypatch.setattr(options.iss, 'security_description', lambda *args: desc)
    out = options.update_registry('2025-01-01', '2026-09-30')
    assert out['contracts_recovered_from_cards'] == 1
    row = options.read_contracts(secids='ABSENT').to_dicts()[0]
    assert row['strike'] == 0.
    assert row['strike_source'] == 'description'
    assert row['underlying_asset'] == 'F'
    assert row['lot_size'] == 1.
    assert options.update_registry('2025-01-01', '2026-09-30')['contracts_recovered_from_cards'] == 0


def test_changed_card_dates_do_not_assign_reused_code(env, monkeypatch):
    monkeypatch.setattr(options, 'fetch_series', series_frame)
    monkeypatch.setattr(options, 'fetch_series_contracts', result)
    lake.write('options', pl.DataFrame({'SECID': ['ABSENT'], 'BOARDID': ['ROPD'],
                                       'date': [dt.date(2025, 2, 1)]}))
    monkeypatch.setattr(options.iss, 'make_session', Session)
    desc = {'SERIES_NAME': 'WRONG', 'STRIKE': '10', 'OPTIONTYPE': 'P',
            'FRSTTRADE': '2026-01-01', 'LSTDELDATE': '2026-03-01',
            'ASSETCODE': 'A', 'UNDERLYINGASSET': 'F', 'SHORTNAME': 'WRONGP10'}
    monkeypatch.setattr(options.iss, 'security_description', lambda *args: desc)
    with pytest.raises(ExceptionGroup):
        options.update_registry('2025-01-01', '2026-09-30')
    assert options.read_contracts(secids='ABSENT').is_empty()
    assert 'options_registry' not in lake.query('SELECT name FROM lake.load_state')['name'].to_list()


def test_series_registry_excludes_anonymous_iss_placeholder(monkeypatch):
    class SeriesResponse(Response):
        def json(self):
            frame = series_frame().with_columns(
                pl.col('start_date', 'expiration_date').cast(pl.String))
            return {'series': {'columns': frame.columns,
                               'data': [list(row) for row in frame.iter_rows()]
                               + [['', None, None, None, None, 0]]}}

    monkeypatch.setattr(Session, 'get', lambda *args, **kwargs: SeriesResponse())
    monkeypatch.setattr(options.iss, 'make_session', Session)
    assert options.fetch_series().equals(series_frame())
