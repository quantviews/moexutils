import datetime as dt

import polars as pl
import pytest

from moexutils import lake, openpositions as oi

DAY = dt.date(2025, 1, 3)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv('MOEX_LAKE_CATALOG', 'ducklake:' + (tmp_path / 'catalog.ducklake').as_posix())
    monkeypatch.setattr(lake, 'LAKE_DATA_PATH', str(tmp_path / 'lake'))


def assets(market):
    return pl.DataFrame({'market': [market], 'asset': ['A'],
                         'date_from': [DAY], 'date_till': [DAY]})


def frame(market):
    data = {'date': [DAY, DAY], 'asset': ['A', 'A'], 'is_fiz': [0, 1],
            'persons_long': [1, 2], 'persons_short': [2, 3],
            'open_position_long': [30, 70], 'open_position_short': [50, 50],
            'oichange_long': [-1, 1], 'oichange_short': [0, 0]}
    if market == 'options':
        data.update({col: [value, value] for col, value in
                     zip(oi.OPTION_DIMS, ['F', 'C', 'M', 'A', 'D'])})
    return pl.DataFrame(data)


def test_backfill_resume_and_read_filters(env, monkeypatch):
    monkeypatch.setattr(oi, 'fetch_assets', assets)
    calls = []

    def fetch(market, asset, start, end):
        calls.append(market)
        return frame(market)

    monkeypatch.setattr(oi, 'fetch_positions', fetch)
    assert oi.update(DAY, DAY)['rows_written'] == 4
    assert oi.read(is_fiz=1)['open_position_long'].to_list() == [70]
    assert oi.read('options', assets='A').height == 2
    assert oi.read(assets=[]).is_empty()
    assert oi.update(DAY, DAY)['assets_processed'] == 0
    assert sorted(calls) == ['forts', 'options']
    assert oi.update(DAY, DAY, force=True)['rows_written'] == 0


def test_failure_does_not_checkpoint_failed_asset(env, monkeypatch):
    monkeypatch.setattr(oi, 'fetch_assets', assets)

    def fetch(market, *args):
        if market == 'options':
            raise ConnectionError('ISS failed')
        return frame(market)

    monkeypatch.setattr(oi, 'fetch_positions', fetch)
    with pytest.raises(ExceptionGroup):
        oi.update(DAY, DAY)
    state = lake.query('SELECT name FROM lake.load_state')['name'].to_list()
    assert state == ['futures_open_positions:A']
    monkeypatch.setattr(oi, 'fetch_positions', lambda market, *args: frame(market))
    assert oi.update(DAY, DAY)['assets_processed'] == 1


class Session:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def get(self, *args, **kwargs):
        class Response:
            def raise_for_status(self):
                pass

            def json(self):
                return Session.payload
        return Response()


def test_source_dimensions_and_raw_group_totals(monkeypatch):
    monkeypatch.setattr(oi.iss, 'make_session', Session)
    source = frame('options').rename({'date': 'tradedate'}).with_columns(
        pl.col('tradedate').cast(pl.String))
    Session.payload = {'open_positions': {'columns': source.columns,
                                         'data': [list(row) for row in source.iter_rows()]}}
    result = oi.fetch_positions('options', 'A', DAY, DAY)
    assert result['is_fiz'].to_list() == [0, 1]
    assert result['date'].dtype == pl.Date
    Session.payload['open_positions']['data'][0][source.columns.index('open_position_long')] = 31
    assert oi.fetch_positions('options', 'A', DAY, DAY)['open_position_long'][0] == 31
    Session.payload['open_positions']['data'] = Session.payload['open_positions']['data'][:1]
    assert oi.fetch_positions('options', 'A', DAY, DAY).height == 1


def test_nightly_hook(monkeypatch):
    import update_data
    calls = []
    monkeypatch.setattr(update_data, '_lake_tables', lambda warnings: ['open_position_assets'])
    monkeypatch.setattr(update_data, '_update_dataset', lambda *args: None)
    monkeypatch.setattr(update_data.openpositions, 'update', lambda: calls.append('oi'))
    assert update_data.main(do_update=False, do_indexes=False, do_bonds=False, do_key_rate=False,
                            do_futures=False, do_markets=True, do_rates=False,
                            do_adj_close=False, do_market_cap=False, do_derived=False,
                            do_check=False, do_maintenance=False, do_backup=False) == 0
    assert calls == ['oi']


def test_refresh_removes_only_unpublished_groups_in_requested_scope(env, monkeypatch):
    monkeypatch.setattr(oi, 'fetch_assets', assets)
    monkeypatch.setattr(oi, 'fetch_positions', lambda market, *args: frame(market))
    oi.update(DAY, DAY)
    older = frame('forts').with_columns(pl.lit(DAY-dt.timedelta(days=1)).alias('date'))
    lake.write('futures_open_positions', older)
    monkeypatch.setattr(oi, 'fetch_positions', lambda market, *args: frame(market).filter(pl.col('is_fiz')==1))
    assert oi.update(DAY, DAY, force=True)['rows_deleted'] == 2
    assert oi.read('forts', start=DAY).height == 1
    assert oi.read('forts', end=DAY-dt.timedelta(days=1)).height == 2


def test_unexpected_empty_refresh_preserves_saved_data_and_checkpoint(env, monkeypatch):
    monkeypatch.setattr(oi, 'fetch_assets', assets)
    monkeypatch.setattr(oi, 'fetch_positions', lambda market, *args: frame(market))
    oi.update(DAY, DAY)
    before = lake.query('SELECT * FROM lake.load_state').sort('name')
    monkeypatch.setattr(oi, 'fetch_positions', lambda *args: pl.DataFrame())
    with pytest.raises(ExceptionGroup):
        oi.update(DAY, DAY, force=True)
    assert oi.read().height == 2
    assert lake.query('SELECT * FROM lake.load_state').sort('name').equals(before)
