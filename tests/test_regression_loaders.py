"""Regression tests for loaders."""
import datetime as dt
from functools import partial

import polars as pl
import pytest

from moexutils import history, indices, lake, rates, refdata, stocks


@pytest.fixture
def lake_env(tmp_path, monkeypatch):
    monkeypatch.setenv('MOEX_LAKE_CATALOG', 'ducklake:' + (tmp_path / 'catalog.ducklake').as_posix())
    monkeypatch.setattr(lake, 'LAKE_DATA_PATH', str(tmp_path / 'data'))



def fail(*args, **kwargs):
    raise ConnectionError('ISS down')



@pytest.mark.parametrize('module,fetch,call', [
    (history, '_fetch', lambda: history.update('bonds', start='2026-09-28', max_days=1)),
    (history, 'iss', None),
    (rates, 'fetch_zcyc', lambda: rates.update_zcyc(start='2026-09-28', max_days=1)),
    (refdata, 'fetch_snapshot', lambda: refdata.update_refdata(start='2026-09-28', max_days=1)),
])
def test_stop_on_error_loaders_propagate(lake_env, monkeypatch, module, fetch, call):
    if call is None:
        lake.write('bonds', pl.DataFrame({'date': [dt.date(2026, 9, 28)], 'SECID': ['A'], 'BOARDID': ['TQCB']}))
        monkeypatch.setattr(history.iss, 'security_description', fail)
        call = partial(history.update_securities, 'bonds')
    else:
        monkeypatch.setattr(module, fetch, fail)
    with pytest.raises(ConnectionError, match='ISS down'):
        call()



@pytest.mark.parametrize('kind', ['stocks', 'indexes', 'weights', 'repair'])
def test_independent_loaders_report_failures(lake_env, monkeypatch, kind):
    day = dt.date(2026, 9, 28)
    if kind == 'stocks':
        monkeypatch.setattr(stocks, 'fetch_stock', fail)
        call = partial(stocks.update_stocks, ['TEST'])
    elif kind == 'indexes':
        monkeypatch.setattr(stocks, 'fetch_index', fail)
        call = partial(stocks.update_indexes, ['IMOEX'])
    elif kind == 'weights':
        monkeypatch.setattr(history, 'trading_calendar', lambda: [day])
        monkeypatch.setattr(indices, 'list_indexes', lambda session: pl.DataFrame({'indexid': ['IMOEX'], 'from': ['2001-01-03']}))
        monkeypatch.setattr(indices, 'fetch_weights', fail)
        call = partial(indices.update_index_weights, ['IMOEX'])
    else:
        lake.write('bonds', pl.DataFrame({'date': [day, day + dt.timedelta(days=2)], 'SECID': ['A', 'A'], 'BOARDID': ['TQCB', 'TQCB']}))
        monkeypatch.setattr(history, '_fetch', fail)
        call = partial(history.repair, 'bonds', calendar=[day + dt.timedelta(days=1)])
    with pytest.raises(ExceptionGroup) as exc:
        call()
    assert isinstance(exc.value.exceptions[0], ConnectionError)



def test_refdata_failure_saves_pending_changes_and_resume_date(lake_env, monkeypatch):
    first = dt.date(2026, 9, 28)

    def fetch(day, session):
        if day != first:
            raise ConnectionError('ISS down')
        return pl.DataFrame({'date': [day], 'secid': ['A'], 'issuesize': [100.]})

    monkeypatch.setattr(refdata, 'fetch_snapshot', fetch)
    with pytest.raises(ConnectionError, match='ISS down'):
        refdata.update_refdata(start=first, flush_every=20)
    assert refdata.read_refdata().select('secid', 'issuesize').rows() == [('A', 100.)]
    assert refdata._processed_until() == first



def test_index_failure_preserves_successful_instrument(lake_env, monkeypatch):
    day = dt.date(2026, 9, 28)

    def fetch(ticker, start, session):
        if ticker == 'BAD':
            raise ConnectionError('ISS down')
        return pl.DataFrame({'date': [day], 'ticker': [ticker], 'BOARDID': ['SNDX'],
                             'close': [100.], 'value_rub': [1.], 'volume': [1.]})

    monkeypatch.setattr(stocks, 'fetch_index', fetch)
    with pytest.raises(ExceptionGroup):
        stocks.update_indexes(['BAD', 'GOOD'])
    assert stocks.read_index('GOOD')['close'].to_list() == [100.]

