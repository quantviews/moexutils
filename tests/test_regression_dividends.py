"""Regression tests for dividends."""
import datetime as dt

import polars as pl
import pytest

from moexutils import lake, stocks


@pytest.fixture
def lake_env(tmp_path, monkeypatch):
    monkeypatch.setenv('MOEX_LAKE_CATALOG', 'ducklake:' + (tmp_path / 'catalog.ducklake').as_posix())
    monkeypatch.setattr(lake, 'LAKE_DATA_PATH', str(tmp_path / 'data'))



def test_corrupt_dividend_csv_blocks_recompute(lake_env, tmp_path):
    day = dt.date(2026, 9, 29)
    frame = pl.DataFrame({'date': [day], 'ticker': ['TEST'], **{c: [100.] for c in stocks.STOCK_COLS if c not in ('date', 'ticker')}})
    lake.write('stocks', frame)
    (tmp_path / 'TEST.csv').write_text('wrong_header\n1\n', encoding='utf-8')
    with pytest.raises(ValueError, match='TEST'):
        stocks.recompute_stocks(div_folder=str(tmp_path))
    assert lake.query('SELECT adj_close FROM lake.stocks')['adj_close'].to_list() == [100.]
    assert stocks.load_dividends('MISSING', str(tmp_path)).is_empty()



def test_supplements_yield_to_external_date_and_preserve_multiple_payments(tmp_path, monkeypatch):
    supplement = tmp_path / 'supplements.csv'
    supplement.write_text('ticker,closing_date,dividend_value\nTEST,2026-07-14,10\n', encoding='utf-8')
    monkeypatch.setattr(stocks, 'DIVIDEND_SUPPLEMENTS_FILE', str(supplement))
    assert stocks.load_dividends('TEST', str(tmp_path))['dividend_value'].to_list() == [10.]
    path = tmp_path / 'TEST.csv'
    path.write_text('closing_date,dividend_value\n2026-07-14,9.9999\n2026-07-14,2\n', encoding='utf-8')
    assert stocks.load_dividends('TEST', str(tmp_path))['dividend_value'].to_list() == [9.9999, 2.]
    path.write_text('closing_date,dividend_value\n2026-07-14,0\n', encoding='utf-8')
    assert stocks.load_dividends('TEST', str(tmp_path)).is_empty()


def test_upstream_arrival_does_not_apply_dividend_twice(tmp_path, monkeypatch):
    supplement = tmp_path / 'supplements.csv'
    supplement.write_text('ticker,closing_date,dividend_value\nTEST,2026-07-14,10\n', encoding='utf-8')
    monkeypatch.setattr(stocks, 'DIVIDEND_SUPPLEMENTS_FILE', str(supplement))
    prices = pl.DataFrame({'ticker': ['TEST'] * 3,
                          'date': [dt.date(2026, 7, d) for d in (13, 14, 15)], 'close': [100., 90., 91.]})
    splits = pl.DataFrame(schema={'ticker': pl.String, 'date': pl.Date, 'ratio': pl.Float64, 'kind': pl.String})
    before, _ = stocks.adj_close(prices, stocks.load_dividends('TEST', str(tmp_path)), splits)
    (tmp_path / 'TEST.csv').write_text('closing_date,dividend_value\n2026-07-14,10\n', encoding='utf-8')
    after, _ = stocks.adj_close(prices, stocks.load_dividends('TEST', str(tmp_path)), splits)
    assert before['adj_close'].to_list() == after['adj_close'].to_list() == [90., 90., 91.]
