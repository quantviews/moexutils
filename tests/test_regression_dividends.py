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

