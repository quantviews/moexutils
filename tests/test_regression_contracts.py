"""Regression tests for contracts."""
import datetime as dt

import polars as pl
import pytest

from moexutils import contracts, lake


@pytest.fixture
def lake_env(tmp_path, monkeypatch):
    monkeypatch.setenv('MOEX_LAKE_CATALOG', 'ducklake:' + (tmp_path / 'catalog.ducklake').as_posix())
    monkeypatch.setattr(lake, 'LAKE_DATA_PATH', str(tmp_path / 'data'))



def test_partial_continuous_update_preserves_other_assets(lake_env, monkeypatch):
    day = dt.date(2026, 1, 2)
    old = pl.DataFrame({'date': [day, day], 'asset': ['Si', 'Eu'], 'settle': [100., 200.]})
    lake.write('futures_continuous', old)
    new = old.filter(pl.col('asset') == 'Si').with_columns(pl.lit(110.).alias('settle'))
    monkeypatch.setattr(contracts, 'build_continuous', lambda assets: new)
    assert contracts.update_continuous(iter(['Si'])) == (1, 0)
    assert lake.query('SELECT asset, settle FROM lake.futures_continuous ORDER BY asset').rows() == [('Eu', 200.), ('Si', 110.)]
    monkeypatch.setattr(contracts, 'build_continuous', lambda assets: new.clear())
    assert contracts.update_continuous(['Si']) == (0, 1)
    assert lake.query('SELECT asset FROM lake.futures_continuous').rows() == [('Eu',)]
    assert contracts.update_continuous([]) == (0, 0)



def test_roll_fallback_uses_price_inside_expiration_window(monkeypatch):
    reg = pl.DataFrame({'secid': ['OLD', 'NEW'], 'asset_code': ['Si', 'Si'],
                        'expiration_date': [dt.date(2026, 1, 10), dt.date(2026, 3, 20)]})
    prices = [100., 100., 200.]
    fut = pl.DataFrame({'date': [dt.date(2026, 1, 2), dt.date(2026, 1, 3), dt.date(2026, 1, 3)],
                        'SECID': ['OLD', 'OLD', 'NEW'], 'OPEN': prices, 'HIGH': prices,
                        'LOW': prices, 'CLOSE': prices, 'SETTLEPRICE': prices,
                        'VOLUME': [10.] * 3, 'OPENPOSITION': [100.] * 3})
    monkeypatch.setattr(contracts, 'read_contracts', lambda assets: reg)
    monkeypatch.setattr(lake, 'query', lambda *args: fut)
    out = contracts.build_continuous(['Si'])
    assert out['SECID'].to_list() == ['OLD', 'NEW']
    assert out['adj_factor'].to_list() == [2., 1.]
    assert out['settle_adj'].to_list() == [200., 200.]

