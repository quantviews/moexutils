"""Regression tests for schema."""
import datetime as dt

import polars as pl
import pytest

from moexutils import lake, refdata


@pytest.fixture
def lake_env(tmp_path, monkeypatch):
    monkeypatch.setenv('MOEX_LAKE_CATALOG', 'ducklake:' + (tmp_path / 'catalog.ducklake').as_posix())
    monkeypatch.setattr(lake, 'LAKE_DATA_PATH', str(tmp_path / 'data'))



def test_new_column_persisted_by_sync(lake_env):
    old = pl.DataFrame({'ticker': ['A', 'B'], 'sector': ['x', 'y']})
    lake.sync('ref_sectors', old)
    new = old.with_columns(pl.Series('new_field', [7., None]))
    assert lake.sync('ref_sectors', new) == (1, 0)
    assert lake.query('SELECT new_field FROM lake.ref_sectors ORDER BY ticker')['new_field'].to_list() == [7., None]
    assert lake.sync('ref_sectors', new) == (0, 0)



def test_refdata_new_parameter_is_a_change():
    state = pl.DataFrame({'secid': ['A', 'B'], 'date': [dt.date(2026, 9, 29)] * 2, 'issuesize': [100., 200.]})
    snap = state.with_columns(pl.lit(dt.date(2026, 9, 30)).alias('date'), pl.Series('new_parameter', [7., None]))
    delta = refdata._changes(state, snap)
    assert delta['secid'].to_list() == ['A']
    assert delta['new_parameter'].to_list() == [7.]

