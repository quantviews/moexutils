"""Regression tests for cashflows."""
import datetime as dt

import polars as pl
import pytest

from moexutils import cashflows, lake


@pytest.fixture
def lake_env(tmp_path, monkeypatch):
    monkeypatch.setenv('MOEX_LAKE_CATALOG', 'ducklake:' + (tmp_path / 'catalog.ducklake').as_posix())
    monkeypatch.setattr(lake, 'LAKE_DATA_PATH', str(tmp_path / 'data'))



def test_empty_cashflow_window_deletes_only_in_scope(lake_env, monkeypatch):
    day = dt.date.today()
    near, far = day + dt.timedelta(days=5), day + dt.timedelta(days=400)
    lake.write('bond_coupons', pl.DataFrame({'secid': ['A', 'A'], 'coupondate': [near, far], 'value': [50., 50.]}))
    lake.write('bond_amortizations', pl.DataFrame({'secid': ['A'], 'amortdate': [near], 'data_source': ['maturity']}))
    lake.write('bond_offers', pl.DataFrame({'secid': ['A'], 'offer_date': [near]}))
    monkeypatch.setattr(cashflows, 'fetch_block', lambda *args: pl.DataFrame())
    cashflows.update_cashflows('window')
    assert cashflows.read_cashflows('coupons')['coupondate'].to_list() == [far]
    assert cashflows.read_cashflows('amortizations').is_empty()
    assert cashflows.read_cashflows('offers').height == 1
    cashflows.update_cashflows('full')
    assert cashflows.read_cashflows('coupons').is_empty()
    assert cashflows.read_cashflows('offers').is_empty()



@pytest.mark.parametrize('payload', [
    {},
    {'coupons': {'columns': ['secid', 'coupondate'], 'data': []},
     'coupons.cursor': {'columns': ['TOTAL'], 'data': [[1]]}},
    {'coupons': {'columns': ['secid', 'coupondate'], 'data': [['A', '2026-10-02']]},
     'coupons.cursor': {'columns': ['TOTAL'], 'data': [[2]]}},
])
def test_incomplete_cashflow_response_cannot_delete_saved_rows(lake_env, monkeypatch, payload):
    saved = pl.DataFrame({'secid': ['A'], 'coupondate': [dt.date.today()]})
    lake.write('bond_coupons', saved)

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    class Session:
        def get(self, *args, **kwargs):
            return Response()

    original = cashflows.fetch_block
    monkeypatch.setattr(cashflows, 'fetch_block',
                        lambda block, start, till, session: original(block, start, till, Session(), max_pages=1))
    with pytest.raises(ValueError):
        cashflows.update_cashflows('full')
    assert cashflows.read_cashflows('coupons').select(saved.columns).equals(saved)

