import polars as pl
import pytest

from scripts.build_cny_cash_followup import select_coefficient


def test_exact_premium_series_and_integer_strike():
    frame = pl.DataFrame([
        dict(isin='CNYP040626CE11', tradedate='14.05.2026', strike='11,00',
             option_type='C', expiration_date='04.06.2026'),
        dict(isin='CNY-6.26M040626CA11', tradedate='14.05.2026', strike='11',
             option_type='C', expiration_date='04.06.2026'),
    ])
    contract = dict(series_name='CNYP040626XE', strike='11.0', option_type='C', expiry='2026-06-04')
    assert select_coefficient(frame, contract, '14.05.2026')['isin'] == 'CNYP040626CE11'
    with pytest.raises(ValueError, match='Expected one'):
        select_coefficient(pl.concat([frame, frame]), contract, '14.05.2026')
    with pytest.raises(ValueError, match='identity mismatch'):
        select_coefficient(frame, {**contract, 'expiry': '2026-06-05'}, '14.05.2026')
