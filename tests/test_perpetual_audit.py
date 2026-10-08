import datetime as dt
from decimal import Decimal
import json

import polars as pl
import pytest

from moexutils import perpetual_audit as audit


def sample():
    return pl.DataFrame({
        'SECID': ['CNYRUBF'] * 3, 'BOARDID': ['RFUD'] * 3,
        'date': [dt.date(2026, 1, d) for d in (5, 6, 8)],
        'registry_present': [False] * 3, 'source_type': ['futures'] * 3,
        'start_date': [dt.date(2026, 1, 5)] * 3,
        'expiration_date': [dt.date(2100, 1, 1)] * 3,
        **{field: [None, 0., 1.] for field in audit.FIELDS},
    })


def candles(board='FUSR'):
    return pl.DataFrame({
        'date': [dt.date(2026, 1, d) for d in (5, 6, 7)],
        'end': ['2026-01-05 23:00:00', '2026-01-06 23:00:00', '2026-07-01 01:00:00'],
        'swap_board': [board] * 3, 'close': [0., 0., 1.],
    })


def test_null_zero_and_missing_registry_are_preserved():
    result = audit.coverage(sample()).row(0, named=True)
    assert result['rows'] == 3
    assert result['SWAPRATE_null'] == result['SWAPRATE_zero'] == 1
    assert result['SWAPRATE_first'] == dt.date(2026, 1, 6)
    assert result['registry_present'] is False


def test_gaps_are_candidates_and_do_not_extend_to_2100():
    dates = pl.DataFrame({'date': [dt.date(2026, 1, d) for d in (5, 6, 7, 8)]})
    result = audit.candidate_gaps(sample(), dates)
    assert result['date'].to_list() == [dt.date(2026, 1, 7)]
    assert result['status'][0] == 'unverified_market_calendar_candidate'


def test_comparison_distinguishes_null_zero_absence_and_timestamp_anomaly():
    result = audit.compare_funding(sample(), candles())
    assert result['status'].to_list() == [
        'null_forts_value', 'equal_numeric_only', 'candle_only_date', 'forts_only_date']
    assert result['candle_end_other_date'][2]


def test_currency_board_uses_currency_field():
    history = sample().with_columns(pl.lit(100.).alias('SWAPRATE_CURR'))
    result = audit.compare_funding(history, candles('FUSC'))
    assert result['status'][1] == 'disputed_no_replacement'
    assert result['difference'][1] == -100
    assert result['reference_field'].unique().to_list() == ['SWAPRATE_CURR']


def test_ambiguous_dates_fail_instead_of_multiplying_rows():
    with pytest.raises(ValueError, match='Ambiguous'):
        audit.compare_funding(pl.concat([sample(), sample()]), candles())
    with pytest.raises(ValueError, match='Duplicate'):
        audit.compare_funding(sample(), pl.concat([candles(), candles()]))


@pytest.mark.parametrize('kind,flag,expected', [
    ('futures_collateral', '1', 'collateral_excluded'),
    ('futures', '1', 'perpetual_card_confirmed'),
    ('futures', None, 'unresolved_candidate'),
])
def test_classification_requires_card_evidence(kind, flag, expected):
    payload = {'description': {'columns': ['name', 'value'], 'data': [
        ['TYPE', kind], ['PERPETUAL_FUTURES', flag], ['LSTTRADE', '2100-01-01']]}}
    assert audit.classify_candidate(json.dumps(payload)) == expected


def test_funding_sign_units_and_zero():
    long = audit.illustrative_cashflow('12.494', '12.757', '.00381', 'long')
    short = audit.illustrative_cashflow('12.494', '12.757', '.00381', 'short')
    assert long == {'price_revaluation': Decimal('263'), 'funding_payment': Decimal('-3.81'),
                    'total': Decimal('259.19')}
    assert short['total'] == -long['total']
    assert audit.illustrative_cashflow(12, 12, '-.01', 'long')['total'] == 10
    assert audit.illustrative_cashflow(12, 12, 0, 'short')['total'] == 0


def test_multiple_days_telescope_without_inventing_weekend_funding():
    # Hypothetical opening immediately after Friday clearing; closing after Tuesday clearing.
    for side, expected in [('long', Decimal('201.35')), ('short', Decimal('-201.35'))]:
        days = [audit.illustrative_cashflow('12.494', '12.757', '.00381', side),
                audit.illustrative_cashflow('12.757', '12.704', '.00484', side)]
        assert sum(day['total'] for day in days) == expected


def test_official_moex_worked_example_page_17():
    # https://fs.moex.com/f/17840/prezentacija-vyhod-i-fanding-v-vechnyh-fjuchersa.pdf
    # Published USD example (December 2022), not a historical CNY specification.
    first = audit.illustrative_cashflow('75.50', '75.35', '-.0144', 'short')
    intermediate = audit.illustrative_cashflow('75.35', '75.45', 0, 'short')
    final = audit.illustrative_cashflow('75.45', '75.05', '.0145', 'short')
    assert first['total'] == Decimal('135.6')
    assert intermediate['total'] == -100
    assert final['total'] == Decimal('414.5')
    assert first['total'] + intermediate['total'] + final['total'] - 1 == Decimal('449.1')


@pytest.mark.parametrize('funding', [None, 'nan', 'Infinity'])
def test_unknown_funding_is_never_zero(funding):
    with pytest.raises(ValueError):
        audit.illustrative_cashflow(12, 13, funding, 'long')
