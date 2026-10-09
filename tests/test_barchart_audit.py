import datetime as dt

import polars as pl
import pytest

from moexutils.barchart_audit import audit


def inputs():
    b = pl.DataFrame({'symbol': ['OJH26', 'OJH26'], 'asset': ['ORANGE', 'ORANGE'],
        'date': [dt.date(2026, 2, 27), dt.date(2026, 3, 2)],
        **{c: [175.7, 999.] for c in ['open', 'high', 'low', 'close']}, 'volume': [0., 5.]})
    h = pl.DataFrame({'SECID': ['OJH6'], 'BOARDID': ['RFUD'], 'date': [dt.date(2026, 3, 2)], 'SETTLEPRICE': [1.757]})
    m = {'contracts': [{'symbol': 'OJH26', 'asset': 'ORANGE', 'moex_secid': 'OJH6', 'moex_expiration': '2026-03-02'}]}
    return b, h, m


def test_expiry_uses_previous_session_not_same_day():
    b, h, m = inputs()
    anchors, coverage, missing = audit(b, h, m)
    assert anchors[0]['status'] == 'matched'
    assert anchors[0]['source_date'] == dt.date(2026, 2, 27)
    assert coverage[0]['zero_volume'] == 1
    assert missing == []


def test_mismatch_and_stale_are_not_matches():
    b, h, m = inputs()
    h = h.with_columns(pl.lit(2.).alias('SETTLEPRICE'))
    assert audit(b, h, m)[0][0]['status'] == 'mismatch'
    b = b.with_columns((pl.col('date') - pl.duration(days=30)).alias('date'))
    assert audit(b, h, m)[0][0]['status'] == 'missing_or_stale'


def test_duplicate_history_rejected():
    b, h, m = inputs()
    with pytest.raises(ValueError, match='Duplicate'):
        audit(b, pl.concat([h, h]), m)
