from datetime import date

import polars as pl

from scripts.audit_positioning_availability import differences


def test_group_disappearance_is_not_zero():
    old = pl.DataFrame({'date': [date(2026, 1, 5)] * 2, 'asset': ['ORANGE'] * 2,
                        'is_fiz': [0, 1], 'open_position_long': [0, 12]})
    result = differences(old, old.filter(pl.col('is_fiz') == 1))
    assert result['removed'] == 1
    assert result['changed'] == 0


def test_revision_is_distinct_from_new_day():
    old = pl.DataFrame({'date': [date(2026, 1, 5)], 'asset': ['Si'],
                        'is_fiz': [1], 'open_position_long': [12]})
    new = pl.concat([old.with_columns(pl.lit(13, dtype=pl.Int64).alias('open_position_long')),
                     old.with_columns(pl.lit(date(2026, 1, 6)).alias('date'))])
    result = differences(old, new)
    assert (result['changed'], result['added'], result['removed']) == (1, 1, 0)
