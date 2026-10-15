from datetime import date

from moexutils.specification_series import interval, third_thursday


def test_quarterly_dates_are_not_delivery_or_settlement_dates():
    assert third_thursday(2023, 9) == date(2023, 9, 21)
    assert third_thursday(2026, 3) == date(2026, 3, 19)


def test_missing_end_is_explicitly_inferred_and_does_not_overlap_successor():
    row = dict(source_start_date=date(2025, 3, 25), source_end_date=None)
    start, end = date(2023, 1, 1), date(2026, 10, 5)
    assert interval(row, date(2025, 8, 19), start, end) == (date(2025, 3, 25), date(2025, 8, 18), True)
    assert row['source_end_date'] is None
    assert interval(row, date(2025, 8, 19), date(2025, 9, 1), end) is None
    assert interval(row, None, start, end) == (date(2025, 3, 25), end, False)
