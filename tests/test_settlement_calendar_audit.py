from datetime import date
import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "settlement_audit", Path(__file__).parents[1] / "scripts/audit_settlement_calendar.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def entry(day, code, value):
    return dict(tradedate=day, currencyid=code, stock_market=value)


def test_all_currency_holiday_and_working_saturday():
    rows = [entry("2024-04-27", "RUB", 1), entry("2024-05-01", "-", 0)]
    result, trace, status = audit.conditional_settlement(date(2024, 4, 26), 1, rows, {2024})
    assert result == date(2024, 4, 27)
    assert trace[0]["basis"] == "explicit_RUB_or_all_currencies"
    assert status == "conditional_weekday_baseline"
    assert audit.conditional_settlement(date(2024, 4, 30), 1, rows, {2024})[0] == date(2024, 5, 2)


def test_null_is_not_false_but_specific_known_value_resolves_it():
    rows = [entry("2023-06-12", "-", None)]
    assert audit.conditional_settlement(date(2023, 6, 9), 1, rows, {2023})[0] is None
    rows.append(entry("2023-06-12", "RUB", 0))
    assert audit.conditional_settlement(date(2023, 6, 9), 1, rows, {2023})[0] == date(2023, 6, 13)


def test_conflict_and_missing_year_stop_calculation():
    rows = [entry("2024-03-25", "-", 0), entry("2024-03-25", "RUB", 1)]
    assert audit.conditional_settlement(date(2024, 3, 22), 1, rows, {2024})[2] == "conflicting_entries"
    assert audit.conditional_settlement(date(2026, 12, 31), 1, [], {2026})[2] == "year_not_covered"


def test_t2_counts_from_delivery_trade_not_expiration():
    candidate, trace, _ = audit.conditional_settlement(date(2023, 6, 16), 2, [], {2023})
    assert candidate == date(2023, 6, 20)
    assert len(trace) == 4
    assert all(t["basis"] == "unverified_weekday_baseline" for t in trace)
