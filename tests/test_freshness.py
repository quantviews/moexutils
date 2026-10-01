"""Freshness policies and successful cashflow load markers."""
import datetime as dt

import polars as pl
import pytest

from moexutils import cashflows, history, indices, lake, quality, rates, refdata

TODAY = dt.date(2026, 10, 1)
CAL = [dt.date(2026, 9, d) for d in (25, 28, 29, 30)]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv('MOEX_LAKE_CATALOG', 'ducklake:' + (tmp_path / 'catalog.ducklake').as_posix())
    monkeypatch.setattr(lake, 'LAKE_DATA_PATH', str(tmp_path / 'lake'))
    monkeypatch.setattr(quality, 'trading_calendar', lambda: CAL)
    return tmp_path


def state(names, days):
    lake.write('load_state', pl.DataFrame({'name': names, 'date': days}))


def checks(today=TODAY):
    return {(c, o) for c, o, _ in quality.freshness_report(today).iter_rows()}


def test_uninitialized_optional_datasets_are_skipped(env):
    assert quality.freshness_report(TODAY).is_empty()


def test_options_registry_requires_successful_checkpoint(env):
    lake.write('options_series', pl.DataFrame({'name': ['S']}))
    assert checks() == {('options_registry_stale', 'options_registry')}
    state(['options_registry'], [TODAY])
    assert checks() == set()


def test_open_positions_checkpoints_and_retired_assets(env):
    lake.write('open_position_assets', pl.DataFrame({'market': ['forts', 'options'],
                'asset': ['ACTIVE', 'RETIRED'], 'date_from': [CAL[0], CAL[0]],
                'date_till': [CAL[-1], CAL[0]]}))
    state(['futures_open_positions:ACTIVE', 'options_open_positions:RETIRED'], [CAL[-2], CAL[0]])
    assert checks() == {('open_positions_stale', 'forts/ACTIVE')}
    state(['futures_open_positions:ACTIVE'], [CAL[-1]])
    assert checks() == set()


def test_ruonia_publication_grace_and_stale(env):
    lake.write('ruonia', pl.DataFrame({'date': [CAL[-2]], 'rate': [14.]}))
    assert checks() == set()
    lake.write('ruonia', pl.DataFrame({'date': [CAL[-3]], 'rate': [14.]}),
               delete=pl.DataFrame({'date': [CAL[-2]]}))
    assert checks() == {('ruonia_stale', 'RUONIA')}


def test_ruonia_weekend_and_working_saturday_grace(env, monkeypatch):
    cal = [dt.date(2025, 10, d) for d in (29, 30, 31)] + [dt.date(2025, 11, 1)]
    monkeypatch.setattr(quality, 'trading_calendar', lambda: cal)
    lake.write('ruonia', pl.DataFrame({'date': [dt.date(2025, 10, 30)], 'rate': [14.]}))
    assert checks(dt.date(2025, 11, 3)) == set()


def test_zcyc_checks_all_tables_and_ignores_current_day(env, monkeypatch):
    monkeypatch.setattr(quality, 'trading_calendar', lambda: CAL + [TODAY])
    for table in rates.ZCYC_TABLES.values():
        key_cols = {'period': [1.]} if table == 'zcyc_yields' else ({'secid': ['OFZ']} if table == 'zcyc_bonds' else {})
        lake.write(table, pl.DataFrame({'date': [CAL[-1]], **key_cols}))
    assert checks() == set()
    lake.write('zcyc_bonds', pl.DataFrame(), delete=pl.DataFrame({'date': [CAL[-1]], 'secid': ['OFZ']}))
    assert checks() == {('zcyc_stale', 'zcyc_bonds')}


def test_zcyc_missing_tables_and_confirmed_empty_day(env):
    lake.write('zcyc_params', pl.DataFrame({'date': [CAL[-2]]}))
    assert ('zcyc_stale', 'zcyc_params') in checks()
    lake.write('empty_dates', pl.DataFrame({'dataset': ['zcyc'], 'date': [CAL[-1]]}))
    assert checks() == {('zcyc_stale', 'zcyc_yields'), ('zcyc_stale', 'zcyc_bonds')}


def test_refdata_no_changes_does_not_mean_stale(env):
    lake.write(refdata.TABLE, pl.DataFrame({'date': [CAL[0]], 'secid': ['A'], 'issuesize': [100.]}))
    state([refdata.TABLE], [CAL[-1]])
    assert checks() == set()
    state([refdata.TABLE], [CAL[-2]])
    assert checks() == {('refdata_stale', refdata.TABLE)}


def test_refdata_ignores_working_saturday_not_polled_by_loader(env, monkeypatch):
    friday, saturday = dt.date(2025, 10, 31), dt.date(2025, 11, 1)
    monkeypatch.setattr(quality, 'trading_calendar', lambda: [friday, saturday])
    state([refdata.TABLE], [friday])
    assert checks(dt.date(2025, 11, 2)) == set()


def test_weights_checked_per_core_index_by_progress_even_when_empty(env):
    names = [indices.TABLE + ':' + idx for idx in indices.CORE_INDEXES]
    state(names, [CAL[-1]] * len(names))
    assert checks() == set()
    state([indices.TABLE + ':IMOEX'], [CAL[-2]])
    assert checks() == {('index_weights_stale', 'IMOEX')}


def test_weights_reports_missing_core_index_progress(env):
    state([indices.TABLE + ':IMOEX'], [CAL[-1]])
    assert checks() == {('index_weights_stale', idx) for idx in indices.CORE_INDEXES if idx != 'IMOEX'}


def cashflow_state(daily, weekly):
    tables = [b.table for b in cashflows.BLOCKS.values()]
    state([cashflows.UPDATE_STATE_PREFIX + t for t in tables] +
          [cashflows.FUTURE_STATE_PREFIX + t for t in tables],
          [daily] * len(tables) + [weekly] * len(tables))


def test_cashflow_dates_in_future_do_not_hide_missing_updates(env):
    lake.write('bond_coupons', pl.DataFrame({'secid': ['A'], 'coupondate': [dt.date(2030, 1, 1)]}))
    got = checks()
    assert ('cashflows_stale', 'bond_coupons') in got
    assert ('cashflows_future_stale', 'bond_coupons') in got


def test_cashflow_daily_and_weekly_policies(env):
    cashflow_state(TODAY, dt.date(2026, 9, 26))
    assert checks() == set()
    state([cashflows.UPDATE_STATE_PREFIX + 'bond_offers'], [CAL[-1]])
    assert checks() == {('cashflows_stale', 'bond_offers')}
    state([cashflows.FUTURE_STATE_PREFIX + 'bond_coupons'], [dt.date(2026, 9, 19)])
    assert ('cashflows_future_stale', 'bond_coupons') in checks()


@pytest.mark.parametrize('today', [dt.date(2026, 9, 27), dt.date(2026, 9, 28)])
def test_cashflows_no_scheduled_run_sunday_or_monday(env, today):
    cashflow_state(dt.date(2026, 9, 26), dt.date(2026, 9, 26))
    assert checks(today) == set()


def test_cashflows_saturday_requires_new_weekly_update(env):
    today = dt.date(2026, 10, 3)
    cashflow_state(today, dt.date(2026, 9, 26))
    assert checks(today) == {('cashflows_future_stale', b.table) for b in cashflows.BLOCKS.values()}


@pytest.mark.parametrize('mode', ['window', 'future', 'full'])
def test_successful_empty_cashflow_fetch_records_progress(env, monkeypatch, mode):
    monkeypatch.setattr(cashflows, 'fetch_block', lambda *args: pl.DataFrame())
    assert cashflows.update_cashflows(mode) == {b.table: 0 for b in cashflows.BLOCKS.values()}
    got = dict(lake.query('SELECT name, date FROM lake.load_state').iter_rows())
    expected = {cashflows.UPDATE_STATE_PREFIX + b.table: dt.date.today() for b in cashflows.BLOCKS.values()}
    if mode != 'window':
        expected.update({cashflows.FUTURE_STATE_PREFIX + b.table: dt.date.today() for b in cashflows.BLOCKS.values()})
    assert got == expected


def test_cashflow_partial_failure_only_marks_successful_block(env, monkeypatch):
    def fetch(block, *args):
        if block == 'amortizations':
            raise ConnectionError('ISS down')
        return pl.DataFrame()

    monkeypatch.setattr(cashflows, 'fetch_block', fetch)
    with pytest.raises(ConnectionError):
        cashflows.update_cashflows('full')
    got = dict(lake.query('SELECT name, date FROM lake.load_state').iter_rows())
    assert got == {cashflows.UPDATE_STATE_PREFIX + 'bond_coupons': dt.date.today(),
                   cashflows.FUTURE_STATE_PREFIX + 'bond_coupons': dt.date.today()}


def test_freshness_issues_integrated_without_calendar(env, monkeypatch):
    monkeypatch.setattr(quality, 'trading_calendar', lambda: [])
    lake.write('bond_coupons', pl.DataFrame({'secid': ['A'], 'coupondate': [dt.date(2030, 1, 1)]}))
    report = quality.data_quality_report()
    assert 'calendar' in report['check'].to_list()
    assert 'cashflows_stale' in report['check'].to_list()
    assert 'cashflows_future_stale' in quality.quality_summary(report)


def test_freshness_checks_are_offline(env, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError('network request')

    monkeypatch.setattr(history.iss, 'make_session', fail)
    lake.write('ruonia', pl.DataFrame({'date': [CAL[-2]], 'rate': [14.]}))
    assert checks() == set()
