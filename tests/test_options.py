"""Options coverage, zero-trade rows and resumable parallel backfills."""
import datetime as dt

import polars as pl
import pytest

from moexutils import history, lake
from scripts import backfill_options
import update_data


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv('MOEX_LAKE_CATALOG', 'ducklake:' + (tmp_path / 'catalog.ducklake').as_posix())
    monkeypatch.setattr(lake, 'LAKE_DATA_PATH', str(tmp_path / 'lake'))


def frame(day):
    return pl.DataFrame({'date': [day, day], 'SECID': ['A', 'A'], 'BOARDID': ['ROPD', 'TEST'],
                         'CLOSE': [0., 0.], 'NUMTRADES': [0., 0.], 'SETTLEPRICE': [100., 101.]})


def test_options_keep_zero_trade_rows_on_all_boards(monkeypatch):
    day = dt.date(2025, 1, 3)
    monkeypatch.setattr(history.iss, 'history_day', lambda *args: frame(day))
    assert history._fetch('options', day, None).equals(frame(day))
    assert 'options' in lake.PARTITIONED_BY_YEAR


def test_backfill_skips_saved_and_confirmed_empty_dates(env, monkeypatch):
    days = [dt.date(2025, 1, i) for i in range(1, 5)]
    lake.write('options', frame(days[0]))
    lake.write('empty_dates', pl.DataFrame({'dataset': ['options'], 'date': [days[3]]}))
    calls = []

    def download(day):
        calls.append(day)
        return frame(day)

    monkeypatch.setattr(backfill_options, 'download', download)
    result = backfill_options.backfill(days[0], days[-1], workers=2)
    assert result == {'dates_processed': 2, 'rows_written': 4}
    assert set(calls) == set(days[1:3])
    assert history.read('options').height == 6
    assert backfill_options.backfill(days[0], days[-1]) == {'dates_processed': 0, 'rows_written': 0}


def test_backfill_preserves_success_and_retries_failed_date(env, monkeypatch):
    days = [dt.date(2025, 1, i) for i in range(1, 4)]

    def download(day):
        if day == days[1]:
            raise ConnectionError('ISS down')
        return frame(day)

    monkeypatch.setattr(backfill_options, 'download', download)
    with pytest.raises(ExceptionGroup):
        backfill_options.backfill(days[0], days[-1], workers=1)
    assert history.dataset_dates('options') == [days[0], days[2]]
    monkeypatch.setattr(backfill_options, 'download', frame)
    assert backfill_options.backfill(days[0], days[-1])['rows_written'] == 2
    assert history.dataset_dates('options') == days


def test_wrong_date_is_not_saved(env, monkeypatch):
    day = dt.date(2025, 1, 3)
    monkeypatch.setattr(backfill_options, 'download', lambda requested: frame(day + dt.timedelta(days=1)))
    with pytest.raises(ExceptionGroup):
        backfill_options.backfill(day, day)
    assert history.dataset_dates('options') == []
    assert history.empty_dates('options') == set()


def test_empty_weekend_is_remembered(env, monkeypatch):
    day = dt.date(2025, 1, 4)
    monkeypatch.setattr(backfill_options, 'download', lambda requested: pl.DataFrame())
    assert backfill_options.backfill(day, day)['rows_written'] == 0
    assert history.empty_dates('options') == {day}


def test_options_included_in_nightly_other_markets(monkeypatch):
    called = []
    monkeypatch.setattr(update_data, '_update_dataset', lambda dataset, title, warnings: called.append(dataset))
    monkeypatch.setattr(update_data, '_lake_tables', lambda warnings: [])
    assert update_data.main(do_update=False, do_indexes=False, do_bonds=False, do_key_rate=False,
                            do_futures=False, do_markets=True, do_rates=False,
                            do_adj_close=False, do_market_cap=False, do_derived=False,
                            do_check=False, do_maintenance=False, do_backup=False) == 0
    assert 'options' in called
