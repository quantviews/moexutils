"""
Тесты iss.py и history.py: ISS замокан (ответы в формате ISS с metadata),
хранилище — файловый DuckLake во временной папке.
"""
import datetime as dt

import polars as pl
import pytest

import history
import iss
import lake


@pytest.fixture
def lake_env(tmp_path, monkeypatch):
    monkeypatch.setenv('MOEX_LAKE_CATALOG', 'ducklake:' + str(tmp_path / 'catalog.ducklake').replace('\\', '/'))
    monkeypatch.setattr(lake, 'LAKE_DATA_PATH', str(tmp_path / 'data'))
    return tmp_path


COLS = ['BOARDID', 'TRADEDATE', 'SECID', 'CLOSE', 'MATDATE']
META = {'BOARDID': {'type': 'string'}, 'TRADEDATE': {'type': 'date'}, 'SECID': {'type': 'string'},
        'CLOSE': {'type': 'double'}, 'MATDATE': {'type': 'date'}}


class FakeISS:
    """ISS по датам: rows_for(date) -> список строк; страницы по page строк."""

    def __init__(self, rows_for, page=2, fail_on=()):
        self.rows_for, self.page, self.fail_on = rows_for, page, set(fail_on)
        self.days = []

    def get(self, url, params=None):
        day = params['date']
        start = int(params.get('start', 0))
        if start == 0:
            self.days.append(dt.date.fromisoformat(day))
        if dt.date.fromisoformat(day) in self.fail_on:
            raise ConnectionError('ISS down')
        rows = self.rows_for(day)
        chunk = rows[start:start + self.page]

        class _R:
            def raise_for_status(_):
                pass

            def json(_):
                return {'history': {'metadata': META, 'columns': COLS, 'data': chunk},
                        'history.cursor': {'columns': ['INDEX', 'TOTAL', 'PAGESIZE'],
                                           'data': [[start, len(rows), self.page]]}}
        return _R()


def two_bonds(day):
    return [['TQCB', day, 'B1', 100.0, '2030-01-01'], ['TQOB', day, 'B2', 99.5, None],
            ['PSOB', day, 'B1', 98.0, '2030-01-01']]


class TestIss:
    def test_to_frame_uses_metadata_types(self):
        block = {'metadata': {'A': {'type': 'int32'}, 'B': {'type': 'string'}},
                 'columns': ['A', 'B'], 'data': [[1, None], [None, 'x']]}
        df = iss.to_frame(block)
        assert df.schema == {'A': pl.Float64, 'B': pl.Utf8}
        assert df['A'].to_list() == [1.0, None]

    def test_to_frame_without_metadata_infers(self):
        df = iss.to_frame({'columns': ['A', 'B'], 'data': [[1, 'x'], [2.5, 3]]})
        assert df.schema == {'A': pl.Float64, 'B': pl.Utf8}

    def test_history_day_paginates_and_builds_date(self):
        df = iss.history_day('stock/markets/bonds', dt.date(2025, 6, 2), FakeISS(two_bonds))
        assert df.height == 3 and df.columns[0] == 'date'
        assert df['date'].dtype == pl.Date and 'TRADEDATE' not in df.columns

    def test_records_to_frame(self):
        df = iss.records_to_frame([{'SECID': 'B1', 'ISSUESIZE': '3000000', 'NAME': 'X'},
                                   {'SECID': 'B2', 'ISSUESIZE': None, 'NAME': 'Y', 'NEW': '1'}])
        assert df['ISSUESIZE'].dtype == pl.Float64 and df['NAME'].dtype == pl.Utf8
        assert df['NEW'].to_list() == [None, 1.0]


class TestHistory:
    def test_update_tail_and_idempotent(self, lake_env):
        start = (dt.date.today() - dt.timedelta(days=6)).isoformat()
        fake = FakeISS(two_bonds)
        n = history.update('bonds', start=start, session=fake)
        weekdays = [d for d in fake.days if d.weekday() < 5]
        assert n == 3 * len(weekdays) and fake.days == weekdays
        df = history.read('bonds')
        assert df.height == n and set(df['BOARDID']) == {'TQCB', 'TQOB', 'PSOB'}
        # повторный запуск — даты не запрашиваются
        fake.days.clear()
        assert history.update('bonds', start=start, session=fake) == 0 and fake.days == []

    def test_backfill_goes_backwards_and_stops_on_failure(self, lake_env):
        today = dt.date.today()
        recent = (today - dt.timedelta(days=3)).isoformat()
        history.update('bonds', start=recent, session=FakeISS(two_bonds))
        lo = history.dataset_dates('bonds')[0]
        weekdays_before = [d for d in (lo - dt.timedelta(days=i) for i in range(1, 40)) if d.weekday() < 5]
        fail_day = weekdays_before[3]
        fake = FakeISS(two_bonds, fail_on=[fail_day])
        history.update('bonds', start=(today - dt.timedelta(days=60)).isoformat(), session=fake,
                       flush_every=2)
        assert fake.days[:4] == weekdays_before[:4]            # назад от истории
        dates = history.dataset_dates('bonds')
        assert dates[0] == weekdays_before[2]                   # до сбойной даты, без дыр

    def test_repair_fills_gaps_and_remembers_empty(self, lake_env):
        def rows(day):
            return [] if day == '2025-06-04' else two_bonds(day)
        cal = [dt.date(2025, 6, d) for d in (2, 3, 4, 5, 6)]
        history.update('bonds', start='2025-06-02', max_days=1, session=FakeISS(rows))
        history.update('bonds', start='2025-06-02', session=FakeISS(rows), max_days=0)
        # сохранены 02 и 06, пропущены 03, 04, 05
        lake.write('bonds', pl.DataFrame({'date': [dt.date(2025, 6, 6)], 'SECID': ['B1'],
                                          'BOARDID': ['TQCB'], 'CLOSE': [1.0]}))
        fake = FakeISS(rows)
        n = history.repair('bonds', session=fake, calendar=cal)
        assert n == 6 and sorted(fake.days) == [dt.date(2025, 6, d) for d in (3, 4, 5)]
        assert history.empty_dates('bonds') == {dt.date(2025, 6, 4)}
        fake.days.clear()
        assert history.repair('bonds', session=fake, calendar=cal) == 0 and fake.days == []

    def test_read_filters(self, lake_env):
        history.update('bonds', start=(dt.date.today() - dt.timedelta(days=6)).isoformat(),
                       session=FakeISS(two_bonds))
        df = history.read('bonds', boards='TQCB', columns=['date', 'SECID', 'CLOSE'])
        assert set(df['SECID']) == {'B1'} and df.columns == ['date', 'SECID', 'CLOSE']
        assert history.read('bonds', secids=['B2'])['BOARDID'].unique().to_list() == ['TQOB']

    def test_read_missing_dataset(self, lake_env):
        with pytest.raises(FileNotFoundError):
            history.read('futures')

    def test_unknown_dataset(self, lake_env):
        with pytest.raises(ValueError):
            history.update('options')

    def test_securities_registry(self, lake_env, monkeypatch):
        history.update('bonds', start=(dt.date.today() - dt.timedelta(days=3)).isoformat(),
                       session=FakeISS(two_bonds))
        asked = []

        def fake_desc(secid, session=None):
            asked.append(secid)
            return {'SECID': secid, 'ISSUESIZE': '1000', 'TYPENAME': 'Биржевая облигация'}

        monkeypatch.setattr(iss, 'security_description', fake_desc)
        assert history.update_securities('bonds', max_new=1) == 1
        assert history.update_securities('bonds') == 1
        reg = history.read_securities('bonds')
        assert reg['SECID'].to_list() == ['B1', 'B2'] and reg['ISSUESIZE'].dtype == pl.Float64
        asked.clear()
        assert history.update_securities('bonds') == 0 and asked == []


def test_nightly_update_without_start_skips_backfill(lake_env):
    recent = (dt.date.today() - dt.timedelta(days=3)).isoformat()
    history.update('bonds', start=recent, session=FakeISS(two_bonds))
    fake = FakeISS(two_bonds)
    history.update('bonds', session=fake)          # как ночью: без start
    assert all(d > dt.date.fromisoformat(recent) for d in fake.days)
