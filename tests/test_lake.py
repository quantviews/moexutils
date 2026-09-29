"""
Тесты хранилища lake.py. Каталог — файловый DuckLake во временной папке
(MOEX_LAKE_CATALOG), Postgres не нужен.
"""
import datetime as dt
import os

import polars as pl
import pytest

import lake


@pytest.fixture
def lake_env(tmp_path, monkeypatch):
    monkeypatch.setenv('MOEX_LAKE_CATALOG', 'ducklake:' + str(tmp_path / 'catalog.ducklake').replace('\\', '/'))
    monkeypatch.setattr(lake, 'LAKE_DATA_PATH', str(tmp_path / 'data'))
    return tmp_path


def frame(dates, secids, close, boards=None):
    return pl.DataFrame({
        'date': [dt.date.fromisoformat(d) for d in dates],
        'SECID': secids,
        'BOARDID': boards or ['TQCB'] * len(secids),
        'CLOSE': close,
    })


class TestPgpass:
    def write(self, tmp_path, monkeypatch, text):
        path = tmp_path / 'pgpass.conf'
        path.write_text(text, encoding='utf-8')
        monkeypatch.setenv('PGPASSFILE', str(path))

    def test_first_matching_line_and_wildcards(self, tmp_path, monkeypatch):
        self.write(tmp_path, monkeypatch,
                   '# comment\nlocalhost:5432:*:postgres:superpw\nlocalhost:5432:moex_lake:moex:moexpw\n'
                   '*:*:*:moex:fallback\n')
        assert lake.pg_password('localhost', 5432, 'moex_lake', 'moex') == 'moexpw'
        assert lake.pg_password('localhost', 5432, 'other', 'postgres') == 'superpw'
        assert lake.pg_password('remote', 5433, 'x', 'moex') == 'fallback'

    def test_escaped_colon_and_backslash(self, tmp_path, monkeypatch):
        self.write(tmp_path, monkeypatch, 'localhost:5432:moex_lake:moex:a\\:b\\\\c\n')
        assert lake.pg_password('localhost', 5432, 'moex_lake', 'moex') == 'a:b\\c'

    def test_missing_entry_raises(self, tmp_path, monkeypatch):
        self.write(tmp_path, monkeypatch, 'localhost:5432:*:postgres:pw\n')
        with pytest.raises(lake.LakeConfigError):
            lake.pg_password('localhost', 5432, 'moex_lake', 'moex')


class TestWrite:
    def test_create_merge_and_read(self, lake_env):
        assert lake.write('bonds', frame(['2025-06-02', '2025-06-02'], ['B1', 'B2'], [100.0, 99.0])) == 2
        # перекачанная дата обновляет строку, новая — добавляется
        lake.write('bonds', frame(['2025-06-02', '2025-06-03'], ['B1', 'B1'], [101.0, 102.0]))
        df = lake.query("SELECT * FROM lake.bonds ORDER BY date, SECID")
        assert isinstance(df, pl.DataFrame)
        assert df.height == 3
        assert df.filter((pl.col('SECID') == 'B1') & (pl.col('date') == dt.date(2025, 6, 2)))['CLOSE'][0] == 101.0
        assert 'bonds' in lake.tables()

    def test_same_secid_on_two_boards_is_two_rows(self, lake_env):
        lake.write('bonds', frame(['2025-06-02', '2025-06-02'], ['B1', 'B1'], [100.0, 99.0],
                                  boards=['TQCB', 'PSOB']))
        assert lake.query("SELECT count(*) AS n FROM lake.bonds")['n'][0] == 2

    def test_new_columns_are_added(self, lake_env):
        lake.write('bonds', frame(['2025-06-02'], ['B1'], [100.0]))
        wider = frame(['2025-06-03'], ['B1'], [101.0]).with_columns(pl.lit(55.5).alias('ZSPREAD'))
        lake.write('bonds', wider)
        df = lake.query("SELECT date, ZSPREAD FROM lake.bonds ORDER BY date")
        assert df['ZSPREAD'].to_list() == [None, 55.5]

    def test_duplicate_keys_rejected(self, lake_env):
        with pytest.raises(ValueError, match='повторяющимся ключом'):
            lake.write('bonds', frame(['2025-06-02', '2025-06-02'], ['B1', 'B1'], [1.0, 2.0]))

    def test_missing_key_column_rejected(self, lake_env):
        with pytest.raises(ValueError, match='ключевых колонок'):
            lake.write('bonds', pl.DataFrame({'SECID': ['B1'], 'CLOSE': [1.0]}))

    def test_big_tables_partitioned_by_year(self, lake_env):
        lake.write('futures', frame(['2024-12-30', '2025-01-03'], ['SiH5', 'SiH5'], [1.0, 2.0],
                                    boards=['RFUD', 'RFUD']))
        with lake.session(read_only=True) as con:
            parts = con.execute("SELECT count(*) FROM ducklake_list_files('lake', 'futures')").fetchone()[0]
        assert parts == 2  # по файлу на год


def test_maintenance_runs(lake_env):
    lake.init()
    for i in range(3):
        lake.write('bonds', frame([f'2025-06-0{i + 2}'], ['B1'], [100.0 + i]))
    lake.maintenance(retention_days=0)
    assert lake.query("SELECT count(*) AS n FROM lake.bonds")['n'][0] == 3
