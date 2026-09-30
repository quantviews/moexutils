"""
Тесты эксплуатации: копия каталога (backup.py, pg_dump подменен), история
прогонов и оповещения (quality.record_run, update_data.main). Каталог хранилища —
файловый DuckLake во временной папке, Postgres не нужен.
"""
import datetime as dt
import os
import subprocess

import polars as pl
import pytest

from moexutils import backup
from moexutils import lake
from moexutils import quality
import update_data


@pytest.fixture
def lake_env(tmp_path, monkeypatch):
    monkeypatch.setenv('MOEX_LAKE_CATALOG', 'ducklake:' + str(tmp_path / 'catalog.ducklake').replace('\\', '/'))
    monkeypatch.setattr(lake, 'LAKE_DATA_PATH', str(tmp_path / 'data'))
    return tmp_path


def issues_frame(rows):
    return pl.DataFrame(rows, schema=quality.ISSUE_SCHEMA, orient='row')


class TestBackup:
    @pytest.fixture
    def fake_pg(self, monkeypatch):
        calls = []
        state = {'dump_rc': 0, 'tables': 3}

        def run(cmd, **kw):
            calls.append(cmd)
            if os.path.basename(cmd[0]).startswith('pg_dump'):
                if state['dump_rc'] == 0:
                    with open(cmd[cmd.index('-f') + 1], 'wb') as f:
                        f.write(b'PGDMP')
                return subprocess.CompletedProcess(cmd, state['dump_rc'], '', 'connection refused')
            listing = "".join(f"1; 0 1 TABLE DATA public t{i} moex\n" for i in range(state['tables']))
            return subprocess.CompletedProcess(cmd, 0, listing, '')

        monkeypatch.setattr(backup, 'pg_tool', lambda name: name)
        monkeypatch.setattr(backup.subprocess, 'run', run)
        return calls, state

    def test_dump_verified_and_old_copies_pruned(self, tmp_path, fake_pg):
        calls, _ = fake_pg
        old = [tmp_path / f'moex_lake-2026010{i}-000000.dump' for i in range(1, 4)]
        for p in old:
            p.write_bytes(b'x')
        path = backup.backup_catalog(str(tmp_path), keep=2)
        assert os.path.exists(path) and not os.path.exists(path + '.tmp')
        # осталась новая копия и самая свежая из старых
        assert backup.list_backups(str(tmp_path)) == [str(old[2]), path]
        dump = calls[0]
        assert '-w' in dump and '-Fc' in dump and lake.PG_DATABASE in dump
        assert not any('password' in str(a).lower() for a in dump)     # пароль — только из pgpass

    def test_failed_dump_raises_and_leaves_nothing(self, tmp_path, fake_pg):
        _, state = fake_pg
        state['dump_rc'] = 1
        with pytest.raises(backup.BackupError, match='connection refused'):
            backup.backup_catalog(str(tmp_path))
        assert os.listdir(tmp_path) == []

    def test_empty_dump_rejected(self, tmp_path, fake_pg):
        _, state = fake_pg
        state['tables'] = 0
        with pytest.raises(backup.BackupError, match='нет данных'):
            backup.backup_catalog(str(tmp_path))
        assert backup.list_backups(str(tmp_path)) == []


class TestRunHistory:
    def test_new_issues_against_previous_run_of_same_mode(self, lake_env):
        t1, t2, t3 = (dt.datetime(2026, 9, 30, 0, 30) + dt.timedelta(days=i) for i in range(3))
        assert quality.previous_issues('update', t1) is None             # истории еще нет
        first = issues_frame([('dividend_gap', 'MSNG', '2026-07-14: гэп'),
                              ('stock_stale', 'ABRD', 'последняя дата 2026-09-01')])
        quality.record_run(t1, 'update', [], first, first.height)
        # в режиме check — свои замечания, на сравнение update не влияют
        quality.record_run(t1 + dt.timedelta(hours=1), 'check', [],
                           issues_frame([('adj_jump', 'SBER', 'x')]), 1)

        second = issues_frame([('dividend_gap', 'MSNG', '2026-07-14: гэп'),
                               ('stock_stale', 'ABRD', 'последняя дата 2026-09-02'),  # detail сменился
                               ('index_stale', 'IMOEX', 'нет данных за 2 дня')])
        fresh = quality.new_issues(second, quality.previous_issues('update', t2))
        assert fresh.rows() == [('index_stale', 'IMOEX', 'нет данных за 2 дня')]
        quality.record_run(t2, 'update', ['Акции: не удалось обновить — timeout'], second, fresh.height)

        runs = lake.query('SELECT * FROM lake.update_runs ORDER BY run_id')
        assert runs['status'].to_list() == ['issues', 'issues', 'error']
        assert runs['new_issues'].to_list() == [2, 1, 1]
        assert runs['messages'][2] == 'Акции: не удалось обновить — timeout'
        log = lake.query('SELECT * FROM lake.quality_log WHERE run_id = ?', [t2])
        assert log.height == 3 and set(log['mode']) == {'update'}
        # прогон без замечаний: в журнале замечаний ничего, в следующий раз новых нет
        quality.record_run(t3, 'update', [], issues_frame([]), 0)
        assert lake.query('SELECT status FROM lake.update_runs WHERE run_id = ?', [t3])['status'][0] == 'ok'
        assert quality.previous_issues('update', t3 + dt.timedelta(days=1)).is_empty()


class TestMainNotifications:
    @pytest.fixture
    def env(self, lake_env, monkeypatch):
        toasts = []
        monkeypatch.setattr(update_data.notify, 'toast', lambda title, text: toasts.append((title, text)))
        monkeypatch.setattr(update_data.backup, 'backup_catalog', lambda: 'copy.dump')
        monkeypatch.setattr(update_data.lake, 'maintenance', lambda: None)
        state = {'issues': issues_frame([])}
        monkeypatch.setattr(update_data.quality, 'data_quality_report', lambda **kw: state['issues'])
        return toasts, state

    def run(self, **kw):
        return update_data.main(do_update=False, do_indexes=False, do_bonds=False, do_key_rate=False,
                                do_futures=False, do_adj_close=False, do_derived=False,
                                do_markets=False, do_rates=False, **kw)

    def test_step_failure_gives_exit_1_and_toast(self, env, monkeypatch):
        toasts, _ = env

        def boom():
            raise OSError('disk full')
        monkeypatch.setattr(update_data.backup, 'backup_catalog', boom)
        assert self.run() == 1
        assert len(toasts) == 1 and 'сбой' in toasts[0][0] and 'disk full' in toasts[0][1]
        assert lake.query('SELECT status FROM lake.update_runs')['status'].to_list() == ['error']

    def test_toast_only_for_new_issues(self, env):
        toasts, state = env
        state['issues'] = issues_frame([('dividend_gap', 'MSNG', 'гэп')])
        assert self.run() == 0 and len(toasts) == 1 and 'MSNG' in toasts[0][1]
        assert self.run() == 0 and len(toasts) == 1                     # то же замечание — тишина
        state['issues'] = issues_frame([('dividend_gap', 'MSNG', 'гэп'), ('stock_gaps', 'SBER', '2 даты')])
        assert self.run() == 0 and len(toasts) == 2 and 'SBER' in toasts[1][1] and 'MSNG' not in toasts[1][1]

    def test_clean_run_is_silent(self, env):
        toasts, _ = env
        assert self.run() == 0 and toasts == []
        assert self.run(notify_on=False) == 0
