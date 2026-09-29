"""
Хранилище moexutils на DuckLake.

Каталог (метаданные, снимки, схема) — PostgreSQL: база `moex_lake`, роль `moex`
на localhost; данные — Parquet в `<MOEX_DATA_ROOT>/lake`. Чтение и запись идут
через DuckDB, результаты — polars DataFrame.

Пароль роли берется из файла паролей PostgreSQL (`%APPDATA%\\postgresql\\pgpass.conf`
на Windows, `~/.pgpass` в других ОС или `PGPASSFILE`) и передается во временный
безымянный секрет DuckDB: встроенная в DuckDB libpq не читает pgpass и переменные
окружения, а пароль в строке подключения попал бы в текст ошибок.

Для тестов и работы без Postgres каталог переопределяется переменной
`MOEX_LAKE_CATALOG`, например `ducklake:C:/tmp/catalog.ducklake` (файловый
каталог DuckDB — один процесс-писатель).
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Iterable, Optional

import duckdb
import polars as pl

logger = logging.getLogger("moex_utils")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("MOEX_DATA_ROOT") or BASE_DIR
LAKE_DATA_PATH = os.path.join(DATA_ROOT, "lake")

PG_HOST = os.environ.get("MOEX_PG_HOST", "localhost")
PG_PORT = int(os.environ.get("MOEX_PG_PORT", "5432"))
PG_DATABASE = os.environ.get("MOEX_PG_DATABASE", "moex_lake")
PG_USER = os.environ.get("MOEX_PG_USER", "moex")

ALIAS = "lake"
# Снимки старше этого срока удаляются при обслуживании (окно для отката)
SNAPSHOT_RETENTION_DAYS = 30

# Ключи таблиц: по ним идет дозапись MERGE и проверка уникальности
TABLE_KEYS = {
    'stocks': ['date', 'ticker'],
    'indexes': ['date', 'ticker'],
    'bonds': ['date', 'SECID', 'BOARDID'],
    'futures': ['date', 'SECID', 'BOARDID'],
    'bonds_securities': ['SECID'],
    'empty_dates': ['dataset', 'date'],
}
# Таблицы, разбитые по годам (дозапись трогает только текущий год)
PARTITIONED_BY_YEAR = ('bonds', 'futures')

_RETRYABLE = (duckdb.IOException, duckdb.TransactionException, duckdb.ConnectionException)
_extensions_installed = False


class LakeConfigError(RuntimeError):
    """Не удалось настроить подключение к хранилищу (нет пароля, каталога и т.п.)."""


def _pgpass_path() -> str:
    if os.environ.get("PGPASSFILE"):
        return os.environ["PGPASSFILE"]
    if os.name == "nt":
        return os.path.join(os.environ.get("APPDATA", ""), "postgresql", "pgpass.conf")
    return os.path.expanduser("~/.pgpass")


def _pgpass_fields(line: str) -> list[str]:
    """Поля строки pgpass с учетом экранирования `\\:` и `\\\\`."""
    fields, cur, i = [], [], 0
    while i < len(line):
        ch = line[i]
        if ch == "\\" and i + 1 < len(line):
            cur.append(line[i + 1])
            i += 2
            continue
        if ch == ":":
            fields.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
        i += 1
    fields.append("".join(cur))
    return fields


def pg_password(host: str = PG_HOST, port: int = PG_PORT, database: str = PG_DATABASE,
                user: str = PG_USER) -> str:
    """Пароль из файла паролей PostgreSQL по правилам pgpass (первая подходящая строка, `*` — любое)."""
    path = _pgpass_path()
    if not os.path.exists(path):
        raise LakeConfigError(f"Нет файла паролей PostgreSQL: {path}")
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n\r")
            if not line or line.startswith("#"):
                continue
            h, p, d, u, pw = (_pgpass_fields(line) + [""] * 5)[:5]
            if (h in ("*", host) and p in ("*", str(port))
                    and d in ("*", database) and u in ("*", user)):
                return pw
    raise LakeConfigError(f"В {path} нет пароля для {user}@{host}:{port}/{database}")


def _sql_str(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def connect(read_only: bool = False, retries: int = 5) -> duckdb.DuckDBPyConnection:
    """
    Новое подключение DuckDB с присоединенным хранилищем под именем `lake`.
    read_only=True — для чтения (ноутбуки, отчеты). При занятом каталоге
    повторяет попытку с растущей паузой.
    """
    global _extensions_installed
    catalog = os.environ.get("MOEX_LAKE_CATALOG")
    data_path = LAKE_DATA_PATH.replace("\\", "/").rstrip("/") + "/"
    last_error = None
    for attempt in range(retries):
        con = duckdb.connect()
        try:
            if not _extensions_installed:
                con.execute("INSTALL ducklake; INSTALL postgres")
                _extensions_installed = True
            con.execute("LOAD ducklake")
            if not catalog:
                con.execute("LOAD postgres")
                con.execute(
                    f"CREATE TEMPORARY SECRET (TYPE postgres, HOST {_sql_str(PG_HOST)}, "
                    f"PORT {PG_PORT}, USER {_sql_str(PG_USER)}, PASSWORD {_sql_str(pg_password())})")
            target = catalog or f"ducklake:postgres:dbname={PG_DATABASE}"
            attach = f"ATTACH {_sql_str(target)} AS {ALIAS} (DATA_PATH {_sql_str(data_path)}"
            if read_only:
                try:
                    con.execute(attach + ", READ_ONLY)")
                except duckdb.IOException as e:
                    # каталога еще нет (новое хранилище): открыть на запись один раз —
                    # DuckLake его создаст; дальше чтение идет как обычно
                    if "does not exist" not in str(e) and "not initialized" not in str(e):
                        raise
                    con.execute(attach + ")")
            else:
                con.execute(attach + ")")
            return con
        except LakeConfigError:
            con.close()
            raise
        except _RETRYABLE as e:
            con.close()
            last_error = e
            time.sleep(0.5 * (attempt + 1))
    raise last_error


@contextmanager
def session(read_only: bool = False):
    """Подключение на время блока with; закрывается при выходе."""
    con = connect(read_only=read_only)
    try:
        yield con
    finally:
        con.close()


def query(sql: str, params: Optional[list] = None) -> pl.DataFrame:
    """SQL-запрос к хранилищу только на чтение; таблицы — `lake.<имя>`."""
    with session(read_only=True) as con:
        return con.execute(sql, params or []).pl()


def tables(con: Optional[duckdb.DuckDBPyConnection] = None) -> list[str]:
    """Имена таблиц хранилища."""
    sql = f"SELECT table_name FROM information_schema.tables WHERE table_catalog = '{ALIAS}' ORDER BY 1"
    if con is not None:
        return [r[0] for r in con.execute(sql).fetchall()]
    with session(read_only=True) as c:
        return [r[0] for r in c.execute(sql).fetchall()]


def _columns(con, table: str) -> dict[str, str]:
    rows = con.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        f"WHERE table_catalog = '{ALIAS}' AND table_name = ? ORDER BY ordinal_position", [table]).fetchall()
    return {name: dtype for name, dtype in rows}


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def write(table: str, df: pl.DataFrame, key: Optional[Iterable[str]] = None,
          retries: int = 5) -> int:
    """
    Дозапись строк в таблицу одной транзакцией. Строки с тем же ключом
    обновляются (MERGE), новые — добавляются; колонки, которых в таблице еще
    нет, добавляются (у ISS набор полей со временем расширяется). Таблица
    создается при первой записи; bonds и futures — с разбиением по годам.

    Returns: число записанных строк.
    """
    if df.is_empty():
        return 0
    if key is None:
        if table in TABLE_KEYS:
            key = TABLE_KEYS[table]
        elif table.endswith('_securities'):
            key = ['SECID']
        else:
            raise ValueError(f"{table}: не задан ключ таблицы (TABLE_KEYS или параметр key)")
    key = list(key)
    missing_key = [k for k in key if k not in df.columns]
    if missing_key:
        raise ValueError(f"{table}: в данных нет ключевых колонок {missing_key}")
    dup = df.select(key).is_duplicated()
    if dup.any():
        raise ValueError(f"{table}: {int(dup.sum())} строк с повторяющимся ключом {key}")

    last_error = None
    for attempt in range(retries):
        con = connect()
        try:
            con.register("_incoming", df)
            con.execute("BEGIN")
            if table not in tables(con):
                con.execute(f"CREATE TABLE {ALIAS}.{_quote(table)} AS SELECT * FROM _incoming LIMIT 0")
                if table in PARTITIONED_BY_YEAR:
                    con.execute(f"ALTER TABLE {ALIAS}.{_quote(table)} SET PARTITIONED BY (year(date))")
            existing = _columns(con, table)
            incoming_types = {r[0]: r[1] for r in con.execute("DESCRIBE _incoming").fetchall()}
            for col, dtype in incoming_types.items():
                if col not in existing:
                    con.execute(f"ALTER TABLE {ALIAS}.{_quote(table)} ADD COLUMN {_quote(col)} {dtype}")
            cols = df.columns
            on = " AND ".join(f"t.{_quote(k)} = s.{_quote(k)}" for k in key)
            upd = ", ".join(f"{_quote(c)} = s.{_quote(c)}" for c in cols if c not in key)
            ins_cols = ", ".join(_quote(c) for c in cols)
            ins_vals = ", ".join(f"s.{_quote(c)}" for c in cols)
            matched = f"WHEN MATCHED THEN UPDATE SET {upd} " if upd else ""
            con.execute(f"MERGE INTO {ALIAS}.{_quote(table)} t USING _incoming s ON {on} "
                        f"{matched}WHEN NOT MATCHED THEN INSERT ({ins_cols}) VALUES ({ins_vals})")
            con.execute("COMMIT")
            return df.height
        except _RETRYABLE as e:
            last_error = e
            try:
                con.execute("ROLLBACK")
            except duckdb.Error:
                pass
            time.sleep(0.5 * (attempt + 1))
        finally:
            con.close()
    raise last_error


def maintenance(retention_days: int = SNAPSHOT_RETENTION_DAYS) -> None:
    """
    Обслуживание: слить мелкие файлы ежедневных дозаписей, удалить снимки
    старше retention_days и файлы, на которые больше не ссылается ни один снимок.
    """
    older = (datetime.now() - timedelta(days=retention_days)).strftime("%Y-%m-%d %H:%M:%S")
    with session() as con:
        con.execute(f"CALL ducklake_merge_adjacent_files('{ALIAS}')")
        con.execute(f"CALL ducklake_expire_snapshots('{ALIAS}', older_than => TIMESTAMP '{older}')")
        con.execute(f"CALL ducklake_cleanup_old_files('{ALIAS}', older_than => TIMESTAMP '{older}')")


def init() -> None:
    """Однократная настройка хранилища: сжатие zstd для новых файлов данных."""
    os.makedirs(LAKE_DATA_PATH, exist_ok=True)
    with session() as con:
        con.execute(f"CALL {ALIAS}.set_option('parquet_compression', 'zstd')")
