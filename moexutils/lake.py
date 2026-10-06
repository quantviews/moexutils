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
from datetime import date, datetime, time as dtime, timedelta
from typing import Iterable, Optional, Union

import duckdb
import polars as pl

logger = logging.getLogger("moexutils")

# корень проекта: metadata/ и (без MOEX_DATA_ROOT) данные — на уровень выше пакета
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    'options': ['date', 'SECID', 'BOARDID'],
    'shares': ['date', 'SECID', 'BOARDID'],
    'indexes_all': ['date', 'SECID', 'BOARDID'],
    'currency': ['date', 'SECID', 'BOARDID'],
    'currency_fixings': ['date', 'SECID', 'BOARDID'],
    'bonds_securities': ['SECID'],
    'empty_dates': ['dataset', 'date'],
    'update_runs': ['run_id'],
    'quality_log': ['run_id', 'check', 'object', 'detail'],
    'ruonia': ['date'],
    'stock_refdata': ['secid', 'date'],
    'index_weights': ['date', 'indexid', 'ticker'],
    # служебная: до какой даты обработан набор, если по данным этого не видно
    'load_state': ['name'],
    'futures_contracts': ['secid'],
    'futures_parameter_observations': ['SECID', 'BOARDID', 'observed_at'],
    'futures_description_observations': ['secid', 'observed_at'],
    'futures_risk_limits': ['date', 'assetcode', 'updatetime', 'observed_at'],
    'futures_staticparams': ['date', 'row_hash', 'observed_at'],
    'futures_staticparamskeyterm': ['date', 'row_hash', 'observed_at'],
    'futures_rclimits': ['date', 'row_hash', 'observed_at'],
    'futures_specification_editions': ['asset_candidate', 'source_url', 'source_valid_from', 'reviewed_on'],
    'options_series': ['name'],
    'options_contracts': ['secid', 'series_name'],
    'open_position_assets': ['market', 'asset'],
    'futures_open_positions': ['date', 'asset', 'is_fiz'],
    'options_open_positions': ['date', 'asset', 'asset_type', 'option_type', 'margin_style', 'exec_type', 'settle_type', 'is_fiz'],
    'futures_continuous': ['date', 'asset'],
    'bond_coupons': ['secid', 'coupondate'],
    'bond_amortizations': ['secid', 'amortdate', 'data_source'],
    'bond_offers': ['secid', 'offer_date'],
    'zcyc_params': ['date'],
    'zcyc_yields': ['date', 'period'],
    'zcyc_bonds': ['date', 'secid'],
    # производная таблица: акции со склейкой переименований и поправкой на сплиты
    'stocks_adjusted': ['date', 'ticker'],
    # реестры metadata/ — копии для SQL-потребителей
    'ref_splits': ['ticker', 'date'],
    'ref_renames': ['old'],
    'ref_delisted': ['ticker'],
    'ref_key_rate': ['date'],
    'ref_sectors': ['ticker'],
}

# Представления хранилища: имя -> (SELECT, таблицы, без которых его не создать)
VIEWS = {
    'bonds_ofz': (
        "SELECT b.* FROM lake.bonds b WHERE b.SECID IN "
        "(SELECT SECID FROM lake.bonds_securities WHERE TYPE = 'ofz_bond')",
        ('bonds', 'bonds_securities')),
    'bonds_corporate': (
        "SELECT b.* FROM lake.bonds b WHERE b.SECID IN "
        "(SELECT SECID FROM lake.bonds_securities WHERE TYPE IN ('corporate_bond', 'exchange_bond'))",
        ('bonds', 'bonds_securities')),
    # фандинг вечных фьючерсов (экспирация 2100-01-01): SWAPRATE — руб., SWAPRATE_CURR — в валюте;
    # те же ставки, что на рынке ISS swaprates (доски FUSR / FUSC)
    'futures_swaprates': (
        "SELECT f.date, f.SECID, c.asset_code, c.underlying_asset, f.SWAPRATE AS swaprate_rub, "
        "f.SWAPRATE_CURR AS swaprate_curr, f.SETTLEPRICE, f.CLOSE, f.OPENPOSITION, f.VALUE "
        "FROM lake.futures f JOIN lake.futures_contracts c ON c.secid = f.SECID "
        "WHERE c.expiration_date = DATE '2100-01-01' "
        "AND (f.SWAPRATE IS NOT NULL OR f.SWAPRATE_CURR IS NOT NULL)",
        ('futures', 'futures_contracts')),
}
# Таблицы, разбитые по годам (дозапись трогает только текущий год)
PARTITIONED_BY_YEAR = ('bonds', 'futures', 'options', 'shares', 'indexes_all', 'currency')

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


def _names(kind: str, con: Optional[duckdb.DuckDBPyConnection]) -> list[str]:
    sql = (f"SELECT table_name FROM information_schema.tables "
           f"WHERE table_catalog = '{ALIAS}' AND table_type = '{kind}' ORDER BY 1")
    if con is not None:
        return [r[0] for r in con.execute(sql).fetchall()]
    with session(read_only=True) as c:
        return [r[0] for r in c.execute(sql).fetchall()]


def tables(con: Optional[duckdb.DuckDBPyConnection] = None) -> list[str]:
    """Имена таблиц хранилища (без представлений)."""
    return _names('BASE TABLE', con)


def views(con: Optional[duckdb.DuckDBPyConnection] = None) -> list[str]:
    """Имена представлений хранилища (VIEWS)."""
    return _names('VIEW', con)


AsOf = Union[None, int, date, datetime, str]


def ref(table: str, as_of: AsOf = None) -> str:
    """
    Ссылка на таблицу для SQL: `lake."stocks"`, с as_of — на момент снимка.
    as_of: номер снимка (int), момент времени (datetime) или дата (date или
    'YYYY-MM-DD' — состояние на конец этого дня). Доступны снимки не старше
    SNAPSHOT_RETENTION_DAYS дней (список — snapshots()).
    """
    name = f"{ALIAS}.{_quote(table)}"
    if as_of is None:
        return name
    if isinstance(as_of, bool):
        raise TypeError("as_of: ожидается номер снимка, дата или момент времени")
    if isinstance(as_of, int):
        return f"{name} AT (VERSION => {as_of})"
    if isinstance(as_of, str):
        as_of = datetime.fromisoformat(as_of) if len(as_of) > 10 else date.fromisoformat(as_of)
    if isinstance(as_of, date) and not isinstance(as_of, datetime):
        as_of = datetime.combine(as_of, dtime(23, 59, 59, 999999))
    return f"{name} AT (TIMESTAMP => TIMESTAMP '{as_of:%Y-%m-%d %H:%M:%S.%f}')"


def snapshots() -> pl.DataFrame:
    """Снимки хранилища: snapshot_id, snapshot_time, changes (что изменил снимок)."""
    return query(f"SELECT snapshot_id, snapshot_time, changes FROM ducklake_snapshots('{ALIAS}') "
                 "ORDER BY snapshot_id")


def _columns(con, table: str) -> dict[str, str]:
    rows = con.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        f"WHERE table_catalog = '{ALIAS}' AND table_name = ? ORDER BY ordinal_position", [table]).fetchall()
    return {name: dtype for name, dtype in rows}


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def write(table: str, df: pl.DataFrame, key: Optional[Iterable[str]] = None,
          retries: int = 5, delete: Optional[pl.DataFrame] = None) -> int:
    """
    Дозапись строк в таблицу одной транзакцией. Строки с тем же ключом
    обновляются (MERGE), новые — добавляются; колонки, которых в таблице еще
    нет, добавляются (у ISS набор полей со временем расширяется). Таблица
    создается при первой записи; таблицы PARTITIONED_BY_YEAR — с разбиением по годам.
    delete — ключи строк, которые удаляются в той же транзакции (до записи).

    Returns: число записанных строк.
    """
    has_delete = delete is not None and not delete.is_empty()
    only_delete = df.is_empty()
    if only_delete and not has_delete:
        return 0
    if key is None:
        if table in TABLE_KEYS:
            key = TABLE_KEYS[table]
        elif table.endswith('_securities'):
            key = ['SECID']
        else:
            raise ValueError(f"{table}: не задан ключ таблицы (TABLE_KEYS или параметр key)")
    key = list(key)
    if has_delete and [k for k in key if k not in delete.columns]:
        raise ValueError(f"{table}: в delete нет ключевых колонок {key}")
    missing_key = [] if only_delete else [k for k in key if k not in df.columns]
    if missing_key:
        raise ValueError(f"{table}: в данных нет ключевых колонок {missing_key}")
    dup = df.select(key).is_duplicated() if not only_delete else pl.Series([], dtype=pl.Boolean)
    if dup.any():
        raise ValueError(f"{table}: {int(dup.sum())} строк с повторяющимся ключом {key}")

    last_error = None
    for attempt in range(retries):
        con = connect()
        try:
            con.execute("BEGIN")
            if has_delete and table in tables(con):
                con.register("_delete", delete.select(key))
                cond = " AND ".join(f"t.{_quote(k)} = d.{_quote(k)}" for k in key)
                con.execute(f"DELETE FROM {ALIAS}.{_quote(table)} t USING _delete d WHERE {cond}")
            if only_delete:
                con.execute("COMMIT")
                return 0
            con.register("_incoming", df)
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


def changed_rows(old: pl.DataFrame, new: pl.DataFrame, key: list[str], rel_tol: float = 1e-9) -> pl.DataFrame:
    """Строки new, которых нет в old или которые отличаются (числа — с относительным допуском)."""
    if old.is_empty():
        return new
    cols = [c for c in new.columns if c not in key]
    missing = [c for c in cols if c not in old.columns]
    if missing:
        old = old.with_columns([pl.lit(None, dtype=new.schema[c]).alias(c) for c in missing])
    joined = new.join(old.select(key + cols).with_columns(pl.lit(True).alias('__was')),
                      on=key, how='left', suffix='__old')
    diff = pl.lit(False)
    for c in cols:
        a, b = pl.col(c), pl.col(f"{c}__old")
        if new.schema[c].is_numeric():
            differs = ((a - b).abs() > rel_tol * pl.max_horizontal(a.abs(), b.abs(), pl.lit(1.0)))
            diff = diff | differs.fill_null(False) | (a.is_null() != b.is_null())
        else:
            diff = diff | a.ne_missing(b)
    return joined.filter(diff | pl.col('__was').is_null()).select(new.columns)


def sync(table: str, df: pl.DataFrame, key: Optional[Iterable[str]] = None) -> tuple[int, int]:
    """
    Приводит таблицу к содержимому df одной транзакцией: пишет новые и
    изменившиеся строки, удаляет строки, ключей которых в df нет. Ничего не
    изменилось — снимок не создается. Returns: (записано, удалено).
    """
    key = list(key or TABLE_KEYS[table])
    old = query(f"SELECT * FROM {ref(table)}") if table in tables() else pl.DataFrame()
    changed = changed_rows(old, df, key)
    stale = old.select(key).join(df.select(key), on=key, how='anti') if not old.is_empty() else None
    if changed.is_empty() and (stale is None or stale.is_empty()):
        return 0, 0
    write(table, changed, key=key, delete=stale)
    return changed.height, 0 if stale is None else stale.height


def ensure_views(replace: bool = False) -> list[str]:
    """
    Создает недостающие представления VIEWS (replace=True — пересоздает все,
    например после изменения их определений в коде). Представление, для
    которого еще нет исходных таблиц, пропускается. Returns: созданные.
    """
    done = []
    with session() as con:
        have = set(tables(con))
        current = set(views(con))
        for name, (select, needs) in VIEWS.items():
            if not set(needs) <= have or (name in current and not replace):
                continue
            con.execute(f"CREATE OR REPLACE VIEW {ALIAS}.{_quote(name)} AS {select}")
            done.append(name)
    return done


def maintenance(retention_days: int = SNAPSHOT_RETENTION_DAYS) -> None:
    """
    Обслуживание: слить мелкие файлы ежедневных дозаписей, удалить снимки
    старше retention_days, файлы, на которые больше не ссылается ни один снимок,
    и файлы-сироты старше того же срока (запись, не зафиксированная в каталоге).
    """
    older = (datetime.now() - timedelta(days=retention_days)).strftime("%Y-%m-%d %H:%M:%S")
    with session() as con:
        con.execute(f"CALL ducklake_merge_adjacent_files('{ALIAS}')")
        con.execute(f"CALL ducklake_expire_snapshots('{ALIAS}', older_than => TIMESTAMP '{older}')")
        con.execute(f"CALL ducklake_cleanup_old_files('{ALIAS}', older_than => TIMESTAMP '{older}')")
        # файлы, которых каталог не знает (оборванная запись, отклоненная транзакция)
        con.execute(f"CALL ducklake_delete_orphaned_files('{ALIAS}', older_than => TIMESTAMP '{older}')")


def init() -> None:
    """Однократная настройка хранилища: сжатие zstd для новых файлов данных."""
    os.makedirs(LAKE_DATA_PATH, exist_ok=True)
    with session() as con:
        con.execute(f"CALL {ALIAS}.set_option('parquet_compression', 'zstd')")
