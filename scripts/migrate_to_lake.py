"""
Разовый перенос данных moexutils из Parquet-файлов в DuckLake (lake.py).

Переносятся: акции (data/), кэш индексов (indexes/), весь рынок облигаций
(bonds/market_ALL), все фьючерсы (futures/history), реестр облигаций
(bonds/securities.parquet) и подтвержденно пустые даты хранилищ. Не переносятся
дубли и ранний механизм: доски market_TQOB/TQCB (построчно совпадают с ALL),
bonds/params.parquet и bonds/<SECID>.parquet.

После переноса — сверка по каждой таблице: число строк, уникальность ключа,
сумма контрольной колонки. Исходные файлы не удаляются.

Запуск: MOEX_DATA_ROOT=F:\\moex-data python scripts/migrate_to_lake.py [--replace]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lake  # noqa: E402

ROOT = lake.DATA_ROOT.replace('\\', '/')

# таблица -> (SQL выборки из исходных файлов, контрольная колонка)
SOURCES = {
    'stocks': (f"""
        SELECT CAST(date AS DATE) AS date, ticker, open, low, high, close, waprice,
               volume, value_rub, adj_close, shares, market_cap
        FROM read_parquet('{ROOT}/data/*/*.parquet', union_by_name=true)""", 'close'),
    'indexes': (f"""
        SELECT CAST(date AS DATE) AS date, ticker, BOARDID, close,
               volume_1 AS value_rub, CAST(VOLUME AS DOUBLE) AS volume
        FROM read_parquet('{ROOT}/indexes/*.parquet', union_by_name=true)""", 'close'),
    'bonds': (f"""
        SELECT CAST(date AS DATE) AS date, * EXCLUDE (date, segment)
        FROM read_parquet('{ROOT}/bonds/market_ALL/*.parquet', union_by_name=true)""", 'CLOSE'),
    'futures': (f"""
        SELECT CAST(date AS DATE) AS date, * EXCLUDE (date)
        FROM read_parquet('{ROOT}/futures/history/*.parquet', union_by_name=true)""", 'SETTLEPRICE'),
    'bonds_securities': (f"""
        SELECT * FROM read_parquet('{ROOT}/bonds/securities.parquet')""", None),
    'empty_dates': (f"""
        SELECT 'bonds' AS dataset, CAST(date AS DATE) AS date
          FROM read_csv('{ROOT}/bonds/market_ALL/_empty_dates.csv')
        UNION ALL
        SELECT 'futures', CAST(date AS DATE) FROM read_csv('{ROOT}/futures/history/_empty_dates.csv')""", None),
}


def main(replace: bool = False) -> None:
    lake.init()
    with lake.session() as con:
        existing = lake.tables(con)
        for table, (select, _) in SOURCES.items():
            if table in existing:
                if not replace:
                    print(f"{table}: уже есть в хранилище — пропуск (--replace для перезаписи)")
                    continue
                con.execute(f"DROP TABLE lake.{table}")
            key = lake.TABLE_KEYS[table]
            order = ", ".join(key)
            con.execute("BEGIN")
            con.execute(f"CREATE TABLE lake.{table} AS {select} LIMIT 0")
            if table in lake.PARTITIONED_BY_YEAR:
                con.execute(f"ALTER TABLE lake.{table} SET PARTITIONED BY (year(date))")
            con.execute(f"INSERT INTO lake.{table} {select} ORDER BY {order}")
            con.execute("COMMIT")
            print(f"{table}: перенесено")

        print("\nСверка:")
        ok = True
        for table, (select, check_col) in SOURCES.items():
            key = ", ".join(lake.TABLE_KEYS[table])
            src_n = con.execute(f"SELECT count(*) FROM ({select})").fetchone()[0]
            dst_n, dst_keys = con.execute(
                f"SELECT count(*), count(DISTINCT ({key})) FROM lake.{table}").fetchone()
            line = f"  {table}: строк {src_n} → {dst_n}, уникальных ключей {dst_keys}"
            good = src_n == dst_n == dst_keys
            if check_col:
                src_s = con.execute(f"SELECT sum({check_col}) FROM ({select})").fetchone()[0]
                dst_s = con.execute(f"SELECT sum({check_col}) FROM lake.{table}").fetchone()[0]
                line += f", сумма {check_col} {src_s:.6g} → {dst_s:.6g}"
                good = good and abs((src_s or 0) - (dst_s or 0)) <= 1e-9 * max(1.0, abs(src_s or 0))
            print(line + ("" if good else "  <-- РАСХОЖДЕНИЕ"))
            ok = ok and good
    print("\nИтог:", "все таблицы сошлись" if ok else "ЕСТЬ РАСХОЖДЕНИЯ")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main(replace='--replace' in sys.argv)
