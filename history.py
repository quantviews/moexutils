"""
История рынков MOEX «все инструменты за дату» в хранилище DuckLake (lake.py).

Один постраничный запрос ISS на торговую дату отдает все инструменты рынка:
для облигаций — все доски с 1997 года (включая погашенные выпуски), для
фьючерсов — все контракты FORTS с 2002 года. Строки пишутся в таблицы lake
транзакциями по ключу date + SECID + BOARDID.

Докачка:
- хвост — даты после последней сохраненной до сегодня;
- бэкфилл — если start раньше истории, даты от истории назад (при обрыве
  скачанное примыкает к истории, следующий запуск продолжит без дыр);
- на сбое даты прогон останавливается (дата не перескакивается);
- скачанное пишется порциями по flush_every дат — прогресс не теряется;
- repair() докачивает пропуски внутри истории по торговому календарю IMOEX,
  даты, за которые ISS подтвержденно пуст, запоминаются в lake.empty_dates.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Iterable, Optional

import polars as pl
import requests

import iss
import lake

logger = logging.getLogger("moex_utils")

# Набор данных -> (путь рынка в ISS, начало истории, подпись для логов)
DATASETS = {
    'bonds': ('stock/markets/bonds', '1997-01-01', 'облигации (весь рынок)'),
    'futures': ('futures/markets/forts', '2002-01-01', 'фьючерсы FORTS'),
}


def _check(dataset: str) -> None:
    if dataset not in DATASETS:
        raise ValueError(f"Неизвестный набор {dataset!r}; доступны: {', '.join(DATASETS)}")


def _as_date(value) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


def _weekdays(start: dt.date, end: dt.date) -> list[dt.date]:
    days = (end - start).days
    return [start + dt.timedelta(days=i) for i in range(days + 1)
            if (start + dt.timedelta(days=i)).weekday() < 5]


def dataset_dates(dataset: str) -> list[dt.date]:
    """Все сохраненные даты набора (по возрастанию); пусто, если набора нет."""
    _check(dataset)
    if dataset not in lake.tables():
        return []
    return lake.query(f"SELECT DISTINCT date FROM lake.{dataset} ORDER BY date")['date'].to_list()


def empty_dates(dataset: str) -> set[dt.date]:
    """Торговые (по IMOEX) даты, за которые ISS подтвержденно не вернул строк."""
    if 'empty_dates' not in lake.tables():
        return set()
    return set(lake.query("SELECT date FROM lake.empty_dates WHERE dataset = ?", [dataset])['date'].to_list())


def trading_calendar() -> list[dt.date]:
    """Торговый календарь: будни, по которым есть IMOEX в lake.indexes."""
    if 'indexes' not in lake.tables():
        return []
    dates = lake.query("SELECT DISTINCT date FROM lake.indexes WHERE ticker = 'IMOEX' ORDER BY date")['date']
    return [d for d in dates.to_list() if d.weekday() < 5]


def _fetch(dataset: str, date: dt.date, session: requests.Session) -> pl.DataFrame:
    return iss.history_day(DATASETS[dataset][0], date, session)


def update(dataset: str, start: Optional[str] = None, max_days: int = 3000,
           session: Optional[requests.Session] = None, flush_every: int = 50) -> int:
    """
    Докачивает торговые даты набора: хвост после последней сохраненной даты и,
    если start раньше истории, начало (назад от истории). Без истории — с start
    (по умолчанию — начало истории ISS для набора). max_days ограничивает число
    дат за прогон.

    Returns: число записанных строк.
    """
    _check(dataset)
    label = DATASETS[dataset][2]
    session = session or iss.make_session()
    today = dt.date.today()
    first = _as_date(start or DATASETS[dataset][1])

    bounds = (lake.query(f"SELECT min(date) AS lo, max(date) AS hi FROM lake.{dataset}")
              if dataset in lake.tables() else None)
    if bounds is not None and bounds['hi'][0] is not None:
        lo, hi = _as_date(bounds['lo'][0]), _as_date(bounds['hi'][0])
        # Бэкфилл — только при явно заданном start: иначе (ночной прогон) каждый
        # запуск заново опрашивал бы пустые даты до первых торгов в ISS.
        # Зазор в начале до 10 дней считаем закрытым (праздники)
        head = ([] if start is None or (lo - first).days <= 10
                else _weekdays(first, lo - dt.timedelta(days=1))[::-1])
        dates = (head + _weekdays(hi + dt.timedelta(days=1), today))[:max_days]
    else:
        dates = _weekdays(first, today)[:max_days]
    if not dates:
        logger.info(f"[INFO] {label}: история актуальна")
        return 0

    frames, written = [], 0

    def flush():
        nonlocal frames, written
        if frames:
            written += lake.write(dataset, pl.concat(frames, how='diagonal_relaxed'))
            frames = []

    for i, day in enumerate(dates, 1):
        try:
            rows = _fetch(dataset, day, session)
        except Exception as e:
            logger.warning(f"[WARN] {label} {day}: {e} — прогон остановлен, сохраняю скачанное")
            break
        if rows.height:
            frames.append(rows)
        if i % flush_every == 0:
            flush()
            logger.info(f"[INFO] {label}: обработано дат {i}/{len(dates)} (до {day}), строк +{written}")
    flush()
    logger.info(f"[OK] {label}: +{written} строк" if written else f"[INFO] {label}: новых торговых дат нет")
    return written


def repair(dataset: str, session: Optional[requests.Session] = None,
           calendar: Optional[Iterable] = None) -> int:
    """
    Докачивает пропущенные торговые даты внутри сохраненной истории (дыры от
    сбоев прошлых прогонов). Календарь — будни IMOEX, если не передан.

    Returns: число записанных строк.
    """
    _check(dataset)
    label = DATASETS[dataset][2]
    have = dataset_dates(dataset)
    if not have:
        return 0
    cal = [_as_date(d) for d in (calendar if calendar is not None else trading_calendar())]
    have_set, skip = set(have), empty_dates(dataset)
    missing = [d for d in cal if have[0] <= d <= have[-1] and d.weekday() < 5
               and d not in have_set and d not in skip]
    if not missing:
        return 0

    logger.info(f"[INFO] {label}: пропущенных торговых дат — {len(missing)}, докачиваю")
    session = session or iss.make_session()
    frames, empty = [], []
    for day in missing:
        try:
            rows = _fetch(dataset, day, session)
        except Exception as e:
            logger.warning(f"[WARN] {label} {day}: {e}")
            continue
        (frames.append(rows) if rows.height else empty.append(day))
    if empty:
        lake.write('empty_dates', pl.DataFrame({'dataset': [dataset] * len(empty), 'date': empty}))
    written = lake.write(dataset, pl.concat(frames, how='diagonal_relaxed')) if frames else 0
    if written:
        logger.info(f"[OK] {label}: +{written} строк за {len(frames)} пропущенных дат")
    return written


def read(dataset: str, start=None, end=None, secids=None, boards=None,
         columns: Optional[list[str]] = None) -> pl.DataFrame:
    """
    История набора за период (включительно). secids / boards — код или список
    кодов бумаг / режимов торгов (BOARDID); columns — нужные колонки (быстрее).
    """
    _check(dataset)
    if dataset not in lake.tables():
        raise FileNotFoundError(f"В хранилище нет набора {dataset!r}: выполните "
                                f"update_data.py --history-init {dataset}")
    where, params = [], []
    if start is not None:
        where.append("date >= ?")
        params.append(_as_date(start))
    if end is not None:
        where.append("date <= ?")
        params.append(_as_date(end))
    for col, val in (('SECID', secids), ('BOARDID', boards)):
        if val is not None:
            val = [val] if isinstance(val, str) else list(val)
            where.append(f"{col} IN ({', '.join('?' * len(val))})")
            params.extend(val)
    cols = ", ".join(f'"{c}"' for c in columns) if columns else "*"
    sql = f"SELECT {cols} FROM lake.{dataset}"
    if where:
        sql += " WHERE " + " AND ".join(where)
    return lake.query(sql + " ORDER BY date", params)


# ------------------------------------------------- реестр карточек бумаг

def read_securities(dataset: str = 'bonds') -> pl.DataFrame:
    """Реестр карточек ISS для бумаг набора (таблица <набор>_securities): строка на SECID."""
    table = f"{dataset}_securities"
    if table not in lake.tables():
        raise FileNotFoundError(f"В хранилище нет реестра {table}: выполните update_securities('{dataset}')")
    return lake.query(f"SELECT * FROM lake.{table} ORDER BY SECID")


def update_securities(dataset: str = 'bonds', max_new: Optional[int] = 500,
                      session: Optional[requests.Session] = None, flush_every: int = 200) -> int:
    """
    Дополняет реестр <набор>_securities карточками ISS для бумаг из истории
    набора, которых в реестре еще нет (включая погашенные). max_new ограничивает
    число запросов за прогон: ночью — 500, первичное наполнение — None.

    Флаги HASDEFAULT / HASTECHNICALDEFAULT в карточках облигаций — текущий статус,
    как правило на уровне эмитента (все его действующие выпуски), а не история:
    у погашенных выпусков они не выставлены.

    Returns: число добавленных бумаг.
    """
    _check(dataset)
    table = f"{dataset}_securities"
    names = lake.tables()
    if dataset not in names:
        logger.info(f"[INFO] Реестр {table}: нет истории набора — пропуск")
        return 0
    known = (f"SELECT SECID FROM lake.{table}" if table in names else "SELECT NULL::VARCHAR AS SECID LIMIT 0")
    todo = lake.query(f"SELECT DISTINCT SECID FROM lake.{dataset} "
                      f"WHERE SECID NOT IN ({known}) ORDER BY SECID")['SECID'].to_list()
    if max_new is not None:
        todo = todo[:max_new]
    if not todo:
        logger.info(f"[INFO] Реестр {table}: актуален")
        return 0

    logger.info(f"[INFO] Реестр {table}: новых бумаг {len(todo)}")
    session = session or iss.make_session()
    fetched = dt.date.today().isoformat()
    rows, added = [], 0

    def flush():
        nonlocal rows, added
        if rows:
            added += lake.write(table, iss.records_to_frame(rows))
            rows = []

    for i, secid in enumerate(todo, 1):
        try:
            desc = iss.security_description(secid, session=session)
        except Exception as e:
            logger.warning(f"[WARN] {secid}: карточка ISS недоступна — {e}; прогон остановлен")
            break
        rows.append({**desc, 'SECID': secid, 'FETCHED': fetched})
        if i % flush_every == 0:
            flush()
            logger.info(f"[INFO] Реестр {table}: {i}/{len(todo)}")
    flush()
    logger.info(f"[OK] Реестр {table}: +{added}")
    return added
