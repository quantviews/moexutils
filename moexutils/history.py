"""
История рынков MOEX «все инструменты за дату» в хранилище DuckLake (lake.py).

Один постраничный запрос ISS на торговую дату отдает все инструменты рынка
(наборы DATASETS): облигации — все доски с 1997 года (включая погашенные
выпуски), фьючерсы — все контракты FORTS с 2002 года, все акции и фонды, все
индексы, валютный рынок, валютные фиксинги. Строки пишутся в таблицы lake
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
from typing import Iterable, NamedTuple, Optional

import polars as pl
import requests

from moexutils import iss
from moexutils import lake

logger = logging.getLogger("moexutils")


class Dataset(NamedTuple):
    path: str                        # путь рынка в ISS
    start: str                       # начало истории в ISS
    label: str                       # подпись для логов
    keep: Optional[pl.Expr] = None   # какие строки ответа хранить (None — все)
    weekends: bool = False           # запрашивать и выходные (часть индексов публикуется по воскресеньям)


# Набор данных = таблица хранилища (ключ date + SECID + BOARDID)
DATASETS = {
    'bonds': Dataset('stock/markets/bonds', '1997-01-01', 'облигации (весь рынок)'),
    'futures': Dataset('futures/markets/forts', '2002-01-01', 'фьючерсы FORTS'),
    'shares': Dataset('stock/markets/shares', '1997-03-24', 'акции и фонды (весь рынок)'),
    # сельскохозяйственные индексы (доска AGRO) публикуются по воскресеньям
    'indexes_all': Dataset('stock/markets/index', '1995-09-01', 'индексы (все)', weekends=True),
    # у валютного рынка много строк-заглушек без сделок (NUMTRADES = 0, цены 0)
    'currency': Dataset('currency/markets/selt', '1997-06-02', 'валютный рынок',
                        keep=pl.col('NUMTRADES') > 0),
    'currency_fixings': Dataset('currency/markets/index', '2019-08-01', 'валютные фиксинги'),
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


def _all_days(start: dt.date, end: dt.date) -> list[dt.date]:
    return [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]


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
    """
    Торговый календарь: даты, по которым есть IMOEX в lake.indexes, включая
    рабочие субботы (перенесенные рабочие дни — 50 дат с 1995 года).
    """
    if 'indexes' not in lake.tables():
        return []
    return lake.query("SELECT DISTINCT date FROM lake.indexes WHERE ticker = 'IMOEX' ORDER BY date")['date'].to_list()


def _fetch(dataset: str, date: dt.date, session: requests.Session) -> pl.DataFrame:
    rows = iss.history_day(DATASETS[dataset].path, date, session)
    keep = DATASETS[dataset].keep
    return rows.filter(keep) if keep is not None and rows.height else rows


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
    days = _all_days if DATASETS[dataset].weekends else _weekdays

    bounds = (lake.query(f"SELECT min(date) AS lo, max(date) AS hi FROM lake.{dataset}")
              if dataset in lake.tables() else None)
    if bounds is not None and bounds['hi'][0] is not None:
        lo, hi = _as_date(bounds['lo'][0]), _as_date(bounds['hi'][0])
        # Бэкфилл — только при явно заданном start: иначе (ночной прогон) каждый
        # запуск заново опрашивал бы пустые даты до первых торгов в ISS.
        # Зазор в начале до 10 дней считаем закрытым (праздники)
        head = ([] if start is None or (lo - first).days <= 10
                else days(first, lo - dt.timedelta(days=1))[::-1])
        dates = (head + days(hi + dt.timedelta(days=1), today))[:max_days]
    else:
        dates = days(first, today)[:max_days]
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
            flush()
            raise
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
    сбоев прошлых прогонов). Календарь — даты торгов IMOEX (с рабочими субботами), если не передан.

    Returns: число записанных строк.
    """
    _check(dataset)
    label = DATASETS[dataset][2]
    have = dataset_dates(dataset)
    if not have:
        return 0
    cal = [_as_date(d) for d in (calendar if calendar is not None else trading_calendar())]
    have_set, skip = set(have), empty_dates(dataset)
    # календарь — даты торгов (в том числе рабочие субботы), выходные из него не отбрасываются
    missing = [d for d in cal if have[0] <= d <= have[-1] and d not in have_set and d not in skip]
    if not missing:
        return 0

    logger.info(f"[INFO] {label}: пропущенных торговых дат — {len(missing)}, докачиваю")
    session = session or iss.make_session()
    frames, empty, errors = [], [], []
    for day in missing:
        try:
            rows = _fetch(dataset, day, session)
        except Exception as e:
            logger.warning(f"[WARN] {label} {day}: {e}")
            errors.append(e)
            continue
        (frames.append(rows) if rows.height else empty.append(day))
    if empty:
        lake.write('empty_dates', pl.DataFrame({'dataset': [dataset] * len(empty), 'date': empty}))
    written = lake.write(dataset, pl.concat(frames, how='diagonal_relaxed')) if frames else 0
    if errors:
        raise ExceptionGroup(f"{label}: ошибки восстановления истории", errors)
    if written:
        logger.info(f"[OK] {label}: +{written} строк за {len(frames)} пропущенных дат")
    return written


def read(dataset: str, start=None, end=None, secids=None, boards=None,
         columns: Optional[list[str]] = None, as_of: lake.AsOf = None) -> pl.DataFrame:
    """
    История набора за период (включительно). secids / boards — код или список
    кодов бумаг / режимов торгов (BOARDID); columns — нужные колонки (быстрее);
    as_of — на момент снимка хранилища (lake.ref).
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
    sql = f"SELECT {cols} FROM {lake.ref(dataset, as_of)}"
    if where:
        sql += " WHERE " + " AND ".join(where)
    return lake.query(sql + " ORDER BY date", params)


# ------------------------------------------------- реестр карточек бумаг

def read_securities(dataset: str = 'bonds', as_of: lake.AsOf = None) -> pl.DataFrame:
    """Реестр карточек ISS для бумаг набора (таблица <набор>_securities): строка на SECID."""
    table = f"{dataset}_securities"
    if table not in lake.tables():
        raise FileNotFoundError(f"В хранилище нет реестра {table}: выполните update_securities('{dataset}')")
    return lake.query(f"SELECT * FROM {lake.ref(table, as_of)} ORDER BY SECID")


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
            flush()
            raise
        rows.append({**desc, 'SECID': secid, 'FETCHED': fetched})
        if i % flush_every == 0:
            flush()
            logger.info(f"[INFO] Реестр {table}: {i}/{len(todo)}")
    flush()
    logger.info(f"[OK] Реестр {table}: +{added}")
    return added
