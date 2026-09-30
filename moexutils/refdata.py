"""
Справочные параметры бумаг фондового рынка MOEX по датам: объем выпуска,
уровень листинга, номинал, параметры купона — с 01.04.2024 (раньше ISS их
по датам не отдает).

Источник — ежедневный срез /iss/referencedata/engines/stock/markets/all/securities
(акции, облигации, фонды — около 5,4 тыс. бумаг). Хранятся только изменения:
lake.stock_refdata — строка на бумагу и дату, с которой ее параметры стали
такими (ключ secid + date). Не хранятся: НКД (accruedint) — меняется каждый
день и есть в истории облигаций; флаги hasprospectus и hastechnicaldefault —
в этой выдаче они «мигают» 0/1 изо дня в день у одних и тех же бумаг (шум
источника, десятки тысяч ложных изменений). Состояние на дату — последняя строка бумаги
не позже этой даты (refdata_at).
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Optional

import polars as pl
import requests

from moexutils import iss, lake

logger = logging.getLogger("moexutils")

URL = f"{iss.ISS_URL}/referencedata/engines/stock/markets/all/securities.json"
START = dt.date(2024, 4, 1)
TABLE = 'stock_refdata'
# служебные поля — в сравнении «изменилось ли» не участвуют; ненадежные — не хранятся
_VOLATILE = ('tradedate', 'updatetime')
_DROPPED = ('tradedate', 'accruedint', 'hasprospectus', 'hastechnicaldefault')


def fetch_snapshot(date, session: Optional[requests.Session] = None, max_pages: int = 100) -> pl.DataFrame:
    """Срез параметров всех бумаг на дату (все страницы по 1000); date — колонка Date."""
    session = session or iss.make_session()
    day = dt.date.fromisoformat(str(date)[:10])
    pages, offset = [], 0
    for _ in range(max_pages):
        resp = session.get(URL, params={'date': day.isoformat(), 'start': offset})
        resp.raise_for_status()
        data = resp.json()
        page = iss.to_frame(data.get('securities'))
        if page.is_empty():
            break
        pages.append(page)
        offset += page.height
        cursor = iss.to_frame(data.get('securities.cursor'))
        if cursor.is_empty() or offset >= int(cursor['TOTAL'][0]):
            break
    if not pages:
        return pl.DataFrame()
    df = pl.concat(pages, how='diagonal_relaxed').unique('secid', keep='last', maintain_order=True)
    return (df.with_columns(pl.lit(day).alias('date'))
              .select(['date', 'secid', *[c for c in df.columns if c not in ('secid', *_DROPPED)]]))


def _compare_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ('date', 'secid', *_VOLATILE)]


def _changes(state: pl.DataFrame, snap: pl.DataFrame) -> pl.DataFrame:
    """Строки среза, у которых параметры отличаются от последнего состояния бумаги (или бумага новая)."""
    if state.is_empty():
        return snap
    cols = [c for c in _compare_cols(snap) if c in state.columns]
    joined = snap.join(state.select('secid', *cols).with_columns(pl.lit(True).alias('__was')),
                       on='secid', how='left', suffix='__old')
    diff = pl.col('__was').is_null()
    for c in cols:
        diff = diff | pl.col(c).ne_missing(pl.col(f"{c}__old"))
    return joined.filter(diff).select(snap.columns)


def _last_state() -> pl.DataFrame:
    if TABLE not in lake.tables():
        return pl.DataFrame()
    return lake.query(f"SELECT * FROM lake.{TABLE} QUALIFY row_number() OVER (PARTITION BY secid ORDER BY date DESC) = 1")


def update_refdata(start=None, max_days: int = 2000, session: Optional[requests.Session] = None,
                   flush_every: int = 20) -> int:
    """
    Срезы с последней сохраненной даты (без истории — с 01.04.2024 или start)
    по вчерашний день, только будни; в хранилище — изменения. На сбое прогон
    останавливается, записанное сохраняется. Returns: число записанных строк.
    """
    session = session or iss.make_session()
    last_day = dt.date.today() - dt.timedelta(days=1)
    done = _processed_until()
    if done is not None:
        first = done + dt.timedelta(days=1)
    else:
        first = dt.date.fromisoformat(str(start)[:10]) if start is not None else START
    days = [first + dt.timedelta(days=i) for i in range((last_day - first).days + 1)]
    days = [d for d in days if d.weekday() < 5][:max_days]
    if not days:
        logger.info("[INFO] Параметры бумаг: срезы актуальны")
        return 0
    state, pending, written, last_done = _last_state(), [], 0, None

    def flush():
        nonlocal pending, written
        if pending:
            written += lake.write(TABLE, pl.concat(pending, how='diagonal_relaxed'))
            pending = []
        if last_done is not None:
            lake.write('load_state', pl.DataFrame({'name': [TABLE], 'date': [last_done]}))

    for i, day in enumerate(days, 1):
        try:
            snap = fetch_snapshot(day, session)
        except Exception as e:
            logger.warning(f"[WARN] Параметры бумаг {day}: {e} — прогон остановлен, сохраняю изменения")
            break
        last_done = day
        if snap.is_empty():
            continue
        delta = _changes(state, snap)
        if delta.height:
            pending.append(delta)
            state = (pl.concat([state, delta], how='diagonal_relaxed') if state.height else delta)
            state = state.sort('date').unique('secid', keep='last')
        if i % flush_every == 0:
            flush()
            logger.info(f"[INFO] Параметры бумаг: обработано дат {i}/{len(days)} (до {day}), изменений +{written}")
    flush()
    logger.info(f"[OK] Параметры бумаг: изменений +{written}" if written else "[INFO] Параметры бумаг: изменений нет")
    return written


def _processed_until() -> Optional[dt.date]:
    """Последний обработанный день: строки пишутся только при изменениях, поэтому
    по max(date) таблицы этого не видно — дата хранится в lake.load_state."""
    names = lake.tables()
    if 'load_state' in names:
        got = lake.query("SELECT date FROM lake.load_state WHERE name = ?", [TABLE])
        if got.height:
            return got['date'][0]
    if TABLE in names:
        return lake.query(f"SELECT max(date) AS d FROM lake.{TABLE}")['d'][0]
    return None


def refdata_at(date, secids=None, as_of: lake.AsOf = None) -> pl.DataFrame:
    """Параметры бумаг на дату (последнее состояние не позже date); secids — код или список."""
    where, params = ["date <= ?"], [dt.date.fromisoformat(str(date)[:10])]
    if secids is not None:
        secids = [secids] if isinstance(secids, str) else list(secids)
        where.append(f"secid IN ({', '.join('?' * len(secids))})")
        params += secids
    return lake.query(f"SELECT * FROM {lake.ref(TABLE, as_of)} WHERE {' AND '.join(where)} "
                      "QUALIFY row_number() OVER (PARTITION BY secid ORDER BY date DESC) = 1 ORDER BY secid", params)


def read_refdata(secids=None, as_of: lake.AsOf = None) -> pl.DataFrame:
    """Вся история изменений параметров (secid, date — с какой даты действуют)."""
    where, params = [], []
    if secids is not None:
        secids = [secids] if isinstance(secids, str) else list(secids)
        where.append(f"secid IN ({', '.join('?' * len(secids))})")
        params += secids
    sql = f"SELECT * FROM {lake.ref(TABLE, as_of)}" + (" WHERE " + " AND ".join(where) if where else "")
    return lake.query(sql + " ORDER BY secid, date", params)
