"""
Состав и веса индексов MOEX по датам: lake.index_weights (ключ date + indexid + ticker).

Источник — /iss/statistics/engines/stock/markets/index/analytics/<индекс>?date=:
бумаги индекса и их веса (% от индекса) на дату. ISS отдает один индекс за
одну дату, веса меняются ежедневно вместе с ценами, поэтому хранятся не все
288 индексов, а основные (CORE_INDEXES): IMOEX с 2001 года, широкий рынок,
голубые фишки, отраслевые, RGBI. RTSI и MCFTR не нужны — их состав совпадает
с IMOEX.

Даты — торговый календарь IMOEX (с рабочими субботами). Последняя обработанная
дата каждого индекса — в lake.load_state (index_weights:<индекс>): в дни, когда
индекс не рассчитывался, строк нет, и по max(date) прогресс не виден.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Iterable, Optional

import polars as pl
import requests

from moexutils import history, iss, lake

logger = logging.getLogger("moexutils")

URL = f"{iss.ISS_URL}/statistics/engines/stock/markets/index/analytics"
TABLE = 'index_weights'
CORE_INDEXES = (
    'IMOEX', 'MOEX10', 'MOEXBC', 'MOEXBMI', 'MCXSM', 'MRBC',
    'MOEXOG', 'MOEXFN', 'MOEXMM', 'MOEXEU', 'MOEXTL', 'MOEXCN', 'MOEXCH', 'MOEXTN', 'MOEXIT', 'MOEXRE',
    'RGBI',
)


def list_indexes(session: Optional[requests.Session] = None) -> pl.DataFrame:
    """Индексы, для которых ISS отдает состав: indexid, shortname, from, till."""
    session = session or iss.make_session()
    resp = session.get(f"{URL}.json")
    resp.raise_for_status()
    return iss.to_frame(resp.json().get('indices'))


def fetch_weights(indexid: str, date, session: Optional[requests.Session] = None,
                  max_pages: int = 50) -> pl.DataFrame:
    """Состав индекса на дату (все страницы): date, indexid, ticker, secids, shortnames, weight, ..."""
    session = session or iss.make_session()
    day = dt.date.fromisoformat(str(date)[:10])
    pages, offset = [], 0
    for _ in range(max_pages):
        resp = session.get(f"{URL}/{indexid}.json", params={'date': day.isoformat(), 'start': offset, 'limit': 100})
        resp.raise_for_status()
        data = resp.json()
        page = iss.to_frame(data.get('analytics'))
        if page.is_empty():
            break
        pages.append(page)
        offset += page.height
        cursor = iss.to_frame(data.get('analytics.cursor'))
        if cursor.is_empty() or offset >= int(cursor['TOTAL'][0]):
            break
    if not pages:
        return pl.DataFrame()
    df = pl.concat(pages, how='diagonal_relaxed')
    # ISS на дату без расчета может отдать ближайшую другую дату — берем только запрошенную
    df = df.with_columns(pl.col('tradedate').str.to_date('%Y-%m-%d').alias('date')).filter(pl.col('date') == day)
    return (df.drop('tradedate').select(['date', 'indexid', 'ticker', *[c for c in df.columns
                                          if c not in ('date', 'tradedate', 'indexid', 'ticker')]])
              .unique(['date', 'indexid', 'ticker'], keep='last', maintain_order=True))


def _state_name(indexid: str) -> str:
    return f"{TABLE}:{indexid}"


def _processed_until(indexid: str) -> Optional[dt.date]:
    names = lake.tables()
    if 'load_state' in names:
        got = lake.query("SELECT date FROM lake.load_state WHERE name = ?", [_state_name(indexid)])
        if got.height:
            return got['date'][0]
    if TABLE in names:
        return lake.query(f"SELECT max(date) AS d FROM lake.{TABLE} WHERE indexid = ?", [indexid])['d'][0]
    return None


def update_index_weights(indexes: Iterable[str] = CORE_INDEXES, start=None, max_days: Optional[int] = 30,
                         session: Optional[requests.Session] = None, flush_every: int = 100) -> int:
    """
    Состав индексов по датам торгового календаря после последней обработанной
    (без истории — с start или с начала выдачи ISS для индекса). max_days —
    сколько дат на индекс за прогон (ночью 30; первичная выгрузка — None).
    На сбое индекс останавливается, записанное сохраняется. Returns: записано строк.
    """
    session = session or iss.make_session()
    cal = history.trading_calendar()
    if not cal:
        logger.warning("[WARN] Состав индексов: нет торгового календаря (lake.indexes) — пропуск")
        return 0
    starts = {}
    try:
        starts = dict(zip(*[list_indexes(session)[c].to_list() for c in ('indexid', 'from')]))
    except Exception as e:
        logger.warning(f"[WARN] Состав индексов: список ISS недоступен — {e}")
    total = 0
    for idx in indexes:
        done = _processed_until(idx)
        if done is not None:
            first = done + dt.timedelta(days=1)
        else:
            first = dt.date.fromisoformat(str(start or starts.get(idx) or '2001-01-03')[:10])
        days = [d for d in cal if d >= first]
        if max_days is not None:
            days = days[:max_days]
        if not days:
            continue
        frames, written, last_done = [], 0, None

        def flush():
            nonlocal frames, written
            if frames:
                written += lake.write(TABLE, pl.concat(frames, how='diagonal_relaxed'))
                frames = []
            if last_done is not None:
                lake.write('load_state', pl.DataFrame({'name': [_state_name(idx)], 'date': [last_done]}))

        for i, day in enumerate(days, 1):
            try:
                df = fetch_weights(idx, day, session)
            except Exception as e:
                logger.warning(f"[WARN] Состав {idx} {day}: {e} — индекс остановлен, сохраняю скачанное")
                break
            last_done = day
            if df.height:
                frames.append(df)
            if i % flush_every == 0:
                flush()
                logger.info(f"[INFO] Состав {idx}: {i}/{len(days)} дат (до {day}), строк +{written}")
        flush()
        if written:
            logger.info(f"[OK] Состав {idx}: +{written} строк, по {last_done}")
        total += written
    if not total:
        logger.info("[INFO] Состав индексов: новых дат нет")
    return total


def read_index_weights(indexes=None, start=None, end=None, as_of: lake.AsOf = None) -> pl.DataFrame:
    """Состав и веса индексов из хранилища; indexes — код или список."""
    where, params = [], []
    if indexes is not None:
        indexes = [indexes] if isinstance(indexes, str) else list(indexes)
        where.append(f"indexid IN ({', '.join('?' * len(indexes))})")
        params += indexes
    for op, val in ((">=", start), ("<=", end)):
        if val is not None:
            where.append(f"date {op} ?")
            params.append(dt.date.fromisoformat(str(val)[:10]))
    sql = f"SELECT * FROM {lake.ref(TABLE, as_of)}" + (" WHERE " + " AND ".join(where) if where else "")
    return lake.query(sql + " ORDER BY indexid, date, ticker", params)


def constituents_at(indexid: str, date, as_of: lake.AsOf = None) -> pl.DataFrame:
    """Состав индекса на дату — последняя дата расчета не позже date."""
    day = dt.date.fromisoformat(str(date)[:10])
    return lake.query(
        f"SELECT * FROM {lake.ref(TABLE, as_of)} WHERE indexid = ? AND date = "
        f"(SELECT max(date) FROM {lake.ref(TABLE, as_of)} WHERE indexid = ? AND date <= ?) ORDER BY weight DESC",
        [indexid, indexid, day])
