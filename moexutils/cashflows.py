"""
Денежные потоки облигаций MOEX в хранилище DuckLake: купоны, амортизации, оферты.

Источник — сводная выдача ISS по всем облигациям
(/iss/statistics/engines/stock/markets/bonds/bondization), включая погашенные
выпуски с 1997 года и будущие выплаты:
- lake.bond_coupons — ключ secid + coupondate;
- lake.bond_amortizations — ключ secid + amortdate + data_source ('amortization' / 'maturity');
- lake.bond_offers — ключ secid + offer_date (offerdate, а если биржа его не
  указала, начало или конец периода предъявления); offertype меняется со
  временем («Оферта» -> «Оферта (состоялось)»), поэтому в ключ не входит.

Особенности биржевых данных:
- facevalue — ТЕКУЩИЙ номинал, а не номинал на дату выплаты: у амортизируемых
  выпусков valueprc прошлых купонов искажен — используйте value;
- value_rub биржа пересчитывает по сегодняшнему курсу — колонка не хранится;
- будущие купоны флоатеров и ипотечных облигаций пусты (value = null), пока не зафиксированы.

Обновление: каждую ночь — окно от 10 дней назад до 60 вперед (зафиксированные
купоны, новые выпуски); по субботам — все будущие потоки; полная выгрузка —
update_cashflows('full') (около 2,6 тыс. запросов). Запрошенный диапазон дат
приходит целиком, поэтому купоны и амортизации в нем, которых биржа больше не
отдает (отмененные, перенесенные), удаляются; оферты — только при полной выгрузке.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import NamedTuple, Optional

import polars as pl
import requests

from moexutils import iss, lake

logger = logging.getLogger("moexutils")

URL = f"{iss.ISS_URL}/statistics/engines/stock/markets/bonds/bondization.json"


class Block(NamedTuple):
    table: str
    date_col: str        # дата потока — по ней окно и отбор будущих
    key: list


BLOCKS = {
    'coupons': Block('bond_coupons', 'coupondate', ['secid', 'coupondate']),
    'amortizations': Block('bond_amortizations', 'amortdate', ['secid', 'amortdate', 'data_source']),
    'offers': Block('bond_offers', 'offer_date', ['secid', 'offer_date']),
}
DATE_COLS = ('coupondate', 'recorddate', 'startdate', 'amortdate', 'offerdate', 'offerdatestart', 'offerdateend')
WINDOW_BACK, WINDOW_FORWARD = 10, 60


def fetch_block(block: str, start=None, till=None, session: Optional[requests.Session] = None,
                max_pages: int = 5000) -> pl.DataFrame:
    """Все страницы блока ('coupons', 'amortizations', 'offers') за период дат потока (None — вся история)."""
    session = session or iss.make_session()
    params = {'iss.only': f"{block},{block}.cursor", 'limit': 100}
    if start is not None:
        params['from'] = str(start)[:10]
    if till is not None:
        params['till'] = str(till)[:10]
    pages, offset = [], 0
    for _ in range(max_pages):
        data = session.get(URL, params={**params, 'start': offset})
        data.raise_for_status()
        data = data.json()
        page = iss.to_frame(data.get(block))
        if page.is_empty():
            break
        pages.append(page)
        offset += page.height
        cursor = iss.to_frame(data.get(f"{block}.cursor"))
        if cursor.is_empty() or offset >= int(cursor['TOTAL'][0]):
            break
    return prepare(block, pl.concat(pages, how='diagonal_relaxed')) if pages else pl.DataFrame()


def prepare(block: str, df: pl.DataFrame) -> pl.DataFrame:
    """Даты -> Date ('0000-00-00' -> null), без value_rub, ключ без пустот и повторов."""
    b = BLOCKS[block]
    df = df.with_columns([pl.col(c).str.to_date('%Y-%m-%d', strict=False).alias(c)
                          for c in DATE_COLS if c in df.columns])
    if 'value_rub' in df.columns:
        df = df.drop('value_rub')
    if block == 'offers':
        df = df.with_columns(pl.coalesce('offerdate', 'offerdatestart', 'offerdateend').alias('offer_date'))
    return df.drop_nulls(b.key).unique(b.key, keep='last', maintain_order=True)


def update_cashflows(mode: str = 'window', session: Optional[requests.Session] = None) -> dict[str, int]:
    """
    mode: 'window' — потоки от сегодня−10 до сегодня+60 дней (ночью);
    'future' — все будущие потоки (раз в неделю); 'full' — вся история.
    Потоки в запрошенном диапазоне, которых биржа больше не отдает, удаляются.
    Returns: {таблица: записано строк}.
    """
    today = dt.date.today()
    start, till = {'window': (today - dt.timedelta(days=WINDOW_BACK), today + dt.timedelta(days=WINDOW_FORWARD)),
                   'future': (today, None), 'full': (None, None)}[mode]
    session = session or iss.make_session()
    out = {}
    for block, b in BLOCKS.items():
        new = fetch_block(block, start, till, session)
        stale, delta = None, new
        if new.height and b.table in lake.tables():
            where, params = [], []
            for op, val in ((">=", start), ("<=", till)):
                if val is not None:
                    where.append(f"{b.date_col} {op} ?")
                    params.append(val)
            old = lake.query(f"SELECT * FROM lake.{b.table}"
                             + (" WHERE " + " AND ".join(where) if where else ""), params)
            stale = old.select(b.key).join(new.select(b.key), on=b.key, how='anti')
            if block == 'offers' and mode != 'full':
                # фильтр ISS по датам идет по offerdate, а у части оферт он пустой
                # (0000-00-00): их нет в выдаче окна — удалять только при полной выгрузке
                stale = None
            delta = lake.changed_rows(old, new, b.key)   # пишутся только новые и изменившиеся
        out[b.table] = lake.write(b.table, delta, delete=stale)
        gone = 0 if stale is None else stale.height
        logger.info(f"[OK] {b.table}: записано {out[b.table]}" + (f", удалено отмененных {gone}" if gone else ""))
    return out


def read_cashflows(kind: str = 'coupons', secids=None, start=None, end=None,
                   as_of: lake.AsOf = None) -> pl.DataFrame:
    """Потоки из хранилища: kind — 'coupons', 'amortizations', 'offers'; фильтр по бумагам и датам потока."""
    b = BLOCKS[kind]
    if b.table not in lake.tables():
        raise FileNotFoundError(f"В хранилище нет таблицы {b.table}: выполните update_cashflows('full')")
    where, params = [], []
    if secids is not None:
        secids = [secids] if isinstance(secids, str) else list(secids)
        where.append(f"secid IN ({', '.join('?' * len(secids))})")
        params += secids
    for op, val in ((">=", start), ("<=", end)):
        if val is not None:
            where.append(f"{b.date_col} {op} ?")
            params.append(dt.date.fromisoformat(str(val)[:10]))
    sql = f"SELECT * FROM {lake.ref(b.table, as_of)}" + (" WHERE " + " AND ".join(where) if where else "")
    return lake.query(sql + f" ORDER BY secid, {b.date_col}", params)
