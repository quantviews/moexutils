"""
Ставки и кривые в хранилище DuckLake.

- RUONIA (Банк России, cbr.ru) — lake.ruonia: ставка, объем, число сделок и
  участников, процентили — с 11.01.2010. В ISS RUONIA нет (RUSFAR — в indexes_all).
- Кривая бескупонной доходности ОФЗ (КБД MOEX, /iss/engines/stock/zcyc) с 06.01.2014:
  lake.zcyc_params — параметры Nelson–Siegel–Svensson (B1–B3, T1, G1–G9) на дату,
  lake.zcyc_yields — доходности на сроки 0.25–20 лет, lake.zcyc_bonds — ОФЗ,
  по которым строилась кривая (цены и доходности bid/ask/сделок/расчетные).
  Грузится по вчерашний день: за сегодня ISS отдает промежуточную кривую.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Optional

import polars as pl
import requests

from moexutils import history, iss, lake
from moexutils.stocks import _html_table_rows

logger = logging.getLogger("moexutils")

RUONIA_URL = "https://www.cbr.ru/hd_base/ruonia/dynamics/"
RUONIA_START = dt.date(2010, 1, 11)
RUONIA_COLUMNS = ['date', 'rate', 'volume_bn', 'deals', 'participants',
                  'rate_min', 'rate_p25', 'rate_p75', 'rate_max', 'status', 'published']

ZCYC_START = dt.date(2014, 1, 6)
ZCYC_TABLES = {'params': 'zcyc_params', 'yearyields': 'zcyc_yields', 'securities': 'zcyc_bonds'}


def _as_date(value) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


def _num(text: str) -> Optional[float]:
    text = text.replace('\xa0', '').replace(' ', '').replace(',', '.')
    try:
        return float(text)
    except ValueError:
        return None  # «—» — нет данных


# ---------------------------------------------------------------- RUONIA

def fetch_ruonia(start=None, end=None, session: Optional[requests.Session] = None) -> pl.DataFrame:
    """RUONIA с cbr.ru за период (одна страница на всю историю), по возрастанию даты."""
    session = session or iss.make_session()
    start = _as_date(start) if start is not None else RUONIA_START
    end = _as_date(end) if end is not None else dt.date.today()
    resp = session.get(RUONIA_URL, headers={'User-Agent': 'Mozilla/5.0'}, params={
        'UniDbQuery.Posted': 'True', 'UniDbQuery.From': f"{start:%d.%m.%Y}", 'UniDbQuery.To': f"{end:%d.%m.%Y}"})
    resp.raise_for_status()
    rows = []
    for cells in _html_table_rows(resp.text):
        if len(cells) < len(RUONIA_COLUMNS):
            continue
        try:
            day = dt.datetime.strptime(cells[0], '%d.%m.%Y').date()
        except ValueError:
            continue  # шапка
        try:
            published = dt.datetime.strptime(cells[10], '%d.%m.%Y').date()
        except ValueError:
            published = None
        rows.append((day, *[_num(c) for c in cells[1:9]], cells[9] or None, published))
    schema = {'date': pl.Date, **{c: pl.Float64 for c in RUONIA_COLUMNS[1:9]},
              'status': pl.Utf8, 'published': pl.Date}
    return pl.DataFrame(rows, schema=schema, orient='row').unique('date', keep='first').sort('date')


def update_ruonia(session: Optional[requests.Session] = None) -> int:
    """
    Вся история RUONIA одним запросом; в хранилище — новые и пересмотренные
    строки (ЦБ может уточнить последние значения). Returns: записано строк.
    """
    new = fetch_ruonia(session=session)
    if new.is_empty():
        raise ValueError("cbr.ru вернул пустую таблицу RUONIA")
    old = lake.query("SELECT * FROM lake.ruonia") if 'ruonia' in lake.tables() else pl.DataFrame()
    delta = lake.changed_rows(old, new, ['date'])
    n = lake.write('ruonia', delta)
    logger.info(f"[OK] RUONIA: записано {n}, последняя дата {new['date'].max()}" if n
                else f"[INFO] RUONIA: изменений нет (по {new['date'].max()})")
    return n


def read_ruonia(start=None, end=None, as_of: lake.AsOf = None) -> pl.DataFrame:
    """RUONIA из хранилища за период (включительно)."""
    return _read('ruonia', start, end, as_of)


# ---------------------------------------------------------------- КБД (zcyc)

def fetch_zcyc(date, session: Optional[requests.Session] = None) -> dict[str, pl.DataFrame]:
    """
    Кривая на дату: {'params', 'yearyields', 'securities'} с колонкой date вместо
    tradedate. Выходной или дата без кривой — пустые таблицы.
    """
    session = session or iss.make_session()
    resp = session.get(f"{iss.ISS_URL}/engines/stock/zcyc.json",
                       params={'date': _as_date(date).isoformat(), 'iss.only': ','.join(ZCYC_TABLES)})
    resp.raise_for_status()
    data = resp.json()
    out = {}
    for block in ZCYC_TABLES:
        df = iss.to_frame(data.get(block))
        if df.is_empty():
            out[block] = pl.DataFrame()
            continue
        out[block] = (df.with_columns(pl.col('tradedate').str.to_date('%Y-%m-%d').alias('date'))
                        .drop('tradedate')
                        .select(['date', *[c for c in df.columns if c != 'tradedate']]))
    return out


def _weekdays(start: dt.date, end: dt.date) -> list[dt.date]:
    """Будни и рабочие субботы из торгового календаря (кривая публикуется и в них)."""
    extra = {d for d in history.trading_calendar() if d.weekday() >= 5 and start <= d <= end}
    days = {start + dt.timedelta(days=i) for i in range((end - start).days + 1)
            if (start + dt.timedelta(days=i)).weekday() < 5}
    return sorted(days | extra)


def update_zcyc(start=None, max_days: int = 5000, session: Optional[requests.Session] = None,
                flush_every: int = 50) -> int:
    """
    Докачивает кривую по вчерашний день: хвост после последней даты и, если
    start раньше истории, начало (назад от истории). Без истории — с start
    (по умолчанию 06.01.2014). На сбое прогон останавливается, скачанное
    сохраняется. Returns: число записанных дат.
    """
    session = session or iss.make_session()
    last_day = dt.date.today() - dt.timedelta(days=1)
    first = _as_date(start) if start is not None else ZCYC_START
    if 'zcyc_params' in lake.tables():
        b = lake.query("SELECT min(date) AS lo, max(date) AS hi FROM lake.zcyc_params")
        lo, hi = b['lo'][0], b['hi'][0]
        head = [] if start is None or (lo - first).days <= 10 else _weekdays(first, lo - dt.timedelta(days=1))[::-1]
        dates = head + _weekdays(hi + dt.timedelta(days=1), last_day)
    else:
        dates = _weekdays(first, last_day)
    dates = dates[:max_days]
    if not dates:
        logger.info("[INFO] КБД: история актуальна")
        return 0

    frames = {b: [] for b in ZCYC_TABLES}
    written = 0

    def flush():
        nonlocal written
        n = 0
        for block, table in ZCYC_TABLES.items():
            if frames[block]:
                df = pl.concat(frames[block], how='diagonal_relaxed').unique(lake.TABLE_KEYS[table], keep='last')
                lake.write(table, df)
                if block == 'params':
                    n = df.height
                frames[block] = []
        written += n

    for i, day in enumerate(dates, 1):
        try:
            blocks = fetch_zcyc(day, session)
        except Exception as e:
            logger.warning(f"[WARN] КБД {day}: {e} — прогон остановлен, сохраняю скачанное")
            flush()
            raise
        for block, df in blocks.items():
            if df.height:
                frames[block].append(df)
        if i % flush_every == 0:
            flush()
            logger.info(f"[INFO] КБД: обработано дат {i}/{len(dates)} (до {day}), дат с кривой +{written}")
    flush()
    logger.info(f"[OK] КБД: +{written} дат" if written else "[INFO] КБД: новых дат нет")
    return written


def repair_zcyc(session: Optional[requests.Session] = None) -> int:
    """Докачивает даты торгового календаря внутри истории кривой, которых нет (рабочие субботы, сбои)."""
    if 'zcyc_params' not in lake.tables():
        return 0
    have = set(lake.query("SELECT date FROM lake.zcyc_params")['date'].to_list())
    lo, hi = min(have), max(have)
    skip = history.empty_dates('zcyc')
    missing = [d for d in history.trading_calendar() if lo <= d <= hi and d not in have and d not in skip]
    if not missing:
        return 0
    session = session or iss.make_session()
    frames, empty = {b: [] for b in ZCYC_TABLES}, []
    for day in missing:
        blocks = fetch_zcyc(day, session)
        if blocks['params'].is_empty():
            empty.append(day)   # ISS подтвержденно без кривой — больше не запрашивать
        for block, df in blocks.items():
            if df.height:
                frames[block].append(df)
    if empty:
        lake.write('empty_dates', pl.DataFrame({'dataset': ['zcyc'] * len(empty), 'date': empty}))
    n = 0
    for block, table in ZCYC_TABLES.items():
        if frames[block]:
            df = pl.concat(frames[block], how='diagonal_relaxed').unique(lake.TABLE_KEYS[table], keep='last')
            lake.write(table, df)
            n = df.height if block == 'params' else n
    logger.info(f"[OK] КБД: докачано дат {n} из {len(missing)} пропущенных")
    return n


def read_zcyc(kind: str = 'params', start=None, end=None, as_of: lake.AsOf = None) -> pl.DataFrame:
    """КБД из хранилища: kind — 'params', 'yields' (доходности по срокам) или 'bonds' (ОФЗ кривой)."""
    table = {'params': 'zcyc_params', 'yields': 'zcyc_yields', 'bonds': 'zcyc_bonds'}[kind]
    return _read(table, start, end, as_of)


def _read(table: str, start, end, as_of) -> pl.DataFrame:
    if table not in lake.tables():
        raise FileNotFoundError(f"В хранилище нет таблицы {table}")
    where, params = [], []
    if start is not None:
        where.append("date >= ?")
        params.append(_as_date(start))
    if end is not None:
        where.append("date <= ?")
        params.append(_as_date(end))
    sql = f"SELECT * FROM {lake.ref(table, as_of)}" + (" WHERE " + " AND ".join(where) if where else "")
    return lake.query(sql + " ORDER BY ALL", params)
