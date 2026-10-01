"""
Доступ к MOEX ISS: HTTP-сессия и разбор ответов в polars.

Ответ ISS — блоки вида {"metadata": {колонка: {"type": ...}}, "columns": [...],
"data": [[...], ...]}. Типы колонок берутся из metadata: числовые ISS-типы —
Float64, остальные — строки. Так одна колонка имеет один тип во все годы
истории (значения в ранних датах часто пустые, и угадывание типа по ним
давало разные типы в разных годах).
"""
from __future__ import annotations

from typing import Optional

import polars as pl
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ISS_URL = "https://iss.moex.com/iss"
# Таймаут запроса к ISS (connect, read), сек: без него зависший сокет
# останавливает весь прогон обновления навсегда
ISS_TIMEOUT = (10, 60)

_NUMERIC_TYPES = {'double', 'int32', 'int64', 'number', 'float'}


class _IssSession(requests.Session):
    """requests.Session с таймаутом по умолчанию и повторами на сетевых сбоях/5xx/429."""

    def __init__(self, timeout=ISS_TIMEOUT, retries: int = 3):
        super().__init__()
        self._timeout = timeout
        retry = Retry(total=retries, backoff_factor=1.0,
                      status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=frozenset(['GET']))
        adapter = HTTPAdapter(max_retries=retry)
        self.mount('https://', adapter)
        self.mount('http://', adapter)

    def request(self, method, url, **kwargs):
        kwargs.setdefault('timeout', self._timeout)
        return super().request(method, url, **kwargs)


def make_session() -> requests.Session:
    """HTTP-сессия для ISS MOEX: keep-alive, таймаут и повторы."""
    return _IssSession()


def _to_float(value):
    if value is None or value == '':
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _looks_numeric(values) -> bool:
    present = [v for v in values if v is not None and v != '']
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in present)


def to_frame(block: Optional[dict]) -> pl.DataFrame:
    """
    Блок ответа ISS -> polars DataFrame. Числовые колонки (по metadata, а без
    нее — если все непустые значения числа) — Float64, остальные — строки.
    """
    if not block:
        return pl.DataFrame()
    columns = block.get('columns') or []
    rows = block.get('data') or []
    meta = block.get('metadata') or {}
    series = []
    for j, col in enumerate(columns):
        values = [row[j] for row in rows]
        iss_type = (meta.get(col) or {}).get('type')
        numeric = iss_type in _NUMERIC_TYPES if iss_type else _looks_numeric(values)
        if numeric:
            series.append(pl.Series(col, [_to_float(v) for v in values], dtype=pl.Float64))
        else:
            series.append(pl.Series(col, [None if v is None else str(v) for v in values],
                                    dtype=pl.Utf8))
    return pl.DataFrame(series)


def records_to_frame(records: list[dict]) -> pl.DataFrame:
    """
    Список словарей (карточки бумаг) -> DataFrame: колонка, все непустые значения
    которой приводятся к числу, — Float64, иначе строки.
    """
    if not records:
        return pl.DataFrame()
    columns = list(dict.fromkeys(k for r in records for k in r))
    series = []
    for col in columns:
        values = [r.get(col) for r in records]
        floats = [_to_float(v) for v in values]
        present = [v for v in values if v is not None and v != '']
        if present and all(f is not None for f, v in zip(floats, values) if v is not None and v != ''):
            series.append(pl.Series(col, floats, dtype=pl.Float64))
        else:
            series.append(pl.Series(col, [None if v is None else str(v) for v in values],
                                    dtype=pl.Utf8))
    return pl.DataFrame(series)


def fetch_pages(url: str, block: str, params: dict, session: requests.Session,
                max_pages: int) -> pl.DataFrame:
    """Полная выдача блока ISS; неполная или поврежденная выдача вызывает ValueError.

    Без TOTAL запрашиваем страницы до подтвержденной пустой страницы. Если
    TOTAL был получен, он остается обязательной границей даже без следующего cursor.
    """
    if max_pages <= 0:
        raise ValueError("max_pages должен быть положительным")
    pages, offset, total = [], 0, None

    def integer(value):
        try:
            result = int(value)
            if isinstance(value, bool) or float(value) != result or result < 0:
                raise ValueError
            return result
        except (TypeError, ValueError, OverflowError) as e:
            raise ValueError(f"ISS: поврежден cursor блока {block}") from e

    for _ in range(max_pages):
        resp = session.get(url, params={**params, 'start': offset})
        resp.raise_for_status()
        data = resp.json()
        payload = data.get(block) if isinstance(data, dict) else None
        if (not isinstance(payload, dict) or not isinstance(payload.get('columns'), list)
                or not isinstance(payload.get('data'), list)
                or any(not isinstance(row, list) or len(row) != len(payload['columns'])
                       for row in payload['data'])
                or (payload['data'] and not payload['columns'])):
            raise ValueError(f"ISS: отсутствует или поврежден блок {block}")
        cursor = data.get(f'{block}.cursor')
        if cursor is not None:
            if (not isinstance(cursor, dict) or not isinstance(cursor.get('columns'), list)
                    or not isinstance(cursor.get('data'), list)
                    or len(cursor['data']) > 1
                    or any(not isinstance(row, list) or len(row) != len(cursor['columns'])
                           for row in cursor['data'])):
                raise ValueError(f"ISS: поврежден cursor блока {block}")
            if cursor['data']:
                fields = dict(zip(cursor['columns'], cursor['data'][0]))
                if 'INDEX' in fields and integer(fields['INDEX']) != offset:
                    raise ValueError(f"ISS: неверное смещение страницы {block}: ожидалось {offset}")
                if 'TOTAL' in fields:
                    count = integer(fields['TOTAL'])
                    if total is not None and count != total:
                        raise ValueError(f"ISS: TOTAL изменился при загрузке {block}: {total} -> {count}")
                    total = count
        page = to_frame(payload)
        if page.is_empty():
            if total is not None and offset != total:
                raise ValueError(f"ISS: неполная выдача {block}: {offset}/{total}")
            break
        pages.append(page)
        offset += page.height
        if total is not None:
            if offset > total:
                raise ValueError(f"ISS: число строк {block} превышает TOTAL: {offset}/{total}")
            if offset == total:
                break
    else:
        raise ValueError(f"ISS: превышен лимит страниц {block}; выдача неполная")
    return pl.concat(pages, how='diagonal_relaxed') if pages else pl.DataFrame()


def history_day(market_path: str, date, session: requests.Session,
                max_pages: int = 1000) -> pl.DataFrame:
    """
    История торгов всех инструментов рынка за дату со всеми страницами
    (ISS отдает по 100 строк). market_path — 'stock/markets/bonds',
    'futures/markets/forts' и т.п. TRADEDATE превращается в колонку date (Date).
    """
    url = f"{ISS_URL}/history/engines/{market_path}/securities.json"
    day = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
    df = fetch_pages(url, 'history', {'date': day}, session, max_pages)
    if df.is_empty():
        return df
    return (df.with_columns(pl.col('TRADEDATE').str.to_date('%Y-%m-%d').alias('date'))
              .drop('TRADEDATE')
              .select(['date', *[c for c in df.columns if c != 'TRADEDATE']]))


def security_history(market_path: str, secid: str, start, end=None,
                     session: Optional[requests.Session] = None,
                     columns: Optional[list[str]] = None, max_pages: int = 2000) -> pl.DataFrame:
    """
    История торгов одной бумаги за период (все режимы торгов, все страницы).
    market_path — 'stock/markets/shares', 'stock/markets/index' и т.п.;
    columns — поля ISS (меньше трафика). TRADEDATE -> колонка date (Date).
    """
    session = session or make_session()
    url = f"{ISS_URL}/history/engines/{market_path}/securities/{secid}.json"
    params = {'from': _day(start), 'till': _day(end) if end is not None else _day(_today())}
    if columns:
        params['history.columns'] = ','.join(dict.fromkeys(['TRADEDATE', *columns]))
    df = fetch_pages(url, 'history', params, session, max_pages)
    if df.is_empty():
        return df
    return (df.with_columns(pl.col('TRADEDATE').str.to_date('%Y-%m-%d').alias('date'))
              .drop('TRADEDATE'))


def _day(value) -> str:
    return value.strftime('%Y-%m-%d') if hasattr(value, 'strftime') else str(value)[:10]


def _today():
    import datetime as _dt
    return _dt.date.today()


def security_description(secid: str, session: Optional[requests.Session] = None) -> dict:
    """
    Карточка бумаги ISS (/iss/securities/<SECID>, блок description) как словарь
    {поле: значение}. Отдается и для погашенных и снятых с торгов бумаг.
    Пустой словарь — ISS бумагу не знает.
    """
    if session is None:
        session = make_session()
    resp = session.get(f"{ISS_URL}/securities/{secid}.json", params={'iss.only': 'description'})
    resp.raise_for_status()
    desc = to_frame(resp.json().get('description'))
    if desc.is_empty() or 'name' not in desc.columns:
        return {}
    return dict(zip(desc['name'].to_list(), desc['value'].to_list()))
