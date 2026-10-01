"""
Проверка качества данных: акции, индексы, рынки history.DATASETS и свежесть
ставок, кривой, параметров бумаг, составов индексов и денежных потоков.

data_quality_report() — замечания (check, object, detail) по торговому календарю
IMOEX; quality_summary() — одна строка итога для лога. Шаг 3 update_data.py; история прогонов
(update_runs, quality_log) — в конце файла.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Optional

import numpy as np
import polars as pl
import requests

from moexutils import cashflows, history, indices, rates, refdata
from moexutils import lake
from moexutils import stocks

logger = logging.getLogger("moexutils")

ISSUE_SCHEMA = {'check': pl.Utf8, 'object': pl.Utf8, 'detail': pl.Utf8}


def _returns(values: np.ndarray) -> np.ndarray:
    """Доходность к предыдущему значению (первое — NaN)."""
    out = np.full(len(values), np.nan)
    if len(values) > 1:
        out[1:] = values[1:] / values[:-1] - 1
    return out


def trading_calendar() -> list[dt.date]:
    """Торговый календарь: даты, по которым есть IMOEX в хранилище (включая рабочие субботы)."""
    return history.trading_calendar()


def find_dividend_gap_candidates(df: pl.DataFrame, div_folder: Optional[str] = None, since=None,
                                 min_gap: float = 0.04, market_returns: Optional[dict] = None,
                                 window_days: int = 5, splits: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    """
    Кандидаты в пропущенные дивиденды одной бумаги: гэп открытия
    (open_t / close_{t-1} − 1) хуже −min_gap, не объясненный рынком (гэп минус
    дневное изменение IMOEX тоже хуже −min_gap), сплитом из реестра или
    дивидендом из CSV в пределах window_days от экс-даты. Крупные новостные
    падения тоже попадут — это кандидаты для ручной проверки, а не факты.

    df — date, ticker, open, close; market_returns — {дата: доходность IMOEX}.
    Returns: date, gap, market.
    """
    empty = pl.DataFrame(schema={'date': pl.Date, 'gap': pl.Float64, 'market': pl.Float64})
    if df.is_empty() or not {'ticker', 'open', 'close'}.issubset(df.columns):
        return empty
    splits = stocks.load_splits() if splits is None else splits
    ticker = df['ticker'][0]
    px = stocks.adjust_for_splits(df.select('date', 'ticker', 'open', 'close')
                                  .drop_nulls(['open', 'close']).sort('date'), splits)
    dates = px['date'].to_list()
    opens, closes = px['open'].to_numpy(), px['close'].to_numpy()
    gap = np.full(len(dates), np.nan)
    if len(dates) > 1:
        gap[1:] = opens[1:] / closes[:-1] - 1
    if market_returns is None:
        idx = stocks.read_index('IMOEX')
        market_returns = dict(zip(idx['date'].to_list(), _returns(idx['close'].to_numpy()).tolist()))
    market = np.array([market_returns.get(d, 0.0) for d in dates], dtype=float)
    market = np.where(np.isnan(market), 0.0, market)
    since = stocks._as_date(since) if since is not None else None
    hits = [i for i in range(len(dates))
            if gap[i] < -min_gap and gap[i] - market[i] < -min_gap and (since is None or dates[i] >= since)]
    if not hits:
        return empty

    # Объясненные даты: экс-даты дивидендов из CSV и даты сплитов из реестра
    explained = []
    for rec in stocks.load_dividends(ticker, div_folder)['closing_date'].to_list():
        pos = stocks.ex_dividend_pos(dates, rec)
        explained.append(dates[pos] if 0 <= pos < len(dates) else rec)
    explained += splits.filter(pl.col('ticker') == ticker)['date'].to_list()

    rows = [{'date': dates[i], 'gap': float(gap[i]), 'market': float(market[i])} for i in hits
            if not any(abs((dates[i] - e).days) <= window_days for e in explained)]
    return pl.DataFrame(rows, schema=empty.schema) if rows else empty


def freshness_report(today=None) -> pl.DataFrame:
    """Офлайн-проверка свежести дополнительных наборов; today — дата проверки.

    Отсутствующий целиком набор пропускается. Для наборов с редкими изменениями
    проверяется load_state, для выплат — успешные ежедневные и еженедельные выгрузки.
    """
    day = stocks._as_date(today) if today is not None else dt.date.today()
    return _freshness_report(trading_calendar(), day)


def _freshness_report(calendar: list[dt.date], today: dt.date) -> pl.DataFrame:
    rows = []

    def add(check, obj, detail):
        rows.append({'check': check, 'object': obj, 'detail': detail})

    def stale(check, obj, last, expected):
        if expected is None:
            return
        if last is None:
            add(check, obj, f"нет данных или отметки успешного обновления; ожидается дата не раньше {expected:%Y-%m-%d}")
        elif last < expected:
            add(check, obj, f"последняя дата {last:%Y-%m-%d}, ожидается не раньше {expected:%Y-%m-%d}")

    names = set(lake.tables())
    state = {}
    if 'load_state' in names:
        state = dict(lake.query('SELECT name, date FROM lake.load_state').iter_rows())
    # Данные текущего дня еще могут быть промежуточными. Календарь IMOEX —
    # ориентир, а не независимый календарь рабочих дней Банка России.
    cal = sorted({d for d in calendar if d < today})
    weekdays = [d for d in cal if d.weekday() < 5]
    latest = cal[-1] if cal else None
    if {'options_series', 'options_contracts'} & names:
        stale('options_registry_stale', 'options_registry', state.get('options_registry'), latest)
        if {'options', 'options_contracts'} <= names:
            missing = lake.query('SELECT count(*) AS n FROM lake.options h '
                                 'ANTI JOIN lake.options_contracts c ON h.SECID=c.secid '
                                 'AND h.date BETWEEN c.history_from AND c.history_till')['n'][0]
            if missing:
                add('options_registry_gaps', 'options_registry',
                    f'{missing} строк истории без параметров контракта на эту дату')
    if 'ruonia' in names:
        last = lake.query('SELECT max(date) AS d FROM lake.ruonia')['d'][0]
        # Ночной прогон: ставка за предыдущий рабочий день может еще не выйти.
        expected = weekdays[-2] if len(weekdays) >= 2 else None
        stale('ruonia_stale', 'RUONIA', last, expected)

    if set(rates.ZCYC_TABLES.values()) & names:
        skip = history.empty_dates('zcyc')
        expected_days = [d for d in cal if d >= rates.ZCYC_START and d not in skip]
        expected = expected_days[-1] if expected_days else None
        for table in rates.ZCYC_TABLES.values():
            last = lake.query(f'SELECT max(date) AS d FROM lake.{table}')['d'][0] if table in names else None
            stale('zcyc_stale', table, last, expected)

    if refdata.TABLE in names or refdata.TABLE in state:
        # Загрузчик параметров опрашивает только будни и хранит лишь изменения.
        expected_days = [d for d in weekdays if d >= refdata.START]
        expected = expected_days[-1] if expected_days else None
        stale('refdata_stale', refdata.TABLE, state.get(refdata.TABLE), expected)

    if indices.TABLE in names or any(n.startswith(indices.TABLE + ':') for n in state):
        for indexid in indices.CORE_INDEXES:
            stale('index_weights_stale', indexid, state.get(indices.TABLE + ':' + indexid), latest)

    prefixes = (cashflows.UPDATE_STATE_PREFIX, cashflows.FUTURE_STATE_PREFIX)
    if any(b.table in names for b in cashflows.BLOCKS.values()) or any(n.startswith(prefixes) for n in state):
        # Планировщик запускает обновление вт–сб; полный будущий горизонт — по субботам.
        daily = next(today - dt.timedelta(days=i) for i in range(7)
                     if (today - dt.timedelta(days=i)).weekday() in (1, 2, 3, 4, 5))
        weekly = today - dt.timedelta(days=(today.weekday() - 5) % 7)
        for b in cashflows.BLOCKS.values():
            stale('cashflows_stale', b.table, state.get(cashflows.UPDATE_STATE_PREFIX + b.table), daily)
            stale('cashflows_future_stale', b.table,
                  state.get(cashflows.FUTURE_STATE_PREFIX + b.table), weekly)
    return pl.DataFrame(rows, schema=ISSUE_SCHEMA)


def data_quality_report(days: Optional[int] = 30, div_folder: Optional[str] = None,
                        div_days: Optional[int] = 120, adj_jump: float = 0.25,
                        check_iss: bool = False,
                        session: Optional[requests.Session] = None) -> pl.DataFrame:
    """
    Проверка данных после обновления. Окно — последние `days` торговых дней
    (None — вся история), для дивидендов — `div_days`.

    Проверки (колонка check): index_stale, stock_stale (20+ торговых дней без
    данных — кандидат в metadata/delisted.csv; с check_iss — со статусом ISS),
    stock_gaps, adj_missing, adj_jump (артефакт корректировки: изменение adj_close
    расходится с ценой), price_spike (скачок с разворотом), dividend_skipped,
    dividend_gap (см. find_dividend_gap_candidates), bonds_stale/bonds_gaps,
    futures_stale/futures_gaps; ruonia_stale, zcyc_stale, refdata_stale,
    index_weights_stale, cashflows_stale и cashflows_future_stale.

    Returns: DataFrame (check, object, detail); пустой — замечаний нет.
    """
    issues = []

    def add(check, obj, detail):
        issues.append({'check': check, 'object': obj, 'detail': detail})

    cal = trading_calendar()
    issues.extend(_freshness_report(cal, dt.date.today()).to_dicts())
    if not cal:
        add('calendar', 'IMOEX', 'нет истории IMOEX в хранилище — проверки по календарю невозможны')
        return pl.DataFrame(issues, schema=ISSUE_SCHEMA)
    last_day = cal[-1]
    win_start = cal[-days] if days is not None and len(cal) >= days else cal[0]
    div_start = cal[-div_days] if div_days is not None and len(cal) >= div_days else cal[0]
    cal_arr = np.array(cal, dtype='datetime64[D]')

    imoex = stocks.read_index('IMOEX')
    market_returns = dict(zip(imoex['date'].to_list(), _returns(imoex['close'].to_numpy()).tolist()))

    idx_last = lake.query("SELECT ticker, max(date) AS last FROM lake.indexes GROUP BY ticker")
    for ticker, last in idx_last.iter_rows():
        if ticker != 'IMOEX' and last < last_day:
            add('index_stale', ticker, f"последняя дата {last:%Y-%m-%d}, IMOEX — {last_day:%Y-%m-%d}")

    delisted = set(stocks.load_delisted()['ticker'].to_list())
    splits = stocks.load_splits()
    div_folder = div_folder or stocks.DIVIDENDS_FOLDER
    import os
    has_divs = os.path.isdir(div_folder)
    if not has_divs:
        add('dividends', div_folder, 'папка дивидендов не найдена — проверки дивидендов пропущены')

    data = stocks.read_stocks(merge_renames=False)
    for df in data.partition_by('ticker', maintain_order=True):
        t = df['ticker'][0]
        if t in delisted or df.is_empty():
            continue
        df = df.sort('date')
        dates = df['date'].to_list()
        last = dates[-1]
        lag = int((cal_arr > np.datetime64(last, 'D')).sum())
        if lag >= 20:
            detail = (f"нет данных {lag} торговых дней (последняя дата {last:%Y-%m-%d}); "
                      f"если бумага снята с торгов — внесите в metadata/delisted.csv")
            if check_iss:
                try:
                    detail += f"; ISS is_traded={stocks.is_traded(t, session=session)}"
                except Exception as e:
                    detail += f"; ISS недоступен ({e})"
            add('stock_stale', t, detail)
            continue
        if lag > 0:
            add('stock_stale', t, f"отстает на {lag} торг. дн. (последняя дата {last:%Y-%m-%d})")

        have = set(dates)
        missing = [d for d in cal if max(win_start, dates[0]) <= d <= last and d not in have]
        if missing:
            add('stock_gaps', t, f"{len(missing)} пропущенных торговых дат в окне, первая {missing[0]:%Y-%m-%d}")

        w = df.filter(pl.col('date') >= win_start)
        for col in ('adj_close', 'market_cap'):
            n_null = w[col].null_count() if col in w.columns else 0
            if n_null:
                add('adj_missing', t, f"{col}: {n_null} пустых значений в окне")

        # Сильные движения самой цены — рынок, а не ошибка; ошибкой считаем
        # (а) скачок adj_close, которого нет в сплит-скорректированной цене —
        # артефакт корректировки, (б) скачок цены с разворотом на следующий день
        px_ret = _returns(stocks.adjust_for_splits(df.select('date', 'ticker', 'close'), splits)['close']
                          .to_numpy().astype(float))
        in_win = np.array([d >= win_start for d in dates])
        if df['adj_close'].null_count() < df.height:
            adj_ret = _returns(df['adj_close'].to_numpy().astype(float))
            with np.errstate(invalid='ignore'):
                bad = (np.abs(adj_ret) > adj_jump) & (np.abs(adj_ret - px_ret) > 0.05) & in_win
            for i in np.flatnonzero(bad):
                add('adj_jump', t, f"{dates[i]:%Y-%m-%d}: adj_close {adj_ret[i]:+.1%} за день "
                                   f"при цене {px_ret[i]:+.1%} — артефакт корректировки")
        nxt = np.append(px_ret[1:], np.nan)
        with np.errstate(invalid='ignore'):
            spike = ((np.abs(px_ret) > adj_jump) & (nxt * px_ret < 0)
                     & (np.abs(nxt) > 0.8 * np.abs(px_ret) / (1 + px_ret)) & in_win)
        for i in np.flatnonzero(spike):
            add('price_spike', t, f"{dates[i]:%Y-%m-%d}: цена {px_ret[i]:+.1%} и {nxt[i]:+.1%} на "
                                  f"следующий день — возможна сбойная цена")

        if has_divs:
            _, skipped = stocks.adj_close(df.select('date', 'ticker', 'close'),
                                          stocks.load_dividends(t, div_folder), splits)
            for d, v in skipped:
                if d >= div_start:
                    add('dividend_skipped', t, f"{d:%Y-%m-%d}: дивиденд {v} не согласуется с ценой — пропущен")
            cands = find_dividend_gap_candidates(df, div_folder, since=div_start,
                                                 market_returns=market_returns, splits=splits)
            # Разворот после сбойной цены выглядит как гэп — это не дивиденд
            reversal_days = {dates[i + 1] for i in np.flatnonzero(spike) if i + 1 < len(dates)}
            for c in cands.iter_rows(named=True):
                if c['date'] in reversal_days:
                    continue
                add('dividend_gap', t, f"{c['date']:%Y-%m-%d}: гэп открытия {c['gap']:+.1%} при IMOEX "
                                       f"{c['market']:+.1%}, дивиденда в {t}.csv рядом нет")

    # Гэпы у многих бумаг в один день — скорее отраслевое/рыночное движение
    gap_days = [i['detail'][:10] for i in issues if i['check'] == 'dividend_gap']
    for i in issues:
        if i['check'] == 'dividend_gap':
            n = gap_days.count(i['detail'][:10])
            if n >= 3:
                i['detail'] += f" (гэп у {n} бумаг в этот день — возможно отраслевое движение)"

    # все рынки «инструменты за дату»; набора нет в хранилище — пропуск
    for dataset, label in ((k, v.label) for k, v in history.DATASETS.items()):
        try:
            have = history.dataset_dates(dataset)
            skip = history.empty_dates(dataset)
        except Exception as e:
            add(f'{dataset}_stale', label, f"хранилище недоступно — {e}")
            continue
        if not have:
            continue
        if have[-1] < last_day:
            add(f'{dataset}_stale', label, f"последняя дата {have[-1]:%Y-%m-%d}, IMOEX — {last_day:%Y-%m-%d}")
        hs = set(have)
        missing = [d for d in cal if max(win_start, have[0]) <= d <= have[-1] and d not in hs and d not in skip]
        if missing:
            add(f'{dataset}_gaps', label, f"{len(missing)} пропущенных торговых дат в окне, первая {missing[0]:%Y-%m-%d}")

    return pl.DataFrame(issues, schema=ISSUE_SCHEMA)


def quality_summary(issues: pl.DataFrame) -> str:
    """Одна строка для лога: итог data_quality_report."""
    if issues is None or issues.is_empty():
        return "Проверка данных: замечаний нет"
    counts = issues['check'].value_counts(sort=True)
    return (f"Проверка данных: замечаний {issues.height} ("
            + ", ".join(f"{k}: {v}" for k, v in counts.iter_rows()) + ")")


# ---------------------------------------------------------------- история прогонов
#
# lake.update_runs — итог каждого прогона update_data.py (режим, статус, число
# сбоев шагов и замечаний); lake.quality_log — замечания проверки по прогонам.
# По ним видно, когда проблема появилась, а оповещение идет только о новых.

RUN_SCHEMA = {'run_id': pl.Datetime('us'), 'finished': pl.Datetime('us'), 'mode': pl.Utf8,
              'status': pl.Utf8, 'warnings': pl.Int64, 'issues': pl.Int64,
              'new_issues': pl.Int64, 'messages': pl.Utf8}


def previous_issues(mode: str, before: dt.datetime) -> Optional[pl.DataFrame]:
    """Замечания (check, object) последнего прогона режима mode до before; None — прогонов не было."""
    if 'update_runs' not in lake.tables():
        return None
    last = lake.query("SELECT max(run_id) AS r FROM lake.update_runs WHERE mode = ? AND run_id < ?",
                      [mode, before])['r'][0]
    if last is None:
        return None
    if 'quality_log' not in lake.tables():
        return pl.DataFrame(schema={'check': pl.Utf8, 'object': pl.Utf8})
    return lake.query('SELECT DISTINCT "check", object FROM lake.quality_log WHERE run_id = ?', [last])


def new_issues(issues: pl.DataFrame, previous: Optional[pl.DataFrame]) -> pl.DataFrame:
    """
    Замечания, которых не было в прошлом прогоне. Сравнение по (check, object):
    detail меняется день ото дня (даты, проценты) у одной и той же проблемы.
    Прошлого прогона нет — новые все.
    """
    if previous is None or issues.is_empty():
        return issues
    return issues.join(previous.select('check', 'object').unique(), on=['check', 'object'], how='anti')


def record_run(run_id: dt.datetime, mode: str, warnings: list[str],
               issues: Optional[pl.DataFrame], n_new: int) -> None:
    """Запись итога прогона и его замечаний в хранилище."""
    n_issues = 0 if issues is None else issues.height
    status = 'error' if warnings else ('issues' if n_issues else 'ok')
    lake.write('update_runs', pl.DataFrame([{
        'run_id': run_id, 'finished': dt.datetime.now(), 'mode': mode, 'status': status,
        'warnings': len(warnings), 'issues': n_issues, 'new_issues': n_new,
        'messages': "\n".join(warnings) or None}], schema=RUN_SCHEMA))
    if n_issues:
        lake.write('quality_log', issues.unique(maintain_order=True).with_columns(
            run_id=pl.lit(run_id, dtype=pl.Datetime('us')), mode=pl.lit(mode)
        ).select('run_id', 'mode', 'check', 'object', 'detail'))
