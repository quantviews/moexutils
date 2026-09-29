"""
moexutils — данные Московской биржи: акции, индексы, облигации, фьючерсы.

Фасад прежнего интерфейса. Реализация — в модулях:
- stocks.py   — акции и индексы в хранилище, корпоративные события, adj_close,
                капитализация, ключевая ставка (polars);
- history.py  — история рынков «все инструменты за дату»: облигации, фьючерсы;
- quality.py  — проверка качества данных;
- lake.py     — хранилище DuckLake (каталог PostgreSQL, результаты — polars);
- iss.py      — доступ к MOEX ISS.

Функции облигаций, фьючерсов, обновления и проверки возвращают polars.
Функции чтения акций и индексов (read_moex_stock, combine_moex_stocks,
read_moex_index, adjust_for_splits, apply_renames, load_renames,
risk_free_monthly) пока возвращают pandas — их используют marimo-ноутбуки;
после перевода ноутбуков на polars обертки будут удалены. Новый код —
через stocks.read_stocks / stocks.read_index (polars).
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import pandas as pd
import polars as pl
import requests

import history
import iss
import quality
import stocks

# Пути привязаны к папке модуля, чтобы импорт из nb/ и scripts/ работал при любом cwd
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Корень рыночных данных; реестры metadata/ остаются в проекте и в git
DATA_ROOT = os.environ.get("MOEX_DATA_ROOT") or BASE_DIR
# Прежние папки Parquet-файлов акций и индексов: заморожены на момент перехода
# на хранилище (ноутбуки перебирают по ним тикеры); данные — в lake.stocks/indexes
DATA_FOLDER = os.path.join(DATA_ROOT, "data")
INDEXES_FOLDER = os.path.join(DATA_ROOT, "indexes")
METADATA_FILE = stocks.METADATA_FILE
SPLITS_FILE = stocks.SPLITS_FILE
RENAMES_FILE = stocks.RENAMES_FILE
KEY_RATE_FILE = stocks.KEY_RATE_FILE
EXTERNAL_SPLITS_FILE = stocks.EXTERNAL_SPLITS_FILE
DELISTED_FILE = stocks.DELISTED_FILE
DIVIDENDS_FOLDER = stocks.DIVIDENDS_FOLDER

logger = logging.getLogger("moex_utils")
# Если логирование в приложении не настроено — сообщения в stdout (прогресс в
# ноутбуках и update_data.bat). Любая внешняя настройка logging имеет приоритет.
if not logger.handlers and not logging.getLogger().handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# HTTP-сессия ISS (таймаут, повторы) — в iss.py
ISS_TIMEOUT = iss.ISS_TIMEOUT
make_session = iss.make_session


# ---------------------------------------------------------------- акции и индексы (polars)

update_stocks = stocks.update_stocks
recompute_stocks = stocks.recompute_stocks
add_stock = stocks.add_stock
update_indexes = stocks.update_indexes
update_key_rate = stocks.update_key_rate
load_delisted = stocks.load_delisted
load_splits = stocks.load_splits
list_tickers = stocks.list_tickers


# ---------------------------------------------------------------- pandas-обертки для ноутбуков
#
# Временные: marimo-ноутбуки пока работают с pandas. Данные читаются из
# хранилища (polars) и отдаются в прежнем виде — индекс date (DatetimeIndex).

def _to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    pdf = df.to_pandas()
    if 'date' in pdf.columns:
        pdf['date'] = pd.to_datetime(pdf['date']).astype('datetime64[ns]')
        pdf = pdf.set_index('date')
    return pdf


def _from_pandas(df: pd.DataFrame) -> pl.DataFrame:
    frame = df.reset_index() if df.index.name == 'date' else df.copy()
    out = pl.from_pandas(frame)
    return out.with_columns(pl.col('date').cast(pl.Date)) if 'date' in out.columns else out


def read_moex_stock(ticker: str, start=None, end=None, session=None) -> pd.DataFrame:
    """Дневные данные тикера (pandas, индекс date); нет в хранилище — загружается с MOEX."""
    ticker = ticker.upper()
    df = stocks.read_stocks(ticker, start=start, end=end, merge_renames=False)
    if df.is_empty():
        stocks.add_stock(ticker, session=session)
        df = stocks.read_stocks(ticker, start=start, end=end, merge_renames=False)
    return _to_pandas(df)


def combine_moex_stocks(data_folder: Optional[str] = None, merge_renames: bool = True) -> pd.DataFrame:
    """Все акции одним DataFrame (pandas, индекс date); merge_renames — склейка переименований."""
    return _to_pandas(stocks.read_stocks(merge_renames=merge_renames))


def adjust_for_splits(df: pd.DataFrame, splits_file: Optional[str] = None) -> pd.DataFrame:
    """Цены в пост-сплитовой базе (pandas-обертка над stocks.adjust_for_splits)."""
    if df.empty or 'ticker' not in df.columns:
        return df
    out = stocks.adjust_for_splits(_from_pandas(df), stocks.load_splits(splits_file))
    return _to_pandas(out)


def load_renames(renames_file: Optional[str] = None) -> pd.DataFrame:
    """Реестр переименований (pandas)."""
    out = stocks.load_renames(renames_file).to_pandas()
    out['date'] = pd.to_datetime(out['date'])
    return out


def apply_renames(df: pd.DataFrame, renames_file: Optional[str] = None) -> pd.DataFrame:
    """Склейка историй переименованных тикеров (pandas-обертка над stocks.apply_renames)."""
    if df.empty or 'ticker' not in df.columns:
        return df
    return _to_pandas(stocks.apply_renames(_from_pandas(df), stocks.load_renames(renames_file)))


def read_moex_index(ticker: str = 'IMOEX') -> pd.DataFrame:
    """История индекса (pandas, индекс date): close, value_rub (оборот), volume."""
    df = stocks.read_index(ticker)
    if df.is_empty():
        raise FileNotFoundError(f"Индекса {ticker} нет в хранилище: stocks.update_indexes(['{ticker}'])")
    return _to_pandas(df)


def load_key_rate(key_rate_file: Optional[str] = None) -> pd.DataFrame:
    """История ключевой ставки ЦБ (pandas)."""
    out = stocks.load_key_rate(key_rate_file).to_pandas()
    out['date'] = pd.to_datetime(out['date'])
    return out


def risk_free_monthly(dates, key_rate_file: Optional[str] = None) -> pd.Series:
    """Месячная безрисковая ставка (в долях) на даты — pandas Series с индексом дат."""
    idx = pd.DatetimeIndex(dates)
    rf = stocks.risk_free_monthly(idx.date, key_rate_file)
    return pd.Series(rf['rf'].to_numpy(), index=idx)


# ---------------------------------------------------------------- проверка качества

data_quality_report = quality.data_quality_report
quality_summary = quality.quality_summary
find_dividend_gap_candidates = quality.find_dividend_gap_candidates


# ---------------------------------------------------------------- облигации и фьючерсы
#
# История «все инструменты за дату» (весь рынок облигаций с 1997 года, все
# фьючерсы FORTS с 2002 года) хранится в DuckLake — см. history.py и lake.py;
# функции ниже сохраняют прежние имена и возвращают polars DataFrame.

get_security_description = iss.security_description


def update_bonds_market(start: Optional[str] = None, session: Optional[requests.Session] = None,
                        max_days: int = 3000) -> int:
    """
    Докачивает историю ВСЕХ облигаций MOEX (все доски, все поля ISS) в
    lake.bonds: хвост после последней даты; если start раньше истории —
    начало (назад от истории). Первичная выгрузка — start='1997-01-01'.
    """
    return history.update('bonds', start=start, max_days=max_days, session=session)


def repair_bonds_market(session: Optional[requests.Session] = None, calendar=None) -> int:
    """Докачивает пропущенные торговые даты внутри истории облигаций."""
    return history.repair('bonds', session=session, calendar=calendar)


def update_bonds_market_all(session: Optional[requests.Session] = None) -> None:
    """Ночное обновление облигаций: докачка хвоста и пропусков."""
    update_bonds_market(session=session)
    repair_bonds_market(session=session)


def read_bonds_market(start=None, end=None, boards=None, secids=None,
                      columns: Optional[list] = None) -> pl.DataFrame:
    """
    История облигаций из хранилища (polars): ключ date + SECID + BOARDID.
    boards — режимы торгов ('TQOB' — гособлигации, 'TQCB' — корпоративные, ...),
    columns — нужные колонки (быстрее на полной истории).
    """
    return history.read('bonds', start=start, end=end, secids=secids, boards=boards, columns=columns)


def update_bonds_securities(max_new: Optional[int] = 500,
                            session: Optional[requests.Session] = None) -> int:
    """Дополняет реестр карточек облигаций lake.bonds_securities (включая погашенные)."""
    return history.update_securities('bonds', max_new=max_new, session=session)


def read_bonds_securities() -> pl.DataFrame:
    """Реестр карточек облигаций: строка на SECID (ISIN, эмитент, даты, объем, тип, дефолты)."""
    return history.read_securities('bonds')


def update_futures_history(start: Optional[str] = None, session: Optional[requests.Session] = None,
                           max_days: int = 3000) -> int:
    """
    Докачивает историю ВСЕХ фьючерсных контрактов FORTS в lake.futures.
    Коды контрактов (SiZ5) повторяются раз в 10 лет — год в SHORTNAME (Si-12.25).
    Первичная выгрузка — start='2002-01-01'.
    """
    return history.update('futures', start=start, max_days=max_days, session=session)


def repair_futures_history(session: Optional[requests.Session] = None, calendar=None) -> int:
    """Докачивает пропущенные торговые даты внутри истории фьючерсов."""
    return history.repair('futures', session=session, calendar=calendar)


def read_futures_history(start=None, end=None, assets=None,
                         columns: Optional[list] = None) -> pl.DataFrame:
    """История фьючерсов (polars); assets — базовый актив или список (ASSETCODE: 'Si', 'RTS', ...)."""
    df = history.read('futures', start=start, end=end, columns=columns)
    if assets is not None:
        assets = [assets] if isinstance(assets, str) else list(assets)
        df = df.filter(pl.col('ASSETCODE').is_in(assets))
    return df


def calculate_ytm(price: float, face_value: float, coupon_rate: float, years_to_maturity: float, coupon_freq: int = 2) -> float:
    """
    Calculates Yield to Maturity (YTM) for a bond.
    
    Parameters:
    price (float): Current price (% of face value).
    face_value (float): Face value.
    coupon_rate (float): Annual coupon rate (%).
    years_to_maturity (float): Years to maturity.
    coupon_freq (int): Coupons per year.
    
    Returns:
    float: YTM (%).
    """
    coupon = face_value * (coupon_rate / 100) / coupon_freq
    periods = int(years_to_maturity * coupon_freq)

    if periods == 0:
        return 0

    target = price / 100 * face_value

    def _pv(annual_rate: float) -> float:
        r = annual_rate / coupon_freq
        pv_coupons = sum(coupon / (1 + r) ** i for i in range(1, periods + 1))
        return pv_coupons + face_value / (1 + r) ** periods

    # Бисекция: PV монотонно убывает по ставке, ищем ставку в [-50%, 500%]
    lo, hi = -0.5, 5.0
    if target >= _pv(lo):
        return lo * 100
    if target <= _pv(hi):
        return hi * 100

    for _ in range(200):
        mid = (lo + hi) / 2
        if _pv(mid) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break

    return (lo + hi) / 2 * 100

def calculate_duration(price: float, face_value: float, coupon_rate: float, years_to_maturity: float, ytm: float, coupon_freq: int = 2) -> float:
    """
    Calculates modified duration.
    
    Returns:
    float: Duration in years.
    """
    coupon = face_value * (coupon_rate / 100) / coupon_freq
    periods = int(years_to_maturity * coupon_freq)
    ytm_period = ytm / 100 / coupon_freq
    
    if periods == 0:
        return 0
    
    pv_coupons = sum((i * coupon) / (1 + ytm_period)**i for i in range(1, periods+1))
    pv_face = (periods * face_value) / (1 + ytm_period)**periods
    total_pv_weighted = pv_coupons + pv_face
    
    total_pv = sum(coupon / (1 + ytm_period)**i for i in range(1, periods+1)) + face_value / (1 + ytm_period)**periods
    
    macaulay_duration = total_pv_weighted / total_pv / coupon_freq
    modified_duration = macaulay_duration / (1 + ytm_period)
    
    return modified_duration

def calculate_convexity(price: float, face_value: float, coupon_rate: float,
                        years_to_maturity: float, ytm: float, coupon_freq: int = 2) -> float:
    """
    Модифицированная выпуклость облигации (в годах²).

    Вторая производная цены по ставке: dP/P ≈ -D·dy + 0.5·C·dy².
    Параметр price не используется (PV восстанавливается из ytm) —
    сигнатура симметрична calculate_duration.
    """
    coupon = face_value * (coupon_rate / 100) / coupon_freq
    periods = int(years_to_maturity * coupon_freq)
    y = ytm / 100 / coupon_freq

    if periods == 0:
        return 0.0

    cash_flows = [(i, coupon + (face_value if i == periods else 0.0))
                  for i in range(1, periods + 1)]
    pv = sum(cf / (1 + y) ** i for i, cf in cash_flows)
    if pv <= 0:
        return float('nan')
    weighted = sum(cf * i * (i + 1) / (1 + y) ** i for i, cf in cash_flows)
    return weighted / (pv * (coupon_freq ** 2) * (1 + y) ** 2)
