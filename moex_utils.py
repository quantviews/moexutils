"""
moexutils — данные Московской биржи: акции, индексы, облигации, фьючерсы.

Фасад прежнего интерфейса. Реализация — в модулях:
- stocks.py   — акции и индексы в хранилище, корпоративные события, adj_close,
                капитализация, ключевая ставка (polars);
- history.py  — история рынков «все инструменты за дату»: облигации, фьючерсы;
- quality.py  — проверка качества данных;
- lake.py     — хранилище DuckLake (каталог PostgreSQL, результаты — polars);
- iss.py      — доступ к MOEX ISS.

Все функции возвращают polars DataFrame; pandas в проекте не используется.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import polars as pl
import requests

import history
import iss
import quality
import stocks

# Пути привязаны к папке модуля, чтобы импорт из marimo/ и scripts/ работал при любом cwd
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Корень рыночных данных; реестры metadata/ остаются в проекте и в git
DATA_ROOT = os.environ.get("MOEX_DATA_ROOT") or BASE_DIR
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
read_stocks = stocks.read_stocks
read_index = stocks.read_index
adjust_for_splits = stocks.adjust_for_splits
load_renames = stocks.load_renames
apply_renames = stocks.apply_renames
load_key_rate = stocks.load_key_rate
risk_free_monthly = stocks.risk_free_monthly


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
