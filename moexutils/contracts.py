"""
Фьючерсные контракты FORTS: реестр, коды контрактов, непрерывные ряды.

Реестр lake.futures_contracts — все контракты с 2001 года одним запросом ISS
(/iss/statistics/engines/futures/markets/forts/series?show_expired=1): secid,
name, start_date, expiration_date, asset_code, underlying_asset, is_traded,
base_secid. Ключ — secid.

Коды контрактов повторяются раз в 10 лет, и при повторном листинге ISS
переименовывает СТАРЫЙ контракт задним числом: SiZ5 декабря 2015 года теперь
SiZ5_2015 — и в реестре, и в истории торгов. Строки lake.futures, загруженные до
переименования, remap_futures_secids() переводит на новый код (по датам
обращения контракта из реестра), чтобы история совпадала с биржей и повторная
загрузка не создавала дублей. base_secid — код без суффикса года.

Непрерывные ряды lake.futures_continuous (ключ date + asset) по основным
активам: на дату берется контракт с наибольшим открытым интересом среди тех,
до экспирации которых больше ROLL_DAYS календарных дней (ликвидный, а не
ближайший месячный), и ряд не возвращается к более раннему контракту. settle_adj — расчетная цена, склеенная по отношению
цен двух контрактов в последний день старого (история приведена к уровню
текущего контракта; при отсутствии парных цен коэффициент остается 1,
и скачок может сохраниться); settle — цена самого контракта без поправки.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Iterable, Optional

import polars as pl
import requests

from moexutils import iss, lake

logger = logging.getLogger("moexutils")

SERIES_URL = f"{iss.ISS_URL}/statistics/engines/futures/markets/forts/series.json"
# основные активы для непрерывных рядов (ASSETCODE): валюта, индексы, товары, акции
MAIN_ASSETS = ('Si', 'Eu', 'CNY', 'RTS', 'MIX', 'MXI', 'BR', 'NG', 'GOLD', 'SILV', 'SBRF', 'GAZR')
ROLL_DAYS = 7
PERPETUAL = dt.date(2100, 1, 1)   # вечные фьючерсы: экспирация 2100-01-01


def fetch_contracts(session: Optional[requests.Session] = None) -> pl.DataFrame:
    """Реестр всех контрактов FORTS (включая истекшие) из ISS."""
    session = session or iss.make_session()
    resp = session.get(SERIES_URL, params={'show_expired': 1})
    resp.raise_for_status()
    df = iss.to_frame(resp.json().get('series'))
    if df.is_empty():
        raise ValueError("ISS вернул пустой реестр контрактов FORTS")
    return (df.with_columns(pl.col('start_date').str.to_date('%Y-%m-%d', strict=False),
                            pl.col('expiration_date').str.to_date('%Y-%m-%d', strict=False),
                            pl.col('secid').str.replace(r'_\d{4}$', '').alias('base_secid'))
              .unique('secid', keep='last').sort('secid'))


def update_contracts(session: Optional[requests.Session] = None) -> tuple[int, int]:
    """Реестр контрактов -> lake.futures_contracts (только изменения); Returns: (записано, удалено)."""
    n = lake.sync('futures_contracts', fetch_contracts(session))
    logger.info(f"[OK] Реестр контрактов FORTS: записано {n[0]}, удалено {n[1]}" if n != (0, 0)
                else "[INFO] Реестр контрактов FORTS: изменений нет")
    return n


def read_contracts(assets=None, as_of: lake.AsOf = None) -> pl.DataFrame:
    """Реестр контрактов из хранилища; assets — ASSETCODE или список."""
    if 'futures_contracts' not in lake.tables():
        raise FileNotFoundError("В хранилище нет реестра futures_contracts: выполните contracts.update_contracts()")
    df = lake.query(f"SELECT * FROM {lake.ref('futures_contracts', as_of)} ORDER BY asset_code, expiration_date")
    if assets is not None:
        assets = [assets] if isinstance(assets, str) else list(assets)
        df = df.filter(pl.col('asset_code').is_in(assets))
    return df


def remap_futures_secids() -> int:
    """
    Переводит строки lake.futures со старым кодом (SiZ5) на код, который биржа
    дала контракту после повторного листинга (SiZ5_2015), по датам обращения из
    реестра. Returns: число перекодированных строк.
    """
    if not {'futures', 'futures_contracts'} <= set(lake.tables()):
        return 0
    moved = lake.query("""
        SELECT c.secid AS new_secid, c.base_secid, c.start_date, c.expiration_date
        FROM lake.futures_contracts c WHERE c.secid <> c.base_secid""")
    if moved.is_empty():
        return 0
    bases = moved['base_secid'].unique().to_list()
    rows = lake.query(f"SELECT * FROM lake.futures WHERE SECID IN ({', '.join('?' * len(bases))})", bases)
    hit = (rows.join(moved, left_on='SECID', right_on='base_secid')
               .filter(pl.col('date') >= pl.col('start_date').fill_null(dt.date(1990, 1, 1)),
                       pl.col('date') <= pl.col('expiration_date')))
    if hit.is_empty():
        return 0
    old_keys = hit.select('date', 'SECID', 'BOARDID')
    new = hit.with_columns(pl.col('new_secid').alias('SECID')).select(rows.columns)
    lake.write('futures', new, delete=old_keys)
    logger.info(f"[OK] Фьючерсы: {new.height} строк переведены на коды после повторного листинга "
                f"({new['SECID'].n_unique()} контрактов)")
    return new.height


def _front(g: pl.DataFrame) -> pl.DataFrame:
    """Контракт ряда на каждую дату одного актива: максимум открытого интереса, без возврата назад."""
    g = g.sort('date', 'expiration_date')
    rows, min_exp = [], None
    for part in g.partition_by('date', maintain_order=True):
        if min_exp is not None:
            part = part.filter(pl.col('expiration_date') >= min_exp)
        if part.is_empty():
            continue
        best = part.sort([pl.col('OPENPOSITION').fill_null(-1.0), pl.col('VOLUME').fill_null(-1.0)],
                         descending=True).head(1)
        min_exp = best['expiration_date'][0]
        rows.append(best)
    return pl.concat(rows) if rows else g.clear()


def build_continuous(assets: Iterable[str] = MAIN_ASSETS, roll_days: int = ROLL_DAYS) -> pl.DataFrame:
    """Непрерывные ряды по активам (см. описание модуля) — расчет без записи."""
    assets = list(assets)
    reg = (read_contracts(assets)
           .filter(pl.col('expiration_date') < PERPETUAL)
           .select(pl.col('secid').alias('SECID'), pl.col('asset_code').alias('asset'), 'expiration_date'))
    secids = reg['SECID'].to_list()
    fut = lake.query(
        "SELECT date, SECID, OPEN, HIGH, LOW, CLOSE, SETTLEPRICE, VOLUME, OPENPOSITION FROM lake.futures "
        f"WHERE SECID IN ({', '.join('?' * len(secids))})", secids)
    df = (fut.join(reg, on='SECID')
             .with_columns(pl.coalesce('SETTLEPRICE', 'CLOSE').alias('settle'))
             .filter(pl.col('settle').is_not_null() & (pl.col('settle') > 0)))
    px = df.select('date', 'SECID', 'settle')
    df = df.filter((pl.col('expiration_date') - pl.col('date')).dt.total_days() > roll_days)
    front = pl.concat([_front(g) for _, g in df.group_by('asset')], how='vertical') if df.height else df
    front = front.sort('asset', 'date').with_columns(
        (pl.col('SECID') != pl.col('SECID').shift(1).over('asset')).fill_null(False).alias('roll'),
        pl.col('SECID').shift(1).over('asset').alias('prev_secid'),
        pl.col('date').shift(1).over('asset').alias('prev_date'))
    # отношение цен нового и старого контракта в последний день старого;
    # нет цены нового в тот день — по ценам обоих в день перехода
    rolls = (front.filter('roll')
                  .join(px.rename({'SECID': 'prev_secid', 'settle': 'old_px', 'date': 'prev_date'}),
                        on=['prev_secid', 'prev_date'], how='left')
                  .join(px.rename({'date': 'prev_date', 'settle': 'new_px'}), on=['SECID', 'prev_date'], how='left')
                  .join(px.rename({'SECID': 'prev_secid', 'settle': 'old_px_now'}), on=['prev_secid', 'date'], how='left')
                  .with_columns(pl.when(pl.col('new_px').is_not_null() & pl.col('old_px').is_not_null())
                                  .then(pl.col('new_px') / pl.col('old_px'))
                                  .otherwise(pl.col('settle') / pl.col('old_px_now'))
                                  .fill_null(1.0).alias('ratio'))
                  .select('asset', 'date', 'ratio'))
    front = (front.join(rolls, on=['asset', 'date'], how='left')
                  .with_columns(pl.col('ratio').fill_null(1.0))
                  .sort('asset', 'date'))
    # поправка даты = произведение отношений всех переходов ПОСЛЕ нее
    front = front.with_columns(
        (pl.col('ratio').reverse().cum_prod().reverse().over('asset') / pl.col('ratio')).alias('adj_factor'))
    return front.select(
        'date', 'asset', 'SECID', 'expiration_date',
        pl.col('OPEN').alias('open'), pl.col('HIGH').alias('high'), pl.col('LOW').alias('low'),
        pl.col('CLOSE').alias('close'), 'settle', pl.col('VOLUME').alias('volume'),
        pl.col('OPENPOSITION').alias('open_interest'), 'roll', 'adj_factor',
        (pl.col('settle') * pl.col('adj_factor')).alias('settle_adj'))


def update_continuous(assets: Iterable[str] = MAIN_ASSETS) -> tuple[int, int]:
    """Пересчет непрерывных рядов -> lake.futures_continuous (только изменения)."""
    assets = list(assets)
    if not assets:
        return 0, 0
    new = build_continuous(assets)
    if 'futures_continuous' in lake.tables():
        old = lake.query(
            "SELECT * FROM lake.futures_continuous "
            f"WHERE asset IN ({', '.join('?' * len(assets))})", assets)
        key = lake.TABLE_KEYS['futures_continuous']
        delta = lake.changed_rows(old, new, key)
        stale = old.select(key).join(new.select(key), on=key, how='anti')
        lake.write('futures_continuous', delta, delete=stale)
        n = (delta.height, stale.height)
    else:
        n = (lake.write('futures_continuous', new), 0)
    logger.info(f"[OK] Непрерывные фьючерсы: записано {n[0]}, удалено {n[1]}")
    return n


def read_continuous(assets=None, start=None, end=None, as_of: lake.AsOf = None) -> pl.DataFrame:
    """Непрерывные ряды из хранилища; assets — актив или список."""
    where, params = [], []
    if assets is not None:
        assets = [assets] if isinstance(assets, str) else list(assets)
        where.append(f"asset IN ({', '.join('?' * len(assets))})")
        params += assets
    for op, val in ((">=", start), ("<=", end)):
        if val is not None:
            where.append(f"date {op} ?")
            params.append(dt.date.fromisoformat(str(val)[:10]))
    sql = f"SELECT * FROM {lake.ref('futures_continuous', as_of)}" + (" WHERE " + " AND ".join(where) if where else "")
    return lake.query(sql + " ORDER BY asset, date", params)
