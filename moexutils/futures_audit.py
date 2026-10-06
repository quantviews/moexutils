"""Read-only FORTS coverage and source inconsistencies; never repairs prices.

Descriptions are observations, not historical identities. Coverage of current
fields is deliberately separated from confirmed historical parameter coverage.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
from pathlib import Path

import polars as pl

from moexutils import lake


def _prepare(con):
    con.execute("SET TimeZone='UTC'")
    # One database transaction pins the tables used by all report sections.
    con.execute('''CREATE OR REPLACE TEMP TABLE audit_descriptions AS
        SELECT * FROM lake.futures_description_observations
        QUALIFY row_number() OVER (PARTITION BY secid ORDER BY observed_at DESC)=1''')
    con.execute('''CREATE OR REPLACE TEMP TABLE audit_instruments AS
        WITH h AS (SELECT SECID, min(date) history_from, max(date) history_till,
                   count(*) history_rows FROM lake.futures GROUP BY SECID)
        SELECT h.*, c.secid IS NOT NULL registry_present, c.asset_code registry_asset,
            c.start_date, c.expiration_date, d.ASSETCODE description_asset,
            d.TYPE source_type, d.TYPENAME source_typename, d."GROUP" source_group,
            d.FRSTTRADE, d.LSTTRADE, d.LSTDELDATE, d.observed_at description_observed_at,
            d.source_url description_source_url,
            CASE WHEN d.TYPE='commodity_futures' THEN 'ntb_standard_observed'
                 WHEN d.TYPE='futures_spread' THEN 'calendar_spread_observed'
                 WHEN d.TYPE='futures_collateral' THEN 'collateral_observed'
                 WHEN d.TYPE='futures' AND c.expiration_date=DATE '2100-01-01'
                    THEN 'perpetual_registry_marker'
                 WHEN d.TYPE='futures' THEN 'futures_observed'
                 WHEN d.TYPE IS NOT NULL THEN 'non_forts_description_unresolved'
                 ELSE 'unavailable' END instrument_class,
            c.asset_code IS NOT NULL AND d.ASSETCODE IS NOT NULL
                AND c.asset_code<>d.ASSETCODE asset_conflict
        FROM h LEFT JOIN lake.futures_contracts c ON h.SECID=c.secid
        LEFT JOIN audit_descriptions d ON h.SECID=d.secid''')


def _reports(con) -> dict[str, pl.DataFrame]:
    _prepare(con)
    queries = {
        'instruments': 'SELECT * FROM audit_instruments ORDER BY SECID',
        'missing_registry': 'SELECT * FROM audit_instruments WHERE NOT registry_present ORDER BY SECID',
        'before_start': '''SELECT f.SECID, count(*) n_rows, min(f.date) first_date,
            max(f.date) last_date, c.start_date FROM lake.futures f
            JOIN lake.futures_contracts c ON f.SECID=c.secid
            WHERE f.date<c.start_date GROUP BY ALL ORDER BY f.SECID''',
        'continuous_before_start': '''SELECT f.asset,f.SECID,f.date,c.start_date
            FROM lake.futures_continuous f JOIN lake.futures_contracts c ON f.SECID=c.secid
            WHERE f.date<c.start_date ORDER BY f.asset,f.date''',
        'after_expiration': '''SELECT f.SECID,f.date,f.BOARDID,f.VOLUME,f.SETTLEPRICE,
            c.expiration_date,d.LSTTRADE,d.LSTDELDATE,d.TYPE source_type
            FROM lake.futures f JOIN lake.futures_contracts c ON f.SECID=c.secid
            LEFT JOIN audit_descriptions d ON f.SECID=d.secid
            WHERE f.date>c.expiration_date AND f.VOLUME>0 ORDER BY f.SECID,f.date''',
        'nonpositive_settlement': '''SELECT f.SECID,f.date,f.SETTLEPRICE,f.VOLUME,d.TYPE source_type,
            CASE WHEN f.SECID='CLJ0' AND f.date=DATE '2020-04-21' AND f.SETTLEPRICE=-37.63
                 THEN 'confirmed_exchange_settlement' ELSE 'unresolved' END status,
            CASE WHEN f.SECID='CLJ0' AND f.date=DATE '2020-04-21' AND f.SETTLEPRICE=-37.63
                 THEN 'https://www.moex.com/n28152' END confirmation_url
            FROM lake.futures f LEFT JOIN audit_descriptions d ON f.SECID=d.secid
            WHERE f.VOLUME>0 AND f.SETTLEPRICE<=0 ORDER BY f.SECID,f.date''',
        'gazr_scale_break': '''SELECT SECID,date,OPEN,HIGH,LOW,CLOSE,SETTLEPRICE,VOLUME
            FROM lake.futures WHERE SECID='GZU2_2002'
            AND date BETWEEN DATE '2002-06-24' AND DATE '2002-07-02' ORDER BY date''',
        'settlement_fallbacks': '''SELECT c.asset,c.SECID,c.date,f.BOARDID,f.CLOSE,c.settle
            FROM lake.futures_continuous c JOIN lake.futures f ON c.SECID=f.SECID AND c.date=f.date
            WHERE f.SETTLEPRICE IS NULL AND f.CLOSE IS NOT NULL ORDER BY c.asset,c.date''',
        'ambiguous_price_dates': '''SELECT SECID,date,count(*) n_rows FROM lake.futures
            GROUP BY SECID,date HAVING count(*)>1 ORDER BY SECID,date''',
        'duplicate_registry': '''SELECT secid,count(*) n_rows FROM lake.futures_contracts
            GROUP BY secid HAVING count(*)>1''',
        'unverified_editions': '''SELECT asset_candidate,count(*) editions,
            count(*) FILTER (WHERE contract_applicability_verified) verified,
            count(*) FILTER (WHERE document_downloaded) downloaded
            FROM lake.futures_specification_editions GROUP BY asset_candidate ORDER BY asset_candidate''',
        'edition_overlaps': '''SELECT a.asset_candidate,a.source_url,a.source_valid_from,
            b.source_valid_from next_start,a.reviewed_on FROM lake.futures_specification_editions a
            JOIN lake.futures_specification_editions b ON a.asset_candidate=b.asset_candidate
            AND a.source_url=b.source_url AND a.reviewed_on=b.reviewed_on
            AND a.source_valid_from<b.source_valid_from
            AND (a.source_valid_to IS NULL OR a.source_valid_to>=b.source_valid_from)''',
        'coverage': '''WITH p AS (
            SELECT SECID,count(*) parameter_observations,min(observed_at) first_observed_at,
                max(observed_at) last_observed_at,
                count(*) FILTER (WHERE MINSTEP>0 AND STEPPRICE>0) point_value_observations,
                count(*) FILTER (WHERE INITIALMARGIN>0) margin_observations
            FROM lake.futures_parameter_observations GROUP BY SECID),
            risk AS (SELECT DISTINCT date,assetcode FROM lake.futures_risk_limits)
            SELECT year(f.date)::INTEGER AS year,f.SECID,i.registry_asset,i.description_asset,
                i.instrument_class,i.registry_present,i.asset_conflict,
                count(*) history_rows,min(f.date) history_from,max(f.date) history_till,
                count(*) FILTER (WHERE f.VOLUME>0) traded_rows,
                count(*) FILTER (WHERE f.SETTLEPRICE IS NULL) missing_settlement_rows,
                i.description_observed_at IS NOT NULL description_available,
                coalesce(p.parameter_observations,0) parameter_observations,
                coalesce(p.point_value_observations,0) point_value_observations,
                coalesce(p.margin_observations,0) margin_observations,
                p.first_observed_at,p.last_observed_at,
                count(*) FILTER (WHERE risk.date IS NOT NULL) risk_archive_matched_rows,
                'unavailable' historical_point_value_status,
                'unavailable' historical_ruble_margin_status,
                'unavailable' contract_specification_intervals_status
            FROM lake.futures f JOIN audit_instruments i ON f.SECID=i.SECID
            LEFT JOIN p ON f.SECID=p.SECID
            LEFT JOIN risk ON f.date=risk.date
                AND CASE WHEN i.asset_conflict THEN NULL ELSE i.registry_asset END=risk.assetcode
            GROUP BY ALL ORDER BY year,f.SECID''',
        'rolls': '''WITH c AS (
                SELECT *,lag(SECID) OVER w prev_secid,lag(date) OVER w prev_date
                FROM lake.futures_continuous WINDOW w AS (PARTITION BY asset ORDER BY date)),
            px AS (SELECT SECID,date,coalesce(SETTLEPRICE,CLOSE) settle FROM lake.futures
                   WHERE coalesce(SETTLEPRICE,CLOSE)>0),
            r AS (SELECT c.asset,c.date,c.SECID,c.prev_secid,c.prev_date,
                    o.settle old_prev,n.settle new_prev,t.settle old_today,c.settle new_today
                FROM c LEFT JOIN px o ON o.SECID=c.prev_secid AND o.date=c.prev_date
                LEFT JOIN px n ON n.SECID=c.SECID AND n.date=c.prev_date
                LEFT JOIN px t ON t.SECID=c.prev_secid AND t.date=c.date WHERE c.roll)
            SELECT *, CASE WHEN old_prev IS NOT NULL AND new_prev IS NOT NULL THEN 'previous_date'
                WHEN old_today IS NOT NULL AND new_today>0 THEN 'roll_date'
                ELSE 'missing_pair_ratio_one' END ratio_source,
                CASE WHEN old_prev IS NOT NULL AND new_prev IS NOT NULL THEN new_prev/old_prev
                WHEN old_today IS NOT NULL AND new_today>0 THEN new_today/old_today
                ELSE 1.0 END ratio FROM r ORDER BY asset,date''',
    }
    reports = {}
    for name, sql in queries.items():
        logging.getLogger('moexutils').info(f'FORTS audit: {name}')
        reports[name] = con.execute(sql).pl()
    reports['asset_year_coverage'] = reports['coverage'].group_by('year', 'registry_asset').agg(
        pl.len().alias('contracts'),
        pl.col('history_rows', 'traded_rows', 'missing_settlement_rows', 'risk_archive_matched_rows').sum(),
        pl.col('description_available').sum().alias('contracts_with_description'),
        (pl.col('point_value_observations') > 0).sum().alias('contracts_with_point_observation'),
        (pl.col('margin_observations') > 0).sum().alias('contracts_with_margin_observation'),
    ).sort('year', 'registry_asset')
    return reports


def report() -> tuple[dict, dict[str, pl.DataFrame]]:
    """One read transaction; no downloads, no writes to lake. Missing tables fail explicitly."""
    with lake.session(read_only=True) as con:
        con.execute('BEGIN TRANSACTION')
        try:
            snapshot = con.execute("SELECT max(snapshot_id) FROM ducklake_snapshots('lake')").fetchone()[0]
            reports = _reports(con)
            con.execute('COMMIT')
        except Exception:
            con.execute('ROLLBACK')
            raise
    return {'generated_at': dt.datetime.now(dt.timezone.utc).isoformat(),
            'snapshot_id': snapshot, 'schema_version': 1,
            'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'historical_parameter_policy': 'unavailable; observations are not effective intervals',
            'classification_policy': 'latest description, not a historical identity guarantee'}, reports


def read_coverage(secid=None, year=None) -> pl.DataFrame:
    """Contract/year coverage; observed fields never count as historical parameters."""
    frame = report()[1]['coverage']
    if secid is not None:
        frame = frame.filter(pl.col('SECID') == secid)
    if year is not None:
        frame = frame.filter(pl.col('year') == int(year))
    return frame


def export(output=None) -> Path:
    """Write a timestamped JSON report outside Git; retain previous reports."""
    metadata, reports = report()
    root = Path(output) if output else Path(lake.DATA_ROOT) / 'reports' / 'futures-audit'
    root.mkdir(parents=True, exist_ok=True)
    path = root / (dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
    payload = {**metadata, 'sections': {name: frame.to_dicts() for name, frame in reports.items()}}
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding='utf-8')
    temp.replace(path)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', help='Report directory; default MOEX_DATA_ROOT/reports/futures-audit')
    print(export(parser.parse_args().output))
