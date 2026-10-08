"""CNY perpetual/dated futures audit. No source repair or historical VM inference."""
from __future__ import annotations

import argparse
import datetime as dt
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import polars as pl

from moexutils import iss, lake

FIELDS = ('OPEN', 'HIGH', 'LOW', 'CLOSE', 'SETTLEPRICE', 'VOLUME', 'NUMTRADES',
          'OPENPOSITION', 'VALUE', 'SWAPRATE', 'SWAPRATE_CURR')


def read_cny_history() -> pl.DataFrame:
    """All raw boards and CNY series, including spreads; LEFT JOIN, no filling."""
    return lake.query('''WITH d AS (SELECT * FROM lake.futures_description_observations
        QUALIFY row_number() OVER (PARTITION BY secid ORDER BY observed_at DESC)=1)
        SELECT f.*,c.asset_code registry_asset,c.start_date,c.expiration_date,
            c.secid IS NOT NULL registry_present,d.TYPE source_type,d.ASSETCODE description_asset,
            d.observed_at description_observed_at,d.source_url description_source_url
        FROM lake.futures f LEFT JOIN lake.futures_contracts c ON f.SECID=c.secid
        LEFT JOIN d ON f.SECID=d.secid
        WHERE f.SECID='CNYRUBF' OR c.asset_code='CNY' OR f.ASSETCODE='CNY' OR d.ASSETCODE='CNY'
        ORDER BY f.SECID,f.BOARDID,f.date''')


def coverage(frame: pl.DataFrame) -> pl.DataFrame:
    """NULL and zero are counted independently for every raw field."""
    return frame.group_by('SECID', 'BOARDID').agg(
        pl.col('date').min().alias('first_date'), pl.col('date').max().alias('last_date'),
        pl.len().alias('rows'), pl.col('date').n_unique().alias('dates'),
        (pl.len() - pl.col('date').n_unique()).alias('duplicate_keys'),
        pl.col('registry_present').all(), pl.col('source_type').first(),
        *[expr for field in FIELDS for expr in (
            pl.col(field).null_count().alias(field + '_null'),
            (pl.col(field) == 0).sum().alias(field + '_zero'),
            pl.col('date').filter(pl.col(field).is_not_null()).min().alias(field + '_first'),
            pl.col('date').filter(pl.col(field).is_not_null()).max().alias(field + '_last'))],
    ).sort('SECID', 'BOARDID')


def candidate_gaps(frame: pl.DataFrame, market_dates: pl.DataFrame) -> pl.DataFrame:
    """Gaps on dates observed elsewhere in FORTS, NOT a verified instrument calendar.

    Bounds use the union of registry life and observed history; contradictory
    source dates must not hide early/late rows. Global history end caps live series.
    """
    last = market_dates['date'].max()
    bounds = frame.group_by('SECID', 'BOARDID').agg(
        pl.col('date').min().alias('first'), pl.col('date').max().alias('last'),
        pl.col('start_date').first(), pl.col('expiration_date').first())
    bounds = bounds.with_columns(
        pl.min_horizontal('first', 'start_date').alias('start'),
        pl.min_horizontal(pl.lit(last), pl.max_horizontal('last', 'expiration_date')).alias('end'))
    grid = bounds.join(market_dates, how='cross').filter(pl.col('date').is_between(pl.col('start'), pl.col('end')))
    return grid.join(frame.select('SECID', 'BOARDID', 'date').unique(),
                     on=['SECID', 'BOARDID', 'date'], how='anti').select(
        'SECID', 'BOARDID', 'date', pl.lit('unverified_market_calendar_candidate').alias('status')
    ).sort('SECID', 'BOARDID', 'date')


def fetch_candles(board, start, end, root, session=None):
    """Audit artifact only; retains every raw page. Never writes a lake table."""
    if board not in ('FUSR', 'FUSC'):
        raise ValueError('Expected FUSR/FUSC')
    session = session or iss.make_session()
    pages = []

    class Recorder:
        def get(self, url, params):
            r = session.get(url, params=params)
            r.raise_for_status()
            pages.append({'url': r.url, 'payload': r.json()})
            return r

    url = f'{iss.ISS_URL}/engines/futures/markets/swaprates/boards/{board}/securities/CNYRUBF/candles.json'
    frame = iss.fetch_pages(url, 'candles', {'from': str(start), 'till': str(end), 'interval': 24},
                            Recorder(), max_pages=100)
    if frame.is_empty():
        raise ValueError(f'Empty candle archive: {board}')
    frame = frame.with_columns(pl.col('begin').str.slice(0, 10).str.to_date().alias('date'),
                               pl.lit(board).alias('swap_board'))
    if frame['date'].n_unique() != frame.height:
        raise ValueError('Duplicate daily candles')
    if not frame['date'].is_between(start, end).all():
        raise ValueError('Candle outside requested interval')
    artifact = {'observed_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'pages': pages}
    (root / f'{board}-raw.json').write_text(json.dumps(artifact, ensure_ascii=False), encoding='utf-8')
    return frame


def compare_funding(history: pl.DataFrame, candles: pl.DataFrame) -> pl.DataFrame:
    boards = candles['swap_board'].unique().to_list()
    if len(boards) != 1 or boards[0] not in ('FUSR', 'FUSC'):
        raise ValueError('One known swap board required')
    if candles['date'].n_unique() != candles.height:
        raise ValueError('Duplicate candle dates')
    field = 'SWAPRATE' if boards[0] == 'FUSR' else 'SWAPRATE_CURR'
    base = history.filter(pl.col('SECID') == 'CNYRUBF').select('date', 'BOARDID', 'SWAPRATE', 'SWAPRATE_CURR')
    if base['date'].n_unique() != base.height:
        raise ValueError('Ambiguous CNYRUBF date/board; select a board explicitly')
    base = base.with_columns(pl.lit(True).alias('forts_present'))
    candles = candles.with_columns(pl.lit(True).alias('candle_present'))
    return base.join(candles, on='date', how='full', coalesce=True).with_columns(
        pl.lit(field).alias('reference_field'),
        (pl.col('close') - pl.col(field)).alias('difference'),
        (pl.col('end').str.slice(0, 10).str.to_date() != pl.col('date')).alias('candle_end_other_date'),
        (pl.col('close') - pl.col('SWAPRATE')).alias('difference_rub'),
        (pl.col('close') - pl.col('SWAPRATE_CURR')).alias('difference_curr'),
    ).with_columns(
        pl.when(pl.col('forts_present').is_null()).then(pl.lit('candle_only_date'))
        .when(pl.col('candle_present').is_null()).then(pl.lit('forts_only_date'))
        .when(pl.col(field).is_null() & pl.col('close').is_null()).then(pl.lit('both_values_null'))
        .when(pl.col(field).is_null()).then(pl.lit('null_forts_value'))
        .when(pl.col('close').is_null()).then(pl.lit('null_candle_value'))
        .when(pl.col('difference').abs() <= 1e-9).then(pl.lit('equal_numeric_only'))
        .otherwise(pl.lit('disputed_no_replacement')).alias('status')
    ).sort('date')


def classify_candidate(raw_json):
    """Latest card evidence, not a historically effective classification."""
    block = json.loads(raw_json or '{}').get('description', {})
    records = [dict(zip(block.get('columns', []), row)) for row in block.get('data', [])]
    values = {row.get('name'): row.get('value') for row in records}
    if values.get('TYPE') == 'futures_collateral':
        return 'collateral_excluded'
    if values.get('TYPE') == 'futures' and (
        str(values.get('PERPETUAL_FUTURES')) == '1'
        or 'автопролонг' in str(values.get('CONTRACTNAME', '')).lower()
    ):
        return 'perpetual_card_confirmed'
    return 'unresolved_candidate'


def illustrative_cashflow(previous_price, next_price, funding_per_unit, side, lot=1000):
    """Unrounded arithmetic of the public simplified formula, NOT historical VM.

    Prices must use RUB/CNY, funding RUB/CNY. Eligibility for a clearing,
    execution prices, effective specification and rounding are external inputs.
    No SWAPRATE_CURR input: it cannot be charged as a second funding payment.
    """
    if side not in ('long', 'short') or any(v is None for v in (previous_price, next_price, funding_per_unit)):
        raise ValueError('Explicit prices, funding and long/short required')
    q = Decimal(str(lot))
    if not q.is_finite() or q <= 0:
        raise ValueError('Positive finite lot required')
    before, after, funding = map(lambda v: Decimal(str(v)), (previous_price, next_price, funding_per_unit))
    if not all(v.is_finite() for v in (before, after, funding)):
        raise ValueError('Finite inputs required')
    sign = Decimal(1 if side == 'long' else -1)
    price = sign * (after - before) * q
    payment = -sign * funding * q
    return {'price_revaluation': price, 'funding_payment': payment, 'total': price + payment}


def export(compare=False, output=None):
    """Snapshot-pinned local audit; optional explicit network comparison of CNY only."""
    now = dt.datetime.now(dt.timezone.utc)
    root = Path(output) if output else Path(lake.DATA_ROOT) / 'reports/perpetual-audit' / now.strftime('%Y%m%dT%H%M%S%fZ')
    root.mkdir(parents=True, exist_ok=True)
    # lake.query opens its own connection; pin every read to this explicit snapshot.
    with lake.session(read_only=True) as con:
        con.execute('BEGIN TRANSACTION')
        snapshot = con.execute("SELECT max(snapshot_id) FROM ducklake_snapshots('lake')").fetchone()[0]
        con.execute('''CREATE TEMP TABLE perp_d AS SELECT * FROM lake.futures_description_observations
            QUALIFY row_number() OVER(PARTITION BY secid ORDER BY observed_at DESC)=1''')
        frame = con.execute('''SELECT f.*,c.start_date,c.expiration_date,c.asset_code registry_asset,
            c.secid IS NOT NULL registry_present,d.TYPE source_type,d.ASSETCODE description_asset,
            d.observed_at description_observed_at,d.source_url description_source_url
            FROM lake.futures f LEFT JOIN lake.futures_contracts c ON f.SECID=c.secid
            LEFT JOIN perp_d d ON f.SECID=d.secid WHERE f.SECID='CNYRUBF'
            OR c.asset_code='CNY' OR f.ASSETCODE='CNY' OR d.ASSETCODE='CNY' ''').pl()
        dates = con.execute('SELECT DISTINCT date FROM lake.futures ORDER BY date').pl()
        candidates = con.execute('''SELECT c.*,d.TYPE source_type,d.raw_json,
            d.source_url,d.observed_at FROM lake.futures_contracts c
            LEFT JOIN perp_d d ON c.secid=d.secid WHERE c.expiration_date=DATE '2100-01-01'
            OR lower(d.raw_json) LIKE '%автопролонг%' ''').pl()
        overview = con.execute('''SELECT f.SECID,f.BOARDID,min(f.date) first_date,max(f.date) last_date,
            count(*) n_rows,count(*) FILTER(WHERE SWAPRATE IS NULL) funding_null,
            count(*) FILTER(WHERE SWAPRATE=0) funding_zero
            FROM lake.futures f LEFT JOIN lake.futures_contracts c ON f.SECID=c.secid
            LEFT JOIN perp_d d ON f.SECID=d.secid WHERE c.expiration_date=DATE '2100-01-01'
            OR lower(d.raw_json) LIKE '%автопролонг%' GROUP BY f.SECID,f.BOARDID ORDER BY f.SECID''').pl()
        con.register('cny_ids', frame.select('SECID').unique())
        descriptions = con.execute('''SELECT d.* FROM lake.futures_description_observations d
            WHERE d.secid IN (SELECT SECID FROM cny_ids)''').pl()
        parameters = con.execute('''SELECT p.* FROM lake.futures_parameter_observations p
            WHERE p.SECID IN (SELECT SECID FROM cny_ids)''').pl()
        con.execute('COMMIT')
    candidates = candidates.with_columns(pl.col('raw_json').map_elements(
        classify_candidate, return_dtype=pl.String, skip_nulls=False).alias('classification'))
    overview = overview.join(candidates.select(pl.col('secid').alias('SECID'), 'classification'), on='SECID', how='left')
    duplicates = frame.group_by('SECID', 'BOARDID', 'date').len().filter(pl.col('len') > 1)
    summary = coverage(frame)
    gaps = candidate_gaps(frame, dates)
    # No chain selection: every ordinary CNY contract on every common raw date.
    perp = frame.filter(pl.col('SECID') == 'CNYRUBF')
    dated = frame.filter((pl.col('source_type') == 'futures') & (pl.col('registry_asset') == 'CNY'))
    pairs = dated.join(perp, on='date', suffix='_perpetual').sort('SECID', 'date')
    for name, df in {'history': frame, 'coverage': summary, 'candidate_gaps': gaps,
                     'duplicates': duplicates, 'common_dates': pairs,
                     'descriptions': descriptions, 'parameter_observations': parameters,
                     'perpetual_candidates': candidates, 'other_perpetual_coverage': overview}.items():
        df.write_parquet(root / (name + '.parquet'))
    metadata = {'bundle_schema_version': 1, 'generated_at': now.isoformat(), 'snapshot_id': snapshot,
                'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'calendar_status': 'unverified; candidates use market-observed dates, not an official instrument calendar',
                'historical_vm_status': 'unavailable', 'history_rows': frame.height,
                'market_last_date': str(dates['date'].max()),
                'instruments': summary.height, 'duplicate_keys': duplicates.height,
                'candidate_gap_rows': gaps.height, 'common_date_rows': pairs.height,
                'candidate_classification': candidates.group_by('classification').len().to_dicts(),
                'comparisons': {}}
    if compare:
        for board in ('FUSR', 'FUSC'):
            candles = fetch_candles(board, perp['date'].min(), perp['date'].max(), root)
            result = compare_funding(perp, candles)
            result.write_parquet(root / (board + '-comparison.parquet'))
            metadata['comparisons'][board] = result.group_by('status').len().to_dicts()
    metadata['artifact_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(root.glob('*.parquet'))}
    (root / 'summary.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    return root


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compare-swaprates', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(export(args.compare_swaprates, args.output))
