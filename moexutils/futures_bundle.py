"""All-market FORTS snapshot and raw funding candles for card-confirmed perpetuals."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

import polars as pl

from moexutils import iss, lake, perpetual_audit

TABLES = ('history', 'contracts', 'descriptions', 'parameters', 'coverage', 'asset_coverage',
          'funding_availability', 'funding_comparison')


def _write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def fetch_candles(secid, board, start, end, root, session=None):
    """Archive every successful page. Empty is explicit availability, not an error."""
    if board not in ('FUSR', 'FUSC'):
        raise ValueError('Expected FUSR or FUSC')
    session = session or iss.make_session()
    pages = []

    class Recorder:
        def get(self, url, params):
            response = session.get(url, params=params)
            response.raise_for_status()
            pages.append({'url': response.url, 'payload': response.json()})
            return response

    url = f'{iss.ISS_URL}/engines/futures/markets/swaprates/boards/{board}/securities/{quote(secid, safe="")}/candles.json'
    frame = iss.fetch_pages(url, 'candles', {'from': str(start), 'till': str(end), 'interval': 24},
                            Recorder(), max_pages=200, empty_page_retries=2)
    if frame.is_empty():
        frame = pl.DataFrame(schema={'date': pl.Date, 'swap_board': pl.String, 'close': pl.Float64,
                                     'begin': pl.String, 'end': pl.String})
    else:
        frame = frame.with_columns(pl.col('begin').str.slice(0, 10).str.to_date().alias('date'),
                                    pl.lit(board).alias('swap_board'))
        if frame['date'].n_unique() != frame.height or not frame['date'].is_between(start, end).all():
            raise ValueError(f'Invalid daily candle dates: {secid}/{board}')
    # Hash names avoid unsafe paths for future security codes.
    name = hashlib.sha256(f'{secid}/{board}'.encode()).hexdigest()
    _write_json(root / (name + '.json'), {'secid': secid, 'board': board,
        'observed_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'from': start, 'till': end, 'pages': pages})
    frame.write_parquet(root / (name + '.parquet'))
    return frame


def classify(raw):
    status = perpetual_audit.classify_candidate(raw)
    if status != 'unresolved_candidate':
        return status
    block = json.loads(raw or '{}').get('description', {})
    records = [dict(zip(block.get('columns', []), row)) for row in block.get('data', [])]
    return next((str(row['value']) for row in records if row.get('name') == 'TYPE'), 'unresolved')


def export(*, compare=True, workers=4, output=None):
    """Export ALL raw FORTS history; never select only continuous/traded series.

    The audit does not certify effective historical specs, currency conversion,
    calendar completeness or funding finality. No universal VM formula is applied.
    """
    if workers < 1 or workers > 8:
        raise ValueError('workers must be between 1 and 8')
    now = dt.datetime.now(dt.timezone.utc)
    root = Path(output) if output else Path(lake.DATA_ROOT) / 'reports/futures-bundles' / now.strftime('%Y%m%dT%H%M%S%fZ')
    root.mkdir(parents=True, exist_ok=False)
    raw_root = root / 'raw'
    raw_root.mkdir()
    with lake.session(read_only=True) as con:
        con.execute('BEGIN TRANSACTION')
        snapshot = con.execute("SELECT max(snapshot_id) FROM ducklake_snapshots('lake')").fetchone()[0]
        history = con.execute('SELECT * FROM lake.futures').pl()
        contracts = con.execute('SELECT * FROM lake.futures_contracts').pl()
        descriptions = con.execute('SELECT * FROM lake.futures_description_observations').pl()
        parameters = con.execute('SELECT * FROM lake.futures_parameter_observations').pl()
        con.execute('COMMIT')
    if history.is_empty() or history.select('date', 'SECID', 'BOARDID').is_duplicated().any():
        raise ValueError('Empty history or duplicate date/SECID/BOARDID')
    latest = descriptions.sort('observed_at').unique('secid', keep='last').with_columns(
        pl.col('raw_json').map_elements(classify, return_dtype=pl.String, skip_nulls=False).alias('instrument_type'))
    history = history.join(contracts.select(pl.col('secid').alias('SECID'),
        pl.col('asset_code').alias('registry_asset'), 'start_date', 'expiration_date',
        pl.lit(True).alias('registry_present')), on='SECID', how='left').join(
        latest.select(pl.col('secid').alias('SECID'), 'instrument_type',
                      pl.col('TYPE').alias('source_type')), on='SECID', how='left').with_columns(
        pl.col('registry_present').fill_null(False), pl.col('instrument_type').fill_null('unresolved'),
        pl.coalesce('registry_asset', 'ASSETCODE').alias('asset'))
    coverage = perpetual_audit.coverage(history).join(
        history.select('SECID', 'BOARDID', 'asset', 'instrument_type').unique(), on=['SECID', 'BOARDID'])
    assets = history.group_by('asset', 'instrument_type').agg(
        pl.len().alias('rows'), pl.col('SECID').n_unique().alias('contracts'),
        pl.col('date').min().alias('first_date'), pl.col('date').max().alias('last_date'))
    for name, frame in {'history': history, 'contracts': contracts, 'descriptions': descriptions,
                        'parameters': parameters, 'coverage': coverage, 'asset_coverage': assets}.items():
        frame.write_parquet(root / (name + '.parquet'))
    perpetuals = latest.filter(pl.col('instrument_type') == 'perpetual_card_confirmed')['secid'].to_list()
    jobs = [(secid, board) for secid in sorted(perpetuals) for board in ('FUSR', 'FUSC')] if compare else []

    def job(args):
        secid, board = args
        base = history.filter(pl.col('SECID') == secid)
        start = base['date'].min() if base.height else history['date'].min()
        end = history['date'].max()
        try:
            candles = fetch_candles(secid, board, start, end, raw_root)
            comparison = None
            if candles.height and base.height and base['date'].n_unique() == base.height:
                comparison = perpetual_audit.compare_funding(base, candles, secid=secid).with_columns(
                    pl.lit(secid).alias('SECID'), pl.lit(board).alias('comparison_board'))
            state = 'available' if candles.height else 'empty_source'
            if candles.height and base.is_empty():
                state = 'no_forts_history'
            if candles.height and base['date'].n_unique() != base.height:
                state = 'ambiguous_forts_boards'
            return {'SECID': secid, 'board': board, 'status': state, 'rows': candles.height,
                    'first_date': str(candles['date'].min()) if candles.height else None,
                    'last_date': str(candles['date'].max()) if candles.height else None, 'error': ''}, comparison
        except Exception as exc:
            return {'SECID': secid, 'board': board, 'status': 'error', 'rows': 0,
                    'first_date': None, 'last_date': None, 'error': str(exc)}, None

    availability, comparisons = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for record, comparison in pool.map(job, jobs):
            availability.append(record)
            if comparison is not None:
                comparisons.append(comparison)
            print(f"{record['SECID']}/{record['board']}: {record['status']} {record['rows']}", flush=True)
    avail = pl.DataFrame(availability, schema={'SECID': pl.String, 'board': pl.String, 'status': pl.String,
        'rows': pl.Int64, 'first_date': pl.String, 'last_date': pl.String, 'error': pl.String})
    avail.write_parquet(root / 'funding_availability.parquet')
    combined = pl.concat(comparisons, how='diagonal_relaxed') if comparisons else pl.DataFrame(
        schema={'SECID': pl.String, 'comparison_board': pl.String, 'date': pl.Date, 'status': pl.String})
    combined.write_parquet(root / 'funding_comparison.parquet')
    errors = avail.filter(pl.col('status') == 'error').height
    manifest = {'bundle_schema_version': 1, 'generated_at': now.isoformat(), 'snapshot_id': snapshot,
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'all_FORTS_history_and_all_card_confirmed_perpetuals',
        'history_rows': history.height, 'instruments': coverage.height, 'perpetuals': len(perpetuals),
        'comparison_requested': compare, 'download_errors': errors,
        'status': 'partial' if errors else 'complete',
        'limitations': ['historical_specs_not_verified', 'official_calendar_not_verified',
                       'funding_units_are_instrument_specific', 'funding_finality_not_verified',
                       'current_parameters_not_backfilled', 'no_universal_VM_or_CNY_lot'],
        'artifact_sha256': {str(p.relative_to(root)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(root.rglob('*')) if p.is_file()}}
    _write_json(root / 'summary.json', manifest)
    return root


def read_table(report_dir, table, *, assets=None):
    """Read a checksummed table from an explicit complete bundle. No network/DB."""
    if table not in TABLES:
        raise ValueError(f'Expected one of {TABLES}')
    root = Path(report_dir)
    meta = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    if meta.get('bundle_schema_version') != 1 or meta.get('status') != 'complete':
        raise ValueError('A complete version 1 bundle is required')
    name = table + '.parquet'
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != meta['artifact_sha256'].get(name):
        raise ValueError(f'Checksum mismatch: {name}')
    frame = pl.scan_parquet(root / name)
    if assets is not None:
        if table not in ('history', 'coverage', 'asset_coverage'):
            raise ValueError('asset filter is only supported for history/coverage/asset_coverage')
        frame = frame.filter(pl.col('asset').is_in([assets] if isinstance(assets, str) else assets))
    return frame.collect()


def export_if_due():
    """Weekly successful full bundle; daily raw updates remain in history pipeline."""
    now = dt.datetime.now(dt.timezone.utc)
    root = Path(lake.DATA_ROOT) / 'reports/futures-bundles'
    for path in root.glob('*/summary.json'):
        try:
            meta = json.loads(path.read_text(encoding='utf-8'))
            stamp = dt.datetime.fromisoformat(meta['generated_at'])
            if (meta.get('bundle_schema_version') == 1 and meta.get('status') == 'complete'
                    and meta.get('comparison_requested') is True and stamp.tzinfo is not None
                    and dt.timedelta(0) <= now - stamp < dt.timedelta(days=7)):
                return None
        except (ValueError, KeyError, OSError):
            continue
    result = export()
    if json.loads((result / 'summary.json').read_text(encoding='utf-8'))['download_errors']:
        raise RuntimeError(f'Incomplete funding archive: {result}')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-compare', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = export(compare=not args.no_compare, workers=args.workers, output=args.output)
    print(result)
    raise SystemExit(int(json.loads((result / 'summary.json').read_text(encoding='utf-8'))['download_errors'] > 0))
