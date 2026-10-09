"""Offline import of the six-contract browser pilot; never reads session cookies."""
import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import polars as pl

SYMBOLS = {'KCH26': 'COFFEE', 'KCK26': 'COFFEE', 'OJH26': 'ORANGE',
           'OJK26': 'ORANGE', 'TGH26': 'TTF', 'TGJ26': 'TTF'}
FIELDS = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'open_interest']


def parse_capture(capture, expected_symbols=None):
    expected_symbols = SYMBOLS if expected_symbols is None else expected_symbols
    if capture.get('schema_version') != 1 or capture.get('stopped'):
        raise ValueError('Complete version 1 capture required')
    rows, seen = [], set()
    for response in capture.get('responses', []):
        symbol = response['symbol']
        if symbol not in expected_symbols or symbol in seen or response['http_status'] != 200:
            raise ValueError('Unexpected, duplicate or failed symbol')
        url = urlsplit(response['url'])
        params = parse_qs(url.query)
        if url.scheme != 'https' or url.netloc != 'www.barchart.com' or url.path != '/proxies/timeseries/historical/queryeod.ashx':
            raise ValueError('Unexpected source URL')
        for key, value in {'symbol': symbol, 'data': 'daily', 'volume': 'contract', 'backadjust': 'false'}.items():
            if params.get(key) != [value]:
                raise ValueError(f'Wrong query parameter: {key}')
        body_rows = list(csv.reader(io.StringIO(response['body'])))
        if not body_rows:
            raise ValueError('Empty response')
        for row in body_rows:
            if len(row) != 8 or row[0] != symbol:
                raise ValueError('Unexpected eight-column record or symbol')
            date = dt.date.fromisoformat(row[1])
            values = [float(x) if x.strip() not in ('', 'null', 'N/A') else None for x in row[2:]]
            if any(v is not None and not math.isfinite(v) for v in values):
                raise ValueError('Non-finite numeric value')
            if any(v is not None and (v < 0 or not v.is_integer()) for v in values[4:]):
                raise ValueError('Invalid volume/open interest')
            rows.append(dict(zip(FIELDS, [symbol, date, *values])))
        seen.add(symbol)
    if seen != set(expected_symbols) or not seen:
        raise ValueError('All expected contracts required')
    frame = pl.DataFrame(rows, schema_overrides={c: pl.Float64 for c in FIELDS[2:]})
    if frame.select('symbol', 'date').is_duplicated().any():
        raise ValueError('Duplicate symbol/date')
    return frame.with_columns(
        pl.col('symbol').replace_strict(expected_symbols).alias('asset'),
        pl.lit('provider_close_not_verified_settlement').alias('price_kind'),
        pl.lit(False).alias('execution_model_ready'),
    ).sort('symbol', 'date')


def import_capture(source, output):
    source, output = Path(source), Path(output)
    raw = source.read_bytes()
    capture = json.loads(raw.decode('utf-8-sig'))
    frame = parse_capture(capture)
    coverage = frame.group_by('symbol', 'asset').agg(
        pl.len().alias('rows'), pl.col('date').min().alias('first'), pl.col('date').max().alias('last'),
        pl.col('close').null_count().alias('null_close'),
        (pl.col('volume') == 0).sum().alias('zero_volume_rows'),
        ((pl.col('open') == pl.col('high')) & (pl.col('high') == pl.col('low')) &
         (pl.col('low') == pl.col('close'))).sum().alias('flat_ohlc_rows'),
    ).sort('symbol')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'capture.json').write_bytes(raw)
    frame.write_parquet(output / 'contracts.parquet')
    metadata = {'schema_version': 1, 'status': 'parsed_not_execution_validated',
        'source_sha256': hashlib.sha256(raw).hexdigest(), 'captured_at': capture.get('captured_at'),
        'coverage': coverage.to_dicts(),
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limitations': ['640_record_cap_not_full_history', 'close_not_verified_settlement',
                        'zero_volume_and_flat_OHLC_not_evidence_of_executable_prices',
                        'historical_vintage_and_publication_time_unknown'],
        'artifact_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}}
    (output / 'summary.json').write_text(json.dumps(metadata, indent=2, default=str), encoding='utf-8')
    return coverage


def read_contracts(folder):
    root = Path(folder)
    meta = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    p = root / 'contracts.parquet'
    if meta.get('schema_version') != 1 or hashlib.sha256(p.read_bytes()).hexdigest() != meta['artifact_sha256'][p.name]:
        raise ValueError('Invalid version or checksum')
    return pl.read_parquet(p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(import_capture(args.source, args.output))
