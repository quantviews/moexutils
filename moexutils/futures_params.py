"""Observed FORTS parameters and published daily risk limits, without backdating.

Raw ISS responses are retained outside the repository. Security descriptions
retrieved for expired contracts are observations today, not historical versions.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import hashlib
import json
from pathlib import Path
import threading
from urllib.parse import quote

import polars as pl

from moexutils import contracts, iss, lake

CURRENT_URL = f'{iss.ISS_URL}/engines/futures/markets/forts/securities.json'
LIMITS_URL = f'{iss.ISS_URL}/rms/engines/futures/objects/limits.json'
_local = threading.local()


def _session():
    if not hasattr(_local, 'session'):
        _local.session = iss.make_session()
    return _local.session


def _root():
    return Path(lake.DATA_ROOT) / 'raw' / 'futures_params'


def _get(url, params=None):
    response = _session().get(url, params=params)
    response.raise_for_status()
    return {'source_url': response.url,
            'observed_at': dt.datetime.now(dt.timezone.utc).isoformat(),
            'payload': response.json()}


def _save(path, envelope):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(envelope, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def _block(envelope, name, required):
    block = envelope['payload'].get(name)
    if not isinstance(block, dict) or not set(required) <= set(block.get('columns', [])):
        raise ValueError(f'ISS: missing or invalid {name}')
    if not isinstance(block.get('data'), list):
        raise ValueError(f'ISS: missing data in {name}')
    if any(len(row) != len(block['columns']) for row in block['data']):
        raise ValueError(f'ISS: malformed row in {name}')
    return iss.to_frame(block)


def _provenance(frame, envelope):
    return frame.with_columns(
        pl.lit(dt.datetime.fromisoformat(envelope['observed_at'])).alias('observed_at'),
        pl.lit(envelope['source_url']).alias('source_url'),
        pl.lit('observed').alias('availability'),
    )


def current_frame(envelope):
    frame = _block(envelope, 'securities', ['SECID', 'BOARDID', 'MINSTEP', 'STEPPRICE', 'INITIALMARGIN'])
    if frame.is_empty() or frame.select(pl.struct('SECID', 'BOARDID').is_duplicated().any()).item():
        raise ValueError('ISS: empty or duplicate current parameters')
    if frame['SECID'].null_count() or frame['BOARDID'].null_count():
        raise ValueError('ISS: null current parameter key')
    # IMTIME is retained verbatim: it is not an effective timestamp for all fields.
    return _provenance(frame, envelope).with_columns(
        pl.lit(None, dtype=pl.Datetime('us', 'UTC')).alias('effective_from'),
        pl.lit(None, dtype=pl.Datetime('us', 'UTC')).alias('effective_to'),
    )


def capture_current():
    envelope = _get(CURRENT_URL)
    frame = current_frame(envelope)
    stamp = dt.datetime.fromisoformat(envelope['observed_at']).strftime('%Y%m%dT%H%M%S%fZ')
    _save(_root() / 'current' / f'{stamp}.json', envelope)
    lake.write('futures_parameter_observations', frame)
    return frame


def risk_frame(envelope, day):
    frame = _block(envelope, 'limits', ['tradedate', 'assetcode', 'updatetime', 'mr1', 'mr2', 'mr3'])
    cursor = envelope['payload'].get('limits.cursor', {})
    rows = cursor.get('data', [])
    if len(rows) != 1:
        raise ValueError('ISS: missing limits cursor')
    fields = dict(zip(cursor['columns'], rows[0]))
    if fields.get('INDEX') != 0 or fields.get('TOTAL') != frame.height:
        raise ValueError('ISS: incomplete risk limits response')
    if frame.is_empty():
        return frame
    if frame['tradedate'].null_count() or frame['tradedate'].unique().to_list() != [day.isoformat()]:
        raise ValueError('ISS: returned risk limits for a different date')
    if frame['assetcode'].null_count() or frame['updatetime'].null_count():
        raise ValueError('ISS: null asset/update time in risk limits')
    frame = frame.unique(maintain_order=True)
    if frame.select(pl.struct('assetcode', 'updatetime').is_duplicated().any()).item():
        raise ValueError(f'ISS: conflicting risk limits with same update time on {day}')
    return _provenance(frame, envelope).with_columns(
        pl.col('tradedate').str.to_date().alias('date'))


def risk_bounds():
    envelope = _get(LIMITS_URL)
    dates = _block(envelope, 'limits.dates', ['from', 'till'])
    if dates.height != 1:
        raise ValueError('ISS: missing risk archive bounds')
    return tuple(dt.date.fromisoformat(dates[c][0]) for c in ('from', 'till'))


def _risk_day(args):
    day, refresh = args
    path = _root() / 'risk_limits' / f'{day}.json'
    envelope = json.loads(path.read_text(encoding='utf-8')) if path.exists() and not refresh else None
    if envelope is None:
        envelope = _get(LIMITS_URL, {'date': day.isoformat(), 'limit': 1000})
        frame = risk_frame(envelope, day)
        _save(path, envelope)
    else:
        frame = risk_frame(envelope, day)
    return frame


def update_risk_limits(start=None, end=None, workers=4):
    """Explicit start backfills every calendar date, including confirmed empty days.

    Default: catch up from the last stored date and refresh the last seven days.
    Revisions retain their observation timestamps; rates are NOT contract margin.
    """
    first, last = risk_bounds()
    stop = min(dt.date.fromisoformat(str(end)), last) if end else last
    if start is None:
        begin = last - dt.timedelta(days=7)
        if 'futures_risk_limits' in lake.tables():
            previous = lake.query('SELECT max(date) AS d FROM lake.futures_risk_limits').item()
            if previous:
                begin = min(begin, previous)
    else:
        begin = dt.date.fromisoformat(str(start))
    begin = max(begin, first)
    days = [begin + dt.timedelta(days=i) for i in range(max(0, (stop - begin).days + 1))]
    total, pending, empty = 0, [], 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, frame in enumerate(pool.map(_risk_day, [(d, d >= last - dt.timedelta(days=7)) for d in days]), 1):
            if frame.is_empty():
                empty += 1
            else:
                pending.append(frame)
            if i % 100 == 0 or i == len(days):
                if pending:
                    batch = pl.concat(pending, how='diagonal_relaxed')
                    total += lake.write('futures_risk_limits', batch)
                    pending.clear()
                print(f'Risk limits: {i}/{len(days)} dates, {total} rows, {empty} empty', flush=True)
    return total


def description_frame(envelope, secid):
    frame = _block(envelope, 'description', ['name', 'value'])
    fields = dict(zip(frame['name'], frame['value']))
    if fields.get('SECID') is not None and fields['SECID'] != secid:
        raise ValueError(f'ISS: missing or mismatched description for {secid}')
    if fields.get('SECID') == secid:
        identity = 'description_secid'
    else:
        boards = _block(envelope, 'boards', ['secid'])
        codes = boards['secid'].drop_nulls().unique().to_list()
        if codes and codes != [secid]:
            raise ValueError(f'ISS: mismatched board identifiers for {secid}')
        identity = 'boards_secid' if codes else 'unverified'
    # Explicit string schema: an all-null field today must not become DOUBLE.
    record = {'secid': secid, 'identity_status': identity, **{key: fields.get(key) for key in (
        'ASSETCODE', 'FRSTTRADE', 'LSTTRADE', 'LSTDELDATE', 'LOTSIZE', 'UNIT',
        'FACEUNIT', 'EXECTYPE', 'GROUP', 'TYPE', 'TYPENAME', 'DELIVERYTYPE')},
        'raw_json': json.dumps(envelope['payload'], ensure_ascii=False)}
    return _provenance(pl.DataFrame([record], schema={key: pl.String for key in record}), envelope).with_columns(
        pl.lit('observed' if frame.height else 'unavailable').alias('availability'))


def _description(args):
    secid, refresh = args
    # Hashed filenames avoid interpreting instrument identifiers as paths.
    key = hashlib.sha256(secid.encode()).hexdigest()
    path = _root() / 'descriptions' / f'{key}.json'
    envelope = json.loads(path.read_text(encoding='utf-8')) if path.exists() and not refresh else None
    if envelope is None:
        envelope = _get(f'{iss.ISS_URL}/securities/{quote(secid, safe="")}.json')
        frame = description_frame(envelope, secid)
        _save(path, envelope)
    else:
        frame = description_frame(envelope, secid)
    return frame


def capture_descriptions(secids, refresh=False, workers=4):
    secids = sorted(set(secids))
    pending, total = [], 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, frame in enumerate(pool.map(_description, [(s, refresh) for s in secids]), 1):
            pending.append(frame)
            if i % 100 == 0 or i == len(secids):
                total += lake.write('futures_description_observations', pl.concat(pending, how='diagonal_relaxed'))
                pending.clear()
                print(f'Contract descriptions: {i}/{len(secids)}', flush=True)
    return total


def read_observations(secid, descriptions=False):
    """Observed snapshots only. No implicit carry-forward or historical validity."""
    table = 'futures_description_observations' if descriptions else 'futures_parameter_observations'
    column = 'secid' if descriptions else 'SECID'
    return lake.query(f'SELECT * FROM lake.{table} WHERE {column} = ? ORDER BY observed_at', [secid])


def read_risk_limits(start, end, asset=None):
    """Latest retrieved response per date, preserving all source update times."""
    args = [start, end] + ([asset] if asset else [])
    return lake.query('SELECT * FROM (SELECT * FROM lake.futures_risk_limits WHERE date BETWEEN ? AND ? '
                      'QUALIFY observed_at = max(observed_at) OVER (PARTITION BY date)) '
                      + ('WHERE assetcode = ? ' if asset else '') +
                      'ORDER BY date, assetcode, updatetime', args)


def update():
    current = capture_current()
    capture_descriptions(current['SECID'].to_list(), refresh=True)
    update_risk_limits()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backfill-risk', action='store_true')
    parser.add_argument('--all-descriptions', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    current = capture_current()
    print(f'Current parameters: {current.height}', flush=True)
    update_risk_limits(start=risk_bounds()[0] if args.backfill_risk else None, workers=args.workers)
    secids = current['SECID'].to_list()
    if args.all_descriptions:
        secids += contracts.fetch_contracts()['secid'].to_list()
        secids += lake.query('SELECT DISTINCT SECID FROM lake.futures')['SECID'].to_list()
    capture_descriptions(secids, workers=args.workers)
