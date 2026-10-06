"""Published FORTS risk inputs. These are not reconstructed contract margins."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import hashlib
import json
from pathlib import Path

import polars as pl

from moexutils import futures_params as fp, iss, lake

DATASETS = ('staticparams', 'staticparamskeyterm', 'rclimits')


def _url(dataset):
    if dataset not in DATASETS:
        raise ValueError(f'Unknown RMS dataset: {dataset}')
    return f'{iss.ISS_URL}/rms/engines/futures/objects/{dataset}.json'


def bounds(dataset):
    envelope = fp._get(_url(dataset))
    frame = fp._block(envelope, dataset + '.dates', ['from', 'till'])
    if frame.height != 1:
        raise ValueError('ISS: missing archive bounds')
    return tuple(dt.date.fromisoformat(frame[c][0]) for c in ('from', 'till'))


def parse(envelope, dataset, day):
    frame = fp._block(envelope, dataset, ['tradedate', 'assetcode', 'updatetime'])
    if frame.is_empty():
        return frame
    if frame['tradedate'].null_count() or frame['tradedate'].unique().to_list() != [str(day)]:
        raise ValueError(f'ISS: wrong date for {dataset}: {day}')
    if frame['assetcode'].null_count():
        raise ValueError('ISS: missing asset')
    raw = envelope['payload'][dataset]
    # Content identity retains conflicting source rows, without guessing their key.
    hashes = [hashlib.sha256(json.dumps(dict(zip(raw['columns'], row)),
                                       sort_keys=True, ensure_ascii=False).encode()).hexdigest()
              for row in raw['data']]
    return fp._provenance(frame, envelope).with_columns(
        pl.col('tradedate').str.to_date().alias('date'),
        pl.Series('row_hash', hashes)).unique(['row_hash', 'observed_at'], maintain_order=True)


def _day(args):
    dataset, day, refresh = args
    path = Path(lake.DATA_ROOT) / 'raw' / 'futures_rms' / dataset / f'{day}.json'
    if path.exists() and not refresh:
        return parse(json.loads(path.read_text(encoding='utf-8')), dataset, day)
    pages = []

    class RecordingSession:
        def get(self, url, params):
            response = fp._session().get(url, params=params)
            response.raise_for_status()
            pages.append(response.json())
            return response

    url = _url(dataset)
    iss.fetch_pages(url, dataset, {'date': str(day), 'limit': 1000}, RecordingSession(), max_pages=100)
    columns = pages[0][dataset]['columns']
    if any(p[dataset]['columns'] != columns for p in pages):
        raise ValueError('ISS: column order changed during pagination')
    block = {**pages[0][dataset], 'data': [row for page in pages for row in page[dataset]['data']]}
    envelope = {'source_url': url + '?date=' + str(day),
                'observed_at': dt.datetime.now(dt.timezone.utc).isoformat(),
                'payload': {dataset: block}, 'pages': pages}
    frame = parse(envelope, dataset, day)
    fp._save(path, envelope)
    return frame


def update_dataset(dataset, backfill=False, workers=4):
    first, last = bounds(dataset)
    table = 'futures_' + dataset
    start = first if backfill else last - dt.timedelta(days=7)
    if not backfill and table in lake.tables():
        previous = lake.query(f'SELECT max(date) FROM lake.{table}').item()
        if previous:
            start = min(start, previous)
    start = max(first, start)
    days = [start + dt.timedelta(days=i) for i in range((last - start).days + 1)]
    pending, total, empty = [], 0, 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, frame in enumerate(pool.map(_day, [(dataset, d, d >= last - dt.timedelta(days=7)) for d in days]), 1):
            if frame.is_empty():
                empty += 1
            else:
                pending.append(frame)
            if i % 50 == 0 or i == len(days):
                if pending:
                    total += lake.write(table, pl.concat(pending, how='diagonal_relaxed'))
                    pending.clear()
                print(f'{dataset}: {i}/{len(days)} dates, {total} rows, {empty} empty', flush=True)
    return total


def read(dataset, start, end, asset=None):
    _url(dataset)  # whitelist SQL identifier
    return lake.query(f'SELECT * FROM (SELECT * FROM lake.futures_{dataset} WHERE date BETWEEN ? AND ? '
                      'QUALIFY observed_at=max(observed_at) OVER (PARTITION BY date)) '
                      + ('WHERE assetcode=? ' if asset else '') + 'ORDER BY date,assetcode,updatetime',
                      [start, end] + ([asset] if asset else []))


def update(backfill=False, workers=4):
    for dataset in DATASETS:
        update_dataset(dataset, backfill=backfill, workers=workers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backfill', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    update(args.backfill, args.workers)
