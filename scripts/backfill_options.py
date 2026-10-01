"""Resumable options backfill: parallel date downloads, sequential DuckLake writes.

python scripts/backfill_options.py --start 2025-01-01 --end 2026-09-30 --workers 8
Each complete date is saved immediately; failed dates are retried on the next run.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import time

import polars as pl

from moexutils import history, iss, lake


def download(day):
    with iss.make_session() as session:
        return history._fetch('options', day, session)


def backfill(start: dt.date, end: dt.date, workers: int = 4) -> dict:
    if start > end:
        raise ValueError('start must not exceed end')
    if not 1 <= workers <= 8:
        raise ValueError('workers must be between 1 and 8')
    if end >= dt.date.today():
        raise ValueError('end must be before today: only completed trading dates')
    stored = set(history.dataset_dates('options'))
    empty = history.empty_dates('options')
    todo = [d for d in history._all_days(start, end) if d not in stored and d not in empty]
    rows, completed, errors = 0, 0, []
    started = time.monotonic()
    print(f'Options: {len(todo)} pending calendar dates, {workers} workers', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(download, day): day for day in todo}
        for future in as_completed(jobs):
            day = jobs[future]
            try:
                frame = future.result()
                if frame.height:
                    if frame['date'].unique().to_list() != [day]:
                        raise ValueError(f'ISS returned a different date for {day}')
                    lake.write('options', frame)
                    rows += frame.height
                else:
                    lake.write('empty_dates', pl.DataFrame({'dataset': ['options'], 'date': [day]}))
                completed += 1
                print(f'{day}: {frame.height} rows; completed {completed}/{len(todo)}; '
                      f'total {rows}; elapsed {time.monotonic() - started:.0f}s', flush=True)
            except Exception as e:
                errors.append(e)
                print(f'ERROR {day}: {e}', flush=True)
    if errors:
        raise ExceptionGroup('Options backfill failed on some dates; rerun to resume', errors)
    return {'dates_processed': completed, 'rows_written': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=dt.date.fromisoformat, default=dt.date(2025, 1, 1))
    parser.add_argument('--end', type=dt.date.fromisoformat, default=dt.date.today() - dt.timedelta(days=1))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    print(backfill(args.start, args.end, args.workers), flush=True)


if __name__ == '__main__':
    main()
