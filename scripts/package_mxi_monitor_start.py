"""Freeze start evidence and a read-only calendar check in a new handoff directory."""
import argparse
from datetime import date
from pathlib import Path
import shutil

import polars as pl

from moexutils import lake, mxi_monitor as m


def build(root, output):
    output.mkdir(parents=True, exist_ok=False)
    for name in ['start.json', 'code-revision.json', 'monitor-initial.py',
                 'task.xml', 'task-check.json', 'summary.json']:
        shutil.copy2(root/name, output/name)
    for name in ['rules', 'attempts/weekly-0', 'deliveries/2026-10-24']:
        shutil.copytree(root/name, output/name)
    shutil.copy2(Path(m.__file__), output/'monitor.py')
    with lake.session(read_only=True) as con:
        snapshot = con.execute("SELECT max(snapshot_id) FROM ducklake_snapshots('lake')").fetchone()[0]
        frame = con.execute(f"SELECT DISTINCT date FROM lake.futures AT (VERSION => {int(snapshot)}) "
            "WHERE BOARDID='RFUD' AND date >= ? AND date <= ? ORDER BY date",
            [date(2026, 4, 1), date(2026, 10, 23)]).pl()
    expected = set(m.calendar(date(2026, 10, 23)))
    actual = set(frame['date'].to_list())
    frame.write_parquet(output/'observed-session-calendar.parquet')
    cache = m.load(root/'cache.json')
    positions = pl.from_dicts(list(cache['rows'].values())).select(m.COLS).with_columns(pl.col('date').str.to_date())
    positions.write_parquet(output/'positions-start-snapshot.parquet')
    needed = m.calendar(date(2026, 10, 26))[:-2][-126:]
    warmup = positions.filter(pl.col('date').is_in(needed))
    m.save(output/'calendar-check.json', dict(checked_at=m.now().isoformat(), snapshot_id=snapshot,
        from_date='2026-04-01', through='2026-10-23', actual_dates=len(actual),
        missing=sorted(expected-actual), extra=sorted(actual-expected),
        hypothetical_next_session='2026-10-26', required_source_dates=len(needed),
        warmup_rows=warmup.height, warmup_nulls=warmup.null_count().to_dicts(),
        current_contract_before_cutoff_confirmed=False,
        reason='October 24 is closed and bootstrap was after 10:00; next session requires live evidence'), True)
    if expected != actual or warmup.height != 252:
        raise ValueError('Calendar or source warmup incomplete; inspect new package')
    shutil.copy2('docs/mxi-monitoring.md', output/'README.md')
    m.manifest(output)
    check = m.load(output/'manifest.json')['artifacts']
    assert all(m.sha(output/name) == value for name, value in check.items())
    print(dict(package=str(output), manifest_sha256=m.sha(output/'manifest.json'), verified_files=len(check)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.root, args.output)
