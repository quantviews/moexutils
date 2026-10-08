"""Nightly CNY data checks; full funding comparison at least every seven days."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import polars as pl

from moexutils import lake, perpetual_audit, perpetual_data

SCHEMA = {'check': pl.String, 'object': pl.String, 'detail': pl.String}


def comparison_due(root: Path, now: dt.datetime) -> bool:
    """Only a completed versioned audit with BOTH comparisons satisfies the interval."""
    latest = None
    for path in root.glob('*/summary.json'):
        try:
            meta = json.loads(path.read_text(encoding='utf-8'))
            stamp = dt.datetime.fromisoformat(meta['generated_at'])
            if (meta.get('bundle_schema_version') == 1
                    and {'FUSR', 'FUSC'} <= set(meta.get('comparisons', {}))
                    and stamp.tzinfo is not None and stamp <= now):
                if all((path.parent / f'{b}-comparison.parquet').is_file() for b in ('FUSR', 'FUSC')):
                    latest = max(latest, stamp) if latest else stamp
        except (ValueError, KeyError, OSError):
            continue
    return latest is None or now - latest >= dt.timedelta(days=7)


def check_bundle(bundle: perpetual_data.CnyBundle) -> pl.DataFrame:
    rows = []

    def add(check, obj, detail):
        rows.append((check, obj, detail))

    frame = bundle.history.filter(pl.col('SECID') == 'CNYRUBF')
    if frame.is_empty():
        return pl.DataFrame([('perpetual_missing', 'CNYRUBF', 'No history')], schema=SCHEMA, orient='row')
    market_last = dt.date.fromisoformat(bundle.metadata['market_last_date'])
    cutoff = market_last - dt.timedelta(days=14)
    for (board,), leg in frame.group_by('BOARDID'):
        if leg['date'].max() < market_last:
            add('perpetual_stale', f'CNYRUBF/{board}', f"last={leg['date'].max()}, FORTS={market_last}")
    recent = frame.filter(pl.col('date') >= cutoff)
    for row in recent.iter_rows(named=True):
        obj = f"CNYRUBF/{row['BOARDID']}/{row['date']}"
        for field, label in [('SWAPRATE', 'funding'), ('SETTLEPRICE', 'settlement')]:
            if row[field] is None:
                add('perpetual_missing_' + label, obj, f'{field}=NULL (not zero)')
        for board in ('fusr', 'fusc'):
            if row[board + '_status'] == 'disputed_no_replacement':
                add('perpetual_funding_disputed', obj, f'{board.upper()} differs; FORTS retained')
    gaps = bundle.candidate_gaps.filter((pl.col('SECID') == 'CNYRUBF') & (pl.col('date') >= cutoff))
    for row in gaps.iter_rows(named=True):
        add('perpetual_gap_candidate', f"CNYRUBF/{row['BOARDID']}/{row['date']}",
            'Absent on market-observed date; official calendar unverified')
    fields = ('MINSTEP', 'STEPPRICE', 'LOTVOLUME', 'INITIALMARGIN')
    params = bundle.parameter_observations
    if not params.is_empty():
        for (secid, board), group in params.group_by('SECID', 'BOARDID'):
            observations = group.sort('observed_at').tail(2).to_dicts()
            if len(observations) != 2:
                continue
            before, after = observations
            for field in fields:
                if before.get(field) != after.get(field):
                    add('perpetual_parameter_changed', f'{secid}/{board}/{field}',
                        f"{before.get(field)} -> {after.get(field)}; observed={after['observed_at']}; not effective date")
    return pl.DataFrame(rows, schema=SCHEMA, orient='row')


def run() -> tuple[Path, pl.DataFrame]:
    root = Path(lake.DATA_ROOT) / 'reports/perpetual-audit'
    compare = comparison_due(root, dt.datetime.now(dt.timezone.utc))
    report = perpetual_audit.export(compare=compare)
    bundle = perpetual_data.read_cny_bundle(report)  # checks hashes and duplicate keys
    issues = check_bundle(bundle)
    issues.write_parquet(report / 'monitor-issues.parquet')
    return report, issues
