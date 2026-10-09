"""Import browser expansion captures, reporting missing contracts and cap risks."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import polars as pl

from moexutils.barchart_pilot import parse_capture

MANIFEST = Path(__file__).parents[1] / 'scripts/barchart_expansion_manifest.json'


def inspect_captures(captures, manifest):
    jobs = {r['symbol']: r for r in manifest['contracts']}
    if len(jobs) != len(manifest['contracts']):
        raise ValueError('Duplicate manifest symbol')
    responses = {}
    for capture in captures:
        if capture.get('schema_version') != 2 or capture.get('campaign') != manifest['campaign']:
            raise ValueError('Wrong capture version/campaign')
        for response in capture.get('responses', []):
            symbol = response['symbol']
            if symbol not in jobs:
                raise ValueError('Unexpected contract')
            if symbol in responses and responses[symbol]['body'] != response['body']:
                raise ValueError('Conflicting repeated contract; keep revisions separate')
            responses[symbol] = response
    if not responses:
        raise ValueError('No successful responses to import')
    frame = parse_capture({'schema_version': 1, 'responses': list(responses.values())},
                          {s: jobs[s]['asset'] for s in responses})
    coverage = []
    for symbol, job in jobs.items():
        g = frame.filter(pl.col('symbol') == symbol)
        if g.is_empty():
            coverage.append({**job, 'status': 'missing', 'rows': 0})
            continue
        limit = int(parse_qs(urlsplit(responses[symbol]['url']).query)['maxrecords'][0])
        first, last = g['date'].min(), g['date'].max()
        flags = []
        if g.height >= limit or g.height == 640:
            flags.append('possible_record_cap')
        if manifest.get('purpose') == 'historical_availability_probe':
            start = dt.date.fromisoformat(job['probe_start'])
            end = dt.date.fromisoformat(job['probe_end'])
            if first > start:
                flags.append('starts_after_probe_window')
            if last < end:
                flags.append('ends_before_probe_window')
            if g.filter(pl.col('date').is_between(start, end)).is_empty():
                flags.append('no_rows_in_probe_window')
        else:
            if first > dt.date.fromisoformat(job['moex_first']):
                flags.append('starts_after_moex')
            # A review threshold, not an exchange expiry rule.
            if last < dt.date.fromisoformat(job['moex_expiration']) - dt.timedelta(days=7):
                flags.append('ends_too_early_for_expiry_check')
        coverage.append({**job, 'status': 'review' if flags else 'coverage_candidate',
            'flags': flags, 'rows': g.height, 'first': str(first), 'last': str(last),
            'requested_maxrecords': limit, 'null_close': g['close'].null_count(),
            'zero_volume_rows': int((g['volume'] == 0).sum())})
    return frame, coverage


def import_captures(sources, output, manifest_path=MANIFEST):
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    raw = [Path(p).read_bytes() for p in sources]
    frame, coverage = inspect_captures([json.loads(b.decode('utf-8-sig')) for b in raw], manifest)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    for i, data in enumerate(raw):
        (root / f'capture-{i + 1}.json').write_bytes(data)
    (root / 'manifest.json').write_bytes(Path(manifest_path).read_bytes())
    frame.write_parquet(root / 'contracts.parquet')
    meta = {'schema_version': 1, 'campaign': manifest['campaign'],
        'status': 'partial_capture' if any(r['status'] == 'missing' for r in coverage) else 'parsed_not_execution_validated',
        'coverage': coverage, 'full_history_verified': False,
        'code_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       [Path(__file__), Path(__file__).with_name('barchart_pilot.py')]},
        'artifact_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}}
    (root / 'summary.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return meta


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--manifest', type=Path, default=MANIFEST)
    a = p.parse_args()
    result = import_captures(a.source, a.output, a.manifest)
    print(json.dumps({'status': result['status'], 'coverage': result['coverage']}, indent=2))
    raise SystemExit(1 if result['status'] == 'partial_capture' else 0)
