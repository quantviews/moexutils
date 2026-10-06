"""Verify a completed scheduled run and its FORTS report without loading data."""
import argparse
import datetime as dt
import json
from pathlib import Path
import re
import time


def check(log_text, not_before):
    starts = list(re.finditer(r'^===== start (\d{2}\.\d{2}\.\d{4})\s+(\d{1,2}:\d{2}:\d{2})', log_text, re.M))
    if not starts:
        return {'status': 'pending', 'reason': 'no_run'}
    match = starts[-1]
    started = dt.datetime.strptime(' '.join(match.groups()), '%d.%m.%Y %H:%M:%S').astimezone()
    if started.date() < dt.date.fromisoformat(not_before):
        return {'status': 'pending', 'reason': 'next_run_not_started'}
    block = log_text[match.end():]
    end = re.search(r'^===== exit (\d+) ', block, re.M)
    if end is None:
        return {'status': 'pending', 'reason': 'run_not_finished'}
    result = {'status': 'failed', 'run_started': started.isoformat(), 'exit_code': int(end[1])}
    if int(end[1]) != 0:
        return {**result, 'reason': 'update_failed'}
    report = re.search(r'^\[OK\].*FORTS: (.+)$', block[:end.start()], re.M)
    if report is None:
        return {**result, 'reason': 'audit_success_line_missing'}
    path = Path(report[1].strip())
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
        generated = dt.datetime.fromisoformat(payload['generated_at'])
        if generated.tzinfo is None or generated < started:
            raise ValueError('stale_or_naive_report_timestamp')
        sections = payload['sections']
        if not sections['coverage'] or not sections['asset_year_coverage'] or not sections['rolls']:
            raise ValueError('empty_report_section')
        if payload['schema_version'] != 1 or payload['snapshot_id'] is None:
            raise ValueError('missing_snapshot_or_unknown_schema')
    except (OSError, ValueError, KeyError, TypeError) as e:
        return {**result, 'reason': 'invalid_report', 'error': str(e), 'report': str(path)}
    missing = [name for name in ('staticparams:', 'staticparamskeyterm:', 'rclimits:')
               if name not in block[:end.start()]]
    if missing:
        return {**result, 'reason': 'rms_updates_missing', 'missing': missing}
    return {**result, 'status': 'passed', 'report': str(path),
            'snapshot_id': payload['snapshot_id'], 'coverage_rows': len(sections['coverage']),
            'rolls': len(sections['rolls'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--not-before', required=True, help='Local date YYYY-MM-DD')
    parser.add_argument('--wait-minutes', type=float, default=0)
    parser.add_argument('--log', type=Path, default=Path(__file__).resolve().parents[1] / 'logs/update.log')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'logs/nightly-futures-verification.json')
    args = parser.parse_args()
    deadline = time.monotonic() + max(0, args.wait_minutes) * 60
    while True:
        text = args.log.read_text(encoding='utf-8', errors='replace') if args.log.exists() else ''
        result = check(text, args.not_before)
        if result['status'] != 'pending' or time.monotonic() >= deadline:
            break
        time.sleep(min(30, max(0, deadline - time.monotonic())))
    result['checked_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix('.tmp')
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(args.output)
    return {'passed': 0, 'failed': 1, 'pending': 2}[result['status']]


if __name__ == '__main__':
    raise SystemExit(main())
