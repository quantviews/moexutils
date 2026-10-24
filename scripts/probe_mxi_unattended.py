"""One-shot dependency check under the scheduled task's Windows logon token."""
import argparse
from datetime import date
import json
import os
from pathlib import Path
import sys

import requests

from moexutils import lake, mxi_monitor as m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = dict(started_at=m.now().isoformat(), user=os.environ.get('USERNAME'),
                  data_root=lake.DATA_ROOT, checks={}, published_forward_input=False)
    checks = result['checks']
    try:
        if Path(lake.DATA_ROOT).resolve() != args.root.parent.parent.resolve():
            raise ValueError('Wrong MOEX_DATA_ROOT')
        state = m.load(args.root/'start.json')
        revision = m.load(args.root/'code-revision.json')
        if revision['from_sha256'] != state['code_sha256'] or revision['to_sha256'] != m.sha(Path(m.__file__)):
            raise ValueError('Frozen code mismatch')
        checks['code_and_state'] = True
        probe = args.output.parent/'write-probe.json'
        m.save(probe, {'step': 1})
        m.save(probe, {'step': 2})
        if m.load(probe)['step'] != 2:
            raise ValueError('Atomic write failed')
        checks['observation_write_replace'] = True
        forward_probe = Path(lake.DATA_ROOT)/'reports/mxi-forward-input'/('write-probe-'+args.output.parent.name+'.json')
        m.save(forward_probe, {'probe': True, 'available_at': m.now().isoformat()}, True)
        checks['forward_directory_write'] = True
        with lake.session(read_only=True) as con:
            snapshot = con.execute("SELECT max(snapshot_id) FROM ducklake_snapshots('lake')").fetchone()[0]
            dates = con.execute(f"SELECT DISTINCT date FROM lake.futures AT (VERSION => {int(snapshot)}) "
                "WHERE BOARDID='RFUD' AND date >= ? AND date <= ? ORDER BY date",
                [date(2026, 4, 1), date(2026, 10, 23)]).pl()['date'].to_list()
        if dates != m.calendar(date(2026, 10, 23)):
            raise ValueError('Calendar mismatch')
        checks['postgres_ducklake_parquet_calendar'] = dict(snapshot_id=snapshot, dates=len(dates))
        with requests.Session() as session:
            session.mount('https://', requests.adapters.HTTPAdapter(max_retries=0))
            response = session.get('https://iss.moex.com/iss/engines.json', timeout=(10, 45), allow_redirects=False)
        (args.output.parent/'iss-probe-response.bin').write_bytes(response.content)
        if response.status_code != 200 or 'engines' not in response.json():
            raise ValueError('ISS probe failed')
        checks['iss_https'] = dict(status=response.status_code, requests=1,
            sha256=m.sha(args.output.parent/'iss-probe-response.bin'))
        # Exercise the real dispatcher; it keeps the original deadline and slot reservations.
        m.tick(args.root)
        checks['actual_monitor_tick'] = True
        result['success'] = True
    except Exception as error:
        result.update(success=False, error_type=type(error).__name__)
        # Do not log connection exceptions: they may contain credentials.
    result['finished_at'] = m.now().isoformat()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return 0 if result['success'] else 1


if __name__ == '__main__':
    sys.exit(main())
