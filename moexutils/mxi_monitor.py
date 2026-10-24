"""Bounded MXI observations and fail-closed forward inputs. No orders or signals."""
import argparse
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import polars as pl
import requests

from moexutils import lake

MSK = ZoneInfo('Europe/Moscow')
POSITION_URL = 'https://iss.moex.com/iss/statistics/engines/futures/markets/forts/openpositions/MXI.json'
BOARD_URL = 'https://iss.moex.com/iss/engines/futures/markets/forts/boards/RFUD/securities.json'
COLS = ['date', 'asset', 'is_fiz', 'open_position_long', 'open_position_short']
COUNTERS = ['persons_long', 'persons_short', 'open_position_long', 'open_position_short', 'oichange_long', 'oichange_short']
CALENDAR_START = date(2026, 4, 1)
EXCLUDED = {date(2026, 5, 1), date(2026, 6, 12), date(2026, 11, 4)}
RULE_URLS = ['https://www.moex.com/n96571', 'https://www.moex.com/n105220', 'https://www.moex.com/n103379']


def now():
    return datetime.now(timezone.utc)


def stamp(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError('Missing timezone')
    return result


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value, immutable=False):
    payload = json.dumps(value, ensure_ascii=False, default=str, indent=2, allow_nan=False)+'\n'
    if immutable:
        with path.open('x', encoding='utf-8') as f:
            f.write(payload)
    else:
        temp = path.with_name(path.name+'.'+uuid4().hex+'.tmp')
        temp.write_text(payload, encoding='utf-8')
        os.replace(temp, path)


@contextmanager
def locked(root):
    import msvcrt
    with (root/'runner.lock').open('a+b') as f:
        f.seek(0)
        if not f.read(1):
            f.write(b'0')
            f.flush()
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def manifest(root):
    save(root/'manifest.json', {'artifacts': {p.relative_to(root).as_posix(): sha(p)
        for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'manifest.json'}})


def schedule(start):
    end = start+timedelta(days=30)
    jobs = []
    for i in range(5):
        due = start+timedelta(days=7*i)
        jobs.append(dict(id=f'weekly-{i}', kind='weekly', due=due.isoformat()))
    first = start.astimezone(MSK).replace(hour=23, minute=40, second=0, microsecond=0)
    if first < start:
        first += timedelta(days=1)
    for i in range(30):
        night = first+timedelta(days=i)
        slots = [night+timedelta(minutes=10*j) for j in range(12)]
        morning = (night+timedelta(days=1)).replace(hour=3, minute=0)
        slots += [morning, morning.replace(hour=9)]
        for j, due in enumerate(slots):
            if due < end:
                jobs.append(dict(id=f'cycle-{i:02d}-{j:02d}', kind='recent', due=due.astimezone(timezone.utc).isoformat()))
    return sorted(jobs, key=lambda r: r['due'])


def calendar(end):
    if not CALENDAR_START <= end <= date(2026, 11, 30):
        raise ValueError('Calendar reviewed only for April-November 2026')
    return [CALENDAR_START+timedelta(days=i) for i in range((end-CALENDAR_START).days+1)
            if (CALENDAR_START+timedelta(days=i)).weekday() < 5 and CALENDAR_START+timedelta(days=i) not in EXCLUDED]


def get(url, params, folder):
    started = now()
    save(folder/'request.json', dict(url=url, params=params, started_at=started.isoformat(), retries=0, redirects=False), True)
    try:
        with requests.Session() as session:
            session.mount('https://', requests.adapters.HTTPAdapter(max_retries=0))
            r = session.get(url, params=params, timeout=(10, 45), allow_redirects=False)
        received = now()
        (folder/'response.bin').write_bytes(r.content)
        save(folder/'http.json', dict(status=r.status_code, url=r.url, observed_at=received.isoformat(),
             observed_moscow=received.astimezone(MSK).isoformat(), response_sha256=sha(folder/'response.bin'),
             published_at=None, content_type=r.headers.get('Content-Type')), True)
        if r.status_code != 200:
            raise ValueError(f'HTTP {r.status_code}')
        return r, received
    except Exception as error:
        save(folder/'error.json', dict(error=str(error), observed_at=now().isoformat(), absence_proven=False), True)
        raise


def block(payload, key):
    obj = payload[key]
    cols, rows = obj['columns'], obj['data']
    if not isinstance(cols, list) or len(set(cols)) != len(cols) or not isinstance(rows, list):
        raise ValueError('Invalid ISS block')
    if any(not isinstance(r, list) or len(r) != len(cols) for r in rows):
        raise ValueError('Invalid ISS row')
    return [dict(zip(cols, r)) for r in rows]


def parse_positions(payload, start, end):
    for key in payload:
        if 'cursor' in key:
            raise ValueError('Unexpected pagination; cannot prove complete response')
    rows = block(payload, 'open_positions')
    required = {'tradedate', 'asset', 'is_fiz', *COUNTERS}
    if not required <= set(payload['open_positions']['columns']):
        raise ValueError('Missing columns')
    result = []
    for row in rows:
        d = date.fromisoformat(row.pop('tradedate'))
        if row['asset'] != 'MXI' or row['is_fiz'] not in (0, 1) or not start <= d <= end:
            raise ValueError('Wrong identity/date')
        for c in ['is_fiz', *COUNTERS]:
            if row[c] is not None:
                if float(row[c]) != int(row[c]):
                    raise ValueError('Non-integral counter')
                row[c] = int(row[c])
        if any(row[c] is not None and row[c] < 0 for c in COUNTERS[:4]):
            raise ValueError('Negative count')
        result.append({**row, 'date': d.isoformat()})
    keys = [(r['date'], r['is_fiz']) for r in result]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate key')
    return result


def key(row):
    return row['date']+'|'+str(row['is_fiz'])


def diff(old, new):
    result = []
    for k in sorted(old.keys() | new.keys()):
        a, b = old.get(k), new.get(k)
        if a == b:
            continue
        op = 'insert' if a is None else 'delete' if b is None else 'update'
        def net(r):
            if r is None or r['open_position_long'] is None or r['open_position_short'] is None:
                return None
            return r['open_position_long']-r['open_position_short']
        x, y = net(a), net(b)
        result.append(dict(key=k, operation=op, before=a, after=b,
            physical_sign_changed=k.endswith('|1') and op == 'update' and x is not None and y is not None
                 and (x > 0)-(x < 0) != (y > 0)-(y < 0)))
    return result


def initialize(root):
    if (root/'start.json').exists():
        return load(root/'start.json')
    root.mkdir(parents=True, exist_ok=True)
    for name in ['attempts', 'deliveries', 'rules']:
        (root/name).mkdir(exist_ok=True)
    start = now()
    state = dict(id=root.name, started_at=start.isoformat(), ends_at=(start+timedelta(days=30)).isoformat(),
        started_moscow=start.astimezone(MSK).isoformat(), ends_moscow=(start+timedelta(days=30)).astimezone(MSK).isoformat(),
        position_request_limit=425, forward_request_limit=30, jobs=schedule(start),
        code_sha256=sha(Path(__file__)), task_name='MOEX MXI observations 30d',
        calendar_start=CALENDAR_START.isoformat(), monitoring_only=True)
    save(root/'start.json', state, True)
    save(root/'cache.json', dict(rows={}, first_seen={}, last_absent={}, last_success=None, last_cache_at=None, last_recent_at=None))
    for i, url in enumerate(RULE_URLS):
        folder = root/'rules'/str(i)
        folder.mkdir()
        try:
            get(url, None, folder)
        except Exception:
            pass
        manifest(folder)
    save(root/'rules/review.json', dict(reviewed_at=now().isoformat(), urls=RULE_URLS,
        calendar_policy='weekday trade dates; May 1, June 12, November 4 are DSVD assigned to next trade date',
        no_session_dates=['2026-10-24', '2026-10-25'], scope_end='2026-11-30'))
    manifest(root/'rules')
    return state


def observe(root, job):
    folder = root/'attempts'/job['id']
    folder.mkdir()  # A reserved slot is never reissued, even after a crash.
    save(folder/'slot.json', job, True)
    moment = now()
    today = moment.astimezone(MSK).date()
    first = date(2020, 9, 15) if job['kind'] == 'weekly' else today-timedelta(days=14)
    last = today-timedelta(days=15 if job['kind'] == 'weekly' and job['id'] != 'weekly-0' else 1)
    cache = load(root/'cache.json')
    outcome = dict(slot=job['id'], kind=job['kind'], start=str(first), end=str(last), status='failed', absence_proven=False)
    try:
        response, received = get(POSITION_URL, {'from': str(first), 'till': str(last)}, folder)
        rows = parse_positions(response.json(), first, last)
        new = {key(r): r for r in rows}
        old = {k: r for k, r in cache['rows'].items() if str(first) <= r['date'] <= str(last)}
        changes = diff(old, new)
        save(folder/'changes.json', changes, True)
        evidence = []
        day = first
        while day <= last:
            for group in (0, 1):
                k = str(day)+'|'+str(group)
                if k in new:
                    cache['first_seen'].setdefault(k, received.isoformat())
                evidence.append(dict(date=str(day), is_fiz=group, present=k in new,
                    first_seen_at=cache['first_seen'].get(k), last_seen_absent_at=cache['last_absent'].get(k),
                    observed_at=received.isoformat(), published_at=None))
                if k not in new:
                    cache['last_absent'][k] = received.isoformat()
            day += timedelta(days=1)
        save(folder/'availability.json', evidence, True)
        save(folder/'positions.json', rows, True)
        for k in old:
            cache['rows'].pop(k)
        cache['rows'].update(new)
        cache['last_success'] = str(folder)
        cache['last_cache_at'] = now().isoformat()
        if job['kind'] == 'recent' or job['id'] == 'weekly-0':
            cache['last_recent_at'] = cache['last_cache_at']
        save(root/'cache.json', cache)
        outcome.update(status='success', absence_proven=True, observed_at=received.isoformat(),
            applied_at=cache['last_cache_at'], rows=len(rows),
            counts={op: sum(r['operation'] == op for r in changes) for op in ['insert','update','delete']},
            revisions=[r for r in changes if r['operation'] != 'insert'],
            physical_sign_changes=sum(r['physical_sign_changed'] for r in changes))
    except Exception as error:
        outcome['error'] = str(error)
    save(folder/'result.json', outcome, True)
    manifest(folder)
    return outcome


def market_evidence(payload, day, received):
    securities = {r['SECID']: r for r in block(payload, 'securities')
        if r['ASSETCODE'] == 'MXI' and r['BOARDID'] == 'RFUD'
        and r['FRSTTRADE'] <= str(day) <= r['LASTTRADEDATE']}
    evidence = []
    for r in block(payload, 'marketdata'):
        if r['SECID'] not in securities or r.get('TRADEDATE') != str(day) or r.get('TRADE_SESSION_DATE') != str(day):
            continue
        systime = datetime.fromisoformat(r['SYSTIME']).replace(tzinfo=MSK)
        trade_time = datetime.combine(day, datetime.strptime(r['TIME'], '%H:%M:%S').time(), MSK)
        if (r.get('NUMTRADES') or 0) > 0 and timedelta(0) <= received-systime <= timedelta(minutes=30) and timedelta(0) <= received-trade_time <= timedelta(minutes=30):
            evidence.append(dict(secid=r['SECID'], systime=r['SYSTIME'], trade_time=r['TIME'],
                                 numtrades=r['NUMTRADES'], session_date=r['TRADE_SESSION_DATE']))
    if not evidence:
        raise ValueError('No fresh MXI trade evidence for current session; tradability unknown')
    return evidence


def forward(root, force_check=False):
    moment = now()
    local = moment.astimezone(MSK)
    day = local.date()
    folder = root/'deliveries'/str(day)
    if folder.exists():
        return
    folder.mkdir()
    status = dict(session_date=str(day), checked_at=moment.isoformat(), status='blocked', published_ready=False)
    try:
        if local.hour >= 10:
            raise ValueError('Past 10:00 MSK; no retrospective publication')
        dates = calendar(day)
        if day not in dates:
            raise ValueError('Not an independent trade date; DSVD must not create a new calendar day')
        if not force_check and not (local.hour == 9 and 30 <= local.minute <= 35):
            raise ValueError('Outside delivery window')
        # Fixed rules must have real archived official responses, not only configured URLs.
        for i in range(3):
            http = root/'rules'/str(i)/'http.json'
            if not http.exists() or load(http)['status'] != 200:
                review = root/'rules/browser-review.json'
                capture = root/'rules/browser-capture.json'
                if not review.exists() or not capture.exists():
                    raise ValueError('Official calendar evidence unavailable')
                evidence_review = load(review)
                if evidence_review['urls'] != RULE_URLS or evidence_review['capture_sha256'] != sha(capture):
                    raise ValueError('Reviewed browser calendar evidence mismatch')
        with lake.session(read_only=True) as con:
            snap = con.execute("SELECT max(snapshot_id) FROM ducklake_snapshots('lake')").fetchone()[0]
            observed = con.execute(f"SELECT DISTINCT date FROM lake.futures AT (VERSION => {int(snap)}) WHERE BOARDID='RFUD' AND date >= ? AND date < ? ORDER BY date", [CALENDAR_START, day]).pl()
        expected = [d for d in dates if d < day]
        if observed['date'].to_list() != expected:
            raise ValueError('Observed FORTS calendar has missing or extra dates relative to reviewed official rules')
        save(folder/'calendar-lineage.json', dict(snapshot_id=snap, observed_at=now().isoformat(),
            source='lake.futures RFUD all assets; independent of openpositions', rules=str(root/'rules'), dates=expected), True)
        if len(dates) < 140:
            raise ValueError('Calendar warmup insufficient')
        if len(list((root/'deliveries').glob('*/request.json'))) >= 30:
            raise ValueError('Forward market-data request budget exhausted')
        response, received = get(BOARD_URL, None, folder)
        evidence = market_evidence(response.json(), day, received)
        cache = load(root/'cache.json')
        if not cache['last_recent_at'] or now()-stamp(cache['last_recent_at']) > timedelta(hours=18):
            raise ValueError('Position cache stale')
        rows = list(cache['rows'].values())
        positions = pl.from_dicts(rows).select(COLS).with_columns(pl.col('date').str.to_date(),
            pl.col('is_fiz').cast(pl.Int8), pl.col('open_position_long','open_position_short').cast(pl.Int64)).sort('date','is_fiz')
        source_days = dates[:-2][-126:]
        subset = positions.filter(pl.col('date').is_in(source_days))
        if len(source_days) < 126 or subset.height != 252 or subset.select(pl.any_horizontal(pl.all().is_null()).any()).item():
            raise ValueError('Missing groups or null counts in 126 source sessions')
        counts = subset.group_by('date').agg(pl.col('is_fiz').n_unique().alias('groups'))
        if counts.filter(pl.col('groups') != 2).height:
            raise ValueError('Group coverage incomplete')
        positions.write_parquet(folder/'positions.parquet')
        pl.DataFrame({'date': dates}).write_parquet(folder/'calendar.parquet')
        pl.DataFrame({'date': [day], 'tradable': [True]}).write_parquet(folder/'tradable.parquet')
        save(folder/'lineage.json', dict(position_cache_at=cache['last_cache_at'],
             position_observation=cache['last_success'], market_evidence=evidence, snapshot_id=snap,
             evidence_scope='recent actual MXI trades, not a guarantee of execution at 10:00',
             monitor_start=str(root/'start.json')), True)
        available = now()
        if available.astimezone(MSK).hour >= 10:
            raise ValueError('Build finished after cutoff')
        pointer = dict(available_at=available.isoformat(), session_date=str(day), calendar_complete_through=str(day),
            calendar_source='MOEX n96571 + observed complete RFUD session dates + fresh current-session MXI trades',
            **{name: dict(path=str((folder/(name+'.parquet')).resolve()), sha256=sha(folder/(name+'.parquet')),
                         available_at=available.isoformat()) for name in ['positions','calendar','tradable']})
        save(folder/'pointer.json', pointer, True)
        target = Path(lake.DATA_ROOT)/'reports/mxi-forward-input'
        target.mkdir(parents=True, exist_ok=True)
        save(target/'latest.json', pointer)
        status.update(status='ready', published_ready=True, available_at=available.isoformat())
    except Exception as error:
        status['reason'] = str(error)
    save(folder/'status.json', status, True)
    manifest(folder)
    target = Path(lake.DATA_ROOT)/'reports/mxi-forward-input'
    target.mkdir(parents=True, exist_ok=True)
    save(target/'status.json', status)


def report(root, state):
    events = []
    for p in sorted((root/'attempts').glob('*/slot.json')):
        result = p.parent/'result.json'
        events.append(load(result) if result.exists() else dict(slot=p.parent.name, status='interrupted_reserved_no_retry'))
    summary = dict(started_at=state['started_at'], ends_at=state['ends_at'], checked_at=now().isoformat(),
        stopped=now() >= stamp(state['ends_at']), attempts=len(events), request_limit=425,
        successful=sum(r['status'] == 'success' for r in events), failed=sum(r['status'] != 'success' for r in events),
        events=events, published_at_known=False)
    save(root/'summary.json', summary)
    if summary['stopped'] and not (root/'final-summary.json').exists():
        save(root/'final-summary.json', summary, True)


def tick(root, bootstrap=False):
    state = load(root/'start.json')
    expected_code = state['code_sha256']
    revision = root/'code-revision.json'
    if revision.exists():
        revised = load(revision)
        if revised['from_sha256'] != expected_code:
            raise ValueError('Code revision does not match original monitor')
        expected_code = revised['to_sha256']
    if sha(Path(__file__)) != expected_code:
        raise ValueError('Frozen monitor code changed; explicit migration required')
    with locked(root):
        moment = now()
        if moment >= stamp(state['ends_at']):
            report(root, state)
            return
        attempted = len(list((root/'attempts').glob('*/slot.json')))
        for job in state['jobs']:
            if (root/'attempts'/job['id']).exists():
                continue
            delay = moment-stamp(job['due'])
            if timedelta(0) <= delay <= timedelta(minutes=5) or bootstrap and job['id'] == 'weekly-0':
                if attempted >= 425:
                    break
                observe(root, job)
                attempted += 1
        local = moment.astimezone(MSK)
        if bootstrap or local.hour == 9 and 30 <= local.minute <= 35:
            forward(root, force_check=bootstrap)
        report(root, state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['init', 'tick'])
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'init':
        initialize(args.root)
        tick(args.root, bootstrap=True)
    else:
        tick(args.root)


if __name__ == '__main__':
    main()
