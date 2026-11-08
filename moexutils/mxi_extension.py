"""Morning MXI delivery after the original research campaign (Moscow dates)."""
from datetime import date, timedelta
from pathlib import Path
import tempfile

import polars as pl

from moexutils import lake, mxi_monitor as m

FIRST, LAST = date(2026, 11, 24), date(2027, 2, 26)
LIMIT = (LAST-FIRST).days+1
SOURCES = ['https://www.moex.com/n96571', 'https://www.moex.com/n104687']
# No independent trade date confirmed for these holidays or for weekends.
EXCLUDE = m.EXCLUDED | {date(2026, 12, 31), date(2027, 1, 1), date(2027, 1, 7), date(2027, 2, 23)}


def calendar(day):
    if not m.CALENDAR_START <= day <= LAST:
        raise ValueError('Outside reviewed calendar')
    days = (m.CALENDAR_START+timedelta(days=i) for i in range((day-m.CALENDAR_START).days+1))
    return [d for d in days if d.weekday() < 5 and d not in EXCLUDE]


def check_calendar(day):
    dates = calendar(day)
    with lake.session(read_only=True) as con:
        snap = int(con.execute("SELECT max(snapshot_id) FROM ducklake_snapshots('lake')").fetchone()[0])
        observed = con.execute(f"SELECT DISTINCT date FROM lake.futures AT (VERSION => {snap}) "
            "WHERE BOARDID='RFUD' AND date >= ? AND date < ? ORDER BY date", [m.CALENDAR_START, day]).pl()
    if observed['date'].to_list() != [d for d in dates if d < day]:
        raise ValueError('RFUD calendar differs from official rules; review required')
    return dates, snap


def positions(folder, day):
    # A single full response keeps warmup independent of the old campaign cache.
    start, end = date(2020, 9, 15), day-timedelta(days=1)
    resp, received = m.get(m.POSITION_URL, {'from': str(start), 'till': str(end)}, folder)
    rows = m.parse_positions(resp.json(), start, end)
    frame = pl.from_dicts(rows).select(m.COLS).with_columns(pl.col('date').str.to_date(),
        pl.col('is_fiz').cast(pl.Int8), pl.col('open_position_long','open_position_short').cast(pl.Int64))
    frame.sort('date','is_fiz').write_parquet(folder/'positions.parquet')
    return {'observed_at': received.isoformat(), 'rows': len(frame), 'published_at': None}


def build(folder, source, day):
    dates, snap = check_calendar(day)
    result = m.load(source/'status.json')
    if result['status'] != 'ready' or m.stamp(result['observed_at']).astimezone(m.MSK).date() != day:
        raise ValueError('No successful morning positions for this date')
    path = source/'positions.parquet'
    if m.sha(path) != m.load(source/'manifest.json')['artifacts']['positions.parquet']:
        raise ValueError('Morning positions checksum mismatch')
    frame = pl.read_parquet(path)
    warm = frame.filter(pl.col('date').is_in(dates[:-2][-126:]))
    if len(dates) < 140 or warm.height != 252 or warm.select(pl.any_horizontal(pl.all().is_null()).any()).item():
        raise ValueError('Incomplete calendar/position warmup')
    response, received = m.get(m.BOARD_URL, None, folder)
    evidence = m.market_evidence(response.json(), day, received)
    frame.write_parquet(folder/'positions.parquet')
    pl.DataFrame({'date': dates}).write_parquet(folder/'calendar.parquet')
    pl.DataFrame({'date':[day], 'tradable':[True]}).write_parquet(folder/'tradable.parquet')
    m.save(folder/'lineage.json', dict(snapshot_id=snap, rules=SOURCES, market_evidence=evidence,
        positions_observation=str(source), observed_at=result['observed_at']), True)
    specs = {k:dict(path=str(folder/(k+'.parquet')), sha256=m.sha(folder/(k+'.parquet')))
             for k in ['positions','calendar','tradable']}
    available = m.now().astimezone(m.MSK)
    if available.date() != day or available.hour != 9 or not 30 <= available.minute <= 35:
        raise ValueError('Missed delivery window')
    when = available.isoformat()
    pointer = dict(session_date=str(day), calendar_complete_through=str(day), available_at=when,
        calendar_source='MOEX n96571/n104687 + observed RFUD dates + current MXI trades',
        **{k:dict(**v, available_at=when) for k,v in specs.items()})
    m.save(folder/'pointer.json', pointer, True)
    return pointer


def tick(root):
    local = m.now().astimezone(m.MSK)
    day = local.date()
    ext = root/'extension-20270226'
    ext.mkdir(exist_ok=True)
    target = root.parent.parent/'reports/mxi-forward-input'
    if day < FIRST:
        # On-demand commissioning under the actual scheduler identity.
        dates, snap = check_calendar(day)
        with tempfile.TemporaryDirectory(dir=target) as tmp:
            p = Path(tmp)/'probe.parquet'
            pl.DataFrame({'date':dates}).write_parquet(p)
            assert pl.read_parquet(p).height == len(dates)
            response, _ = m.get('https://iss.moex.com/iss/engines.json', None, Path(tmp))
            assert 'engines' in response.json()
        m.save(ext/'commissioning.json', dict(checked_at=m.now().isoformat(), snapshot_id=snap,
            calendar_dates=len(dates), iss_status=response.status_code, parquet_write_read=True))
        return
    if day > LAST:
        return
    if not (root/'final-summary.json').exists():
        with m.locked(root):
            m.report(root, m.load(root/'start.json'))
    if local.hour != 9:
        return
    morning = local.minute <= 5
    if not morning and not 30 <= local.minute <= 35:
        return
    source = ext/'positions'/str(day)
    folder = source if morning else root/'deliveries'/str(day)
    if folder.exists():
        return
    folder.mkdir(parents=True)
    pointer = None
    status = dict(session_date=str(day), checked_at=m.now().isoformat(), status='blocked', published_ready=False)
    try:
        if day not in calendar(day):
            raise ValueError('No confirmed independent session')
        if morning:
            status.update(positions(folder, day))
        else:
            pointer = build(folder, source, day)
            status.update(available_at=pointer['available_at'], published_ready=True)
        status['status'] = 'ready'
    except Exception as exc:
        status['reason'] = str(exc)
    m.save(folder/'status.json', status, True)
    m.manifest(folder)
    if not morning:
        if pointer:
            m.save(target/'latest.json', pointer)
        m.save(target/'status.json', status)
    if day in calendar(day) and status['status'] != 'ready':
        raise RuntimeError(status['reason'])


if __name__ == '__main__':
    tick(Path(lake.DATA_ROOT)/'observations/mxi-20261024')
