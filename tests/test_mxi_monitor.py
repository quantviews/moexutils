from datetime import date, datetime, timedelta, timezone

import pytest
import json
from contextlib import contextmanager

from moexutils.mxi_monitor import calendar, diff, market_evidence, parse_positions, schedule
from moexutils import mxi_monitor


def test_schedule_budget_midnight_and_fixed_end():
    start = datetime(2026, 10, 24, 12, tzinfo=timezone.utc)
    jobs = schedule(start)
    assert len(jobs) == 425
    assert len({j['id'] for j in jobs}) == 425
    assert sum(j['kind'] == 'recent' for j in jobs) == 420
    assert all(start <= datetime.fromisoformat(j['due']) < start+timedelta(days=30) for j in jobs)
    slots = [j for j in jobs if j['id'].startswith('cycle-00')]
    assert len(slots) == 14
    assert datetime.fromisoformat(slots[0]['due']).hour == 20


def test_calendar_not_positions_or_calendar_weekends():
    days = calendar(date(2026, 11, 5))
    assert date(2026, 5, 1) not in days
    assert date(2026, 6, 12) not in days
    assert date(2026, 11, 4) not in days
    assert date(2026, 10, 24) not in days
    assert date(2026, 5, 11) in days
    assert len(days) >= 140


def payload():
    columns = ['tradedate', 'asset', 'is_fiz', 'persons_long', 'persons_short',
        'open_position_long', 'open_position_short', 'oichange_long', 'oichange_short']
    return {'open_positions': {'columns': columns, 'data': [['2026-10-23', 'MXI', 1, 1, 2, 10, 4, None, -2]]}}


def test_null_preserved_and_bad_responses_not_absence():
    p = payload()
    r = parse_positions(p, date(2026, 10, 10), date(2026, 10, 23))
    assert r[0]['oichange_long'] is None
    p['open_positions']['data'] *= 2
    with pytest.raises(ValueError, match='Duplicate'):
        parse_positions(p, date(2026, 10, 10), date(2026, 10, 23))
    with pytest.raises(ValueError, match='pagination'):
        parse_positions({**payload(), 'cursor': {}}, date(2026, 10, 10), date(2026, 10, 23))


def test_historical_inserts_are_not_sign_revisions():
    r = dict(date='2020-09-15', is_fiz=1, open_position_long=2, open_position_short=5)
    assert diff({}, {'2020-09-15|1': r})[0]['physical_sign_changed'] is False
    changed = {**r, 'open_position_long': 10}
    assert diff({'2020-09-15|1': r}, {'2020-09-15|1': changed})[0]['physical_sign_changed'] is True


def market(day='2026-10-26'):
    return {'securities': {'columns': ['SECID', 'ASSETCODE', 'BOARDID', 'FRSTTRADE', 'LASTTRADEDATE'],
            'data': [['MMZ6', 'MXI', 'RFUD', '2026-01-01', '2026-12-17']]},
        'marketdata': {'columns': ['SECID', 'TRADEDATE', 'TRADE_SESSION_DATE', 'SYSTIME', 'TIME', 'NUMTRADES'],
            'data': [['MMZ6', day, day, day+' 09:15:00', '09:14:00', 100]]}}


def test_current_activity_required_and_weekend_session_not_shifted():
    received = datetime(2026, 10, 26, 6, 30, tzinfo=timezone.utc)
    assert market_evidence(market(), date(2026, 10, 26), received)
    with pytest.raises(ValueError, match='No fresh'):
        market_evidence(market('2026-10-23'), date(2026, 10, 26), received)
    with pytest.raises(ValueError, match='No fresh'):
        market_evidence(market(), date(2026, 10, 26), received+timedelta(hours=1))


def test_restart_does_not_extend_period(tmp_path):
    original = {'started_at': '2026-10-24T12:00:00+00:00', 'ends_at': '2026-11-23T12:00:00+00:00'}
    (tmp_path/'start.json').write_text(json.dumps(original))
    assert mxi_monitor.initialize(tmp_path) == original


def test_late_delivery_does_not_publish_or_download(tmp_path, monkeypatch):
    (tmp_path/'deliveries').mkdir()
    monkeypatch.setattr(mxi_monitor, 'now', lambda: datetime(2026, 10, 26, 8, tzinfo=timezone.utc))
    monkeypatch.setattr(mxi_monitor.lake, 'DATA_ROOT', tmp_path)
    def forbidden(*a, **kw):
        raise AssertionError('Unexpected network request')
    monkeypatch.setattr(mxi_monitor, 'get', forbidden)
    mxi_monitor.forward(tmp_path, force_check=True)
    report = json.loads((tmp_path/'deliveries/2026-10-26/status.json').read_text())
    assert report['status'] == 'blocked'
    assert 'Past 10:00' in report['reason']
    assert not (tmp_path/'reports/mxi-forward-input/latest.json').exists()


def test_forward_publishes_only_verified_current_inputs(tmp_path, monkeypatch):
    import polars as pl
    moment = datetime(2026, 10, 26, 6, 30, tzinfo=timezone.utc)
    day = moment.date()
    dates = calendar(day)
    (tmp_path/'deliveries').mkdir()
    (tmp_path/'rules').mkdir()
    capture = tmp_path/'rules/browser-capture.json'
    capture.write_text('{}')
    mxi_monitor.save(tmp_path/'rules/browser-review.json', {
        'urls': mxi_monitor.RULE_URLS, 'capture_sha256': mxi_monitor.sha(capture)})
    rows = {f'{d}|{g}': dict(date=str(d), asset='MXI', is_fiz=g,
        open_position_long=20, open_position_short=10) for d in dates[:-1] for g in (0, 1)}
    mxi_monitor.save(tmp_path/'cache.json', dict(rows=rows, last_recent_at=moment.isoformat(),
        last_cache_at=moment.isoformat(), last_success='fixture'))
    class Connection:
        def execute(self, *args):
            return self
        def fetchone(self):
            return (123,)
        def pl(self):
            return pl.DataFrame({'date': dates[:-1]})
    @contextmanager
    def session(**kwargs):
        yield Connection()
    class Response:
        def json(self):
            return market()
    monkeypatch.setattr(mxi_monitor, 'now', lambda: moment)
    monkeypatch.setattr(mxi_monitor.lake, 'DATA_ROOT', tmp_path)
    monkeypatch.setattr(mxi_monitor.lake, 'session', session)
    monkeypatch.setattr(mxi_monitor, 'get', lambda *a, **k: (Response(), moment))
    mxi_monitor.forward(tmp_path)
    pointer = mxi_monitor.load(tmp_path/'reports/mxi-forward-input/latest.json')
    assert pointer['session_date'] == str(day)
    assert pointer['available_at'] == moment.isoformat()
    for name in ['positions', 'calendar', 'tradable']:
        from pathlib import Path
        assert mxi_monitor.sha(Path(pointer[name]['path'])) == pointer[name]['sha256']
    assert pl.read_parquet(pointer['tradable']['path'])['tradable'].to_list() == [True]
