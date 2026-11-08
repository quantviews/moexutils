from datetime import date, datetime, timezone

import polars as pl
import pytest

from moexutils import mxi_extension as x


def test_extension_calendar():
    dates = x.calendar(x.LAST)
    assert x.LIMIT == 95
    assert all(date.fromisoformat(d) in dates for d in ['2027-01-04','2027-01-05',
        '2027-01-06','2027-01-08','2027-02-22','2027-02-26'])
    assert all(date.fromisoformat(d) not in dates for d in ['2026-12-31','2027-01-01',
        '2027-01-07','2027-02-23','2027-01-09'])
    with pytest.raises(ValueError):
        x.calendar(date(2027,2,27))


def test_extension_handoff_and_missing_morning(tmp_path, monkeypatch):
    root = tmp_path/'observations/mxi-20261024'
    (root/'deliveries').mkdir(parents=True)
    (root/'final-summary.json').write_text('{}')
    target = tmp_path/'reports/mxi-forward-input'
    target.mkdir(parents=True)
    day = x.FIRST
    dates = x.calendar(day)
    clock = [datetime(2026,11,24,6,0,tzinfo=timezone.utc)]
    monkeypatch.setattr(x.m, 'now', lambda: clock[0])
    def mock_positions(folder, day):
        data = [dict(date=d,asset='MXI',is_fiz=g,open_position_long=10,open_position_short=10)
                for d in dates[:-1] for g in (0,1)]
        pl.from_dicts(data).write_parquet(folder/'positions.parquet')
        return {'observed_at':clock[0].isoformat()}
    monkeypatch.setattr(x,'positions',mock_positions)
    monkeypatch.setattr(x,'check_calendar',lambda d:(x.calendar(d),123))
    class Response:
        def json(self):
            return {}
    monkeypatch.setattr(x.m,'get',lambda *args:(Response(),clock[0]))
    monkeypatch.setattr(x.m,'market_evidence',lambda *args:[{'secid':'fixture'}])
    x.tick(root)
    clock[0] = clock[0].replace(minute=30)
    x.tick(root)
    p = x.m.load(target/'latest.json')
    assert p['session_date'] == str(day)
    for k in ['positions','calendar','tradable']:
        assert x.m.sha(root/'deliveries'/str(day)/(k+'.parquet')) == p[k]['sha256']
    x.tick(root)
    clock[0] = clock[0].replace(day=25)
    with pytest.raises(RuntimeError):
        x.tick(root)
    assert x.m.load(target/'status.json')['status'] == 'blocked'
    assert x.m.load(target/'latest.json') == p
