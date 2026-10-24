from datetime import date

import polars as pl

from scripts.audit_mxi_versions import availability, compare


def frame(day='2026-10-19', long=10, short=5):
    return pl.DataFrame([dict(date=date.fromisoformat(day), asset='MXI', is_fiz=1,
        open_position_long=long, open_position_short=short)])


def observation(start, end, time):
    return dict(start=start, end=end, observed_at_utc=time, observed_at_msk=time,
                version=time, source_path='fixture', source_sha256='fixture')


def test_insert_is_not_revision_or_sign_change():
    f = frame()
    rows = compare(f.clear(), f)
    assert rows[0]['operation'] == 'insert'
    assert rows[0]['physical_sign_changed'] is False


def test_zero_crossing_and_delete():
    rows = compare(frame(), frame(long=3))
    assert rows[0]['physical_sign_changed'] is True
    assert rows[0]['long_short_changed'] is True
    assert compare(frame(), frame().clear())[0]['operation'] == 'delete'


def test_negative_observation_requires_requested_range():
    f = frame()
    result = availability([
        (observation('2026-10-18', '2026-10-18', '2026-10-20T00:00:00+00:00'), f.clear()),
        (observation('2026-10-19', '2026-10-19', '2026-10-20T01:00:00+00:00'), f),
    ])
    row = next(r for r in result if r['tradedate'] == '2026-10-19' and r['is_fiz'] == 1)
    assert row['last_seen_absent_at'] is None
    assert row['first_seen_at'] == '2026-10-20T01:00:00+00:00'


def test_real_absence_is_retained_without_inventing_publication():
    f = frame()
    result = availability([
        (observation('2026-10-19', '2026-10-19', '2026-10-20T00:00:00+00:00'), f.clear()),
        (observation('2026-10-19', '2026-10-19', '2026-10-20T01:00:00+00:00'), f),
    ])
    row = result[-1]
    assert row['last_seen_absent_at'] == '2026-10-20T00:00:00+00:00'
    assert row['published_at'] is None
