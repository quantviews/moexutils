from scripts.check_options_listing import classify_series, wing


def test_future_series_is_not_never_listed():
    assert classify_series('2025-01-03', '2025-01-22')[0] == 'not_yet_listed'
    assert classify_series('2025-01-03', '2025-01-22', True)[0] == 'unknown'


def test_series_start_does_not_prove_strike_listing():
    assert classify_series('2026-10-01', '2026-09-09')[0] == 'unknown'
    assert classify_series('2026-10-01', None)[0] == 'unknown'


def test_exact_wing_thresholds():
    assert wing('P', 11.4)
    assert not wing('P', 11.5)
    assert wing('C', 13.5)
    assert not wing('C', 13.4)
    assert wing('P', 11.4961382)
    assert wing('C', 13.4051447)
