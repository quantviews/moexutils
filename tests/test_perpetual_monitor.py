import datetime as dt
import json
from types import SimpleNamespace

import polars as pl
import pytest

from moexutils import perpetual_monitor as monitor
import update_data


def test_comparison_schedule_ignores_partial_and_future_reports(tmp_path):
    now = dt.datetime(2026, 10, 8, tzinfo=dt.timezone.utc)
    assert monitor.comparison_due(tmp_path, now)
    folder = tmp_path / 'report'
    folder.mkdir()
    meta = {'bundle_schema_version': 1, 'generated_at': now.isoformat(), 'comparisons': {'FUSR': [], 'FUSC': []}}
    (folder / 'summary.json').write_text(json.dumps(meta))
    assert monitor.comparison_due(tmp_path, now)
    for board in ('FUSR', 'FUSC'):
        (folder / f'{board}-comparison.parquet').touch()
    assert not monitor.comparison_due(tmp_path, now)
    assert monitor.comparison_due(tmp_path, now + dt.timedelta(days=7))
    assert monitor.comparison_due(tmp_path, now - dt.timedelta(days=1))


def test_missing_not_zero_and_parameter_changes():
    history = pl.DataFrame({'SECID': ['CNYRUBF'] * 2, 'BOARDID': ['RFUD'] * 2,
        'date': [dt.date(2026, 10, 5), dt.date(2026, 10, 6)],
        'SWAPRATE': [0., None], 'SETTLEPRICE': [12., None],
        'fusr_status': ['not_checked', 'disputed_no_replacement'], 'fusc_status': ['not_checked'] * 2})
    params = pl.DataFrame({'SECID': ['CNYRUBF'] * 2, 'BOARDID': ['RFUD'] * 2,
        'observed_at': [dt.datetime(2026, 10, 6), dt.datetime(2026, 10, 7)],
        'MINSTEP': [.001, .002]})
    gaps = pl.DataFrame({'SECID': ['CNYRUBF'], 'BOARDID': ['RFUD'], 'date': [dt.date(2026, 10, 7)]})
    bundle = SimpleNamespace(history=history, parameter_observations=params, candidate_gaps=gaps,
                             metadata={'market_last_date': '2026-10-07'})
    issues = monitor.check_bundle(bundle)
    assert set(issues['check']) == {'perpetual_stale', 'perpetual_missing_funding',
        'perpetual_missing_settlement', 'perpetual_funding_disputed', 'perpetual_gap_candidate',
        'perpetual_parameter_changed'}
    assert issues.filter(pl.col('check') == 'perpetual_missing_funding').height == 1


@pytest.mark.parametrize('failure', [False, True])
def test_nightly_monitor_reaches_quality_or_error(monkeypatch, tmp_path, failure):
    empty = pl.DataFrame(schema=monitor.SCHEMA)
    findings = pl.DataFrame([('perpetual_missing_funding', 'CNYRUBF', 'NULL')], schema=monitor.SCHEMA, orient='row')
    monkeypatch.setattr(update_data, '_update_dataset', lambda *args: None)
    monkeypatch.setattr(update_data, '_lake_tables', lambda *args: [])
    for module in (update_data.futures_params, update_data.futures_rms):
        monkeypatch.setattr(module, 'update', lambda: None)
    monkeypatch.setattr(update_data.futures_audit, 'export', lambda: tmp_path)
    monkeypatch.setattr(update_data.quality, 'data_quality_report', lambda **kw: empty)
    captured = []
    monkeypatch.setattr(update_data, '_finish', lambda *args: captured.append(args))

    def run():
        if failure:
            raise ValueError('duplicate history')
        return tmp_path, findings

    monkeypatch.setattr(update_data.perpetual_monitor, 'run', run)
    result = update_data.main(do_update=False, do_indexes=False, do_bonds=False, do_key_rate=False,
        do_futures=True, do_markets=False, do_rates=False, do_adj_close=False, do_market_cap=False,
        do_derived=False, do_maintenance=False, do_backup=False, notify_on=False)
    assert result == int(failure)
    if failure:
        assert 'duplicate history' in str(captured[0][2])
    else:
        assert captured[0][3].equals(findings)
