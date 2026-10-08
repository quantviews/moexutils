import datetime as dt
import json

from scripts.verify_nightly_futures import check


def test_previous_run_is_not_evidence_of_new_report():
    log = '===== start 07.10.2026  0:30:00,88\n===== exit 0 07.10.2026 0:37:00\n'
    assert check(log, '2026-10-08')['status'] == 'pending'


def test_exit_zero_without_report_is_failure():
    log = '===== start 08.10.2026  0:30:00\n===== exit 0 08.10.2026 0:37:00\n'
    assert check(log, '2026-10-08')['reason'] == 'audit_success_line_missing'


def test_actual_report_required_and_stale_report_rejected(tmp_path):
    path = tmp_path / 'report.json'
    log = f'===== start 08.10.2026  0:30:00\n[OK] фьючерсы FORTS: +842 строк\n[OK] Реестр контрактов FORTS: записано 2, удалено 0\nstaticparams: 1\nstaticparamskeyterm: 1\nrclimits: 1\n[OK] Покрытие параметров и аудит FORTS: {path}\n===== exit 0 08.10.2026 0:37:00\n'
    payload = {'generated_at': dt.datetime(2026, 10, 8, 0, 35).astimezone().isoformat(),
               'schema_version': 1, 'snapshot_id': 6582,
               'sections': {'coverage': [{}], 'asset_year_coverage': [{}], 'rolls': [{}]}}
    path.write_text(json.dumps(payload), encoding='utf-8')
    assert check(log, '2026-10-08')['status'] == 'passed'
    payload['generated_at'] = '2026-10-01T00:00:00+00:00'
    path.write_text(json.dumps(payload), encoding='utf-8')
    assert check(log, '2026-10-08')['reason'] == 'invalid_report'
