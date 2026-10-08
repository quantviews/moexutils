import datetime as dt
import hashlib
import json

import polars as pl
import pytest

from moexutils.perpetual_data import read_cny_bundle


def make_bundle(root, *, comparison=True, duplicate=False):
    dates = [dt.date(2026, 10, d) for d in (1, 2, 2, 3)]
    frame = pl.DataFrame({
        'date': dates, 'SECID': ['CNYRUBF', 'CNYRUBF', 'CRZ6', 'CRZ6'],
        'BOARDID': ['RFUD'] * 4, 'registry_present': [True] * 4,
        'registry_asset': ['CNYRUBTOM', 'CNYRUBTOM', 'CNY', 'CNY'],
        'source_type': ['futures'] * 4, 'SWAPRATE': [None, 0., None, None],
        'SETTLEPRICE': [12., 13., None, 14.], 'CLOSE': [12., 13., 13., 14.],
        'VOLUME': [None, 0., 2., 3.],
    })
    if duplicate:
        frame = pl.concat([frame, frame.head(1)])
    frame.write_parquet(root / 'history.parquet')
    for name in ('descriptions', 'parameter_observations', 'candidate_gaps'):
        pl.DataFrame({'sentinel': [1]}).write_parquet(root / f'{name}.parquet')
    if comparison:
        pl.DataFrame({'date': [dates[1]], 'BOARDID': ['RFUD'], 'forts_present': [True],
                      'status': ['disputed_no_replacement']}).write_parquet(root / 'FUSR-comparison.parquet')
    metadata = {'bundle_schema_version': 1, 'snapshot_id': 123,
                'comparisons': {'FUSR': []} if comparison else {},
                'artifact_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.parquet')}}
    (root / 'summary.json').write_text(json.dumps(metadata), encoding='utf-8')


def test_full_join_retains_missing_legs_and_raw_values(tmp_path):
    make_bundle(tmp_path)
    bundle = read_cny_bundle(tmp_path)
    pair = bundle.pair('CRZ6')
    assert pair['both_legs_present'].to_list() == [False, True, False]
    assert pair['perpetual_funding_state'].to_list() == ['missing', 'zero', None]
    assert pair['perpetual_fusr_status'][1] == 'disputed_no_replacement'
    assert pair['perpetual_SWAPRATE'][1] == 0
    assert pair['dated_SETTLEPRICE'][1] is None  # no CLOSE substitution
    assert bundle.metadata['snapshot_id'] == 123
    assert bundle.metadata['estimation_policy']['mode'] == 'estimated_vm'
    assert bundle.parameter_observations['sentinel'][0] == 1


def test_no_comparison_never_implies_pass(tmp_path):
    make_bundle(tmp_path, comparison=False)
    bundle = read_cny_bundle(tmp_path)
    assert bundle.history['fusr_status'].unique().to_list() == ['not_checked']
    assert not bundle.history['historical_parameters_verified'].any()


def test_duplicate_keys_rejected(tmp_path):
    make_bundle(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match='Duplicate'):
        read_cny_bundle(tmp_path)


def test_mixed_or_modified_artifacts_rejected(tmp_path):
    make_bundle(tmp_path)
    pl.DataFrame({'wrong': [1]}).write_parquet(tmp_path / 'parameter_observations.parquet')
    with pytest.raises(ValueError, match='checksum'):
        read_cny_bundle(tmp_path)


def test_explicit_contract_board_and_interval(tmp_path):
    make_bundle(tmp_path)
    bundle = read_cny_bundle(tmp_path)
    assert bundle.pair('CRZ6', start='2026-10-02', end='2026-10-02').height == 1
    for ticker, board in [('unknown', 'RFUD'), ('CNYRUBF', 'RFUD'), ('CRZ6', 'UNKNOWN')]:
        with pytest.raises(ValueError):
            bundle.pair(ticker, board=board)
    with pytest.raises(ValueError, match='start'):
        bundle.pair('CRZ6', start='2026-10-03', end='2026-10-01')
