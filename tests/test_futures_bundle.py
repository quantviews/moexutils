import datetime as dt
import hashlib
import json

import polars as pl
import pytest

from moexutils import futures_bundle as bundle


class Session:
    def __init__(self, pages):
        self.pages = iter(pages)

    def get(self, url, params):
        data = next(self.pages)

        class Response:
            def raise_for_status(self):
                pass

            def json(self):
                return {'candles': {'columns': ['begin', 'end', 'close'], 'data': data}}

        r = Response()
        r.url = url + '?start=' + str(params['start'])
        return r


def test_candle_pages_and_provenance(tmp_path):
    session = Session([[['2026-10-01 00:00:00', '2026-10-01 23:00:00', 0]],
                       [['2026-10-02 00:00:00', '2026-10-02 23:00:00', 1]], []])
    df = bundle.fetch_candles('BTCUSDF', 'FUSC', dt.date(2026, 10, 1), dt.date(2026, 10, 2), tmp_path, session)
    assert df.height == 2
    raw = json.loads(next(tmp_path.glob('*.json')).read_text(encoding='utf-8'))
    assert len(raw['pages']) == 3 and raw['secid'] == 'BTCUSDF'
    assert df['close'].to_list() == [0., 1.]


def test_empty_archive_is_preserved(tmp_path):
    df = bundle.fetch_candles('TEST', 'FUSR', dt.date(2026, 1, 1), dt.date(2026, 1, 2), tmp_path, Session([[]]))
    assert df.is_empty() and list(tmp_path.glob('*.json'))


def test_duplicate_candles_not_accepted(tmp_path):
    row = ['2026-10-01 00:00:00', '2026-10-01 23:00:00', 0]
    with pytest.raises(ValueError, match='Invalid'):
        bundle.fetch_candles('TEST', 'FUSR', dt.date(2026, 10, 1), dt.date(2026, 10, 2), tmp_path, Session([[row, row], []]))


def test_reader_filters_and_checks_hash_and_completion(tmp_path):
    path = tmp_path / 'history.parquet'
    pl.DataFrame({'asset': ['COFFEE', 'BTC'], 'SWAPRATE': [None, 0.]}).write_parquet(path)
    meta = {'bundle_schema_version': 1, 'status': 'complete',
            'artifact_sha256': {'history.parquet': hashlib.sha256(path.read_bytes()).hexdigest()}}
    (tmp_path / 'summary.json').write_text(json.dumps(meta))
    assert bundle.read_table(tmp_path, 'history', assets='COFFEE')['SWAPRATE'][0] is None
    meta['status'] = 'partial'
    (tmp_path / 'summary.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='complete'):
        bundle.read_table(tmp_path, 'history')
    meta['status'] = 'complete'
    meta['artifact_sha256']['history.parquet'] = 'wrong'
    (tmp_path / 'summary.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='Checksum'):
        bundle.read_table(tmp_path, 'history')


def test_weekly_export_retries_partial_but_skips_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(bundle.lake, 'DATA_ROOT', tmp_path)
    path = tmp_path / 'reports/futures-bundles/test'
    path.mkdir(parents=True)
    meta = {'bundle_schema_version': 1, 'status': 'complete', 'comparison_requested': True,
            'generated_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'download_errors': 0}
    (path / 'summary.json').write_text(json.dumps(meta))
    calls = []
    monkeypatch.setattr(bundle, 'export', lambda: calls.append(1) or path)
    assert bundle.export_if_due() is None and not calls
    meta['status'] = 'partial'
    meta['download_errors'] = 1
    (path / 'summary.json').write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match='Incomplete'):
        bundle.export_if_due()
    assert calls == [1]
