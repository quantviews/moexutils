"""Offline coverage and expiry-anchor checks; no strategy or price model."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import polars as pl

from moexutils.barchart_pilot import read_contracts
from moexutils.futures_bundle import read_table


def audit(contracts, history, manifest):
    anchors, coverage, missing = [], [], []
    for job in manifest['contracts']:
        symbol, secid, asset = job['symbol'], job['moex_secid'], job['asset']
        expiry = dt.date.fromisoformat(job['moex_expiration'])
        b = contracts.filter(pl.col('symbol') == symbol).sort('date')
        m = history.filter((pl.col('SECID') == secid) & (pl.col('BOARDID') == 'RFUD')).sort('date')
        if b['date'].n_unique() != b.height or m['date'].n_unique() != m.height:
            raise ValueError('Duplicate dates')
        previous = b.filter(pl.col('date') < expiry).tail(1)
        end = m.filter(pl.col('date') == expiry)
        source_date = previous['date'][0] if previous.height else None
        price = previous['close'][0] if previous.height else None
        price = price * (.01 if asset in ('COFFEE', 'ORANGE') else 1.) if price is not None else None
        settle = end['SETTLEPRICE'][0] if end.height else None
        fresh = source_date is not None and (expiry - source_date).days <= 7
        diff = price - settle if price is not None and settle is not None and fresh else None
        anchors.append(dict(asset=asset, symbol=symbol, SECID=secid, expiry=expiry,
            source_date=source_date, source_price=price, moex_settle=settle, difference=diff,
            status='missing_or_stale' if diff is None else ('matched' if abs(diff) < 1e-9 else 'mismatch')))
        dates = set(b['date'])
        absent = sorted(set(m['date']) - dates)
        peer_dates = set(contracts.filter(pl.col('asset') == asset)['date'])
        # Peer dates are only a diagnostic reference, not an official exchange calendar.
        internal = sorted(d for d in peer_dates - dates if b.height and b['date'].min() < d < b['date'].max())
        for kind, ds in [('moex_date_without_foreign_bar', absent), ('peer_date_inside_foreign_span', internal)]:
            missing.extend(dict(symbol=symbol, SECID=secid, date=d, kind=kind) for d in ds)
        coverage.append(dict(asset=asset, symbol=symbol, SECID=secid, rows=b.height,
            moex_rows=m.height, common_dates=len(set(m['date']) & dates),
            moex_dates_without_bar=len(absent), peer_gap_candidates=len(internal),
            null_close=b['close'].null_count(), zero_volume=int((b['volume'] == 0).sum()),
            flat_ohlc=b.filter((pl.col('open') == pl.col('high')) & (pl.col('high') == pl.col('low')) &
                              (pl.col('low') == pl.col('close'))).height))
    return anchors, coverage, missing


def run(package, moex, output, pilot=None):
    root, source = Path(output), Path(package)
    manifest_path = source / 'manifest.json'
    manifest_bytes = manifest_path.read_bytes()
    package_meta = json.loads((source / 'summary.json').read_text())
    if hashlib.sha256(manifest_bytes).hexdigest() != package_meta['artifact_sha256']['manifest.json']:
        raise ValueError('Manifest checksum mismatch')
    manifest = json.loads(manifest_bytes)
    b = read_contracts(source)
    h = read_table(moex, 'history', assets=['COFFEE', 'ORANGE', 'TTF'])
    anchors, coverage, missing = audit(b, h, manifest)
    comparison = None
    if pilot:
        old = read_contracts(pilot)
        cols = ['open', 'high', 'low', 'close', 'volume', 'open_interest']
        joined = old.join(b, on=['symbol', 'date'], how='inner', suffix='_new')
        changed = joined.filter(pl.any_horizontal([~pl.col(c).eq_missing(pl.col(c + '_new')) for c in cols]))
        comparison = dict(old_rows=old.height, common_rows=joined.height, changed_rows=changed.height,
                          missing_old_rows=old.height - joined.height)
    root.mkdir(parents=True, exist_ok=False)
    pl.DataFrame(anchors).write_parquet(root / 'expiry_anchors.parquet')
    pl.DataFrame(coverage).write_parquet(root / 'coverage.parquet')
    pl.DataFrame(missing, schema={'symbol': pl.String, 'SECID': pl.String, 'date': pl.Date, 'kind': pl.String}).write_parquet(root / 'date_checks.parquet')
    inputs = [source / 'summary.json', Path(moex) / 'summary.json']
    if pilot:
        inputs.append(Path(pilot) / 'summary.json')
    meta = dict(status='data_audit_not_execution_validation', rows=b.height, contracts=len(coverage),
        matched_expiries=sum(r['status'] == 'matched' for r in anchors), anchors=anchors, coverage=coverage,
        pilot_comparison=comparison, full_history_verified=False,
        input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        artifact_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.parquet')})
    (root / 'summary.json').write_text(json.dumps(meta, indent=2, default=str), encoding='utf-8')
    print(json.dumps({k: meta[k] for k in ['rows', 'contracts', 'matched_expiries', 'pilot_comparison']}))
    print('Date differences:', len(missing), '(calendar diagnostics, not proven missing data)')
    return meta


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['package', 'moex', 'output']:
        p.add_argument('--' + key, required=True, type=Path)
    p.add_argument('--pilot', type=Path)
    a = p.parse_args()
    run(a.package, a.moex, a.output, a.pilot)
