"""Offline evidence supplement; never promote an unverified fee input."""
import argparse
from datetime import date
from decimal import Decimal
import json
from pathlib import Path
import shutil
import zipfile

import polars as pl

from scripts.build_cny_cash_audit import sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def select_coefficient(frame, contract, day):
    strike = format(Decimal(contract['strike']).normalize(), 'f')
    isin = contract['series_name'].replace('XE', contract['option_type'] + 'E') + strike
    selected = frame.filter((pl.col('isin') == isin) & (pl.col('tradedate') == day))
    if selected.height != 1:
        raise ValueError(f'Expected one archived coefficient: {isin} {day}, got {selected.height}')
    row = selected.row(0, named=True)
    if (row['option_type'] != contract['option_type']
            or Decimal(row['strike'].replace(',', '.')) != Decimal(contract['strike'])
            or date.fromisoformat(contract['expiry']).strftime('%d.%m.%Y') != row['expiration_date']):
        raise ValueError(f'Contract identity mismatch: {isin}')
    return row


def run(evidence, base, output):
    evidence, base, out = map(Path, (evidence, base, output))
    # Verify the existing package before using it as input.
    manifest = json.loads((base / 'manifest.json').read_text(encoding='utf-8'))
    # Its format is inspected explicitly; fail rather than silently skipping hashes.
    for name, expected in manifest['artifacts'].items():
        if sha(base / name) != expected:
            raise ValueError(f'Input hash mismatch: {name}')
    contracts = pl.read_parquet(base / 'contracts.parquet')
    history = pl.read_parquet(base / 'history.parquet')
    rows = []
    for month, day in [('05', '14.05.2026'), ('08', '14.08.2026')]:
        filename = f'options_coefficients_archive_2026-{month}.csv.zip'
        with zipfile.ZipFile(evidence / filename) as archive:
            frame = pl.read_csv(archive.read('options_coefficients_archive.csv'),
                                separator=';', skip_rows=2, infer_schema=False)
        for contract in contracts.filter(pl.col('entry_date').str.starts_with(f'2026-{month}')).to_dicts():
            row = select_coefficient(frame, contract, day)
            day_iso = date(2026, int(month), 14)
            hist = history.filter((pl.col('SECID') == contract['secid']) & (pl.col('date') == day_iso))
            if hist.height != 1:
                raise ValueError('Expected one history record for comparison')
            h = hist.row(0, named=True)
            price = row['price'].replace(',', '.')
            rows.append(dict(secid=contract['secid'], series_name=contract['series_name'],
                             entry_date=contract['entry_date'], coefficient_date=day_iso.isoformat(),
                             archive_isin=row['isin'], archive_price=price,
                             archive_updatetime=row['updatetime'], historical_publication_at=None,
                             history_settleprice=h['SETTLEPRICE'], history_theor_price=h['THEOR_PRICE'],
                             equals_settleprice=Decimal(price) == Decimal(str(h['SETTLEPRICE'])),
                             source_file=filename, source_sha256=sha(evidence / filename),
                             required_main_session_verified=False, verified_premium=None,
                             verified_price_rub=None, opening_fee_taker=None,
                             opening_fee_negotiated=None,
                             status='coefficient_found_fee_inputs_not_confirmed'))
    if len(rows) != 8:
        raise ValueError('Expected exactly eight contracts')
    out.mkdir(parents=True, exist_ok=False)
    sources = out / 'sources'
    sources.mkdir()
    source_urls = {
        'coefficients.html': 'https://www.moex.com/ru/forts/coefficients-values',
        'archive.json': 'https://iss.moex.com/iss/archives/files/options_coefficients_archive.json?iss.json=extended&iss.meta=off',
        'archive-filtered.json': 'https://iss.moex.com/iss/archives/files/options_coefficients_archive.json?iss.json=extended&iss.meta=off&from=2026-05-01&till=2026-08-31',
        'limits-2026-05-14.json': 'https://iss.moex.com/iss/rms/engines/futures/objects/limits.json?date=2026-05-14&limit=1000',
        'august-weekend.html': 'https://www.moex.com/n103333',
        'ets.html': 'https://www.moex.com/ru/derivatives/unified-trading-session',
        'theory-method-index.html': 'https://www.moex.com/s1452',
    }
    for month in ['05', '08']:
        filename = f'options_coefficients_archive_2026-{month}.csv.zip'
        source_urls[filename] = 'https://iss.moex.com/iss/downloads/rms/engines/futures/objects/optionscoefficients/' + filename
    write_json(out / 'source_urls.json', source_urls)
    for path in sorted(evidence.iterdir()):
        if path.is_file():
            shutil.copyfile(path, sources / path.name)
    for name in ['contracts.parquet', 'history.parquet', 'manifest.json']:
        shutil.copyfile(base / name, out / ('input_' + name))
    pl.DataFrame(rows).write_parquet(out / 'coefficient_checks.parquet')
    write_json(out / 'coefficient_checks.json', rows)
    candidates = []
    for event, day, candidate in [
        ('entry_premium_and_fees', '2026-05-15', '2026-05-18'),
        ('exercise_and_fee', '2026-06-04', '2026-06-05'),
        ('entry_premium_and_fees', '2026-08-17', '2026-08-18'),
        ('exercise_and_fee', '2026-09-03', '2026-09-04'),
    ]:
        candidates.append(dict(event=event, obligation_date=day,
                               candidate_settlement_date=candidate, actual_settlement_date=None,
                               status='weekday_T_plus_1_candidate_not_calendar_confirmed',
                               reason='Futures settlement calendar and historical exceptions not exhaustively verified',
                               rule_source='https://www.moex.com/ru/derivatives/unified-trading-session'))
    write_json(out / 'settlement_candidates.json', candidates)
    write_json(out / 'summary.json', dict(
        status='partial_followup', contracts=8, coefficients_found=len(rows),
        equal_to_history_settleprice=sum(r['equals_settleprice'] for r in rows),
        exact_opening_fees_confirmed=0, actual_payment_dates_confirmed=0,
        missing=['main_session_provenance', 'historical_PriceRub', 'three_NCC_original_PDFs',
                 'historical_futures_settlement_calendar_exceptions', 'actual_clearing_reports'],
        base_package=str(base), base_manifest_sha256=sha(base / 'manifest.json')))
    shutil.copyfile(ROOT / 'docs/cny-cash-followup.md', out / 'README.md')
    shutil.copyfile(__file__, out / 'build_script.py')
    write_json(out / 'manifest.json', dict(files=[dict(path=p.relative_to(out).as_posix(),
                                                      sha256=sha(p), bytes=p.stat().st_size)
                                                 for p in sorted(out.rglob('*')) if p.is_file()]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--base', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.evidence, args.base, args.output)
