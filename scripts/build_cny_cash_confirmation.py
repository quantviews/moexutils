"""Offline, rule-derived confirmation; no fabricated PriceRub or payment dates."""
import argparse
import json
from pathlib import Path
import shutil

import polars as pl
from pypdf import PdfReader

from scripts.build_cny_cash_audit import sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def run(evidence, previous, output):
    evidence, previous, out = map(Path, (evidence, previous, output))
    manifest = json.loads((previous / 'manifest.json').read_text(encoding='utf-8'))
    for r in manifest['files']:
        if sha(previous / r['path']) != r['sha256']:
            raise ValueError('Input checksum mismatch: ' + r['path'])
    method = PdfReader(evidence / 'theory-method.pdf')
    text = ' '.join(method.pages[12].extract_text().split())
    if 'по итогам основной торговой сессии' not in text:
        raise ValueError('Expected methodology clause missing')
    coefficients = pl.read_parquet(previous / 'coefficient_checks.parquet')
    if coefficients.height != 8 or not coefficients['equals_settleprice'].all():
        raise ValueError('Eight reconciled archive prices required')
    coefficients = coefficients.with_columns(
        pl.col('archive_price').alias('verified_premium'),
        pl.lit(True).alias('required_main_session_verified'),
        pl.lit('confirmed_by_disclosure_methodology').alias('premium_confirmation'),
        pl.lit('premium_confirmed_PriceRub_missing').alias('status'),
    )
    dates = json.loads((previous / 'settlement_candidates.json').read_text(encoding='utf-8'))
    for row in dates:
        row['planned_settlement_date'] = row.pop('candidate_settlement_date')
        row['status'] = 'rule_derived_from_official_calendar'
        row['reason'] = 'MOEX annual calendar page 3, specification T+1, NCC historical clearing schedule'
        row['calendar_source'] = 'calendar-2026.pdf page 3'
        row['planned_session_start_moscow'] = '20:00:00'
        row['planned_session_end_moscow'] = '20:30:00'
        row['all_individual_exceptions_checked'] = False
    out.mkdir(parents=True, exist_ok=False)
    sources = out / 'sources'
    sources.mkdir()
    for p in evidence.iterdir():
        if p.is_file():
            shutil.copyfile(p, sources / ('calendar-subscription-response.html' if p.name == 'calendar.json' else p.name))
    rules_root = evidence.parents[1] / 'specifications/ncc-review-20261015-archive/documents'
    # reports/<evidence> is under the data root; preserve exact historical originals.
    for prefix in ['ac0e81f6df25', 'eaa23c120451']:
        paths = list(rules_root.glob(prefix + '*.pdf'))
        if len(paths) != 1 or not sha(paths[0]).startswith(prefix):
            raise ValueError('Missing historical NCC rules: ' + prefix)
        shutil.copyfile(paths[0], sources / paths[0].name)
    shutil.copyfile(ROOT / 'scripts/ncc_review_registry.json', sources / 'ncc_rules_review.json')
    shutil.copyfile(previous / 'manifest.json', out / 'previous_manifest.json')
    for p in (previous / 'sources').glob('options_coefficients_archive*.zip'):
        shutil.copyfile(p, sources / p.name)
    shutil.copyfile(previous / 'sources/coefficients.html', sources / 'coefficients.html')
    coefficients.write_parquet(out / 'premium_inputs.parquet')
    write_json(out / 'premium_inputs.json', coefficients.to_dicts())
    write_json(out / 'planned_settlements.json', dates)
    write_json(out / 'summary.json', dict(
        status='rule_confirmed_premium_and_calendar_PriceRub_missing',
        premium_inputs_confirmed_by_methodology=8, planned_settlement_dates=4,
        historical_PriceRub_confirmed=0, exact_opening_fees_confirmed=0,
        actual_cash_events_confirmed=0, previous_manifest_sha256=sha(previous / 'manifest.json'),
        rule_confirmation_is_not_actual_account_confirmation=True))
    write_json(out / 'source_urls.json', {
        'theory-method.pdf': 'https://www.moex.com/files/4fxyc4skgfk3s47ets49jxkzd5',
        'theory-method.raw': 'https://fs.moex.com/files/4720',
        'calendar-2026.pdf': 'https://fs.moex.com/f/23804/june-july-26-rus-eng-fin.pdf',
        'calendar-public.html': 'https://www.moex.com/ru/tradingcalendar?market=derivatives-market&type=trading',
        'calendar-year-news.html': 'https://www.moex.com/n94172',
        'calendar-weekend-news.html': 'https://www.moex.com/n95564',
        'calendar-subscription-response.html': 'https://iss.moex.com/iss/calendars/futures.json?from=2026-05-01&till=2026-09-10',
        'plaza-prod.html': 'https://ftp.moex.com/pub/ClientsAPI/Spectra/CGate/prod/docs/p2gate_ru.html',
        'futures-archive-catalog.json': 'https://iss.moex.com/iss/archives/files/futures_coefficients_archive.json?iss.json=extended&iss.meta=off&from=2026-05-01&till=2026-08-31',
        'futures_coefficients_archive_2026-05.csv.zip': 'https://iss.moex.com/iss/downloads/rms/engines/futures/objects/fortscoefficients/futures_coefficients_archive_2026-05.csv.zip',
        'futures_coefficients_archive_2026-08.csv.zip': 'https://iss.moex.com/iss/downloads/rms/engines/futures/objects/fortscoefficients/futures_coefficients_archive_2026-08.csv.zip',
    })
    for src, dst in [('cny-cash-confirmation.md', 'README.md'), ('cny-pricerub-request.md', 'PriceRub-request.md')]:
        shutil.copyfile(ROOT / 'docs' / src, out / dst)
    shutil.copyfile(__file__, out / 'build_script.py')
    write_json(out / 'manifest.json', dict(artifacts={p.relative_to(out).as_posix(): sha(p)
                                                    for p in sorted(out.rglob('*')) if p.is_file()}))
    print(sha(out / 'manifest.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['evidence', 'previous', 'output']:
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    run(args.evidence, args.previous, args.output)
