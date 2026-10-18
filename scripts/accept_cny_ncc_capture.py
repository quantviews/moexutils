"""Validate the three requested NCC originals and preserve an immutable supplement."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import shutil

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def run(capture, output):
    capture, output = Path(capture), Path(output)
    payload = json.loads(capture.read_text(encoding='utf-8'))
    urls = json.loads((ROOT / 'scripts/cny_cash_sources.json').read_text(encoding='utf-8'))
    review = json.loads((ROOT / 'scripts/cny_cash_review.json').read_text(encoding='utf-8'))
    expected = {r['source_id']: r for r in review['ncc_tariffs']}
    rows = payload['files']
    if len(rows) != 3 or {r['name'] for r in rows} != set(expected):
        raise ValueError('Expected exactly the three requested NCC tariff PDFs')
    approvals = {'ncc-march.pdf': '«17» марта 2026',
                 'ncc-june.pdf': '«27» мая 2026', 'ncc-july.pdf': '«24» июня 2026'}
    checked = []
    for row in rows:
        name = row['name']
        if row.get('error') or row['status'] != 200 or row['url'] != urls[name]:
            raise ValueError(f'Failed response or unexpected source: {name}')
        data = base64.b64decode(row['base64'], validate=True)
        if (not data.startswith(b'%PDF-') or len(data) != row['bytes']
                or digest(data) != row['sha256']):
            raise ValueError(f'Invalid PDF signature, size or SHA-256: {name}')
        pdf = PdfReader(io.BytesIO(data))
        pages = [p.extract_text() or '' for p in pdf.pages]
        if len(pages) != 65 or approvals[name] not in ' '.join(pages[0].split()):
            raise ValueError(f'Unexpected tariff edition: {name}')
        for index, tokens in {52: ['0,002', '0,006', '0,17', 'PriceRub'],
                              53: ['0,51'], 57: ['0,004']}.items():
            if not all(token in pages[index] for token in tokens):
                raise ValueError(f'Tariff review requires attention: {name}, page {index+1}')
        checked.append((row, data, pages))
    output.mkdir(parents=True, exist_ok=False)
    originals = output / 'sources'
    originals.mkdir()
    shutil.copyfile(capture, output / capture.name)
    accepted = []
    for row, data, pages in checked:
        name = row['name']
        (originals / name).write_bytes(data)
        (originals / (name + '.txt')).write_text(
            '\n\n'.join(f'PAGE {i+1}\n{p}' for i, p in enumerate(pages)), encoding='utf-8')
        accepted.append(dict(**expected[name], sha256=digest(data), pages=len(pages),
                             source_url=row['url'], final_url=row.get('final_url'),
                             capture_observed_at=payload['observed_at'],
                             status='original_pdf_verified', inspected_pages=[1, 52, 53, 54, 57, 58]))
        accepted[-1]['original_saved'] = True
    summary = dict(status='NCC_originals_accepted_other_gaps_remain',
                   capture_sha256=digest(capture.read_bytes()), originals=accepted,
                   previous_tariff_conclusions_changed=False,
                   remaining=['historical_PriceRub', 'main_session_price_provenance',
                              'historical_futures_settlement_calendar_exceptions',
                              'actual_clearing_reports'])
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    shutil.copyfile(__file__, output / 'accept_script.py')
    shutil.copyfile(ROOT / 'docs/cny-cash-ncc-acceptance.md', output / 'README.md')
    artifacts = {p.relative_to(output).as_posix(): digest(p.read_bytes())
                 for p in sorted(output.rglob('*')) if p.is_file()}
    (output / 'manifest.json').write_text(json.dumps(dict(artifacts=artifacts), indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(originals=len(accepted), package=str(output),
                         manifest_sha256=digest((output / 'manifest.json').read_bytes())), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.capture, args.output)
