"""Reviewed document-edition dates, NOT resolved contract specifications.

MOEX legacy pages may still label superseded families as current. Open ends are
unknown. Shared specifications and series exceptions require document review;
the API deliberately returns candidates, never a guessed single specification.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import polars as pl

from moexutils import lake

REGISTRY = Path(__file__).with_name('specification_sources.json')


def reviewed_editions(path=REGISTRY):
    registry = json.loads(Path(path).read_text(encoding='utf-8'))
    rows = []
    for source in registry['sources']:
        previous_end = None
        for start, end in sorted(source['periods']):
            start = dt.date.fromisoformat(start)
            end = dt.date.fromisoformat(end) if end else None
            if end is not None and end < start:
                raise ValueError('Reversed source validity interval')
            if previous_end is not None and start <= previous_end:
                raise ValueError('Overlapping source validity intervals')
            if rows and rows[-1]['source_url'] == source['url'] and previous_end is None:
                raise ValueError('Open interval before another edition')
            previous_end = end
            rows.append({'asset_candidate': source['asset'], 'source_url': source['url'],
                         'source_valid_from': start, 'source_valid_to': end,
                         'reviewed_on': dt.date.fromisoformat(registry['reviewed_on']),
                         'evidence_kind': 'web_index_of_official_page',
                         'document_downloaded': False, 'contract_applicability_verified': False,
                         'notes': source['notes']})
    return pl.DataFrame(rows, schema={
        'asset_candidate': pl.String, 'source_url': pl.String,
        'source_valid_from': pl.Date, 'source_valid_to': pl.Date,
        'reviewed_on': pl.Date, 'evidence_kind': pl.String,
        'document_downloaded': pl.Boolean, 'contract_applicability_verified': pl.Boolean,
        'notes': pl.String})


def install():
    """Import explicitly reviewed evidence; no network and no guessed applicability."""
    frame = reviewed_editions()
    return lake.write('futures_specification_editions', frame)


def read_editions(asset=None, on=None):
    """Document candidates, possibly several families for the same date.

    Unknown open ends do not establish continuing validity. This is a research
    registry, not a contract-parameters-on-date API.
    """
    where, params = [], []
    if asset is not None:
        where.append('asset_candidate = ?')
        params.append(asset)
    if on is not None:
        where.append('source_valid_from <= ? AND (source_valid_to IS NULL OR source_valid_to >= ?)')
        day = dt.date.fromisoformat(str(on))
        params.extend([day, day])
    return lake.query('SELECT * FROM lake.futures_specification_editions'
                      + (' WHERE ' + ' AND '.join(where) if where else '')
                      + ' ORDER BY asset_candidate, source_url, source_valid_from', params)


if __name__ == '__main__':
    print(f'Reviewed document editions imported: {install()}')
