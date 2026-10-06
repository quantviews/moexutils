import json

import pytest

from moexutils import specification_editions as specs


def test_reviewed_dates_are_not_contract_applicability():
    frame = specs.reviewed_editions()
    assert frame.height == 41
    assert not frame['contract_applicability_verified'].any()
    assert not frame['document_downloaded'].any()
    assert 'BR' not in frame['asset_candidate'].to_list()


@pytest.mark.parametrize('periods', [
    [['2020-01-01', '2021-01-01'], ['2021-01-01', None]],
    [['2020-01-01', None], ['2021-01-01', None]],
    [['2021-01-01', '2020-01-01']],
])
def test_invalid_document_intervals_rejected(tmp_path, periods):
    path = tmp_path / 'registry.json'
    path.write_text(json.dumps({'reviewed_on': '2026-10-06', 'sources': [
        {'asset': 'Si', 'url': 'https://www.moex.com/test', 'notes': '', 'periods': periods}]}))
    with pytest.raises(ValueError):
        specs.reviewed_editions(path)
