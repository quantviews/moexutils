import pytest

from moexutils.specification_review import appendix_rows


def xml(rows):
    table = "".join(
        "<w:tr>" + "".join(f"<w:tc><w:p><w:r><w:t>{x}</w:t></w:r></w:p></w:tc>" for x in row) + "</w:tr>"
        for row in rows
    )
    return (
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body><w:tbl>{table}</w:tbl></w:body></w:document>"
    ).encode()


def rows():
    return [
        ["", "contract", asset, "share", lot, "1 RUB", "1 RUB", "19:00 МСК"]
        for asset, lot in [("SBRF", "100"), ("GAZR", "100"), ("LKOH", "10")]
    ]


def test_exact_asset_not_preferred_and_preserve_missing_time():
    source = rows()
    source[0] = source[0][:-1]
    source.append(["", "preferred", "SBPR", "share", "100", "1 RUB", "1 RUB"])
    result = appendix_rows(xml(source))
    assert [r["lot_shares"] for r in result] == [100, 100, 10]
    assert result[0]["trading_end_msk"] is None


def test_changed_units_or_missing_asset_require_review():
    source = rows()
    source[0][5] = "1 USD"
    with pytest.raises(ValueError, match="format"):
        appendix_rows(xml(source))
    with pytest.raises(ValueError, match="exactly"):
        appendix_rows(xml(rows()[:2]))
