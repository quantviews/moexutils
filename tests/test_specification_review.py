import pytest

from moexutils.specification_review import appendix_rows, source_dates


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


def test_legacy_alternative_codes_do_not_shift_units_or_time():
    source = rows()
    for row, alias in zip(source, ["SBRx", "GAZx", "LKOx"]):
        row.pop()
        row.insert(3, alias)
    result = appendix_rows(xml(source))
    assert [r["lot_shares"] for r in result] == [100, 100, 10]
    assert all(r["trading_end_msk"] is None for r in result)
    source[0][3] = "UNKNOWN"
    with pytest.raises(ValueError, match="format"):
        appendix_rows(xml(source))


def test_source_interval_ignores_dates_of_separate_amendments():
    start, end = source_dates("Период действия:с 01.07.2015 по 08.11.2022\nИзменения от 20.10.2016")
    assert str(start) == "2015-07-01"
    assert str(end) == "2022-11-08"
    start, end = source_dates("Вступает в силу 25.03.2025 с момента начала вечерней клиринговой сессии")
    assert str(start) == "2025-03-25"
    assert end is None
    with pytest.raises(ValueError, match="explicit"):
        source_dates("Загружено 15.10.2026")
