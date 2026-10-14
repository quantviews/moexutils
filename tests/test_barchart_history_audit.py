import datetime as dt

import polars as pl
import pytest

pytest.importorskip("holidays")

from moexutils.barchart_history_audit import audit, candidate_expiry  # noqa: E402


def test_historical_anchor_and_uk_bank_holidays():
    assert candidate_expiry("COFFEE", "2020-03") == dt.date(2020, 3, 19)
    # UK spring bank holiday on May 31 is excluded.
    assert candidate_expiry("TTF", "2021-06") == dt.date(2021, 5, 27)


def test_peer_gap_does_not_label_common_absence_as_gap():
    frame = pl.DataFrame(
        {
            "asset": ["COFFEE"] * 5,
            "symbol": ["KCH20"] * 2 + ["KCK20"] * 3,
            "date": [dt.date(2020, 1, d) for d in [2, 6, 2, 3, 6]],
        }
    )
    manifest = {
        "contracts": [
            dict(asset="COFFEE", symbol=s, delivery_month=m) for s, m in [("KCH20", "2020-03"), ("KCK20", "2020-05")]
        ]
    }
    coverage, gaps = audit(frame, manifest)
    assert gaps.to_dicts() == [dict(asset="COFFEE", symbol="KCH20", date=dt.date(2020, 1, 3))]
    assert coverage.filter(pl.col("symbol") == "KCH20")["official_expiry"][0] == dt.date(2020, 3, 19)
    with pytest.raises(ValueError, match="Duplicate"):
        audit(pl.concat([frame, frame.head(1)]), manifest)
