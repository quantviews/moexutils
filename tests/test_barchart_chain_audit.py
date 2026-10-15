import datetime as dt

import polars as pl
import pytest

pytest.importorskip("holidays")
from moexutils.barchart_chain_audit import audit


def test_no_fallback_and_unverified_common_gap():
    frame = pl.DataFrame(dict(asset=["COFFEE"], symbol=["KCK25"], date=[dt.date(2025, 2, 13)], close=[12.0]))
    manifest = {"contracts": [{"symbol": "KCK25"}, {"symbol": "KCH25"}]}
    rows = audit(frame, manifest, dt.date(2025, 2, 13), dt.date(2025, 2, 14))
    coffee = rows.filter(pl.col("asset") == "COFFEE")
    assert coffee["symbol"].to_list() == ["KCH25", "KCH25"]
    assert coffee["coverage_status"].to_list() == ["missing_bar", "no_peer_session_unverified"]
    absent = audit(frame, {"contracts": [{"symbol": "KCK25"}]}, dt.date(2025, 2, 13), dt.date(2025, 2, 13))
    assert absent.filter(pl.col("asset") == "COFFEE")["coverage_status"].item() == "missing_contract"


def test_roll_requires_same_day_old_mark():
    frame = pl.DataFrame(
        dict(
            asset=["COFFEE"] * 2,
            symbol=["KCH25", "KCK25"],
            date=[dt.date(2025, 2, 14), dt.date(2025, 2, 17)],
            close=[10.0, 12.0],
        )
    )
    rows = audit(
        frame, {"contracts": [{"symbol": s} for s in frame["symbol"]]}, dt.date(2025, 2, 14), dt.date(2025, 2, 17)
    )
    roll = rows.filter((pl.col("asset") == "COFFEE") & pl.col("roll_event"))
    assert roll.height == 1
    assert roll["missing_held_mark"].item()
    assert roll["previous_symbol"].item() == "KCH25"
