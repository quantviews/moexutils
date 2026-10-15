import datetime as dt

import polars as pl
import pytest

from moexutils.spot_basis_intraday import OUTRIGHT, common_marks, download_one


def candles(minutes, volumes):
    begin = [dt.datetime(2023, 1, 9, 18, m) for m in minutes]
    return pl.DataFrame(
        dict(begin=begin, end=[b + dt.timedelta(seconds=59) for b in begin], close=[10.0] * len(begin), volume=volumes)
    )


def test_same_minute_not_exact_time_and_no_forward_fill():
    s = candles([45], [1.0])
    f = candles([44], [1.0])
    marks = common_marks(s, f)
    assert marks.height == 1
    assert not marks["same_active_minute"].item()
    assert marks["future_close"].item() is None
    matched = common_marks(s, s)
    assert matched["same_active_minute"].item()
    assert not matched["exact_simultaneity_verified"].item()
    assert not common_marks(s, candles([45], [0.0]))["same_active_minute"].item()


def test_calendar_spreads_are_excluded():
    assert OUTRIGHT.fullmatch("GZM3")
    assert not OUTRIGHT.fullmatch("GZH3GZM3")
    assert pl.Series(["GZM3", "GZH3GZM3"]).str.contains(OUTRIGHT.pattern).to_list() == [True, False]


def test_pagination_resume_and_checksum(monkeypatch, tmp_path):
    from moexutils import spot_basis_intraday as module

    calls = []

    class Response:
        status_code = 200
        url = "https://iss.moex.com/test"

        def raise_for_status(self):
            pass

        def json(self):
            rows = [[10, 11, 12, 9, 100, 2, "2023-01-09 18:45:00", "2023-01-09 18:45:59"]]
            return {
                "candles": {
                    "columns": ["open", "close", "high", "low", "value", "volume", "begin", "end"],
                    "data": rows if len(calls) == 1 else [],
                }
            }

    class Session:
        def get(self, url, params, timeout):
            calls.append(params["start"])
            return Response()

    monkeypatch.setattr(module.requests, "Session", Session)
    job = dict(secid="SBER", engine="stock", market="shares", board="TQBR", start="2023-01-09", end="2023-01-09")
    assert download_one(job, tmp_path)["rows"] == 1
    assert calls == [0, 1]  # Stop only after an explicit empty page.
    assert download_one(job, tmp_path)["rows"] == 1
    assert calls == [0, 1]
    (tmp_path / "SBER/candles.parquet").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        download_one(job, tmp_path)
