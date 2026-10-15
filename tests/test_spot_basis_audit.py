import datetime as dt
import hashlib
import json

import polars as pl
import pytest

from moexutils.spot_basis_audit import contract_coverage, read_package, timing_report


def test_equal_prices_and_dates_do_not_prove_simultaneity():
    dates = [dt.date(2026, 3, 20), dt.date(2026, 3, 23)]
    shares = pl.DataFrame(
        dict(
            date=dates,
            SECID=["SBER"] * 2,
            CLOSE=[100.0, 101.0],
            LEGALCLOSEPRICE=[100.0, 100.0],
            TRADINGSESSION=[3.0, 3.0],
        )
    )
    futures = pl.DataFrame(dict(date=dates, asset_code=["SBRF"] * 2, CLOSE=[100.0, 101.0], SETTLEPRICE=[100.0, 100.0]))
    report = timing_report(shares, futures)
    sber = report.filter(pl.col("asset") == "SBRF")
    assert sber["common_observed_dates"].to_list() == [1, 1]
    assert sber["stock_close_vs_legal_different"].to_list() == [0, 1]
    assert not report["synchronized_close_verified"].any()
    assert sber["stock_sessions"].to_list() == ["[3.0]", "[3.0]"]


def test_observation_does_not_establish_effective_history():
    contracts = pl.DataFrame(dict(secid=["SRH3", "GZH3"], asset_code=["SBRF", "GAZR"]))
    descriptions = pl.DataFrame(
        dict(
            secid=["SRH3", "SRH3"],
            LOTSIZE=["100", "1000"],
            DELIVERYTYPE=["delivery", "delivery"],
            LSTTRADE=["2023-03-16"] * 2,
            LSTDELDATE=["2023-03-17"] * 2,
        )
    )
    parameters = pl.DataFrame(dict(SECID=["SRH3"], MINSTEP=[1.0], STEPPRICE=[1.0]))
    coverage = contract_coverage(contracts, descriptions, parameters)
    assert coverage["conflicting_observed_lots"].to_list() == [True, False]
    assert coverage["missing_lot"].to_list() == [False, True]
    assert coverage["effective_from"].null_count() == 2
    assert not coverage["historical_specification_verified"].any()


def test_package_reader_rejects_tampered_evidence(tmp_path):
    path = tmp_path / "evidence.parquet"
    pl.DataFrame({"value": [1]}).write_parquet(path)
    manifest = dict(
        schema_version=1,
        status="partial_evidence_not_execution_ready",
        rows={"evidence": 1},
        artifact_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()},
    )
    raw = json.dumps(manifest).encode()
    (tmp_path / "manifest.json").write_bytes(raw)
    (tmp_path / "summary.json").write_text(json.dumps({"manifest_sha256": hashlib.sha256(raw).hexdigest()}))
    tables, info = read_package(tmp_path)
    assert tables["evidence"].height == 1
    assert info["status"] == "partial_evidence_not_execution_ready"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        read_package(tmp_path)
