"""Immutable evidence package for equity futures specifications and daily CLOSE.

Observation time is never inferred to be effective time. Daily prices alone
cannot certify simultaneous marks, including when their numerical values match.
"""

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re

import polars as pl
import requests

from moexutils import lake, specification_editions

ASSETS = {"SBRF": "SBER", "GAZR": "GAZP", "LKOH": "LKOH"}
TRANSITION = dt.date(2026, 3, 23)
OUTRIGHT = re.compile(r"^(?:SR|GZ|LK)[HMUZ]\d{1,2}$")


def read_package(folder):
    """Verify all evidence hashes; return tables without upgrading partial status."""
    root = Path(folder).resolve()
    manifest_bytes = (root / "manifest.json").read_bytes()
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if hashlib.sha256(manifest_bytes).hexdigest() != summary["manifest_sha256"]:
        raise ValueError("Manifest checksum mismatch")
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != 1 or manifest.get("status") != "partial_evidence_not_execution_ready":
        raise ValueError("Unsupported evidence package")
    for name, digest in manifest["artifact_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Artifact checksum/path mismatch")
    for name in manifest["rows"]:
        if name + ".parquet" not in manifest["artifact_sha256"]:
            raise ValueError("Unverified table")
    tables = {name: pl.read_parquet(root / (name + ".parquet")) for name in manifest["rows"]}
    if any(frame.height != manifest["rows"][name] for name, frame in tables.items()):
        raise ValueError("Row count mismatch")
    return tables, manifest


def timing_report(shares, futures):
    """Report observed sessions and price definitions without inventing clocks."""
    rows = []
    for asset, ticker in ASSETS.items():
        for regime, before in [("before_2026_03_23", True), ("from_2026_03_23", False)]:
            s = shares.filter((pl.col("SECID") == ticker) & ((pl.col("date") < TRANSITION) == before))
            f = futures.filter((pl.col("asset_code") == asset) & ((pl.col("date") < TRANSITION) == before))
            both_s = s.filter(pl.col("CLOSE").is_not_null() & pl.col("LEGALCLOSEPRICE").is_not_null())
            both_f = f.filter(pl.col("CLOSE").is_not_null() & pl.col("SETTLEPRICE").is_not_null())
            dates_s, dates_f = set(s["date"]), set(f["date"])
            rows.append(
                dict(
                    asset=asset,
                    ticker=ticker,
                    regime=regime,
                    stock_rows=s.height,
                    futures_rows=f.height,
                    stock_sessions=json.dumps(sorted(s["TRADINGSESSION"].drop_nulls().unique().to_list())),
                    stock_null_session_rows=s["TRADINGSESSION"].null_count(),
                    stock_null_close=s["CLOSE"].null_count(),
                    futures_null_close=f["CLOSE"].null_count(),
                    stock_close_vs_legal_pairs=both_s.height,
                    stock_close_vs_legal_different=both_s.filter(pl.col("CLOSE") != pl.col("LEGALCLOSEPRICE")).height,
                    futures_close_vs_settle_pairs=both_f.height,
                    futures_close_vs_settle_different=both_f.filter(pl.col("CLOSE") != pl.col("SETTLEPRICE")).height,
                    common_observed_dates=len(dates_s & dates_f),
                    stock_only_dates=len(dates_s - dates_f),
                    futures_only_dates=len(dates_f - dates_s),
                    close_timestamp_status="not_available_in_daily_tables",
                    synchronized_close_verified=False,
                )
            )
    return pl.DataFrame(rows)


def contract_coverage(contracts, descriptions, parameters):
    """Preserve absence/conflicts instead of backdating current descriptions."""
    rows = []
    for contract in contracts.to_dicts():
        secid = contract["secid"]
        d = descriptions.filter(pl.col("secid") == secid)
        p = parameters.filter(pl.col("SECID") == secid)
        lots = sorted(d["LOTSIZE"].drop_nulls().unique().to_list())
        steps = p.select("MINSTEP", "STEPPRICE").drop_nulls().unique().to_dicts()
        rows.append(
            dict(
                secid=secid,
                asset=contract["asset_code"],
                description_observations=d.height,
                parameter_observations=p.height,
                observed_lots=json.dumps(lots),
                observed_tick_pairs=json.dumps(steps),
                conflicting_observed_lots=len(lots) > 1,
                missing_lot=not lots,
                missing_tick_pair=not steps,
                missing_delivery_rule=d.filter(pl.col("DELIVERYTYPE").is_not_null()).is_empty(),
                last_trade_dates=json.dumps(sorted(d["LSTTRADE"].drop_nulls().unique().to_list())),
                last_delivery_dates=json.dumps(sorted(d["LSTDELDATE"].drop_nulls().unique().to_list())),
                effective_from=None,
                effective_to=None,
                historical_specification_verified=False,
                gap="edition_originals_and_series_applicability_unverified",
            )
        )
    return pl.DataFrame(rows, schema_overrides={"effective_from": pl.String, "effective_to": pl.String})


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def capture(url, root, name):
    """Record failures as failures; never interpret a 403 as a document."""
    record = dict(source_url=url, observed_at=dt.datetime.now(dt.timezone.utc).isoformat())
    try:
        response = requests.get(url, timeout=(10, 40))
        path = root / (name + ".raw")
        path.write_bytes(response.content)
        record.update(
            http_status=response.status_code,
            artifact=path.name,
            sha256=hashlib.sha256(response.content).hexdigest(),
            content_type=response.headers.get("Content-Type"),
            final_url=response.url,
        )
        record["status"] = "downloaded" if response.ok else "http_error"
    except requests.RequestException as exc:
        record.update(status="request_error", error_type=type(exc).__name__)
    return record


def export(output, start=dt.date(2023, 1, 1), end=dt.date(2026, 10, 5)):
    if start > end:
        raise ValueError("Reversed date interval")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    with lake.session(read_only=True) as con:
        con.execute("BEGIN TRANSACTION")
        snapshot = con.execute("SELECT max(snapshot_id) FROM ducklake_snapshots('lake')").fetchone()[0]
        contracts = con.execute(
            """SELECT DISTINCT c.* FROM lake.futures_contracts c
            JOIN lake.futures f ON f.SECID=c.secid WHERE c.asset_code IN ('SBRF','GAZR','LKOH')
            AND f.date BETWEEN ? AND ?""",
            [start, end],
        ).pl()
        excluded = contracts.filter(~pl.col("secid").str.contains(OUTRIGHT.pattern))
        contracts = contracts.filter(pl.col("secid").str.contains(OUTRIGHT.pattern))
        con.register("chosen", contracts)
        descriptions = con.execute("""SELECT d.* FROM lake.futures_description_observations d
            JOIN chosen c ON d.secid=c.secid""").pl()
        parameters = con.execute("""SELECT p.* FROM lake.futures_parameter_observations p
            JOIN chosen c ON p.SECID=c.secid""").pl()
        shares = con.execute(
            """SELECT * FROM lake.shares WHERE SECID IN ('SBER','GAZP','LKOH')
            AND BOARDID='TQBR' AND date BETWEEN ? AND ?""",
            [start, end],
        ).pl()
        futures = con.execute(
            """SELECT f.*, c.asset_code FROM lake.futures f
            JOIN chosen c ON f.SECID=c.secid WHERE f.date BETWEEN ? AND ?""",
            [start, end],
        ).pl()
        con.execute("COMMIT")
    if contracts.is_empty() or shares.is_empty() or futures.is_empty():
        raise ValueError("Missing input data")
    for frame, keys in [
        (contracts, ["secid"]),
        (shares, ["date", "SECID", "BOARDID"]),
        (futures, ["date", "SECID", "BOARDID"]),
    ]:
        if frame.select(keys).is_duplicated().any():
            raise ValueError("Duplicate input keys")
    tables = dict(
        contracts=contracts,
        excluded_non_outright=excluded,
        descriptions=descriptions,
        parameters=parameters,
        shares=shares,
        futures=futures,
        contract_coverage=contract_coverage(contracts, descriptions, parameters),
        close_timing=timing_report(shares, futures),
    )
    # Shared-document candidates, not three verified histories of lot/tick values.
    family = specification_editions.reviewed_editions().filter(
        pl.col("source_url") == "https://www.moex.com/ru/documents/4979"
    )
    tables["edition_candidates"] = pl.concat(
        [family.with_columns(pl.lit(asset).alias("asset_candidate")) for asset in ASSETS]
    ).with_columns(
        pl.when(pl.col("source_valid_from") == dt.date(2025, 3, 25))
        .then(pl.lit("evening_clearing_start_not_midnight"))
        .otherwise(pl.lit("date_only"))
        .alias("boundary_precision")
    )
    for name, frame in tables.items():
        frame.write_parquet(root / (name + ".parquet"))
    sources = root / "sources"
    sources.mkdir()
    (sources / "spot_basis_audit.py").write_bytes(Path(__file__).read_bytes())
    (sources / "specification_editions.py").write_bytes(Path(specification_editions.__file__).read_bytes())
    records = []
    for name, url in [
        ("shares_columns", "https://iss.moex.com/iss/history/engines/stock/markets/shares/securities/columns.json"),
        ("futures_columns", "https://iss.moex.com/iss/history/engines/futures/markets/forts/securities/columns.json"),
        ("edition_index", "https://www.moex.com/ru/documents/4979"),
        ("current_specification", "https://fs.moex.com/files/4979"),
        ("session_transition", "https://www.moex.com/n98363"),
    ]:
        records.append(capture(url, sources, name))
    _json(sources / "requests.json", records)
    # Pin the reviewed registry used to generate candidates.
    (sources / "specification_sources.json").write_bytes(specification_editions.REGISTRY.read_bytes())
    summary = dict(
        schema_version=1,
        status="partial_evidence_not_execution_ready",
        snapshot_id=snapshot,
        start=start,
        end=end,
        generated_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        rows={name: frame.height for name, frame in tables.items()},
        historical_specification_verified=False,
        synchronized_close_verified=False,
        missing_lot_contracts=tables["contract_coverage"].filter(pl.col("missing_lot")).height,
        missing_tick_contracts=tables["contract_coverage"].filter(pl.col("missing_tick_pair")).height,
        limitations=[
            "Document candidates do not establish contract applicability",
            "Observed descriptions cannot be backdated to listing dates",
            "Daily CLOSE has no exact last-trade timestamp",
            "Current source definitions are not historical editions",
            "Unobserved trading dates are not verified exchange holidays",
        ],
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        artifact_sha256={
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*")
            if p.is_file()
        },
    )
    _json(root / "manifest.json", summary)
    _json(
        root / "summary.json",
        {**summary, "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()},
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", type=dt.date.fromisoformat, default=dt.date(2023, 1, 1))
    parser.add_argument("--end", type=dt.date.fromisoformat, default=dt.date(2026, 10, 5))
    args = parser.parse_args()
    result = export(args.output, args.start, args.end)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ["status", "snapshot_id", "rows", "missing_lot_contracts", "missing_tick_contracts"]
            },
            indent=2,
        )
    )
