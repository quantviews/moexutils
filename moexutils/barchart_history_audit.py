"""Historical series diagnostics. Calendar estimates are not official expiries."""

import argparse
import calendar
import datetime as dt
import hashlib
import json
from pathlib import Path

import holidays
import polars as pl

from moexutils.barchart_pilot import read_contracts

ANCHORS = {"KCH20": "2020-03-19"}
ANCHOR_SOURCE = "https://www.ice.com/publicdocs/futures_us/exchange_notices/ICE_Futures_US_DST_Start_2020324.pdf"


def read_package(package):
    root = Path(package)
    meta = json.loads((root / "summary.json").read_text())
    if meta.get("status") != "parsed_not_execution_validated":
        raise ValueError("Complete capture required")
    for name, digest in meta["artifact_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Checksum mismatch: {name}")
    return read_contracts(root), json.loads((root / "manifest.json").read_text())


def candidate_expiry(asset, month):
    """Current rules backcast with GB England / NYSE proxy holidays, unverified."""
    first = dt.date.fromisoformat(month + "-01")
    years = [first.year - 1, first.year]
    closed = (
        holidays.country_holidays("GB", subdiv="ENG", years=years)
        if asset == "TTF"
        else holidays.financial_holidays("XNYS", years=years)
    )

    def previous(day, count):
        while count:
            day -= dt.timedelta(days=1)
            if day.weekday() < 5 and day not in closed:
                count -= 1
        return day

    if asset == "TTF":
        return previous(first, 2)
    last = first.replace(day=calendar.monthrange(first.year, first.month)[1])
    while last.weekday() >= 5 or last in closed:
        last -= dt.timedelta(days=1)
    return previous(last, {"COFFEE": 8, "ORANGE": 14}[asset])


def audit(frame, manifest):
    if frame.select("symbol", "date").is_duplicated().any():
        raise ValueError("Duplicate series dates")
    peers = {a: set(g["date"]) for (a,), g in frame.group_by("asset")}
    rows, gaps = [], []
    for job in manifest["contracts"]:
        asset, symbol = job["asset"], job["symbol"]
        g = frame.filter(pl.col("symbol") == symbol).sort("date")
        if g.is_empty():
            raise ValueError(f"Missing series: {symbol}")
        dates = set(g["date"])
        first, last = min(dates), max(dates)
        missing = sorted(d for d in peers[asset] if first < d < last and d not in dates)
        gaps.extend({"asset": asset, "symbol": symbol, "date": d} for d in missing)
        estimate = candidate_expiry(asset, job["delivery_month"])
        official = dt.date.fromisoformat(ANCHORS[symbol]) if symbol in ANCHORS else None
        rows.append(
            dict(
                asset=asset,
                symbol=symbol,
                delivery_month=job["delivery_month"],
                first_bar=first,
                last_bar=last,
                candidate_expiry=estimate,
                last_minus_candidate_days=(last - estimate).days,
                official_expiry=official,
                official_source=ANCHOR_SOURCE if official else None,
                calendar_status="official_notice" if official else "current_rule_calendar_proxy_unverified",
                peer_missing_dates=len(missing),
                rows=g.height,
            )
        )
    return pl.DataFrame(rows), pl.DataFrame(gaps, schema={"asset": pl.String, "symbol": pl.String, "date": pl.Date})


def run(package, output):
    f, manifest = read_package(package)
    coverage, gaps = audit(f, manifest)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    coverage.write_parquet(root / "calendar.parquet")
    gaps.write_parquet(root / "peer_gaps.parquet")
    summary = dict(
        contracts=coverage.height,
        rows=f.height,
        peer_gap_rows=gaps.height,
        candidate_matches=coverage.filter(pl.col("last_minus_candidate_days") == 0).height,
        official_anchors=coverage["official_expiry"].drop_nulls().len(),
        official_mismatches=coverage.filter(pl.col("official_expiry") != pl.col("last_bar")).height,
        holidays_version=holidays.__version__,
        full_history_verified=False,
        limitations=[
            "NYSE_is_not_IFUS_calendar",
            "current_rules_not_historical_editions",
            "peer_gaps_do_not_detect_dates_missing_from_all_series",
            "last_bar_not_proof_of_expiry_or_listing",
        ],
        input_summary_sha256=hashlib.sha256((Path(package) / "summary.json").read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        artifact_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob("*.parquet")},
    )
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.package, args.output)
