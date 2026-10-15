"""Check fixed-calendar contract coverage without building strategy returns."""

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import polars as pl

from moexutils.barchart_history_audit import read_package

MONTHS = {"COFFEE": ("KC", "HKNUZ"), "ORANGE": ("OJ", "FHKNUX"), "TTF": ("TG", "FGHJKMNQUVXZ")}


def audit(frame, manifest, start, end):
    if start > end or frame.select("symbol", "date").is_duplicated().any():
        raise ValueError("Invalid dates or duplicate bars")
    known = {j["symbol"] for j in manifest["contracts"]}
    records = []
    for asset, (prefix, codes) in MONTHS.items():
        # Expected series are generated independently of available packages.
        candidates = []
        for year in range(start.year - 1, end.year + 2):
            for code in codes:
                month = "FGHJKMNQUVXZ".index(code) + 1
                first = dt.date(year, month, 1)
                cutoff = (first - dt.timedelta(days=1)).replace(day=15)
                candidates.append((cutoff, f"{prefix}{code}{year % 100:02d}"))
        candidates.sort()
        bars = {(r["symbol"], r["date"]): r["close"] for r in frame.filter(pl.col("asset") == asset).to_dicts()}
        observed = {d for _, d in bars}
        previous = None
        day = start
        while day <= end:
            if day.weekday() < 5 or day in observed:
                symbol = next(s for cutoff, s in candidates if day < cutoff)
                peer = day in observed
                close = bars.get((symbol, day))
                old_close = bars.get((previous, day)) if previous else None
                status = (
                    "missing_contract"
                    if symbol not in known
                    else "no_peer_session_unverified"
                    if not peer
                    else "missing_bar"
                    if close is None
                    else "available"
                )
                records.append(
                    dict(
                        asset=asset,
                        date=day,
                        symbol=symbol,
                        coverage_status=status,
                        close=close,
                        peer_session=peer,
                        previous_symbol=previous,
                        previous_contract_close=old_close,
                        roll_event=bool(peer and previous and previous != symbol),
                        missing_held_mark=bool(peer and previous and old_close is None),
                    )
                )
                if peer:
                    previous = symbol
            day += dt.timedelta(days=1)
    return pl.DataFrame(
        records,
        schema_overrides={"close": pl.Float64, "previous_contract_close": pl.Float64, "previous_symbol": pl.String},
    )


def run(package, output, start, end):
    frame, manifest = read_package(package)
    rows = audit(frame, manifest, start, end)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    rows.write_parquet(root / "coverage_dates.parquet")
    rolls = rows.filter(pl.col("roll_event"))
    rolls.write_parquet(root / "roll_marks.parquet")
    summary = dict(
        start=str(start),
        snapshot_end=str(end),
        rows=frame.height,
        contracts=frame["symbol"].n_unique(),
        coverage=rows.group_by("asset", "coverage_status").len().sort("asset", "coverage_status").to_dicts(),
        missing_held_marks=int(rows["missing_held_mark"].sum()),
        roll_events=rolls.height,
        incomplete_roll_marks=rolls.filter(
            pl.col("close").is_null() | pl.col("previous_contract_close").is_null()
        ).height,
        full_history_verified=False,
        calendar_basis="observed_peer_sessions_not_official",
        input_summary_sha256=hashlib.sha256((Path(package) / "summary.json").read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        artifact_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob("*.parquet")},
    )
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--start", type=dt.date.fromisoformat, default=dt.date(2020, 1, 1))
    p.add_argument("--end", type=dt.date.fromisoformat, required=True)
    args = p.parse_args()
    run(args.package, args.output, args.start, args.end)
