"""Publish the agreed rule-based settlement model for the historical basis request."""

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path

import polars as pl

from audit_settlement_calendar import conditional_settlement


def reconstruct(source, output):
    root = Path(source).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["artifact_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Input checksum/path mismatch")
    entries = pl.read_parquet(root / "calendar_entries.parquet").to_dicts()
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    years = {r["year"] for r in summary["comparisons"]
             if r["excel_is_iss_subset"] and r["excel_rows"] and r["year"] >= 2023}
    contracts = pl.read_parquet(root / "series_settlement_audit.parquet").filter(
        pl.col("matured_by_snapshot") & pl.col("last_trade_date").is_between(date(2023, 1, 1), date(2026, 10, 5)))
    if contracts.height != 45 or contracts["secid"].n_unique() != 45:
        raise ValueError("Expected 45 unique historical contracts")
    rows, trace_rows = [], []
    for r in contracts.to_dicts():
        lag = 2 if r["delivery_trade_date"] < date(2023, 7, 31) else 1
        result, trace, reason = conditional_settlement(r["delivery_trade_date"], lag, entries, years)
        if result is None:
            raise ValueError(f"Cannot reconstruct {r['secid']}: {reason}")
        rows.append(dict(secid=r["secid"], asset=r["asset"],
                         last_trade_date=r["last_trade_date"], delivery_trade_date=r["delivery_trade_date"],
                         planned_settlement_date=result, settlement_lag=lag,
                         assumed_settlecode=f"Y{lag}", settlement_date_method="rules_reconstruction_v1",
                         calendar_baseline="mon_fri_with_explicit_RUB_and_all_currency_overrides",
                         usable_for_backtest=True, source_date_verified=False,
                         historical_known_at_verified=False, actual_execution_verified=False))
        trace_rows.extend(t | dict(secid=r["secid"]) for t in trace)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    pl.DataFrame(rows).sort("asset", "last_trade_date").write_parquet(out / "planned_settlements.parquet")
    pl.DataFrame(trace_rows).write_parquet(out / "calculation_trace.parquet")
    for name in ["calendar_entries.parquet", "rule_sources.json", "observations.json", "calendar_differences.json"]:
        (out / name).write_bytes((root / name).read_bytes())
    for name in ["reconstruct_speq_settlements.py", "audit_settlement_calendar.py"]:
        (out / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report = dict(contracts=len(rows), unique_delivery_dates=len({r["delivery_trade_date"] for r in rows}),
                  scope_start="2023-01-01", scope_end="2026-10-05",
                  calendar_part_accepted_under_assumptions=True, request_fully_closed=False,
                  acceptance_basis="User accepted reconstruction for backtesting in this conversation",
                  source_package=str(root), source_manifest_sha256=hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
                  method="rules_reconstruction_v1", limitations=[
                      "Ordinary weekdays are assumed settlement days unless an explicit override exists",
                      "Weekend trading sessions do not create settlement days without explicit override",
                      "Null/conflicting marks or uncovered years block reconstruction",
                      "Standard SPEQ Y2 before 2023-07-31 and Y1 thereafter; participant-specific Nn not modeled",
                      "No claim of direct date verification, historical availability or actual delivery",
                      "Dividend event history and remaining specification review are separate"])
    (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    (out / "manifest.json").write_text(json.dumps(dict(artifact_sha256=hashes), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(reconstruct(**vars(parser.parse_args())), indent=2))
