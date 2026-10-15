"""Read-only acceptance audit of a pinned dividends handoff for the spot-basis request.

Uses the producer's schema/evidence validator; records its exact code hashes.
Passing structural checks never overrides integration_ready=false or missing coverage.
"""
import argparse
from datetime import datetime, timedelta
import importlib
import json
from pathlib import Path
import sys

import polars as pl

from .reference_metadata import sha256

TARGETS = {"SBER": "RU0009029540", "GAZP": "RU0007661625", "LKOH": "RU0009024277"}


def timestamp(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("Timezone required")
    return result


def assess(handoff, securities, coverage, conflicts, events):
    """Return explicit blocking issues; absent rows never imply no dividends."""
    issues = []

    def issue(ticker, check, detail):
        issues.append(dict(ticker=ticker, check=check, detail=detail))

    if handoff.get("integration_ready") is not True:
        issue("ALL", "producer_not_ready", handoff.get("pending_fix") or "integration_ready is not true")
    for ticker, isin in TARGETS.items():
        sec = securities.filter((pl.col("ticker") == ticker) & (pl.col("isin") == isin))
        if sec.height != 1 or sec["identity_status"][0] != "identified":
            issue(ticker, "identity", "Expected exactly one identified ordinary issue with pinned ISIN")
            continue
        sid = sec["security_id"][0]
        cov = coverage.filter(pl.col("security_id") == sid)
        # Producer schema currently only allows partial/not_checked/no_events_found/unavailable.
        # Do not invent a completeness flag from counts or successful HTTP responses.
        if cov.height != 1 or cov["status"][0] != "complete":
            issue(ticker, "coverage_not_complete", "Needs a documented target-specific coverage review for 2023-01-01..2026-10-05")
        for r in conflicts.filter(pl.col("security_id") == sid).to_dicts():
            if r["status"] != "resolved":
                issue(ticker, "unresolved_conflict", r["conflict_id"])
        rows = [r for r in events.to_dicts() if r["security_id"] == sid and r["event_type"] != "reference"]
        if not rows:
            issue(ticker, "no_primary_events", "No primary events; this does not prove no dividend")
        for r in rows:
            # Preserve pre-period announcements and later revisions in exports; scope field review
            # to publications/record dates in requested period plus earlier still-relevant facts.
            published = (r.get("published_at") or "")[:10]
            record = r.get("record_date") or ""
            if not ("2023-01-01" <= published <= "2026-10-05" or "2023-01-01" <= record <= "2026-10-05"):
                continue
            required = ["published_at", "publication_evidence", "edition_evidence", "public_available_at"]
            if r["verification_status"] != "verified" or r["historical_availability"] != "proven":
                issue(ticker, "historical_evidence_missing", r["event_id"])
            if r["event_type"] in ("recommendation", "approval", "payment") and r.get("amount") != "0":
                required += ["amount", "currency", "record_date", "share_basis_date"]
            if r["event_type"] == "payment":
                required += ["actual_payment_date", "recipient_category"]
            if r["event_type"] in ("amendment", "cancellation"):
                required += ["supersedes_id"]
            for key in required:
                if r.get(key) is None or r.get(key) == "":
                    issue(ticker, "missing_" + key, r["event_id"])
    return issues


def audit(project, handoff, output):
    project, handoff, output = Path(project).resolve(), Path(handoff).resolve(), Path(output)
    handoff_bytes = handoff.read_bytes()
    transfer = json.loads(handoff_bytes)
    package = Path(transfer["package_path"]).resolve()
    if sha256(package / "manifest.json") != transfer["manifest_sha256"]:
        raise ValueError("Handoff manifest mismatch")
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    if manifest["package_id"] != transfer["package_id"]:
        raise ValueError("Handoff package ID mismatch")
    sys.path.insert(0, str(project))
    validator = importlib.import_module("dividend_events.package")
    reader = importlib.import_module("dividend_events.reader")
    if not Path(validator.__file__).resolve().is_relative_to(project):
        raise ValueError("Wrong producer code loaded")
    validation = validator.validate_package(package)
    tables = {name: pl.read_parquet(package / (name + ".parquet"))
              for name in ("securities", "coverage", "conflicts", "events", "sources", "reconciliation")}
    securities = tables["securities"].filter(pl.col("ticker").is_in(list(TARGETS)))
    ids = securities["security_id"].to_list()
    subset = {name: f.filter(pl.col("security_id").is_in(ids)) for name, f in tables.items()
              if "security_id" in f.columns}
    events = subset["events"]
    issues = assess(transfer, securities, subset["coverage"], subset["conflicts"], events)
    selected = reader.select_as_of(events.to_dicts(), timestamp(transfer["snapshot_at"]), "public_as_of")
    examples = []
    all_rows = events.to_dicts()
    for ticker in TARGETS:
        for year in range(2023, 2027):
            candidates = [r for r in all_rows if r["ticker"] == ticker and r["verification_status"] == "verified"
                          and r["public_available_at"] and r["public_available_at"][:4] == str(year)
                          and timestamp(r["public_available_at"]) <= timestamp(transfer["snapshot_at"])]
            if not candidates:
                issues.append(dict(ticker=ticker, check="missing_causal_example", detail=str(year)))
                continue
            event = min(candidates, key=lambda r: r["public_available_at"])
            boundary = timestamp(event["public_available_at"])
            before = reader.select_as_of(all_rows, boundary - timedelta(microseconds=1), "public_as_of")
            after = reader.select_as_of(all_rows, boundary, "public_as_of")
            passed = event["event_id"] not in before["event_id"].to_list() and event["event_id"] in after["event_id"].to_list()
            examples.append(dict(ticker=ticker, year=year, event_id=event["event_id"],
                boundary=boundary.isoformat(), absent_before_present_at=passed,
                source_url=event["source_url"], sha256=event["sha256"]))
            if not passed:
                issues.append(dict(ticker=ticker, check="causality_failed", detail=event["event_id"]))
    output.mkdir(parents=True, exist_ok=False)
    for name, f in subset.items():
        f.write_parquet(output / (name + ".parquet"))
    selected.write_parquet(output / "public_as_of.parquet")
    source_ids = set(events["source_id"].to_list())
    tables["sources"].filter(pl.col("source_id").is_in(source_ids)).write_parquet(output / "event_sources.parquet")
    for name, value in [("issues", issues), ("causal_examples", examples), ("producer_validation", validation)]:
        (output / (name + ".json")).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "handoff.json").write_bytes(handoff_bytes)
    (output / "input_manifest.json").write_bytes((package / "manifest.json").read_bytes())
    (output / "auditor.py").write_bytes(Path(__file__).read_bytes())
    summary = dict(package_id=transfer["package_id"], package_path=str(package),
        manifest_sha256=transfer["manifest_sha256"], target_acceptance_complete=not issues,
        integration_ready=transfer.get("integration_ready") is True,
        issue_count=len(issues), public_as_of_rows=selected.height,
        strict_counts={t: selected.filter(pl.col("ticker") == t).height for t in TARGETS},
        causal_examples=len(examples),
        scope="SBER/GAZP/LKOH 2023-01-01..2026-10-05; exports retain earlier/later history",
        producer_code_sha256={p.name: sha256(p) for p in (project / "dividend_events").glob("*.py")},
        limitations=["Producer validation proves structural/evidence consistency, not completeness or parser correctness",
                     "No automatic acceptance while target coverage is partial",
                     "No ex-date/last-buy derivation or share-basis normalization in this preflight",
                     "Raw documents remain in the pinned producer package; this is an audit, not a replacement data package"])
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    hashes = {p.name: sha256(p) for p in output.iterdir() if p.is_file()}
    (output / "manifest.json").write_text(json.dumps(dict(artifact_sha256=hashes), indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--handoff", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(audit(**vars(parser.parse_args())), ensure_ascii=False, indent=2))
