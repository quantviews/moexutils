"""Pinned research delivery with disclosed gaps; not an execution-ready backtest."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import polars as pl

from .dividends_acceptance import timestamp
from .reference_metadata import sha256, verify_package


def select_events(events, at):
    """Select known versions before resolving supersession, preserving earlier states."""
    boundary = timestamp(at)
    rows = events.to_dicts()
    eligible = {r["event_id"]: r for r in rows if r["verification_status"] == "verified"
                and r["historical_availability"] == "proven" and r["public_available_at"]
                and r["published_at"] and r["publication_evidence"] and r["edition_evidence"]
                and timestamp(r["public_available_at"]) <= boundary}
    all_rows = {r["event_id"]: r for r in rows}
    removed = set()
    for row in eligible.values():
        parent = row["supersedes_id"]
        seen = set()
        while parent:
            if parent in seen or parent not in all_rows:
                raise ValueError("Broken revision chain")
            seen.add(parent)
            removed.add(parent)
            parent = all_rows[parent]["supersedes_id"]
    return events.filter(pl.col("event_id").is_in(sorted(set(eligible) - removed))).sort("event_id")


def read_dividends(package, at):
    root = Path(package)
    verify_package(root)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if timestamp(at) > timestamp(summary["snapshot_at"]):
        raise ValueError("Requested time exceeds pinned snapshot")
    return select_events(pl.read_parquet(root / "dividends/events.parquet"), at)


def build(audit, prices, minutes, metadata, output):
    roots = {k: Path(v).resolve() for k, v in dict(audit=audit, prices=prices, minutes=minutes, metadata=metadata).items()}
    references = {}
    for name, root in roots.items():
        verify_package(root)
        references[name] = dict(path=str(root), manifest_sha256=sha256(root / "manifest.json"))
    review = json.loads((roots["audit"] / "summary.json").read_text(encoding="utf-8"))
    if review.get("acceptance_policy") != "original_request_disclosed_gaps_v2" or review["blocking_count"]:
        raise ValueError("Acceptance blocked or wrong policy")
    handoff = json.loads((roots["audit"] / "handoff.json").read_text(encoding="utf-8"))
    producer = Path(handoff["package_path"]).resolve()
    if sha256(producer / "manifest.json") != handoff["manifest_sha256"]:
        raise ValueError("Producer manifest changed")
    producer_manifest = json.loads((producer / "manifest.json").read_text(encoding="utf-8"))
    # Validate every original, including policy evidence. Do not rely on an old validation result.
    for name, expected in producer_manifest["artifacts"].items():
        path = (producer / name).resolve()
        if not path.is_relative_to(producer) or sha256(path) != expected:
            raise ValueError("Producer artifact mismatch")
    references["dividends"] = dict(path=str(producer), manifest_sha256=handoff["manifest_sha256"])
    events = pl.read_parquet(roots["audit"] / "events.parquet")
    actual = select_events(events, handoff["snapshot_at"])
    expected = pl.read_parquet(roots["audit"] / "public_as_of.parquet").sort("event_id")
    if not actual.equals(expected):
        raise ValueError("Consumer and producer causal readers disagree")
    common = pl.read_parquet(roots["minutes"] / "common_minutes.parquet")
    if common.select("secid", "begin").is_duplicated().any():
        raise ValueError("Duplicate common-minute keys")
    bad = common.filter(pl.col("same_active_minute") & (
        pl.col("spot_close").is_null() | pl.col("future_close").is_null()
        | (pl.col("spot_volume").fill_null(0) <= 0) | (pl.col("future_volume").fill_null(0) <= 0)))
    if bad.height:
        raise ValueError("Active-minute flag contradicts data")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)

    def copy(source, relative):
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha256(source) != sha256(target):
            raise ValueError("Copy checksum mismatch")

    for name, root in roots.items():
        copy(root / "manifest.json", f"source_manifests/{name}.json")
    copy(producer / "manifest.json", "source_manifests/dividends.json")
    for path in roots["audit"].iterdir():
        if path.is_file():
            copy(path, "dividends/" + path.name)
    for path in roots["metadata"].glob("meta_*.parquet"):
        copy(path, "metadata/" + path.name)
    for name in ("contracts", "shares", "futures", "contract_coverage", "close_timing"):
        copy(roots["prices"] / (name + ".parquet"), "prices/" + name + ".parquet")
    for name in ("common_minutes.parquet", "coverage.parquet"):
        copy(roots["minutes"] / name, "minutes/" + name)
    # Archive the raw documents directly used by target events. Additional policy/coverage
    # documents remain in the hash-pinned external package, explicitly listed in references.
    for r in events.select("raw_path", "sha256").unique().to_dicts():
        if producer_manifest["artifacts"].get(r["raw_path"]) != r["sha256"]:
            raise ValueError("Target source not present in producer manifest")
        copy(producer / r["raw_path"], "dividends/" + r["raw_path"])
    issues = json.loads((roots["audit"] / "issues.json").read_text(encoding="utf-8"))
    gaps = [dict(dataset="dividends", object=r["ticker"], field=r["check"], detail=r["detail"],
                 severity=r["severity"]) for r in issues]
    gaps += [dict(dataset=ds, object="SBRF/GAZR/LKOH", field=field, detail=detail, severity="limitation")
             for ds, field, detail in [
                 ("specifications", "historical_archive_complete", "Three unreviewed NCC texts, four editions without local originals; per-series exceptions not exhaustively proven"),
                 ("prices", "exact_simultaneity", "Same active 18:45 Moscow minute only; earliest research signal 18:46, actual publication delay unknown"),
                 ("calendar", "source_date_verified", "Settlement dates use accepted rules reconstruction; not actual execution evidence"),
                 ("dividends", "official_ex_date_and_last_buy_date", "Preserve source fields and NULLs; derived dates not substituted for official dates")]]
    pl.DataFrame(gaps).write_parquet(out / "limitations.parquet")
    snapshot = handoff["snapshot_at"]
    summary = dict(status="accepted_with_limitations", acceptance_policy=review["acceptance_policy"],
        history_complete=False, execution_ready=False, full_backtest_ready=False,
        created_at=datetime.now(timezone.utc).isoformat(), snapshot_at=snapshot,
        scope="SBER/GAZP/LKOH and SBRF/GAZR/LKOH; 2023-01-01..2026-10-05",
        dividends_package_id=handoff["package_id"], metadata_version=roots["metadata"].name,
        causal_examples=review["causal_examples"], strict_counts=review["strict_counts"],
        constraints=["Do not treat unknown dividends as zero; proposed is not approved or paid",
                     "No automatic share-basis adjustment when basis is unknown",
                     "Use read_dividends(at) for each entry date; fixed public_as_of snapshot is not an earlier signal",
                     "Use common minutes only where same_active_minute=true; do not forward fill missing legs",
                     "Metadata observed in October 2026 is not proven historically known",
                     "Official ex/last-buy dates and payment dates may be NULL; no derived dates claimed official"],
        external_dependencies="Complete minute archive and supplemental source/policy documents remain in pinned packages; paths and hashes in references.json",
        verified_input_manifests=references)
    for name, value in [("summary.json", summary), ("references.json", references)]:
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    copy(Path(__file__), "builder.py")
    (out / "README.txt").write_text(
        "Research delivery accepted with disclosed gaps, NOT full history or execution-ready.\n"
        "Read summary.json and limitations.parquet first. For causal events use\n"
        "moexutils.spot_basis_delivery.read_dividends(package, timezone_aware_at).\n"
        "dividends/public_as_of.parquet is only the fixed end snapshot.\n"
        "External originals are pinned in references.json; retain their directories.\n", encoding="utf-8")
    hashes = {p.relative_to(out).as_posix(): sha256(p) for p in out.rglob("*") if p.is_file()}
    (out / "manifest.json").write_text(json.dumps(dict(artifact_sha256=hashes), indent=2), encoding="utf-8")
    verify_package(out)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audit", "prices", "minutes", "metadata", "output"):
        parser.add_argument("--" + name, required=True)
    print(json.dumps(build(**vars(parser.parse_args())), ensure_ascii=False, indent=2))
