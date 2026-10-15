"""Package manual NCC review; unavailable editions remain explicit gaps."""

import argparse
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

import polars as pl


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    manifest = root / "manifest.json"
    for name, expected in json.loads(manifest.read_text(encoding="utf-8"))["artifact_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError(f"Input checksum/path mismatch: {name}")
    return digest(manifest)


def edition_for(day, editions):
    matches = [r for r in editions if r["effective_from"] <= day
               and (r["effective_to"] is None or day <= r["effective_to"])]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one edition for {day}")
    return matches[0]


def finalize(source, series, output, registry):
    source, series, output = map(Path, (source, series, output))
    parents = {str(p.resolve()): verify(p) for p in (source, series)}
    review = json.loads(Path(registry).read_text(encoding="utf-8"))
    docs = {r["sha256"]: r for r in json.loads((source / "documents.json").read_text(encoding="utf-8"))}

    def resolve(prefix):
        found = [r for sha, r in docs.items() if sha.startswith(prefix)]
        if len(found) != 1:
            raise ValueError(f"Ambiguous/missing original: {prefix}")
        r = found[0]
        if digest(source / r["artifact"]) != r["sha256"]:
            raise ValueError("Original checksum mismatch")
        return dict(document_sha256=r["sha256"], source_url=r["url"], artifact=r["artifact"])

    decisions = [r | resolve(r["sha256_prefix"]) | dict(
        target_lot_or_expiry_change_identified=False, historical_known_at_verified=False,
        reviewed_pages=list(range(1, docs[resolve(r["sha256_prefix"])["document_sha256"]]["pages"] + 1)))
        for r in review["decisions"]]
    pages = json.loads((source / "pages.json").read_text(encoding="utf-8"))
    scanned = {(r["document_sha256"], r["page"]) for r in pages if r["requires_visual_review"]}
    viewed = {(r["document_sha256"], p) for r in decisions if r["review_method"] == "visual"
              for p in r["reviewed_pages"]}
    if not scanned <= viewed:
        raise ValueError("Unreviewed scanned pages remain")
    editions = review["editions"]
    for previous, following in zip(editions, editions[1:]):
        if date.fromisoformat(previous["effective_to"]) + timedelta(days=1) != date.fromisoformat(following["effective_from"]):
            raise ValueError("Edition timeline is not contiguous")
    for r in editions:
        if r["sha256_prefix"]:
            r.update(resolve(r["sha256_prefix"]))
    contracts = pl.read_parquet(series / "series_calendar.parquet").filter(pl.col("matured_by_snapshot"))
    rows = []
    for r in contracts.to_dicts():
        ed = edition_for(r["delivery_trade_date"].isoformat(), editions)
        rows.append(dict(secid=r["secid"], asset=r["asset"], delivery_trade_date=r["delivery_trade_date"],
                         edition_approved_date=ed["approved_date"], body_evidence=ed["body_evidence"],
                         scope="edition_at_delivery_only_not_entire_contract_lifetime",
                         series_exceptions_verified=False, historical_known_at_verified=False))
    if len(rows) != 45 or len({r["secid"] for r in rows}) != 45:
        raise ValueError("Expected 45 historical contracts")
    summary = dict(status="available_sources_reviewed_with_gaps", decisions_reviewed=len(decisions),
                   scanned_pages_reviewed=len(scanned), active_editions=len(editions),
                   body_unavailable=[r["approved_date"] for r in editions if r["body_evidence"] == "unavailable"],
                   contracts=len(rows), target_lot_or_expiry_exceptions_identified=0,
                   absence_of_all_exceptions_proven=False, historical_archive_complete=False,
                   request_fully_closed=False, source_manifest_sha256=parents)
    output.mkdir(parents=True, exist_ok=False)
    for name, obj in [("decisions.json", decisions), ("editions.json", editions), ("summary.json", summary)]:
        (output / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    pl.DataFrame(rows).write_parquet(output / "series_delivery_edition.parquet")
    (output / "registry.json").write_bytes(Path(registry).read_bytes())
    (output / "reviewer.py").write_bytes(Path(__file__).read_bytes())
    manifest = {p.name: digest(p) for p in output.iterdir() if p.is_file()}
    (output / "manifest.json").write_text(json.dumps(dict(artifact_sha256=manifest), indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "series", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--registry", default=str(Path(__file__).with_name("ncc_review_registry.json")))
    print(json.dumps(finalize(**vars(parser.parse_args())), indent=2))
