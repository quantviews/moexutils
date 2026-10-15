"""Reconcile observed contract dates and archived family editions without backdating evidence."""

import argparse
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import re
import zipfile

from lxml import etree
import polars as pl

from .specification_review import NS, source_dates
from .specification_news import records


def verify(root, filename="manifest.json"):
    root = Path(root).resolve()
    manifest = json.loads((root / filename).read_text(encoding="utf-8"))
    for name, expected in manifest["artifact_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Input checksum/path mismatch")
    return hashlib.sha256((root / filename).read_bytes()).hexdigest()


def third_thursday(year, month):
    first = date(year, month, 1)
    return first + timedelta(days=(3 - first.weekday()) % 7 + 14)


def interval(row, next_start, start, end):
    """Return candidate date overlap, retaining an explicit inferred end boundary."""
    source_end = row["source_end_date"]
    bound = source_end
    inferred = bound is None and next_start is not None
    if inferred:
        bound = next_start - timedelta(days=1)
    left, right = max(start, row["source_start_date"]), min(end, bound or end)
    return (left, right, inferred) if left <= right else None


def audit(package, originals, review, news, supplement, output, start=date(2023, 1, 1), end=date(2026, 10, 5)):
    roots = {k: Path(v) for k, v in dict(package=package, originals=originals, review=review, news=news, supplement=supplement).items()}
    hashes = {k: verify(v, "summary.json" if k == "review" else "manifest.json") for k, v in roots.items()}
    contracts = pl.read_parquet(roots["package"] / "contracts.parquet")
    descriptions = pl.read_parquet(roots["package"] / "descriptions.parquet")
    params = pl.read_parquet(roots["review"] / "appendix_parameters.parquet")
    calendar, links, rules = [], [], []
    for contract in contracts.to_dicts():
        secid, asset = contract["secid"], contract["asset_code"]
        obs = descriptions.filter(pl.col("secid") == secid)
        dates = obs.select("LSTTRADE", "LSTDELDATE").unique().to_dicts()
        if len(dates) != 1:
            raise ValueError(f"Conflicting/missing dates for {secid}")
        last, delivery = (date.fromisoformat(dates[0][f]) for f in ["LSTTRADE", "LSTDELDATE"])
        expected = third_thursday(last.year, last.month)
        calendar.append(dict(secid=secid, asset=asset, last_trade_date=last, delivery_trade_date=delivery,
                             nominal_third_thursday=expected, last_trade_matches_rule=last == expected,
                             delivery_is_next_calendar_day=delivery == last + timedelta(days=1),
                             matured_by_snapshot=last <= end, cash_settlement_date=None,
                             settlement_calendar_verified=False,
                             observed_at_min=str(obs["observed_at"].min()),
                             observed_at_max=str(obs["observed_at"].max()),
                             exception_search_status="no_individual_exception_identified_in_reviewed_sources",
                             series_exceptions_verified=False))
        editions = params.filter(pl.col("asset") == asset).sort("source_start_date").to_dicts()
        for i, row in enumerate(editions):
            next_start = editions[i + 1]["source_start_date"] if i + 1 < len(editions) else None
            overlap = interval(row, next_start, max(start, contract["start_date"]), min(last, end))
            if overlap:
                left, right, inferred = overlap
                links.append(row | dict(secid=secid, candidate_from=left, candidate_to=right,
                                        end_inferred_from_next_edition=inferred,
                                        applicability="family_rule_subject_to_series_decisions",
                                        historical_known_at_verified=False))
    seen = set()
    for doc in json.loads((roots["originals"] / "documents.json").read_text(encoding="utf-8")):
        if doc["sha256"] in seen or doc["sha256"] not in params["document_sha256"].to_list():
            continue
        seen.add(doc["sha256"])
        path = roots["originals"] / doc["artifact"] if doc["format"] == "docx" else roots["review"] / "converted" / (doc["sha256"] + ".docx")
        with zipfile.ZipFile(path) as z:
            tree = etree.fromstring(z.read("word/document.xml"), parser=etree.XMLParser(resolve_entities=False, no_network=True))
        for i, p in enumerate(tree.xpath("//w:body/w:p", namespaces=NS)):
            text = re.sub(r"\s+", " ", "".join(p.xpath(".//w:t/text()", namespaces=NS))).strip()
            if text:
                rules.append(dict(document_sha256=doc["sha256"], source_url=doc["url"],
                                  source_start_date=source_dates(doc["context"])[0], paragraph=i, text=text))
    decisions_path = Path(__file__).with_name("specification_decisions.json")
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    notices = []
    for decision in decisions["decisions"]:
        n = decision["id"]
        path = roots["news"] / "content" / f"{n}.json"
        if not path.exists():
            path = roots["supplement"] / f"{n}.json"
        content = records(json.loads(path.read_text(encoding="utf-8"), strict=False)["content"])
        if len(content) != 1 or content[0]["id"] != n:
            raise ValueError("Reviewed news identity mismatch")
        meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        notices.append(decision | content[0] | dict(source_url=meta["url"], observed_at=meta["observed_at"],
                                                   source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    pl.DataFrame(calendar, schema_overrides={"cash_settlement_date": pl.Date}).write_parquet(out / "series_calendar.parquet")
    pl.DataFrame(links).write_parquet(out / "series_editions.parquet")
    pl.DataFrame(rules).write_parquet(out / "rule_paragraphs.parquet")
    pl.DataFrame(notices).write_parquet(out / "reviewed_notices.parquet")
    (out / "specification_decisions.json").write_bytes(decisions_path.read_bytes())
    pl.DataFrame([dict(secid=r["secid"], missing_field=field, reason=reason)
                  for r in calendar for field, reason in [
                      ("cash_settlement_date", "Historical settlement calendar and applicable clearing rules not archived"),
                      ("historical_known_at", "Family edition publication timestamp not established"),
                      ("absence_of_all_exceptions", "Title-filtered current news archive is not proof of absence of deleted/revised decisions")]]).write_parquet(out / "missing_fields.parquet")
    (out / "auditor.py").write_bytes(Path(__file__).read_bytes())
    summary = dict(status="series_review_complete_with_disclosed_gaps", contracts=len(calendar),
                   matured_contracts=sum(r["matured_by_snapshot"] for r in calendar),
                   date_rule_deviations=sum(not r["last_trade_matches_rule"] for r in calendar),
                   candidate_intervals=len(links), input_manifest_sha256=hashes,
                   manually_classified_notices=len(notices), target_individual_exceptions_identified=0,
                   inputs={k: str(v.resolve()) for k,v in roots.items()},
                   start=str(start), end=str(end), request_fully_closed=False,
                   limits=["News selection is not proof of absence of all individual decisions",
                           "Historical publication times of family editions are unknown",
                           "Delivery trade date is not cash/share settlement date",
                           "Minute bars do not prove exact trade simultaneity"])
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = dict(artifact_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["package", "originals", "review", "news", "supplement", "output"]:
        p.add_argument("--" + key, required=True)
    print(json.dumps(audit(**vars(p.parse_args())), indent=2))
