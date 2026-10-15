"""Extract exact appendix rows from originals; never infer series exceptions."""

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import zipfile

from lxml import etree
import polars as pl

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
ASSETS = {"SBRF", "GAZR", "LKOH"}
LEGACY_CODES = {"SBRF": "SBRx", "GAZR": "GAZx", "LKOH": "LKOx"}


def appendix_rows(xml):
    tree = etree.fromstring(xml, parser=etree.XMLParser(resolve_entities=False, no_network=True))
    result = []
    for index, row in enumerate(tree.xpath("//w:tbl/w:tr", namespaces=NS)):
        cells = [
            re.sub(r"\s+", " ", "".join(c.xpath(".//w:t/text()", namespaces=NS))).strip()
            for c in row.xpath("./w:tc", namespaces=NS)
        ]
        if len(cells) >= 7 and cells[2] in ASSETS:
            lot_index = 5 if cells[3] == LEGACY_CODES[cells[2]] else 4
            if len(cells) < lot_index + 3 or not cells[lot_index].isdigit() or cells[lot_index + 1:lot_index + 3] != ["1 RUB", "1 RUB"]:
                raise ValueError("Changed numeric/unit format requires review")
            result.append(
                dict(
                    asset=cells[2],
                    lot_shares=int(cells[lot_index]),
                    minstep=1.0,
                    stepprice=1.0,
                    currency="RUB",
                    trading_end_msk=cells[lot_index + 3] if len(cells) > lot_index + 3 else None,
                    table_row=index,
                    source_cells=json.dumps(cells, ensure_ascii=False),
                )
            )
    if len(result) != 3 or {r["asset"] for r in result} != ASSETS:
        raise ValueError("Expected exactly one appendix row for each requested asset")
    return result


def source_dates(context):
    date = r"(\d{2}\.\d{2}\.\d{4})"
    match = re.search(r"(?:Период действия|Действует):\s*с\s*" + date + r"(?:\s+по\s+" + date + r")?", context)
    if match:
        values = match.groups()
    else:
        match = re.search(r"Вступает в силу\s+" + date, context)
        if not match:
            raise ValueError("Missing explicit page applicability dates")
        values = (match[1], None)
    return tuple(dt.datetime.strptime(v, "%d.%m.%Y").date() if v else None for v in values)


def review(package, output, converted=None):
    root = Path(package).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for name, sha in manifest["artifact_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError("Source checksum/path mismatch")
    rows, coverage, conversions = [], [], {}
    seen = set()
    for doc in json.loads((root / "documents.json").read_text(encoding="utf-8")):
        if doc["sha256"] in seen:
            continue
        seen.add(doc["sha256"])
        dates = source_dates(doc.get("context", ""))
        document = root / doc["artifact"]
        if doc["format"] == "doc" and converted is not None:
            document = Path(converted) / (doc["sha256"] + ".docx")
            conversions[doc["sha256"]] = document.read_bytes()
        elif doc["format"] != "docx":
            raise ValueError("Legacy DOC requires a separately converted DOCX; PDF requires review")
        with zipfile.ZipFile(document) as z:
            xml = z.read("word/document.xml")
        tree = etree.fromstring(xml, parser=etree.XMLParser(resolve_entities=False, no_network=True))
        has_asset = any(t in ASSETS for t in tree.xpath("//w:t/text()", namespaces=NS))
        absent_old_appendix = not has_asset and dates[1] is not None and dates[1] < dt.date(2023, 1, 1)
        extracted = [] if absent_old_appendix else appendix_rows(xml)
        coverage.append(dict(document_sha256=doc["sha256"], source_start_date=str(dates[0]),
                             source_end_date=str(dates[1]) if dates[1] else None,
                             status="separate_appendix_not_captured" if absent_old_appendix else "appendix_verified",
                             context=doc.get("context"), source_url=doc["url"]))
        for row in extracted:
            rows.append(
                row
                | dict(
                    source_url=doc["url"],
                    document_sha256=doc["sha256"],
                    source_start_date=dates[0],
                    source_end_date=dates[1],
                    boundary_precision="evening_clearing" if "вечерней клиринговой" in doc["context"] else "date_only",
                    original_appendix_verified=True,
                    series_exceptions_verified=False,
                    published_at=None,
                    observed_at=doc.get("received_at"),
                )
            )
    frame = pl.DataFrame(rows, schema_overrides={"source_end_date": pl.Date, "published_at": pl.String})
    frame = frame.unique(["asset", "document_sha256"]).sort("asset", "source_start_date")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    frame.write_parquet(out / "appendix_parameters.parquet")
    (out / "document_coverage.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")
    if conversions:
        (out / "converted").mkdir()
        for original_sha, body in conversions.items():
            (out / "converted" / (original_sha + ".docx")).write_bytes(body)
    (out / "reviewer.py").write_bytes(Path(__file__).read_bytes())
    summary = dict(
        status="appendix_verified_series_exceptions_pending",
        rows=frame.height,
        documents=frame["document_sha256"].n_unique(),
        original_documents=len(coverage),
        documents_without_appendix=sum(r["status"] != "appendix_verified" for r in coverage),
        conversions={sha: dict(artifact="converted/" + sha + ".docx",
                               sha256=hashlib.sha256(body).hexdigest(),
                               provenance="Supplied DOCX conversion; original DOC retained in source package")
                     for sha, body in conversions.items()},
        first_source_start=str(frame["source_start_date"].min()),
        source_manifest_sha256=hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        complete_2023_2026_history=False,
        artifact_sha256={p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob("*") if p.is_file()},
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--converted", help="Reviewed DOC-to-DOCX conversions named by original SHA-256")
    args = parser.parse_args()
    print(json.dumps(review(args.package, args.output, args.converted), indent=2))
