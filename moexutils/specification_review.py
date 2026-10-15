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


def appendix_rows(xml):
    tree = etree.fromstring(xml, parser=etree.XMLParser(resolve_entities=False, no_network=True))
    result = []
    for index, row in enumerate(tree.xpath("//w:tbl/w:tr", namespaces=NS)):
        cells = [
            re.sub(r"\s+", " ", "".join(c.xpath(".//w:t/text()", namespaces=NS))).strip()
            for c in row.xpath("./w:tc", namespaces=NS)
        ]
        if len(cells) >= 7 and cells[2] in ASSETS:
            if not cells[4].isdigit() or cells[5:7] != ["1 RUB", "1 RUB"]:
                raise ValueError("Changed numeric/unit format requires review")
            result.append(
                dict(
                    asset=cells[2],
                    lot_shares=int(cells[4]),
                    minstep=1.0,
                    stepprice=1.0,
                    currency="RUB",
                    trading_end_msk=cells[7] if len(cells) > 7 else None,
                    table_row=index,
                    source_cells=json.dumps(cells, ensure_ascii=False),
                )
            )
    if len(result) != 3 or {r["asset"] for r in result} != ASSETS:
        raise ValueError("Expected exactly one appendix row for each requested asset")
    return result


def review(package, output):
    root = Path(package).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for name, sha in manifest["artifact_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError("Source checksum/path mismatch")
    rows = []
    for doc in json.loads((root / "documents.json").read_text(encoding="utf-8")):
        if doc["format"] != "docx":
            raise ValueError("Legacy DOC/PDF requires separate review")
        with zipfile.ZipFile(root / doc["artifact"]) as z:
            extracted = appendix_rows(z.read("word/document.xml"))
        dates = re.findall(r"\d{2}\.\d{2}\.\d{4}", doc.get("context", ""))
        if not dates or len(dates) > 2:
            raise ValueError("Ambiguous page dates")
        dates = [dt.datetime.strptime(v, "%d.%m.%Y").date() for v in dates]
        for row in extracted:
            rows.append(
                row
                | dict(
                    source_url=doc["url"],
                    document_sha256=doc["sha256"],
                    source_start_date=dates[0],
                    source_end_date=dates[1] if len(dates) == 2 else None,
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
    (out / "reviewer.py").write_bytes(Path(__file__).read_bytes())
    summary = dict(
        status="appendix_verified_series_exceptions_pending",
        rows=frame.height,
        documents=frame["document_sha256"].n_unique(),
        first_source_start=str(frame["source_start_date"].min()),
        source_manifest_sha256=hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        complete_2023_2026_history=False,
        artifact_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()},
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(review(args.package, args.output), indent=2))
