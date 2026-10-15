"""Validate NCC browser PDFs and preserve originals plus page-level text.

Requires PyMuPDF in the local document-review environment, not in the data pipeline.
No date applicability is inferred from the browser's broad parent context.
"""

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse, parse_qs


def review(sources, output):
    import pymupdf

    originals, documents, captures, pages = {}, [], [], []
    for source in sources:
        raw = Path(source).read_bytes()
        capture = json.loads(raw)
        page = urlparse(capture["page_url"])
        if capture.get("kind") != "ncc_clearing_browser_capture" or page.scheme != "https" or page.netloc != "www.nationalclearingcentre.ru":
            raise ValueError("Unexpected capture origin/schema")
        captures.append(raw)
        for row in capture["documents"]:
            if row.get("error"):
                raise ValueError("Incomplete capture requires separate review")
            url = urlparse(row["url"])
            if url.scheme != "https" or url.netloc != page.netloc or url.path != "/connector" or parse_qs(url.query).get("cmd") != ["file"]:
                raise ValueError("Unexpected document URL")
            body = base64.b64decode(row["base64"], validate=True)
            sha = hashlib.sha256(body).hexdigest()
            if sha != row["sha256"] or not body.startswith(b"%PDF-"):
                raise ValueError("PDF checksum/signature mismatch")
            pdf = pymupdf.open(stream=body, filetype="pdf")
            if sha not in originals:
                for number, pdf_page in enumerate(pdf, 1):
                    pages.append(dict(document_sha256=sha, page=number, text=pdf_page.get_text()))
            originals[sha] = body
            documents.append({k: v for k, v in row.items() if k != "base64"} | dict(
                pages=len(pdf), source_capture=len(captures), artifact=f"documents/{sha}.pdf",
                context_date_association_verified=False))
            pdf.close()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "documents").mkdir()
    for sha, body in originals.items():
        (out / "documents" / f"{sha}.pdf").write_bytes(body)
    for i, body in enumerate(captures, 1):
        (out / f"capture-{i}.json").write_bytes(body)
    (out / "documents.json").write_text(json.dumps(documents, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "pages.json").write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "reviewer.py").write_bytes(Path(__file__).read_bytes())
    summary = dict(documents=len(originals), pages=len(pages), captures=len(captures),
                   status="originals_validated_applicability_requires_review",
                   historical_archive_complete=False, settlement_calendar_received=False,
                   extractor="PyMuPDF " + pymupdf.VersionBind)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = dict(generated_at=datetime.now(timezone.utc).isoformat(), artifact_sha256={
        p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in out.rglob("*") if p.is_file()})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", nargs="+", required=True)
    p.add_argument("--output", required=True)
    print(json.dumps(review(**vars(p.parse_args())), indent=2))
