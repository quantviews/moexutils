"""Import browser-saved originals without inferring edition applicability."""

import argparse
import base64
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
from urllib.parse import urlparse
import zipfile


def import_captures(sources, output):
    captures, documents, rows = {}, {}, []
    versions = defaultdict(set)
    failures = []
    for index, source in enumerate(sources, 1):
        raw = Path(source).read_bytes()
        capture = json.loads(raw)
        if capture.get("schema_version") != 1 or capture.get("kind") != "moex_specification_browser_capture":
            raise ValueError("Unexpected capture schema")
        page = urlparse(capture["page_url"])
        if page.scheme != "https" or page.netloc != "www.moex.com" or page.path != "/ru/documents/4979":
            raise ValueError("Unexpected source page")
        captures[f"capture-{index}.json"] = raw
        for item in capture["documents"]:
            if item.get("error") or "base64" not in item:
                failures.append({"capture": index, "url": item.get("url"), "error": item.get("error", "missing_body")})
                continue
            url = urlparse(item["url"])
            if url.scheme != "https" or url.netloc != "www.moex.com" or not url.path.startswith("/files/"):
                raise ValueError("Unexpected document URL")
            body = base64.b64decode(item["base64"], validate=True)
            sha = hashlib.sha256(body).hexdigest()
            if sha != item["sha256"]:
                raise ValueError("Document checksum mismatch")
            fmt = item["format"]
            if fmt == "docx":
                with zipfile.ZipFile(io.BytesIO(body)) as z:
                    if "word/document.xml" not in z.namelist():
                        raise ValueError("Not a Word document")
            elif not (
                (fmt == "pdf" and body.startswith(b"%PDF"))
                or (fmt == "doc" and body.startswith(bytes.fromhex("d0cf11e0")))
            ):
                raise ValueError("Unexpected file signature")
            name = sha + "." + fmt
            documents[name] = body
            versions[item["url"]].add(sha)
            rows.append(
                {k: v for k, v in item.items() if k != "base64"}
                | {"artifact": "documents/" + name, "capture": index, "historical_applicability_verified": False}
            )
    if not rows:
        raise ValueError("No valid originals in captures")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    (root / "documents").mkdir()
    for name, raw in captures.items():
        (root / name).write_bytes(raw)
    for name, raw in documents.items():
        (root / "documents" / name).write_bytes(raw)
    (root / "importer.py").write_bytes(Path(__file__).read_bytes())
    (root / "documents.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = dict(
        schema_version=1,
        status="originals_received_pending_review",
        unique_documents=len(documents),
        observations=len(rows),
        errors=failures,
        conflicting_urls={url: sorted(shas) for url, shas in versions.items() if len(shas) > 1},
        complete_archive_verified=False,
        artifact_sha256={
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*")
            if p.is_file()
        },
    )
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = import_captures(args.source, args.output)
    print(json.dumps({k: result[k] for k in ["status", "unique_documents", "observations", "errors"]}))
