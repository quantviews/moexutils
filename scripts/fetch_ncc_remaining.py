"""One attempt per known public original; preserve failures instead of retrying access denials."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import requests


def fetch(output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).parent
    rows = json.loads((here / "ncc_missing_originals.json").read_text(encoding="utf-8"))
    registry = json.loads((here / "ncc_review_registry.json").read_text(encoding="utf-8"))
    rows += [dict(approved_date=r["approved_date"], url=r["source_url"])
             for r in registry["editions"] if r["body_evidence"].startswith("web_")]
    results = []
    with requests.Session() as session:
        for row in rows:
            result = row | dict(observed_at=datetime.now(timezone.utc).isoformat())
            try:
                response = session.get(row["url"], timeout=(10, 25))
                result.update(status=response.status_code, final_url=response.url,
                              content_type=response.headers.get("Content-Type"))
                if response.status_code == 200 and response.content.startswith(b"%PDF-"):
                    sha = hashlib.sha256(response.content).hexdigest()
                    (out / f"{sha}.pdf").write_bytes(response.content)
                    result.update(sha256=sha, artifact=f"{sha}.pdf", outcome="original_downloaded")
                else:
                    result["outcome"] = "original_unavailable"
            except requests.RequestException as exc:
                result.update(outcome="request_failed", error=type(exc).__name__)
            results.append(result)
            print(row["approved_date"], result["outcome"], result.get("status"), flush=True)
    (out / "attempts.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = dict(requested=len(rows), downloaded=sum(r["outcome"] == "original_downloaded" for r in results),
                   archive_complete=False, rule_content_verified=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "fetcher.py").write_bytes(Path(__file__).read_bytes())
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    (out / "manifest.json").write_text(json.dumps(dict(artifact_sha256=hashes), indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    fetch(parser.parse_args().output)
