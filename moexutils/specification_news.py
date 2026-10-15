"""Capture ISS news for a bounded specification audit; absence is not proof."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import threading

from lxml import html as lhtml
import polars as pl

from .iss import make_session

LOCAL = threading.local()
TOPIC = re.compile(r"фьючер|срочн|дериватив|спецификац|экспирац|поставоч|клиринг|лот[а-я\s]|шаг[а-я\s]|сбербанк|газпром|лукойл|sbrf|gazr|lkoh", re.I)
TARGET = re.compile(r"sbrf|gazr|lkoh|сбербанк|газпром|лукойл|\b(?:SR|GZ|LK)[HMUZ]\d{1,2}\b", re.I)


def selection(title):
    title = html.unescape(title)
    if re.search(r"риск.параметр|ценового коридора|диапазона оценки", title, re.I):
        return "risk_notice_not_contract_terms"
    return "selected" if TOPIC.search(title) else "other_title"


def records(block):
    return [dict(zip(block["columns"], row)) for row in block["data"]]


def fetch(root, relative, url):
    path = root / relative
    if path.exists():
        metadata = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        if metadata["url"] != url or hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError("Cached news checksum/URL mismatch")
        return json.loads(path.read_text(encoding="utf-8"), strict=False)
    if not hasattr(LOCAL, "session"):
        LOCAL.session = make_session()
    response = LOCAL.session.get(url)
    response.raise_for_status()
    # Some archived ISS bodies contain literal control characters inside JSON
    # strings. Preserve exact bytes and allow only that JSON decoder extension.
    data = json.loads(response.content.decode("utf-8"), strict=False)
    path.write_bytes(response.content)
    path.with_suffix(".meta.json").write_text(json.dumps(dict(
        url=url, observed_at=datetime.now(timezone.utc).isoformat(),
        sha256=hashlib.sha256(response.content).hexdigest(), json_strict=False), indent=2), encoding="utf-8")
    return data


def capture(output, start="2022-01-01", end="2026-10-15", workers=4):
    root = Path(output)
    if (root / "manifest.json").exists():
        raise FileExistsError("Completed package is immutable")
    root.mkdir(parents=True, exist_ok=True)
    settings = dict(start=start, end=end, workers=workers)
    config = root / "request.json"
    if config.exists() and json.loads(config.read_text()) != settings:
        raise ValueError("Resume configuration mismatch")
    config.write_text(json.dumps(settings), encoding="utf-8")
    for folder in ["index", "content"]:
        (root / folder).mkdir(exist_ok=True)
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        offset = 0
        while True:
            offsets = list(range(offset, offset + workers * 50, 50))
            pages = list(pool.map(lambda n: fetch(root, f"index/{n:06}.json", f"https://iss.moex.com/iss/sitenews.json?start={n}"), offsets))
            stop = False
            for n, page in zip(offsets, pages):
                cursor = records(page["sitenews.cursor"])[0]
                if cursor["INDEX"] != n or cursor["PAGESIZE"] != 50:
                    raise ValueError("Unexpected news pagination")
                batch = records(page["sitenews"])
                rows.extend(batch)
                if not batch or min(r["published_at"][:10] for r in batch) < start:
                    stop = True
            if offset % 2000 == 0:
                print("index", offset, rows[-1]["published_at"], flush=True)
            if stop:
                break
            offset += workers * 50
            if offset > 200000:
                raise ValueError("Date boundary not reached")
        # Pagination can shift during a live crawl: preserve duplicate observations,
        # report conflicts and do not claim an exhaustive historical archive.
        index = pl.DataFrame(rows)
        index.write_parquet(root / "index_observations.parquet")
        chosen = index.filter(pl.col("published_at").str.slice(0, 10).is_between(pl.lit(start), pl.lit(end))).unique("id", keep="first", maintain_order=True)
        chosen = chosen.with_columns(pl.col("title").map_elements(selection, return_dtype=pl.String).alias("selection"))
        chosen.write_parquet(root / "title_review.parquet")
        ids = chosen.filter(pl.col("selection") == "selected")["id"].to_list()
        bodies = []
        for i, page in enumerate(pool.map(lambda n: fetch(root, f"content/{n}.json", f"https://iss.moex.com/iss/sitenews/{n}.json"), ids)):
            data = records(page["content"])
            if len(data) != 1 or data[0]["id"] != ids[i]:
                raise ValueError("News identity mismatch")
            row = data[0]
            text = lhtml.fromstring(row["body"] or "<p></p>").text_content()
            bodies.append(row | dict(text=text, target_mentioned=bool(TARGET.search(text + " " + row["title"]))))
            if i % 100 == 0:
                print("content", i, "of", len(ids), flush=True)
    pl.DataFrame(bodies).write_parquet(root / "news.parquet")
    (root / "collector.py").write_bytes(Path(__file__).read_bytes())
    summary = dict(status="candidate_news_captured_pending_manual_review", index_observations=index.height,
                   unique_titles=chosen.height, bodies=len(bodies),
                   target_mentions=sum(r["target_mentioned"] for r in bodies),
                   exhaustive_archive_verified=False, historical_versions_verified=False,
                   skipped_title_categories=chosen.group_by("selection").len().to_dicts())
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = dict(request=settings, observed_at=datetime.now(timezone.utc).isoformat(),
                    artifact_sha256={p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in root.rglob("*") if p.is_file()})
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-10-15")
    args = parser.parse_args()
    print(json.dumps(capture(args.output, args.start, args.end), indent=2))
