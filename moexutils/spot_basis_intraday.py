"""Resumable raw ISS minute archive and common-minute marks (not tick execution)."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import time

import polars as pl
import requests

from moexutils.spot_basis_audit import ASSETS, OUTRIGHT, read_package


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_one(job, root):
    secid = job["secid"]
    folder = root / secid
    folder.mkdir(exist_ok=True)
    request_file = folder / "request.json"
    if request_file.exists() and json.loads(request_file.read_text()) != job:
        raise ValueError("Resume request differs")
    write_json(request_file, job)
    if (folder / "complete.json").exists():
        meta = json.loads((folder / "complete.json").read_text())
        for name, sha in meta["artifact_sha256"].items():
            if digest(folder / name) != sha:
                raise ValueError("Resume checksum mismatch")
        return meta
    base = f"https://iss.moex.com/iss/engines/{job['engine']}/markets/{job['market']}/boards/{job['board']}/securities/{secid}"
    session = requests.Session()
    frames, offset, previous = [], 0, None
    while True:
        path = folder / f"page-{offset:09d}.json.gz"
        if path.exists():
            envelope = json.loads(gzip.decompress(path.read_bytes()))
        else:
            params = dict(interval=1, start=offset, **{"from": job["start"], "till": job["end"]})
            for attempt in range(3):
                try:
                    response = session.get(base + "/candles.json", params=params, timeout=(10, 40))
                    response.raise_for_status()
                    payload = response.json()
                    break
                except (requests.RequestException, ValueError):
                    if attempt == 2 or ("response" in locals() and response.status_code in (401, 403, 429)):
                        raise
                    time.sleep(1 + attempt)
            envelope = dict(url=response.url, observed_at=dt.datetime.now(dt.timezone.utc).isoformat(), payload=payload)
            path.write_bytes(gzip.compress(json.dumps(envelope, ensure_ascii=False).encode(), mtime=0))
        block = envelope["payload"]["candles"]
        if not block["data"]:
            break
        frame = pl.DataFrame(block["data"], schema=block["columns"], orient="row", infer_schema_length=None)
        frame = frame.with_columns(
            pl.col("begin", "end").str.to_datetime(),
            pl.col("open", "close", "high", "low", "value", "volume").cast(pl.Float64),
        )
        if frame["begin"].n_unique() != frame.height or not frame["begin"].is_sorted():
            raise ValueError("Nonunique/unsorted candle page")
        if previous is not None and frame["begin"].min() <= previous:
            raise ValueError("Repeated/overlapping pagination")
        if (
            not frame["begin"]
            .dt.date()
            .is_between(dt.date.fromisoformat(job["start"]), dt.date.fromisoformat(job["end"]))
            .all()
        ):
            raise ValueError("Server ignored requested date range")
        if frame.filter(pl.col("end") < pl.col("begin")).height:
            raise ValueError("Invalid candle interval")
        frames.append(frame)
        previous = frame["begin"].max()
        offset += frame.height
        if offset > 5_000_000:
            raise ValueError("Pagination safety limit")
    schema = {
        **{k: pl.Float64 for k in ["open", "close", "high", "low", "value", "volume"]},
        "begin": pl.Datetime,
        "end": pl.Datetime,
    }
    result = pl.concat(frames) if frames else pl.DataFrame(schema=schema)
    result.write_parquet(folder / "candles.parquet")
    meta = dict(
        **job,
        rows=result.height,
        first=result["begin"].min(),
        last=result["begin"].max(),
        timezone="Europe/Moscow",
        interval=1,
        exact_trade_time_available=False,
        artifact_sha256={p.name: digest(p) for p in folder.iterdir() if p.is_file()},
    )
    write_json(folder / "complete.json", meta)
    print(f"{secid}: {result.height} minutes", flush=True)
    return meta


def common_marks(stock, future, minute="18:45"):
    """Require both active candles in the SAME minute; never carry forward."""

    def select(frame, leg):
        return frame.filter(pl.col("begin").dt.strftime("%H:%M") == minute).select(
            "begin",
            pl.col("end").alias(leg + "_end"),
            pl.col("close").alias(leg + "_close"),
            pl.col("volume").alias(leg + "_volume"),
        )

    marks = select(stock, "spot").join(select(future, "future"), on="begin", how="full", coalesce=True)
    return marks.with_columns(
        (
            (pl.col("spot_volume") > 0)
            & (pl.col("future_volume") > 0)
            & pl.col("spot_close").is_not_null()
            & pl.col("future_close").is_not_null()
        )
        .fill_null(False)
        .alias("same_active_minute"),
        pl.lit(False).alias("exact_simultaneity_verified"),
    ).sort("begin")


def run(package, output, workers=3):
    if not 1 <= workers <= 4:
        raise ValueError("workers must be 1..4")
    tables, source = read_package(package)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        raise FileExistsError("Completed archive is immutable")
    identity = dict(
        source_manifest_sha256=digest(Path(package) / "manifest.json"),
        start=source["start"],
        end=source["end"],
        interval=1,
        cutoff_minute="18:45",
    )
    if (root / "request.json").exists() and json.loads((root / "request.json").read_text()) != identity:
        raise ValueError("Archive request differs")
    write_json(root / "request.json", identity)
    contracts = tables["contracts"].filter(pl.col("secid").str.contains(OUTRIGHT.pattern))
    jobs = [
        dict(secid=ticker, engine="stock", market="shares", board="TQBR", start=source["start"], end=source["end"])
        for ticker in ASSETS.values()
    ]
    for row in contracts.to_dicts():
        # Bound by observed daily history, not assumed expiry minus N days.
        history = tables["futures"].filter(pl.col("SECID") == row["secid"])
        jobs.append(
            dict(
                secid=row["secid"],
                engine="futures",
                market="forts",
                board="RFUD",
                start=str(history["date"].min()),
                end=str(history["date"].max()),
            )
        )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda job: download_one(job, root), jobs))
    coverage, marks = [], []
    stocks = {ticker: pl.read_parquet(root / ticker / "candles.parquet") for ticker in ASSETS.values()}
    for row in contracts.to_dicts():
        secid, asset = row["secid"], row["asset_code"]
        future = pl.read_parquet(root / secid / "candles.parquet")
        history = tables["futures"].filter(pl.col("SECID") == secid)
        stock = stocks[ASSETS[asset]].filter(
            pl.col("begin").dt.date().is_between(history["date"].min(), history["date"].max())
        )
        paired = common_marks(stock, future).with_columns(pl.lit(secid).alias("secid"), pl.lit(asset).alias("asset"))
        # Keep dates with neither cutoff candle explicit, not silently dropped.
        expected = history.select(pl.col("date").cast(pl.Datetime).dt.offset_by("18h45m").alias("begin"))
        paired = expected.join(paired, on="begin", how="full", coalesce=True).with_columns(
            pl.col("same_active_minute", "exact_simultaneity_verified").fill_null(False),
            pl.lit(secid).alias("secid"),
            pl.lit(asset).alias("asset"),
        )
        marks.append(paired)
        coverage.append(
            dict(
                secid=secid,
                asset=asset,
                daily_rows=history.height,
                minute_rows=future.height,
                dates_with_minutes=future["begin"].dt.date().n_unique(),
                same_active_minute=int(paired["same_active_minute"].sum()),
                missing_cutoff=int((~paired["same_active_minute"]).sum()),
            )
        )
    pl.concat(marks).sort("secid", "begin").write_parquet(root / "common_minutes.parquet")
    pl.DataFrame(coverage).write_parquet(root / "coverage.parquet")
    (root / "exporter.py").write_bytes(Path(__file__).read_bytes())
    manifest = dict(
        schema_version=1,
        status="minute_archive_not_tick_execution",
        **identity,
        excluded_non_outright=tables["contracts"].height - contracts.height,
        instruments=len(jobs),
        rows=sum(r["rows"] for r in results),
        coverage=coverage,
        exact_trade_time_available=False,
        publication_time_verified=False,
        historical_specifications_verified=False,
        artifact_sha256={p.relative_to(root).as_posix(): digest(p) for p in root.rglob("*") if p.is_file()},
    )
    write_json(root / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ["status", "instruments", "rows", "excluded_non_outright"]}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=3)
    args = p.parse_args()
    run(args.package, args.output, args.workers)
