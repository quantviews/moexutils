"""Combine disjoint, verified Barchart packages without changing source captures."""

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

import polars as pl

from moexutils.barchart_pilot import read_contracts


def normalize_contract(job):
    """Delivery month from explicit provider symbol, never an inferred expiry."""
    match = re.fullmatch(r"(KC|OJ|TG)([FGHJKMNQUVXZ])(\d{2})", job["symbol"])
    if match is None:
        raise ValueError("Unsupported contract symbol")
    prefix, month, year = match.groups()
    if job["asset"] != {"KC": "COFFEE", "OJ": "ORANGE", "TG": "TTF"}[prefix]:
        raise ValueError("Symbol/asset mismatch")
    delivery = f"{2000 + int(year):04d}-{'FGHJKMNQUVXZ'.index(month) + 1:02d}"
    if job.get("delivery_month", delivery) != delivery:
        raise ValueError("Symbol/delivery month mismatch")
    return {
        **job,
        "delivery_month": delivery,
        "calendar_status": job.get("calendar_status", "delivery_month_candidate_expiry_unverified"),
    }


def merge_packages(packages, output, snapshot_end=None):
    if snapshot_end is not None:
        snapshot_end = dt.date.fromisoformat(str(snapshot_end)).isoformat()
    frames, jobs, coverage, inputs, copies = [], [], [], [], {}
    for index, package in enumerate(packages, 1):
        root = Path(package).resolve()
        summary = (root / "summary.json").read_bytes()
        meta = json.loads(summary)
        if meta.get("status") != "parsed_not_execution_validated":
            raise ValueError("Complete capture required")
        for name, digest in meta["artifact_sha256"].items():
            source = (root / name).resolve()
            if not source.is_relative_to(root):
                raise ValueError("Artifact outside package")
            data = source.read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError("Checksum mismatch")
            copies[f"inputs/{index}/{source.relative_to(root).as_posix()}"] = data
        if "manifest.json" not in meta["artifact_sha256"]:
            raise ValueError("Unverified manifest")
        copies[f"inputs/{index}/summary.json"] = summary
        manifest = json.loads(copies[f"inputs/{index}/manifest.json"])
        frame = read_contracts(root)
        expected = {j["symbol"] for j in manifest["contracts"]}
        if expected != set(frame["symbol"]):
            raise ValueError("Manifest and series differ")
        frames.append(frame)
        jobs.extend(normalize_contract(j) for j in manifest["contracts"])
        coverage.extend(normalize_contract(j) for j in meta["coverage"])
        inputs.append(dict(package=str(root), summary_sha256=hashlib.sha256(summary).hexdigest()))
    if not frames:
        raise ValueError("No packages")
    if len({j["symbol"] for j in jobs}) != len(jobs):
        raise ValueError("Overlapping series; reconcile revisions separately")
    frame = pl.concat(frames).sort("asset", "symbol", "date")
    if frame.select("symbol", "date").is_duplicated().any():
        raise ValueError("Duplicate series dates")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, data in copies.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    manifest = dict(
        schema_version=2, purpose="historical_availability_probe", campaign="merged_verified_packages", contracts=jobs
    )
    if snapshot_end is not None:
        manifest["snapshot_end"] = snapshot_end
        manifest["snapshot_policy"] = "consumer_cutoff_raw_history_preserved"
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    frame.write_parquet(output / "contracts.parquet")
    meta = dict(
        schema_version=1,
        status="parsed_not_execution_validated",
        full_history_verified=False,
        snapshot_end=snapshot_end,
        coverage=coverage,
        inputs=inputs,
        rows=frame.height,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        artifact_sha256={
            p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in output.rglob("*")
            if p.is_file()
        },
    )
    (output / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packages", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--snapshot-end", type=dt.date.fromisoformat)
    args = parser.parse_args()
    result = merge_packages(args.packages, args.output, args.snapshot_end)
    print(json.dumps(dict(rows=result["rows"], contracts=len(result["coverage"]))))
