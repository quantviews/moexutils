"""Combine disjoint, verified Barchart packages without changing source captures."""

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl

from moexutils.barchart_pilot import read_contracts


def merge_packages(packages, output):
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
        jobs.extend(manifest["contracts"])
        coverage.extend(meta["coverage"])
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
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    frame.write_parquet(output / "contracts.parquet")
    meta = dict(
        schema_version=1,
        status="parsed_not_execution_validated",
        full_history_verified=False,
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
    args = parser.parse_args()
    result = merge_packages(args.packages, args.output)
    print(json.dumps(dict(rows=result["rows"], contracts=len(result["coverage"]))))
