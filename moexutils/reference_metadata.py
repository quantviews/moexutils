"""Immutable reference releases backed by verified source packages and DuckLake."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import polars as pl

from . import lake

FORMAT_VERSION = 1


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def verify_package(root, manifest_name="manifest.json"):
    root = Path(root).resolve()
    manifest = json.loads((root / manifest_name).read_text(encoding="utf-8"))
    hashes = manifest["artifact_sha256"]
    if not hashes:
        raise ValueError("Empty manifest")
    for name, expected in hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha256(path) != expected:
            raise ValueError(f"Checksum/path mismatch: {name}")
    return hashes


def frame(path):
    if path.suffix == ".parquet":
        result = pl.read_parquet(path)
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Expected nonempty array: {path}")
        rows = [{k: dump(v) if isinstance(v, (list, dict)) else v for k, v in r.items()} for r in rows]
        result = pl.DataFrame(rows, infer_schema_length=None)
    # Preserve unknown values as SQL NULL, never let all-null columns become INTEGER.
    return result.with_columns([pl.col(c).cast(pl.String) for c, t in result.schema.items() if t == pl.Null])


def build(config, data_root, destination=None):
    """Verify sources, preserve full input packages, and publish one content-addressed release."""
    data_root = Path(data_root).resolve()
    destination = Path(destination or data_root / "reference/metadata")
    config = json.loads(Path(config).read_text(encoding="utf-8"))
    packages, identities = {}, {}
    for name, item in config["packages"].items():
        root = (data_root / item["path"]).resolve()
        if not root.is_relative_to(data_root):
            raise ValueError("Source outside data root")
        manifest_name = item.get("manifest", "manifest.json")
        hashes = verify_package(root, manifest_name)
        packages[name] = (root, manifest_name, hashes)
        identities[name] = sha256(root / manifest_name)
    identity = dict(format_version=FORMAT_VERSION, config=config, sources=identities,
                    builder_sha256=sha256(Path(__file__)))
    version = hashlib.sha256(dump(identity).encode()).hexdigest()
    output = destination / version
    if output.exists():
        verify_package(output)
        return output
    destination.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".building-", dir=destination))
    try:
        artifacts = []
        for name, (root, manifest_name, hashes) in packages.items():
            for relative in sorted(set(hashes) | {manifest_name}):
                target = staging / "sources" / name / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / relative, target)
                actual = sha256(target)
                expected = hashes.get(relative, identities[name])
                if actual != expected:
                    raise ValueError("Source changed during copy")
                artifacts.append(dict(package=name, source_artifact=relative, sha256=actual,
                                      original_path=str(root / relative),
                                      archived_path=target.relative_to(staging).as_posix()))
        frames = {}
        for table, spec in config["tables"].items():
            if not table.startswith("meta_") or not table.replace("_", "").isalnum():
                raise ValueError("Invalid metadata table name")
            name, artifact = spec["package"], spec["artifact"]
            if artifact not in packages[name][2]:
                raise ValueError(f"Unverified input: {artifact}")
            df = frame(staging / "sources" / name / artifact)
            reserved = {"version_id", "record_id", "package_id", "input_artifact"}
            if reserved.intersection(df.columns):
                raise ValueError("Input uses reserved provenance columns")
            if df.select(spec["key"]).is_duplicated().any():
                raise ValueError(f"Duplicate logical key: {table}")
            ids = [hashlib.sha256(dump(r).encode()).hexdigest() for r in df.select(spec["key"]).to_dicts()]
            frames[table] = df.with_columns(pl.Series("record_id", ids), pl.lit(name).alias("package_id"),
                                            pl.lit(artifact).alias("input_artifact"))
        frames["meta_artifacts"] = pl.DataFrame(artifacts).with_columns(
            pl.Series("record_id", [hashlib.sha256(dump([r["package"], r["source_artifact"]]).encode()).hexdigest()
                                    for r in artifacts]))
        schemas, counts = {}, {}
        for table, df in frames.items():
            df = df.with_columns(pl.lit(version).alias("version_id"))
            df.write_parquet(staging / f"{table}.parquet")
            schemas[table] = {k: str(v) for k, v in df.schema.items()}
            counts[table] = df.height
        summary = dict(version_id=version, created_at=datetime.now(timezone.utc).isoformat(),
                       scope=config["scope"], identity=identity, tables=counts, schemas=schemas,
                       historical_known_at_verified=False)
        (staging / "summary.json").write_text(dump(summary), encoding="utf-8")
        (staging / "builder.py").write_bytes(Path(__file__).read_bytes())
        hashes = {p.relative_to(staging).as_posix(): sha256(p) for p in staging.rglob("*") if p.is_file()}
        (staging / "manifest.json").write_text(dump(dict(artifact_sha256=hashes)), encoding="utf-8")
        staging.rename(output)
    except BaseException:
        # Keep failed staging for diagnosis; never publish or delete source packages.
        raise
    return output


def publish(package):
    """Append all tables and release marker in ONE transaction; repeat is a no-op."""
    package = Path(package)
    hashes = verify_package(package)
    summary = json.loads((package / "summary.json").read_text(encoding="utf-8"))
    version = summary["version_id"]
    frames = {table: pl.read_parquet(package / f"{table}.parquet") for table in summary["tables"]}
    for name, df in frames.items():
        if f"{name}.parquet" not in hashes or df.height != summary["tables"][name]:
            raise ValueError("Release table mismatch")
        if df["version_id"].null_count() or df["version_id"].unique().to_list() != [version]:
            raise ValueError("Mixed versions")
    manifest_hash = sha256(package / "manifest.json")
    frames["meta_versions"] = pl.DataFrame([dict(version_id=version, manifest_sha256=manifest_hash,
        package_path=str(package.resolve()), created_at=summary["created_at"],
        imported_at=datetime.now(timezone.utc).isoformat(), scope=summary["scope"],
        format_version=FORMAT_VERSION, historical_known_at_verified=False)])
    with lake.session() as con:
        con.execute("BEGIN")
        try:
            existing = set(lake.tables(con))
            if "meta_versions" in existing:
                prior = con.execute("SELECT manifest_sha256 FROM lake.meta_versions WHERE version_id=?", [version]).fetchall()
                if prior:
                    if prior != [(manifest_hash,)]:
                        raise ValueError("Version already exists with a different manifest")
                    con.execute("ROLLBACK")
                    return False
            for table, df in frames.items():
                con.register("_metadata_input", df)
                target = "lake." + lake._quote(table)
                if table not in existing:
                    con.execute(f"CREATE TABLE {target} AS SELECT * FROM _metadata_input LIMIT 0")
                else:
                    columns = lake._columns(con, table)
                    for column, dtype, *_ in con.execute("DESCRIBE _metadata_input").fetchall():
                        if column not in columns:
                            con.execute(f"ALTER TABLE {target} ADD COLUMN {lake._quote(column)} {dtype}")
                con.execute(f"INSERT INTO {target} BY NAME SELECT * FROM _metadata_input")
            con.execute("COMMIT")
        except BaseException:
            con.execute("ROLLBACK")
            raise
    return True


def read(table, version_id):
    """Read a pinned metadata release; deliberately no implicit latest/as-of inference."""
    if not table.startswith("meta_") or table not in lake.tables():
        raise ValueError("Unknown metadata table")
    return lake.query(f"SELECT * FROM lake.{lake._quote(table)} WHERE version_id = ?", [version_id])
