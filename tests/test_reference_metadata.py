import hashlib
import json

import polars as pl
import pytest

from moexutils import lake, reference_metadata as metadata


def source(tmp_path, value=0):
    root = tmp_path / "input"
    root.mkdir(exist_ok=True)
    pl.DataFrame({"day": ["2023-06-12"], "flag": [value], "unknown": [None]}).write_parquet(root / "calendar.parquet")
    hashes = {"calendar.parquet": metadata.sha256(root / "calendar.parquet")}
    (root / "manifest.json").write_text(json.dumps({"artifact_sha256": hashes}))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"scope": "test", "packages": {"calendar": {"path": "input"}},
        "tables": {"meta_test_calendar": {"package": "calendar", "artifact": "calendar.parquet", "key": ["day"]}}}))
    return config


def test_versions_preserve_sources_and_reject_tampering(tmp_path):
    config = source(tmp_path)
    first = metadata.build(config, tmp_path)
    assert metadata.build(config, tmp_path) == first
    metadata.verify_package(first)
    assert (first / "sources/calendar/calendar.parquet").read_bytes() == (tmp_path / "input/calendar.parquet").read_bytes()
    source(tmp_path, 1)
    second = metadata.build(config, tmp_path)
    assert first != second
    assert pl.read_parquet(first / "meta_test_calendar.parquet")["flag"].to_list() == [0]
    (tmp_path / "input/calendar.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Checksum"):
        metadata.build(config, tmp_path)


def test_path_escape_rejected(tmp_path):
    root = tmp_path / "package"
    root.mkdir()
    (tmp_path / "outside").write_bytes(b"x")
    (root / "manifest.json").write_text(json.dumps({"artifact_sha256": {"../outside": hashlib.sha256(b"x").hexdigest()}}))
    with pytest.raises(ValueError, match="path"):
        metadata.verify_package(root)


def test_publish_repeat_and_history(tmp_path, monkeypatch):
    monkeypatch.setenv("MOEX_LAKE_CATALOG", "ducklake:" + (tmp_path / "catalog.ducklake").as_posix())
    monkeypatch.setattr(lake, "LAKE_DATA_PATH", str(tmp_path / "lake"))
    first = metadata.build(source(tmp_path), tmp_path)
    assert metadata.publish(first)
    assert not metadata.publish(first)
    second = metadata.build(source(tmp_path, 1), tmp_path)
    assert metadata.publish(second)
    assert metadata.read("meta_test_calendar", first.name)["flag"].to_list() == [0]
    assert metadata.read("meta_test_calendar", second.name)["flag"].to_list() == [1]
    assert lake.query("SELECT count(*) n FROM lake.meta_versions")["n"][0] == 2


def test_transaction_rolls_back_all_tables(tmp_path, monkeypatch):
    monkeypatch.setenv("MOEX_LAKE_CATALOG", "ducklake:" + (tmp_path / "catalog.ducklake").as_posix())
    monkeypatch.setattr(lake, "LAKE_DATA_PATH", str(tmp_path / "lake"))
    package = metadata.build(source(tmp_path), tmp_path)
    # Force failure after the first table was inserted, before the release marker.
    with lake.session() as con:
        con.execute("CREATE TABLE lake.meta_artifacts (version_id INTEGER)")
    with pytest.raises(Exception):
        metadata.publish(package)
    assert "meta_versions" not in lake.tables()
    assert "meta_test_calendar" not in lake.tables()
