import json
from pathlib import Path

import pytest

from scripts.finalize_ncc_review import edition_for, verify


def registry():
    return json.loads((Path(__file__).parents[1] / "scripts/ncc_review_registry.json").read_text(encoding="utf-8"))


def test_never_effective_edition_and_transition():
    editions = registry()["editions"]
    assert edition_for("2024-03-15", editions)["approved_date"] == "2023-08-28"
    assert edition_for("2023-07-30", editions)["approved_date"] == "2023-03-01"
    assert edition_for("2023-07-31", editions)["approved_date"] == "2023-06-23"
    with pytest.raises(ValueError):
        edition_for("2022-01-01", editions)
    with pytest.raises(ValueError):
        edition_for("2023-07-31", editions + [editions[2]])


def test_clearing_time_cancelled_early():
    decisions = {r["number"]: r for r in registry()["decisions"]}
    assert decisions["01-02/182"]["effective_to"] == "2023-03-20"
    assert decisions["01-02/203"]["effective_from"] == "2023-03-21"


def test_corrupt_source_rejected(tmp_path):
    (tmp_path / "source.pdf").write_bytes(b"changed")
    (tmp_path / "manifest.json").write_text(json.dumps({"artifact_sha256": {"source.pdf": "0" * 64}}))
    with pytest.raises(ValueError, match="checksum/path"):
        verify(tmp_path)
