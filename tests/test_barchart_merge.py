import hashlib
import json

import pytest

from moexutils.barchart_expansion import import_captures
from moexutils.barchart_merge import merge_packages, normalize_contract
from moexutils.barchart_pilot import read_contracts


def package(tmp_path, symbol):
    manifest = dict(
        campaign=symbol,
        purpose="historical_availability_probe",
        contracts=[
            dict(
                symbol=symbol,
                asset="COFFEE",
                delivery_month="2025-03" if symbol == "KCH25" else "2025-05",
                probe_start="2024-12-30",
                probe_end="2024-12-31",
            )
        ],
    )
    response = dict(
        symbol=symbol,
        http_status=200,
        url=f"https://www.barchart.com/proxies/timeseries/historical/queryeod.ashx?symbol={symbol}&data=daily&volume=contract&backadjust=false&maxrecords=10000",
        body=f"{symbol},2024-12-30,10,12,9,11,2,3\n{symbol},2024-12-31,11,13,10,12,3,4",
    )
    source, meta, root = tmp_path / (symbol + ".json"), tmp_path / (symbol + "-manifest.json"), tmp_path / symbol
    source.write_text(json.dumps(dict(schema_version=2, campaign=symbol, responses=[response])))
    meta.write_text(json.dumps(manifest))
    import_captures([source], root, meta)
    return root


def test_merge_preserves_provenance_and_rejects_overwrite(tmp_path):
    a, b = package(tmp_path, "KCH25"), package(tmp_path, "KCK25")
    out = tmp_path / "merged"
    meta = merge_packages([a, b], out)
    assert read_contracts(out).height == 4
    assert (out / "inputs/1/capture-1.json").read_bytes() == (a / "capture-1.json").read_bytes()
    assert all(hashlib.sha256((out / p).read_bytes()).hexdigest() == h for p, h in meta["artifact_sha256"].items())
    with pytest.raises(FileExistsError):
        merge_packages([a, b], out)


def test_overlapping_series_are_not_silently_deduplicated(tmp_path):
    a = package(tmp_path, "KCH25")
    with pytest.raises(ValueError, match="Overlapping"):
        merge_packages([a, a], tmp_path / "merged")
    assert not (tmp_path / "merged").exists()


def test_tampered_manifest_is_rejected_before_writing(tmp_path):
    a = package(tmp_path, "KCH25")
    with (a / "manifest.json").open("ab") as f:
        f.write(b" ")
    with pytest.raises(ValueError, match="Checksum"):
        merge_packages([a], tmp_path / "merged")
    assert not (tmp_path / "merged").exists()


def test_delivery_metadata_and_snapshot(tmp_path):
    assert normalize_contract(dict(symbol="TGX26", asset="TTF"))["delivery_month"] == "2026-11"
    for extra in [dict(asset="ORANGE"), dict(delivery_month="2025-04")]:
        with pytest.raises(ValueError, match="mismatch"):
            normalize_contract(dict(dict(symbol="KCK25", asset="COFFEE"), **extra))
    a = package(tmp_path, "KCH25")
    out = tmp_path / "snapshot"
    merge_packages([a], out, "2024-12-30")
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["snapshot_end"] == "2024-12-30"
    assert read_contracts(out).height == 2  # Raw bars after cutoff remain intact.
