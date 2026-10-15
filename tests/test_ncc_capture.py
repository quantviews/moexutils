import base64
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def test_partial_capture_preserves_valid_pdf_and_failure(tmp_path):
    pdf_module = pytest.importorskip("pymupdf")
    spec = importlib.util.spec_from_file_location(
        "ncc_review", Path(__file__).parents[1] / "scripts/review_ncc_capture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pdf_module.open() as pdf:
        pdf.new_page().insert_text((72, 72), "Test document")
        body = pdf.tobytes()
    url = "https://www.nationalclearingcentre.ru/connector?cmd=file&target=test"
    valid = dict(url=url, base64=base64.b64encode(body).decode(),
                 sha256=hashlib.sha256(body).hexdigest())
    capture = dict(kind="ncc_clearing_browser_capture",
                   page_url="https://www.nationalclearingcentre.ru/catalog/020415/93",
                   stopped="HTTP 404", documents=[valid, dict(url=url, error="HTTP 404")])
    source = tmp_path / "input.json"
    source.write_text(json.dumps(capture))
    out = tmp_path / "out"
    result = module.review([source], out)
    assert result["documents"] == 1
    assert result["failed_downloads"] == result["stopped_captures"] == 1
    assert not result["historical_archive_complete"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert all(hashlib.sha256((out / name).read_bytes()).hexdigest() == sha
               for name, sha in manifest["artifact_sha256"].items())
    valid["sha256"] = "0" * 64
    source.write_text(json.dumps(capture))
    with pytest.raises(ValueError, match="checksum"):
        module.review([source], tmp_path / "bad")
