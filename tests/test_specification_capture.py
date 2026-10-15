import base64
import hashlib
import json

import pytest

from moexutils.specification_capture import import_captures


def capture(tmp_path, body=b"%PDF-test", sha=None):
    doc = dict(
        url="https://www.moex.com/files/test",
        format="pdf",
        base64=base64.b64encode(body).decode(),
        sha256=sha or hashlib.sha256(body).hexdigest(),
        received_at="2026-10-15T00:00:00Z",
    )
    source = tmp_path / "capture.json"
    source.write_text(
        json.dumps(
            dict(
                schema_version=1,
                kind="moex_specification_browser_capture",
                page_url="https://www.moex.com/ru/documents/4979",
                documents=[doc, doc],
            )
        )
    )
    return source


def test_duplicates_preserve_observations_not_applicability(tmp_path):
    source = capture(tmp_path)
    result = import_captures([source], tmp_path / "out")
    assert result["unique_documents"] == 1
    assert result["observations"] == 2
    assert not result["complete_archive_verified"]
    with pytest.raises(FileExistsError):
        import_captures([source], tmp_path / "out")


@pytest.mark.parametrize("body,sha", [(b"%PDF-test", "wrong"), (b"<html>login</html>", None)])
def test_bad_hash_or_html_rejected_before_output(tmp_path, body, sha):
    source = capture(tmp_path, body, sha)
    with pytest.raises(ValueError):
        import_captures([source], tmp_path / "out")
    assert not (tmp_path / "out").exists()
