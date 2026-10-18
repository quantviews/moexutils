import json

import polars as pl
import pytest

from moexutils.dividends_acceptance import TARGETS, assess, audit, timestamp, classify_issues


def inputs():
    securities = pl.DataFrame([dict(ticker=t, isin=i, security_id=t, identity_status="identified")
                               for t, i in TARGETS.items()])
    coverage = pl.DataFrame([dict(security_id=t, status="partial") for t in TARGETS])
    conflicts = pl.DataFrame(schema={"security_id": pl.String, "status": pl.String, "conflict_id": pl.String})
    events = pl.DataFrame(schema={"security_id": pl.String, "event_type": pl.String})
    return securities, coverage, conflicts, events


def test_partial_coverage_and_not_ready_block_acceptance():
    issues = assess({"integration_ready": False}, *inputs())
    assert any(r["check"] == "producer_not_ready" for r in issues)
    assert {r["ticker"] for r in issues if r["check"] == "coverage_not_complete"} == set(TARGETS)
    assert {r["ticker"] for r in issues if r["check"] == "no_primary_events"} == set(TARGETS)


def test_no_dividend_does_not_require_record_or_payment_date():
    securities, coverage, conflicts, _ = inputs()
    event = dict(security_id="GAZP", event_type="no_dividend", event_id="zero",
                 published_at="2026-06-29T13:33:05+03:00", publication_evidence="p", edition_evidence="e",
                 public_available_at="2026-06-29T13:33:05+03:00", verification_status="verified",
                 historical_availability="proven", amount="0", record_date=None)
    issues = assess({"integration_ready": True}, securities, coverage, conflicts, pl.DataFrame([event]))
    assert not any(r["ticker"] == "GAZP" and r["check"].startswith("missing_") for r in issues)


def test_handoff_hash_mismatch_stops_before_import(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "manifest.json").write_text("{}")
    handoff = tmp_path / "handoff.json"
    handoff.write_text(json.dumps({"package_path": str(package), "manifest_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="manifest mismatch"):
        audit(tmp_path, handoff, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_timezone_required():
    with pytest.raises(ValueError, match="Timezone"):
        timestamp("2023-03-17T14:35:00")


def test_disclosed_gaps_do_not_override_real_errors():
    issues = classify_issues([dict(check=c) for c in (
        "coverage_not_complete", "missing_share_basis_date", "historical_evidence_missing",
        "missing_actual_payment_date", "missing_causal_example", "identity",
        "producer_not_ready", "unresolved_conflict", "causality_failed")])
    assert [r["severity"] for r in issues] == ["limitation"] * 5 + ["blocking"] * 4
