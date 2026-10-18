import polars as pl
import pytest

from moexutils.spot_basis_delivery import select_events


def events():
    common = dict(verification_status="verified", historical_availability="proven",
                  published_at="2023-01-01T12:00:00+03:00", publication_evidence="source", edition_evidence="archive")
    return pl.DataFrame([
        common | dict(event_id="old", supersedes_id=None, public_available_at="2023-01-01T12:01:00+03:00"),
        common | dict(event_id="fix", supersedes_id="old", published_at="2023-02-01T12:00:00+03:00",
                      public_available_at="2023-02-01T12:01:00+03:00"),
        common | dict(event_id="current_table", supersedes_id=None, historical_availability="unproven",
                      public_available_at=None),
    ])


def test_future_correction_does_not_change_earlier_state():
    f = events()
    assert select_events(f, "2023-01-01T12:00:59+03:00").is_empty()
    assert select_events(f, "2023-01-01T12:01:00+03:00")["event_id"].to_list() == ["old"]
    assert select_events(f, "2023-02-01T12:01:00+03:00")["event_id"].to_list() == ["fix"]


def test_naive_time_and_broken_revision_rejected():
    with pytest.raises(ValueError, match="Timezone"):
        select_events(events(), "2023-02-02T12:00:00")
    with pytest.raises(ValueError, match="revision"):
        select_events(events().filter(pl.col("event_id") != "old"), "2023-02-02T12:00:00+03:00")
