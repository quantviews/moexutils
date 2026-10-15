from moexutils.specification_news import TARGET, selection
import json

import pytest

from moexutils import specification_news


def test_risk_and_perpetual_mentions_are_not_series_exceptions():
    assert selection("Об изменении риск-параметров на срочном рынке") == "risk_notice_not_contract_terms"
    assert selection("Об изменении параметров срочных контрактов") == "selected"
    # Target mention is a review candidate, not a finding about a quarterly series.
    assert TARGET.search("Вечный фьючерс на акции Сбербанка")
    assert TARGET.search("SRH3")
    assert not TARGET.search("SRH3X")


def test_html_title_and_general_rules_are_selected():
    assert selection("О лоте фьючерса на акции &quot;Полюс&quot;") == "selected"
    assert selection("Изменение правил клиринга") == "selected"
    assert selection("О внесении изменений в список облигаций") == "other_title"


def test_capture_filters_dates_as_values_and_refuses_completed_output(tmp_path, monkeypatch):
    def fake_fetch(root, relative, url):
        if relative.startswith("index/"):
            return {"sitenews.cursor": {"columns": ["INDEX", "TOTAL", "PAGESIZE"], "data": [[0, 2, 50]]},
                    "sitenews": {"columns": ["id", "title", "published_at"], "data": [
                        [1, "Об исполнении фьючерса", "2023-03-01 10:00:00"],
                        [2, "Об исполнении фьючерса", "2021-01-01 10:00:00"]]}}
        return {"content": {"columns": ["id", "title", "published_at", "body"],
                            "data": [[1, "Об исполнении фьючерса", "2023-03-01 10:00:00", "<p>SRH3</p>"]]}}
    monkeypatch.setattr(specification_news, "fetch", fake_fetch)
    result = specification_news.capture(tmp_path, workers=1)
    assert result['bodies'] == result['unique_titles'] == result['target_mentions'] == 1
    assert result['exhaustive_archive_verified'] is False
    with pytest.raises(FileExistsError):
        specification_news.capture(tmp_path, workers=1)


def test_resume_checks_raw_checksum_before_reading_cache(tmp_path):
    (tmp_path / 'cached.json').write_text('{}')
    (tmp_path / 'cached.meta.json').write_text(json.dumps({'url': 'https://iss.moex.com/iss/sitenews.json', 'sha256': 'bad'}))
    with pytest.raises(ValueError, match='checksum'):
        specification_news.fetch(tmp_path, 'cached.json', 'https://iss.moex.com/iss/sitenews.json')
