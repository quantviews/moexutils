from datetime import date
import json

from moexutils.quality import trading_restriction


def test_restriction_expires_and_never_masks_resumed_prices(tmp_path):
    p = tmp_path/'restrictions.json'
    p.write_text(json.dumps([dict(ticker='X',last_price_date='2026-10-15',
        no_regular_prices_from='2026-10-16',review_through='2026-11-30',
        reason='reviewed suspension',sources=['https://example.org/source'])]))
    assert trading_restriction('X',date(2026,10,15),date(2026,11,6),p)
    assert trading_restriction('X',date(2026,10,16),date(2026,11,6),p) is None
    assert trading_restriction('X',date(2026,10,15),date(2026,12,1),p) is None
    assert trading_restriction('X',date(2026,10,15),date(2026,10,15),p) is None
