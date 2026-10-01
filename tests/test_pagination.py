"""Complete ISS pagination and failure behavior across paginated loaders."""
import datetime as dt

import polars as pl
import pytest

from moexutils import cashflows, history, indices, iss, lake, refdata

DAY = dt.date(2026, 9, 28)


class Session:
    def __init__(self, pages):
        self.pages = iter(pages)
        self.offsets = []

    def get(self, url, params):
        self.offsets.append(params['start'])
        payload = next(self.pages)

        class Response:
            def raise_for_status(self):
                pass

            def json(self):
                return payload

        return Response()


def coupon_page(index, total, rows):
    return {'coupons': {'columns': ['secid', 'coupondate'], 'data': rows},
            'coupons.cursor': {'columns': ['INDEX', 'TOTAL'], 'data': [[index, total]]}}


def test_transient_empty_page_retried_at_same_offset():
    session = Session([coupon_page(0, 2, [['A', '2026-10-01']]),
                       coupon_page(1, 2, []), coupon_page(1, 2, [['B', '2026-10-01']])])
    frame = cashflows.fetch_block('coupons', session=session)
    assert frame['secid'].to_list() == ['A', 'B']
    assert session.offsets == [0, 1, 1]


def test_empty_page_retry_exhaustion_still_rejects_partial_data():
    session = Session([coupon_page(0, 1, [])] * 3)
    with pytest.raises(ValueError, match='неполная выдача'):
        cashflows.fetch_block('coupons', session=session)
    assert session.offsets == [0, 0, 0]


def test_total_must_remain_stable_on_empty_page_retry():
    session = Session([coupon_page(0, 1, []), coupon_page(0, 2, [['A', '2026-10-01']])])
    with pytest.raises(ValueError, match='TOTAL изменился'):
        cashflows.fetch_block('coupons', session=session)


@pytest.fixture(params=['day', 'security', 'refdata', 'weights', 'coupons'])
def loader(request):
    kind = request.param
    block = {'day': 'history', 'security': 'history', 'refdata': 'securities',
             'weights': 'analytics', 'coupons': 'coupons'}[kind]
    columns = {'history': ['TRADEDATE', 'SECID'], 'securities': ['secid'],
               'analytics': ['tradedate', 'indexid', 'ticker'], 'coupons': ['secid', 'coupondate']}[block]

    def page(index, total, empty=False, cursor=True):
        row = {'history': [DAY.isoformat(), str(index)], 'securities': [str(index)],
               'analytics': [DAY.isoformat(), 'IMOEX', str(index)],
               'coupons': [str(index), DAY.isoformat()]}[block]
        data = {block: {'columns': columns, 'data': [] if empty else [row]}}
        if cursor:
            data[block + '.cursor'] = {'columns': ['INDEX', 'TOTAL'], 'data': [[index, total]]}
        return data

    def fetch(session, max_pages=3):
        if kind == 'day':
            return iss.history_day('stock/markets/bonds', DAY, session, max_pages=max_pages)
        if kind == 'security':
            return iss.security_history('stock/markets/shares', 'A', DAY, DAY,
                                        session=session, max_pages=max_pages)
        if kind == 'refdata':
            return refdata.fetch_snapshot(DAY, session=session, max_pages=max_pages)
        if kind == 'weights':
            return indices.fetch_weights('IMOEX', DAY, session=session, max_pages=max_pages)
        return cashflows.fetch_block('coupons', session=session, max_pages=max_pages,
                                     empty_page_retries=0)

    return block, page, fetch


def test_complete_response_finishes_at_limit(loader):
    _, page, fetch = loader
    session = Session([page(0, 2), page(1, 2)])
    assert fetch(session, max_pages=2).height == 2
    assert session.offsets == [0, 1]


def test_limit_exhaustion_is_an_error(loader):
    _, page, fetch = loader
    with pytest.raises(ValueError, match='лимит страниц'):
        fetch(Session([page(0, 2)]), max_pages=1)


def test_empty_before_total_is_an_error(loader):
    _, page, fetch = loader
    with pytest.raises(ValueError, match='неполная выдача'):
        fetch(Session([page(0, 2), page(1, 2, empty=True)]))


def test_empty_first_page_with_positive_total_is_an_error(loader):
    _, page, fetch = loader
    with pytest.raises(ValueError, match='неполная выдача'):
        fetch(Session([page(0, 1, empty=True)]))


def test_empty_dataset_is_valid(loader):
    _, page, fetch = loader
    assert fetch(Session([page(0, 0, empty=True)])).is_empty()


def test_no_cursor_requires_empty_page(loader):
    _, page, fetch = loader
    session = Session([page(0, None, cursor=False), page(1, None, cursor=False),
                       page(2, None, empty=True, cursor=False)])
    assert fetch(session).height == 2
    assert session.offsets == [0, 1, 2]


def test_missing_cursor_keeps_previous_total(loader):
    _, page, fetch = loader
    with pytest.raises(ValueError, match='неполная выдача'):
        fetch(Session([page(0, 2), page(1, None, empty=True, cursor=False)]))


@pytest.mark.parametrize('total', [None, -1, 1.5, 'bad'])
def test_invalid_total_is_an_error(loader, total):
    _, page, fetch = loader
    with pytest.raises(ValueError, match='cursor'):
        fetch(Session([page(0, total)]))


def test_wrong_cursor_offset_is_an_error(loader):
    _, page, fetch = loader
    with pytest.raises(ValueError, match='смещение'):
        fetch(Session([page(0, 2), page(0, 2)]))


def test_changing_total_is_an_error(loader):
    _, page, fetch = loader
    with pytest.raises(ValueError, match='TOTAL изменился'):
        fetch(Session([page(0, 2), page(1, 3)]))


def test_more_rows_than_total_is_an_error(loader):
    _, page, fetch = loader
    with pytest.raises(ValueError, match='превышает TOTAL'):
        fetch(Session([page(0, 0)]))


@pytest.mark.parametrize('payload', [{}, None, {'columns': [], 'data': [[1]]},
                                    {'columns': ['A'], 'data': None}])
def test_malformed_block_is_an_error(loader, payload):
    block, _, fetch = loader
    with pytest.raises(ValueError, match='блок'):
        fetch(Session([{} if payload is None else {block: payload}]))


@pytest.mark.parametrize('kind', ['refdata', 'weights', 'history'])
def test_incomplete_day_is_not_saved_or_marked_done(tmp_path, monkeypatch, kind):
    monkeypatch.setenv('MOEX_LAKE_CATALOG', 'ducklake:' + (tmp_path / 'catalog.ducklake').as_posix())
    monkeypatch.setattr(lake, 'LAKE_DATA_PATH', str(tmp_path / 'data'))
    block = {'refdata': 'securities', 'weights': 'analytics', 'history': 'history'}[kind]
    columns = {'securities': ['secid'], 'analytics': ['tradedate', 'indexid', 'ticker'],
               'history': ['TRADEDATE', 'SECID', 'BOARDID']}[block]
    row = {'securities': ['A'], 'analytics': [DAY.isoformat(), 'IMOEX', 'A'],
           'history': [DAY.isoformat(), 'A', 'TQCB']}[block]
    session = Session([
        {block: {'columns': columns, 'data': [row]},
         block + '.cursor': {'columns': ['INDEX', 'TOTAL'], 'data': [[0, 2]]}},
        {block: {'columns': columns, 'data': []},
         block + '.cursor': {'columns': ['INDEX', 'TOTAL'], 'data': [[1, 2]]}},
    ])
    if kind == 'refdata':
        with pytest.raises(ValueError):
            refdata.update_refdata(start=DAY, max_days=1, session=session)
        assert refdata._processed_until() is None
    elif kind == 'weights':
        monkeypatch.setattr(history, 'trading_calendar', lambda: [DAY])
        monkeypatch.setattr(indices, 'list_indexes', lambda session: pl.DataFrame({'indexid': ['IMOEX'], 'from': [DAY.isoformat()]}))
        with pytest.raises(ExceptionGroup):
            indices.update_index_weights(['IMOEX'], start=DAY, session=session)
        assert indices._processed_until('IMOEX') is None
    else:
        with pytest.raises(ValueError):
            history.update('bonds', start=DAY.isoformat(), max_days=1, session=session)
    assert lake.tables() == []
