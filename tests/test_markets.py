"""
Тесты новых наборов: RUONIA и КБД (rates), денежные потоки облигаций
(cashflows), реестр и непрерывные ряды фьючерсов (contracts), фильтр строк
набора в history. ISS и cbr.ru подменены, каталог хранилища — файловый DuckLake.
"""
import datetime as dt

import polars as pl
import pytest

from moexutils import cashflows, contracts, history, lake, rates


@pytest.fixture
def lake_env(tmp_path, monkeypatch):
    monkeypatch.setenv('MOEX_LAKE_CATALOG', 'ducklake:' + str(tmp_path / 'catalog.ducklake').replace('\\', '/'))
    monkeypatch.setattr(lake, 'LAKE_DATA_PATH', str(tmp_path / 'data'))
    return tmp_path


class Resp:
    def __init__(self, payload=None, text=''):
        self._payload, self.text = payload, text

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class FakeSession:
    """Ответы по функции handler(url, params) -> Resp; запросы запоминаются."""

    def __init__(self, handler):
        self.handler, self.calls = handler, []

    def get(self, url, params=None, headers=None):
        self.calls.append((url, dict(params or {})))
        return self.handler(url, dict(params or {}))


def block(columns, rows, types=None):
    return {'columns': columns, 'data': rows,
            'metadata': {c: {'type': (types or {}).get(c, 'string')} for c in columns}}


# ---------------------------------------------------------------- RUONIA

RUONIA_HTML = """<table class="data"><tr><th>Дата ставки</th><th>Ставка</th><th>Объем</th><th>Сделок</th>
<th>Участников</th><th>Мин</th><th>P25</th><th>P75</th><th>Макс</th><th>Статус</th><th>Публикация</th></tr>
<tr><td>28.09.2026</td><td>14,12</td><td>448,28</td><td>46</td><td>18</td><td>13,90</td><td>14,10</td>
<td>14,15</td><td>14,20</td><td>Стандартный</td><td>29.09.2026</td></tr>
<tr><td>25.09.2026</td><td>14,13</td><td>588,70</td><td>49</td><td>18</td><td>—</td><td>14,05</td>
<td>14,15</td><td>14,25</td><td>Стандартный</td><td>28.09.2026</td></tr></table>"""


class TestRuonia:
    def test_parse_and_update_only_changes(self, lake_env):
        html = {'text': RUONIA_HTML}
        s = FakeSession(lambda url, p: Resp(text=html['text']))
        df = rates.fetch_ruonia(session=s)
        assert df['date'].to_list() == [dt.date(2026, 9, 25), dt.date(2026, 9, 28)]
        assert df['rate'].to_list() == [14.13, 14.12] and df['rate_min'][0] is None
        assert df['published'][1] == dt.date(2026, 9, 29)
        assert rates.update_ruonia(session=s) == 2
        assert rates.update_ruonia(session=s) == 0
        html['text'] = RUONIA_HTML.replace('14,12', '14,11')          # ЦБ уточнил значение
        assert rates.update_ruonia(session=s) == 1
        assert rates.read_ruonia()['rate'].to_list() == [14.13, 14.11]


# ---------------------------------------------------------------- КБД

def zcyc_payload(day):
    if day.weekday() >= 5:
        return {b: block(['tradedate'], []) for b in ('params', 'yearyields', 'securities')}
    d = day.isoformat()
    num = {c: 'double' for c in ('B1', 'period', 'value', 'clcyield')}
    return {
        'params': block(['tradedate', 'tradetime', 'B1'], [[d, '18:00:00', 1000.0]], num),
        'yearyields': block(['tradedate', 'tradetime', 'period', 'value'],
                            [[d, '18:00:00', 1.0, 13.0], [d, '18:00:00', 10.0, 16.0]], num),
        'securities': block(['tradedate', 'tradetime', 'secid', 'clcyield'],
                            [[d, '18:00:00', 'SU26238RMFS4', 15.0]], num),
    }


class TestZcyc:
    def test_backfill_and_tail_to_yesterday(self, lake_env, monkeypatch):
        s = FakeSession(lambda url, p: Resp(zcyc_payload(dt.date.fromisoformat(p['date']))))
        yesterday = dt.date.today() - dt.timedelta(days=1)
        start = yesterday - dt.timedelta(days=9)
        n = rates.update_zcyc(start=start, session=s)
        weekdays = sum((start + dt.timedelta(days=i)).weekday() < 5 for i in range(10))
        assert n == weekdays
        params = rates.read_zcyc('params')
        assert params['date'].max() <= yesterday and params.height == weekdays
        assert rates.read_zcyc('yields').height == 2 * weekdays
        assert rates.read_zcyc('bonds')['secid'].unique().to_list() == ['SU26238RMFS4']
        s.calls.clear()
        assert rates.update_zcyc(session=s) == 0 and s.calls == []  # по вчерашний день актуально
        # бэкфилл: даты раньше истории — назад от нее
        rates.update_zcyc(start=start - dt.timedelta(days=20), session=s)
        assert rates.read_zcyc('params')['date'].min() < start


# ---------------------------------------------------------------- денежные потоки

def flows_handler(state):
    num = {c: 'double' for c in ('value', 'valueprc', 'value_rub', 'facevalue', 'price')}

    def handler(url, p):
        name = p['iss.only'].split(',')[0]
        rows = state[name]
        lo, hi = p.get('from'), p.get('till')
        date_col = {'coupons': 'coupondate', 'amortizations': 'amortdate', 'offers': 'offerdate'}[name]
        cols = list(rows[0].keys()) if rows else [date_col]
        sel = [r for r in rows if (lo is None or (r[date_col] or '9999') >= lo) and (hi is None or (r[date_col] or '') <= hi)]
        page = sel[p['start']:p['start'] + 100]
        return Resp({name: block(cols, [[r[c] for c in cols] for r in page], num),
                     f'{name}.cursor': block(['INDEX', 'TOTAL', 'PAGESIZE'], [[p['start'], len(sel), 100]],
                                             {'INDEX': 'int64', 'TOTAL': 'int64', 'PAGESIZE': 'int64'})})
    return handler


def coupon(secid, date, value):
    return {'secid': secid, 'coupondate': date, 'recorddate': date, 'startdate': '2026-01-01',
            'facevalue': 1000.0, 'value': value, 'valueprc': 10.0, 'value_rub': value}


class TestCashflows:
    def test_window_writes_changes_and_drops_cancelled(self, lake_env):
        today = dt.date.today()
        d1, d2 = (today + dt.timedelta(days=5)).isoformat(), (today + dt.timedelta(days=20)).isoformat()
        far = (today + dt.timedelta(days=400)).isoformat()
        state = {
            'coupons': [coupon('A', d1, 50.0), coupon('B', d2, None), coupon('A', far, 50.0)],
            'amortizations': [{'secid': 'A', 'amortdate': far, 'data_source': 'maturity', 'value': 1000.0}],
            'offers': [{'secid': 'B', 'offerdate': '0000-00-00', 'offerdatestart': d2, 'offerdateend': d2,
                        'offertype': 'Оферта', 'price': 100.0}],
        }
        s = FakeSession(flows_handler(state))
        out = cashflows.update_cashflows('full', session=s)
        assert out == {'bond_coupons': 3, 'bond_amortizations': 1, 'bond_offers': 1}
        cp = cashflows.read_cashflows('coupons')
        assert 'value_rub' not in cp.columns and cp['coupondate'].dtype == pl.Date
        offers = cashflows.read_cashflows('offers')
        assert offers['offerdate'][0] is None and offers['offer_date'][0] == dt.date.fromisoformat(d2)
        # купон флоатера зафиксирован, купон A отменен, оферта состоялась
        state['coupons'] = [coupon('B', d2, 42.0), coupon('A', far, 50.0)]
        out = cashflows.update_cashflows('window', session=s)
        assert out['bond_coupons'] == 1
        cp = cashflows.read_cashflows('coupons')
        assert cp.select('secid', 'value').rows() == [('A', 50.0), ('B', 42.0)]   # d1 удален, far вне окна цел
        # оферты без offerdate нет в выдаче окна — она не удаляется
        assert cashflows.read_cashflows('offers').height == 1
        # оферта с датой состоялась: тип меняется, ключ (secid, offer_date) тот же
        state['offers'] = [{'secid': 'B', 'offerdate': d2, 'offerdatestart': d2, 'offerdateend': d2,
                            'offertype': 'Оферта (состоялось)', 'price': 100.0}]
        assert cashflows.update_cashflows('window', session=s)['bond_offers'] == 1
        assert cashflows.read_cashflows('offers')['offertype'].to_list() == ['Оферта (состоялось)']
        assert cashflows.update_cashflows('window', session=s)['bond_coupons'] == 0


# ---------------------------------------------------------------- фьючерсы

def fut_rows(secid, dates, settle, oi, board='RFUD'):
    return pl.DataFrame({'date': dates, 'SECID': [secid] * len(dates), 'BOARDID': [board] * len(dates),
                         'OPEN': settle, 'HIGH': settle, 'LOW': settle, 'CLOSE': settle,
                         'SETTLEPRICE': settle, 'VOLUME': [10.0] * len(dates), 'OPENPOSITION': oi})


class TestContracts:
    def test_registry_and_remap_after_relisting(self, lake_env):
        payload = {'series': block(
            ['secid', 'name', 'start_date', 'expiration_date', 'asset_code', 'underlying_asset', 'is_traded'],
            [['SiZ5_2015', 'Si-12.15', '2014-12-01', '2015-12-15', 'Si', 'USD', 0],
             ['SiZ5', 'Si-12.25', '2023-12-15', '2025-12-18', 'Si', 'USD', 0]], {'is_traded': 'int32'})}
        s = FakeSession(lambda url, p: Resp(payload))
        assert contracts.update_contracts(session=s) == (2, 0)
        reg = contracts.read_contracts()
        assert set(reg['base_secid']) == {'SiZ5'} and reg['expiration_date'].dtype == pl.Date
        # 2015 загружен до повторного листинга — под старым кодом; 2025 — новый контракт
        lake.write('futures', pl.concat([fut_rows('SiZ5', [dt.date(2015, 11, 2)], [66000.0], [1.0]),
                                         fut_rows('SiZ5', [dt.date(2025, 11, 3)], [81000.0], [1.0])]))
        assert contracts.remap_futures_secids() == 1
        got = lake.query('SELECT date, SECID FROM lake.futures ORDER BY date').rows()
        assert got == [(dt.date(2015, 11, 2), 'SiZ5_2015'), (dt.date(2025, 11, 3), 'SiZ5')]
        assert contracts.remap_futures_secids() == 0

    def test_continuous_rolls_by_open_interest_and_back_adjusts(self, lake_env):
        days = [dt.date(2026, 3, d) for d in (2, 3, 4, 5, 6)]
        lake.write('futures_contracts', pl.DataFrame({
            'secid': ['SiH6', 'SiJ6', 'SiM6'], 'asset_code': ['Si'] * 3, 'base_secid': ['SiH6', 'SiJ6', 'SiM6'],
            'expiration_date': [dt.date(2026, 3, 19), dt.date(2026, 4, 16), dt.date(2026, 6, 18)]}))
        lake.write('futures', pl.concat([
            fut_rows('SiH6', days, [100.0, 101.0, 102.0, 103.0, 104.0], [900.0, 800.0, 100.0, 50.0, 10.0]),
            fut_rows('SiJ6', days, [110.0] * 5, [5.0] * 5),                    # неликвидный месячный
            fut_rows('SiM6', days, [200.0, 202.0, 204.0, 206.0, 208.0], [100.0, 200.0, 700.0, 800.0, 900.0])]))
        c = contracts.build_continuous(['Si'])
        assert c['SECID'].to_list() == ['SiH6', 'SiH6', 'SiM6', 'SiM6', 'SiM6']
        assert c['roll'].to_list() == [False, False, True, False, False]
        # отношение в последний день SiH6: 202/101 = 2 — история в уровне SiM6, без скачка
        assert c['settle_adj'].to_list() == pytest.approx([200.0, 202.0, 204.0, 206.0, 208.0])
        assert c['settle'].to_list() == [100.0, 101.0, 204.0, 206.0, 208.0]
        assert contracts.update_continuous(['Si']) == (5, 0)
        assert contracts.update_continuous(['Si']) == (0, 0)


# ---------------------------------------------------------------- наборы history

class TestDatasetFilter:
    def test_currency_rows_without_trades_dropped(self, monkeypatch):
        frame = pl.DataFrame({'date': [dt.date(2026, 9, 29)] * 3, 'SECID': ['USD000UTSTOM', 'X', 'Y'],
                              'BOARDID': ['CETS', 'LICU', 'CNGD'], 'NUMTRADES': [28.0, 0.0, 3.0]})
        monkeypatch.setattr(history.iss, 'history_day', lambda path, day, session: frame)
        assert history._fetch('currency', dt.date(2026, 9, 29), None)['SECID'].to_list() == ['USD000UTSTOM', 'Y']
        assert history._fetch('shares', dt.date(2026, 9, 29), None).height == 3


# ---------------------------------------------------------------- параметры бумаг

def refdata_payload(day, sizes):
    """Срез на дату: две бумаги, объем выпуска по словарю, «мигающий» флаг hasprospectus."""
    cols = ['tradedate', 'updatetime', 'secid', 'issuesize', 'listlevel', 'hasprospectus', 'accruedint']
    rows = [[day, f'{day} 04:00:00', sec, sizes.get(sec, 1000.0), '1', float(dt.date.fromisoformat(day).day % 2), 5.0]
            for sec in ('AAA', 'BBB')]
    num = {c: 'double' for c in ('issuesize', 'hasprospectus', 'accruedint')}
    return {'securities': block(cols, rows, num),
            'securities.cursor': block(['INDEX', 'TOTAL', 'PAGESIZE'], [[0, 2, 1000]],
                                       {'INDEX': 'int64', 'TOTAL': 'int64', 'PAGESIZE': 'int64'})}


class TestRefdata:
    def test_only_changes_stored_and_state_at_date(self, lake_env, monkeypatch):
        from moexutils import refdata
        yesterday = dt.date.today() - dt.timedelta(days=1)
        days = [yesterday - dt.timedelta(days=i) for i in range(14)]
        days = sorted(d for d in days if d.weekday() < 5)[-6:]
        change_day = days[3]

        def handler(url, p):
            sizes = {'AAA': 2000.0} if p['date'] >= change_day.isoformat() else {}
            return Resp(refdata_payload(p['date'], sizes))
        s = FakeSession(handler)
        n = refdata.update_refdata(start=days[0], session=s)
        got = refdata.read_refdata()
        # первая дата — обе бумаги; потом только смена объема AAA; флаг и НКД не хранятся
        assert n == 3 and 'hasprospectus' not in got.columns and 'accruedint' not in got.columns
        assert got.select('secid', 'date', 'issuesize').rows() == [
            ('AAA', days[0], 1000.0), ('AAA', change_day, 2000.0), ('BBB', days[0], 1000.0)]
        assert refdata.refdata_at(days[2], 'AAA')['issuesize'].to_list() == [1000.0]
        assert refdata.refdata_at(days[-1])['issuesize'].to_list() == [2000.0, 1000.0]
        s.calls.clear()
        assert refdata.update_refdata(session=s) == 0 and s.calls == []   # по вчерашний день актуально


class TestZcycWorkingSaturday:
    def test_repair_fills_calendar_days_and_remembers_empty(self, lake_env):
        sat, empty_day = dt.date(2025, 11, 1), dt.date(2025, 11, 5)
        cal = [dt.date(2025, 10, 31), sat, dt.date(2025, 11, 3), empty_day, dt.date(2025, 11, 6)]
        lake.write('indexes', pl.DataFrame({'date': cal, 'ticker': ['IMOEX'] * 5, 'BOARDID': ['SNDX'] * 5,
                                            'close': [1.0] * 5, 'value_rub': [1.0] * 5, 'volume': [1.0] * 5}))

        def payload(day):
            if day == empty_day:
                return {b: block(['tradedate'], []) for b in ('params', 'yearyields', 'securities')}
            p = zcyc_payload(dt.date(2025, 11, 3))          # будний шаблон с датой day
            for b in p.values():
                for row in b['data']:
                    row[0] = day.isoformat()
            return p
        s = FakeSession(lambda url, p: Resp(payload(dt.date.fromisoformat(p['date']))))
        for d in (cal[0], cal[2], cal[4]):
            lake.write('zcyc_params', rates.fetch_zcyc(d, s)['params'])
        s.calls.clear()
        assert rates.repair_zcyc(session=s) == 1                       # суббота докачана
        assert sat in rates.read_zcyc('params')['date'].to_list()
        s.calls.clear()
        assert rates.repair_zcyc(session=s) == 0 and s.calls == []     # пустой день больше не запрашивается


# ---------------------------------------------------------------- состав индексов

def weights_payload(indexid, day, tickers):
    cols = ['indexid', 'tradedate', 'ticker', 'shortnames', 'secids', 'weight', 'tradingsession', 'trade_session_date']
    rows = [[indexid, day, t, t, t, w, 3.0, day] for t, w in tickers]
    return {'analytics': block(cols, rows, {'weight': 'double', 'tradingsession': 'double'}),
            'analytics.cursor': block(['INDEX', 'TOTAL', 'PAGESIZE'], [[0, len(rows), 100]],
                                      {'INDEX': 'int64', 'TOTAL': 'int64', 'PAGESIZE': 'int64'})}


class TestIndexWeights:
    def test_calendar_days_progress_and_constituents(self, lake_env):
        from moexutils import indices
        cal = [dt.date(2025, 10, 30), dt.date(2025, 10, 31), dt.date(2025, 11, 1), dt.date(2025, 11, 3)]
        lake.write('indexes', pl.DataFrame({'date': cal, 'ticker': ['IMOEX'] * 4, 'BOARDID': ['SNDX'] * 4,
                                            'close': [1.0] * 4, 'value_rub': [1.0] * 4, 'volume': [1.0] * 4}))

        def handler(url, p):
            if url.endswith('analytics.json'):
                return Resp({'indices': block(['indexid', 'shortname', 'from', 'till'],
                                              [['IMOEX', 'Индекс', '2025-10-30', '2025-11-03']])})
            day = p['date']
            if day == '2025-10-31':                                  # индекс в этот день не рассчитывался
                return Resp(weights_payload("IMOEX", day, []))
            tickers = [('SBER', 15.0), ('GAZP', 10.0)] if day < '2025-11-03' else [('SBER', 16.0), ('LKOH', 9.0)]
            return Resp(weights_payload('IMOEX', day, tickers))
        s = FakeSession(handler)
        n = indices.update_index_weights(['IMOEX'], session=s)
        assert n == 6                                               # 3 даты с составом × 2 бумаги
        w = indices.read_index_weights('IMOEX')
        assert sorted(set(w['date'].to_list())) == [cal[0], cal[2], cal[3]]   # рабочая суббота есть
        assert indices.constituents_at('IMOEX', '2025-11-02')['ticker'].to_list() == ['SBER', 'GAZP']
        assert indices.constituents_at('IMOEX', cal[3])['ticker'].to_list() == ['SBER', 'LKOH']
        s.calls.clear()
        assert indices.update_index_weights(['IMOEX'], session=s) == 0
        assert [c for c in s.calls if 'analytics/' in c[0]] == []    # прогресс в load_state, пустой день не повторяется


class TestSwapratesView:
    def test_perpetual_futures_only(self, lake_env):
        lake.write('futures_contracts', pl.DataFrame({
            'secid': ['USDRUBF', 'SiZ6'], 'asset_code': ['USDRUBF', 'Si'], 'underlying_asset': ['USD', 'USD'],
            'expiration_date': [dt.date(2100, 1, 1), dt.date(2026, 12, 17)], 'base_secid': ['USDRUBF', 'SiZ6']}))
        f = fut_rows('USDRUBF', [dt.date(2026, 9, 29)], [81.0], [1.0]).vstack(fut_rows('SiZ6', [dt.date(2026, 9, 29)], [82000.0], [1.0]))
        lake.write('futures', f.with_columns(SWAPRATE=pl.Series([0.03, 0.0]), SWAPRATE_CURR=pl.Series([0.0004, 0.0]),
                                             VALUE=pl.Series([1e6, 1e6])))
        assert 'futures_swaprates' in lake.ensure_views()
        got = lake.query('SELECT SECID, swaprate_rub, swaprate_curr FROM lake.futures_swaprates')
        assert got.rows() == [('USDRUBF', 0.03, 0.0004)]
