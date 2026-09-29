"""
Тесты quality.py: отчет о качестве данных и кандидаты в пропущенные дивиденды.
Хранилище — файловый DuckLake во временной папке, реестры — временные файлы.
"""
import datetime as dt

import polars as pl
import pytest

import lake
import quality
import stocks


def bdays(start, n):
    out, day = [], dt.date.fromisoformat(start)
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day)
        day += dt.timedelta(days=1)
    return out


def stock(ticker, dates, closes, adj=None, opens=None):
    n = len(dates)
    return pl.DataFrame({
        'date': dates, 'ticker': [ticker] * n,
        'open': [float(x) for x in (opens or closes)], 'low': [float(c) for c in closes],
        'high': [float(c) for c in closes], 'close': [float(c) for c in closes],
        'waprice': [float(c) for c in closes], 'volume': [10.0] * n,
        'value_rub': [float(c) * 10 for c in closes],
        'adj_close': [float(a) for a in (adj or closes)], 'shares': [10.0] * n,
        'market_cap': [float(c) * 10 for c in closes],
    })


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv('MOEX_LAKE_CATALOG', 'ducklake:' + str(tmp_path / 'catalog.ducklake').replace('\\', '/'))
    monkeypatch.setattr(lake, 'LAKE_DATA_PATH', str(tmp_path / 'lake'))
    for name in ('SPLITS_FILE', 'EXTERNAL_SPLITS_FILE', 'RENAMES_FILE', 'DELISTED_FILE'):
        monkeypatch.setattr(stocks, name, str(tmp_path / f'{name}.csv'))
    divs = tmp_path / 'divs'
    divs.mkdir()
    cal = bdays('2025-01-01', 60)
    lake.write('indexes', pl.DataFrame({'date': cal, 'ticker': ['IMOEX'] * 60, 'BOARDID': ['SNDX'] * 60,
                                        'close': [1000.0] * 60, 'value_rub': [1.0] * 60, 'volume': [1.0] * 60}))
    return {'cal': cal, 'divs': str(divs), 'tmp': tmp_path}


def report(env, days=30):
    return quality.data_quality_report(days=days, div_folder=env['divs'])


def test_clean_data_has_no_issues(env):
    lake.write('stocks', stock('OK', env['cal'], [100.0] * 60))
    issues = report(env)
    assert issues.is_empty() and quality.quality_summary(issues) == 'Проверка данных: замечаний нет'


def test_stale_gaps_and_adj_artefact(env):
    cal = env['cal']
    adj = [100.0] * 60
    adj[55] = 150.0                                                   # скачок только в adj_close
    hole = [d for i, d in enumerate(cal) if i not in (50, 51)]
    lake.write('stocks', pl.concat([
        stock('LAG', cal[:-3], [100.0] * 57),                         # отстает на 3 дня
        stock('OLD', cal[:30], [100.0] * 30),                         # 30 дней без данных
        stock('HOLE', hole, [100.0] * 58),
        stock('ADJ', cal, [100.0] * 60, adj=adj)]))
    issues = report(env)
    got = {(c, o) for c, o, _ in issues.iter_rows()}
    assert {('stock_stale', 'LAG'), ('stock_stale', 'OLD'), ('stock_gaps', 'HOLE'), ('adj_jump', 'ADJ')} <= got
    assert 'delisted.csv' in issues.filter(pl.col('object') == 'OLD')['detail'][0]
    assert 'замечаний' in quality.quality_summary(issues)


def test_delisted_and_real_moves_not_reported(env):
    cal = env['cal']
    (env['tmp'] / 'DELISTED_FILE.csv').write_text('ticker,last_date,note\nGONE,2025-01-14,x\n', encoding='utf-8')
    lake.write('stocks', pl.concat([stock('GONE', cal[:10], [100.0] * 10),
                                    stock('MOVE', cal, [100.0] * 50 + [150.0] * 10)]))  # рост без разворота
    assert report(env).is_empty()


def test_price_spike_and_no_gap_on_reversal(env):
    closes = [100.0] * 60
    closes[55] = 200.0                                                 # +100% и обратно
    lake.write('stocks', stock('SPIKE', env['cal'], closes))
    assert report(env)['check'].to_list() == ['price_spike']


def test_bonds_stale(env):
    cal = env['cal']
    lake.write('stocks', stock('OK', cal, [100.0] * 60))
    lake.write('bonds', pl.DataFrame({'date': cal[:-2], 'SECID': ['B1'] * 58, 'BOARDID': ['TQOB'] * 58}))
    assert report(env)['check'].to_list() == ['bonds_stale']


def test_dividend_skipped_and_gap(env):
    cal = env['cal']
    opens = [100.0] * 60
    opens[40] = 90.0                                                   # гэп открытия −10% без дивиденда
    closes = [100.0] * 40 + [90.0] * 20
    lake.write('stocks', stock('GAP', cal, closes, opens=opens))
    (env['tmp'] / 'divs' / 'GAP.csv').write_text(
        f'closing_date,dividend_value\n{cal[20]},80\n', encoding='utf-8')   # 80% — неправдоподобно
    checks = report(env, days=60)['check'].to_list()
    assert 'dividend_gap' in checks and 'dividend_skipped' in checks


class TestGapCandidates:
    def df(self):
        dates = [dt.date(2025, 1, 2), dt.date(2025, 1, 3), dt.date(2025, 1, 6), dt.date(2025, 1, 7)]
        return pl.DataFrame({'date': dates, 'ticker': ['TEST'] * 4,
                             'open': [100.0, 100.0, 90.0, 90.0], 'close': [100.0, 100.0, 90.0, 90.0]})

    @pytest.fixture(autouse=True)
    def no_splits(self, tmp_path, monkeypatch):
        monkeypatch.setattr(stocks, 'SPLITS_FILE', str(tmp_path / 'no.csv'))
        monkeypatch.setattr(stocks, 'EXTERNAL_SPLITS_FILE', str(tmp_path / 'no.json'))

    def test_unexplained_gap(self, tmp_path):
        res = quality.find_dividend_gap_candidates(self.df(), str(tmp_path), market_returns={})
        assert res['date'].to_list() == [dt.date(2025, 1, 6)] and res['gap'][0] == pytest.approx(-0.10)

    def test_explained_by_dividend(self, tmp_path):
        (tmp_path / 'TEST.csv').write_text('closing_date,dividend_value\n2025-01-06,10\n', encoding='utf-8')
        assert quality.find_dividend_gap_candidates(self.df(), str(tmp_path), market_returns={}).is_empty()

    def test_explained_by_market(self, tmp_path):
        res = quality.find_dividend_gap_candidates(self.df(), str(tmp_path),
                                                   market_returns={dt.date(2025, 1, 6): -0.09})
        assert res.is_empty()
