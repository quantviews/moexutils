"""
Тесты фасада moex_utils: математика облигаций, HTTP-сессия, корень данных и
pandas-обертки для ноутбуков (read_moex_stock, combine_moex_stocks, ...).
Логика акций, индексов и качества — в test_stocks.py / test_quality.py,
облигаций и фьючерсов — в test_history.py, хранилища — в test_lake.py.
"""
import datetime as dt
import os

import pandas as pd
import polars as pl
import pytest

import lake
import moex_utils as mu
import stocks


class TestBondMetrics:
    def test_ytm_par_bond_equals_coupon(self):
        # Облигация по номиналу: YTM = купонной ставке
        ytm = mu.calculate_ytm(price=100, face_value=1000, coupon_rate=10,
                               years_to_maturity=1, coupon_freq=2)
        assert ytm == pytest.approx(10.0, abs=0.1)

    def test_ytm_premium_bond_below_coupon(self):
        ytm = mu.calculate_ytm(price=105, face_value=1000, coupon_rate=10,
                               years_to_maturity=1, coupon_freq=2)
        assert 0 < ytm < 10

    def test_ytm_discount_bond_above_coupon(self):
        ytm = mu.calculate_ytm(price=95, face_value=1000, coupon_rate=10,
                               years_to_maturity=1, coupon_freq=2)
        assert ytm > 10

    def test_ytm_zero_periods(self):
        assert mu.calculate_ytm(price=100, face_value=1000, coupon_rate=10,
                                years_to_maturity=0) == 0

    def test_ytm_independent_of_face_value_scale(self):
        """Регрессия: старый солвер сходился только при номинале ~1000."""
        for face in (1, 100, 1000, 100000):
            ytm = mu.calculate_ytm(price=100, face_value=face, coupon_rate=10,
                                   years_to_maturity=1, coupon_freq=2)
            assert ytm == pytest.approx(10.0, abs=1e-4), f"face_value={face}"

    def test_ytm_long_maturity_converges(self):
        ytm = mu.calculate_ytm(price=100, face_value=1000, coupon_rate=8,
                               years_to_maturity=30, coupon_freq=2)
        assert ytm == pytest.approx(8.0, abs=1e-4)

    def test_duration_zero_coupon_bond(self):
        # Для бескупонной облигации Маколей = сроку, модифицированная = срок / (1 + ytm/freq)
        duration = mu.calculate_duration(price=90, face_value=1000, coupon_rate=0,
                                         years_to_maturity=1, ytm=10, coupon_freq=2)
        assert duration == pytest.approx(1 / 1.05, abs=1e-6)

    def test_duration_below_maturity_for_coupon_bond(self):
        duration = mu.calculate_duration(price=100, face_value=1000, coupon_rate=10,
                                         years_to_maturity=5, ytm=10, coupon_freq=2)
        assert 0 < duration < 5

    def test_convexity_zero_coupon(self):
        # Бескупонная: единственный CF в периоде n → C = n(n+1)/(f²(1+y_p)²)
        n, f, ytm = 2, 2, 10.0
        y_p = ytm / 100 / f
        expected = n * (n + 1) / (f ** 2 * (1 + y_p) ** 2)
        convexity = mu.calculate_convexity(price=90, face_value=1000, coupon_rate=0,
                                           years_to_maturity=1, ytm=ytm, coupon_freq=f)
        assert convexity == pytest.approx(expected, rel=1e-9)

    def test_convexity_properties(self):
        # Положительна и растет со сроком
        c5 = mu.calculate_convexity(100, 1000, 10, 5, 10)
        c10 = mu.calculate_convexity(100, 1000, 10, 10, 10)
        assert 0 < c5 < c10
        assert mu.calculate_convexity(100, 1000, 10, 0, 10) == 0.0

# ---------------------------------------------------------------- infrastructure


class TestInfrastructure:
    def test_session_has_default_timeout(self, monkeypatch):
        seen = {}

        def fake_request(self, method, url, **kwargs):
            seen.update(kwargs)
            return None

        monkeypatch.setattr(mu.requests.Session, 'request', fake_request)
        session = mu.make_session()
        session.get('https://iss.moex.com/iss/x.json')
        assert seen['timeout'] == mu.ISS_TIMEOUT
        session.get('https://iss.moex.com/iss/x.json', timeout=5)
        assert seen['timeout'] == 5  # явный таймаут не перекрывается


class TestDataRoot:
    def test_env_var_moves_data_root(self, tmp_path, monkeypatch):
        import importlib
        monkeypatch.setenv('MOEX_DATA_ROOT', str(tmp_path))
        try:
            m = importlib.reload(mu)
            assert m.DATA_ROOT == str(tmp_path)
            assert m.DATA_FOLDER == os.path.join(str(tmp_path), 'data')
            assert importlib.reload(lake).LAKE_DATA_PATH == os.path.join(str(tmp_path), 'lake')
            assert m.SPLITS_FILE.startswith(m.BASE_DIR)                 # реестры остаются в проекте
        finally:
            monkeypatch.delenv('MOEX_DATA_ROOT')
            importlib.reload(lake)
            importlib.reload(mu)


class TestPandasWrappers:
    @pytest.fixture
    def env(self, tmp_path, monkeypatch):
        monkeypatch.setenv('MOEX_LAKE_CATALOG', 'ducklake:' + str(tmp_path / 'c.ducklake').replace(chr(92), '/'))
        monkeypatch.setattr(lake, 'LAKE_DATA_PATH', str(tmp_path / 'lake'))
        for name in ('SPLITS_FILE', 'EXTERNAL_SPLITS_FILE', 'RENAMES_FILE', 'DELISTED_FILE'):
            monkeypatch.setattr(stocks, name, str(tmp_path / f'{name}.csv'))
        (tmp_path / 'RENAMES_FILE.csv').write_text('old,new,date\nOLD,NEW,2025-01-03\n', encoding='utf-8')
        (tmp_path / 'SPLITS_FILE.csv').write_text('ticker,date,ratio,kind\nNEW,2025-01-03,10,price\n',
                                                  encoding='utf-8')
        days = [dt.date(2025, 1, 2), dt.date(2025, 1, 3)]
        rows = pl.DataFrame({'date': days, 'ticker': ['OLD', 'NEW'], 'open': [1000.0, 100.0],
                             'low': [1000.0, 100.0], 'high': [1000.0, 100.0], 'close': [1000.0, 100.0],
                             'waprice': [1000.0, 100.0], 'volume': [1.0, 10.0], 'value_rub': [1e3, 1e3],
                             'adj_close': [100.0, 100.0], 'shares': [None, None], 'market_cap': [None, None]},
                            schema_overrides={'shares': pl.Float64, 'market_cap': pl.Float64})
        lake.write('stocks', rows)
        lake.write('indexes', pl.DataFrame({'date': days, 'ticker': ['IMOEX'] * 2, 'BOARDID': ['SNDX'] * 2,
                                            'close': [2800.0, 2810.0], 'value_rub': [1.0, 1.0], 'volume': [1.0, 1.0]}))
        return tmp_path

    def test_combine_and_splits_like_notebooks(self, env):
        df = mu.combine_moex_stocks()
        assert isinstance(df, pd.DataFrame) and isinstance(df.index, pd.DatetimeIndex)
        assert set(df['ticker']) == {'NEW'} and set(df['source_ticker']) == {'OLD', 'NEW'}
        adj = mu.adjust_for_splits(df)
        assert adj['close'].tolist() == pytest.approx([100.0, 100.0])
        raw = mu.combine_moex_stocks(merge_renames=False)
        assert set(raw['ticker']) == {'OLD', 'NEW'}
        assert set(mu.apply_renames(raw)['ticker']) == {'NEW'}

    def test_read_stock_and_index(self, env):
        s = mu.read_moex_stock('new')
        assert s.index.name == 'date' and s['close'].tolist() == [100.0]
        i = mu.read_moex_index('IMOEX')
        assert i['close'].tolist() == [2800.0, 2810.0] and str(i.index.dtype) == 'datetime64[ns]'

    def test_risk_free_monthly_series(self, tmp_path):
        path = tmp_path / 'kr.csv'
        path.write_text('date,rate\n2025-02-15,12.0\n', encoding='utf-8')
        rf = mu.risk_free_monthly(pd.to_datetime(['2025-01-31', '2025-03-31']), str(path))
        assert isinstance(rf, pd.Series) and rf.tolist() == pytest.approx([0.01, 0.01])
