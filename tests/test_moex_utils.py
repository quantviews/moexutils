import os
import tempfile
import pandas as pd
import pytest
import moex_utils as mu


class DummyResponse:
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code
    def raise_for_status(self):
        if self.status_code != 200:
            raise RuntimeError(f"HTTP {self.status_code}")
    def json(self):
        return self._json_data


def test_calculate_ytm_simple():
    ytm = mu.calculate_ytm(price=100, face_value=1000, coupon_rate=10, years_to_maturity=1, coupon_freq=2)
    assert isinstance(ytm, float)
    assert ytm > 0


def test_calculate_duration_simple():
    ytm = 10.0
    duration = mu.calculate_duration(price=100, face_value=1000, coupon_rate=10, years_to_maturity=1, ytm=ytm)
    assert isinstance(duration, float)
    assert duration >= 0


def test_add_bond_metrics():
    dates = pd.date_range('2025-01-01', periods=3, freq='D')
    df = pd.DataFrame({'CLOSE': [102, 103, 104]}, index=dates)
    params = pd.Series({'FACEVALUE': 1000, 'COUPONPERCENT': 10, 'MATDATE': '2026-01-01'})
    result = mu.add_bond_metrics(df, params)
    assert 'ytm' in result.columns
    assert 'duration' in result.columns
    assert len(result) == 3


def test_get_moex_bonds_list(monkeypatch):
    expected = {'securities': [['SECID', 'SHORTNAME'], ['BOND1', 'Test Bond']]}

    class FakeSession:
        def get(self, url, params=None):
            assert 'bonds/securities.json' in url
            return DummyResponse(expected)

    df = mu.get_moex_bonds_list(segment='TQCB', session=FakeSession())
    assert df.loc[0, 'SECID'] == 'BOND1'


def test_get_moex_bond_params(monkeypatch):
    data = {'securities': [['SECID', 'COUPONPERCENT'], ['BOND1', '10.0']]}

    class FakeSession:
        def get(self, url, params=None):
            assert 'bonds/securities/BOND1.json' in url
            return DummyResponse(data)

    df = mu.get_moex_bond_params('BOND1', session=FakeSession())
    assert df.loc[0, 'SECID'] == 'BOND1'


def test_get_moex_bond_prices(monkeypatch):
    data = {
        'history': [
            ['TRADEDATE', 'CLOSE', 'WAPRICE'],
            ['2025-01-01', '101', '100'],
            ['2025-01-02', '102', '101']
        ]
    }

    class FakeSession:
        def get(self, url, params=None):
            assert 'history/engines/stock/markets/bonds/securities/' in url
            return DummyResponse(data)

    df = mu.get_moex_bond_prices('BOND1', start='2025-01-01', end='2025-01-02', session=FakeSession())
    assert 'CLOSE' in df.columns
    assert df.index[0] == pd.Timestamp('2025-01-01')


def test_save_read_update_bond(monkeypatch, tmp_path):
    # Настроим bonda папку во временной директории
    monkeypatch.setattr(mu, 'BONDS_FOLDER', str(tmp_path))

    # Мокаем данные цены
    prices = pd.DataFrame(
        {'CLOSE': [100, 101], 'WAPRICE': [100, 101]},
        index=pd.to_datetime(['2025-01-01', '2025-01-02'])
    )

    monkeypatch.setattr(mu, 'get_moex_bond_prices', lambda secid, start, end, session=None: prices)

    mu.save_moex_bond('BOND1', start='2025-01-01', end='2025-01-02')
    df_read = mu.read_moex_bond('BOND1')
    assert len(df_read) == 2

    # апдейт сделает выброс, так как добавлена та же дата и сравнительное start > end
    mu.update_moex_bond('BOND1')
    df_read2 = mu.read_moex_bond('BOND1')
    assert len(df_read2) == 2


def test_save_moex_stock_with_mock(monkeypatch, tmp_path):
    monkeypatch.setattr(mu, 'DATA_FOLDER', str(tmp_path))

    sample_df = pd.DataFrame(
        {'value_rub': [100, 101], 'volume': [10, 11], 'ticker': ['TEST', 'TEST']},
        index=pd.to_datetime(['2025-01-01', '2025-01-02'])
    )

    def fake_get_moex_stock(ticker, start, end, session, frequency):
        return sample_df

    monkeypatch.setattr(mu, 'get_moex_stock', fake_get_moex_stock)
    monkeypatch.setattr(mu, 'calculate_market_cap', lambda df, ticker, metadata_file: df.assign(market_cap=[1000, 1010]))

    out_path = mu.save_moex_stock('TEST', start='2025-01-01', end='2025-01-02')
    assert out_path is not None
    assert os.path.exists(out_path)

    loaded = pd.read_parquet(out_path)
    assert 'market_cap' in loaded.columns


def test_calculate_adj_close_no_dividends():
    dates = pd.to_datetime(['2025-01-01', '2025-01-02'])
    df = pd.DataFrame({'close': [100.0, 101.0], 'ticker': ['TEST', 'TEST']}, index=dates)
    tmp_dir = tempfile.mkdtemp()
    result = mu.calculate_adj_close(df, div_folder=tmp_dir)
    assert 'adj_close' in result.columns
    assert all(result['adj_close'] == result['close'])
