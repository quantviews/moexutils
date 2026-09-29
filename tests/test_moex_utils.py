"""
Тесты moex_utils. Все офлайновые: вызовы MOEX ISS API замоканы.

Организация:
- TestGetMoexStock / TestGetMoexIndex — парсинг ответов API (apimoex замокан)
- TestSaveReadUpdateStock — сохранение/чтение/инкрементальное обновление Parquet
- TestCombineStocks — объединение локальных данных
- TestSharesAndMarketCap — разбор Excel-метаданных и расчет капитализации
- TestAdjClose — математика корректировки цены на дивиденды
- TestBondsApi — запросы по облигациям (мокнутая сессия)
- TestBondsStorage — сохранение/чтение/обновление облигаций
- TestBondMetrics — YTM и дюрация против эталонных значений
"""
import os

import pandas as pd
import pytest

import moex_utils as mu


# ---------------------------------------------------------------- helpers

class DummyResponse:
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code != 200:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._json_data


def make_stock_df(dates, closes, ticker='TEST'):
    idx = pd.to_datetime(dates)
    return pd.DataFrame(
        {
            'value_rub': [float(c) for c in closes],
            'close': [float(c) for c in closes],
            'volume': [10.0] * len(idx),
            'ticker': [ticker] * len(idx),
        },
        index=pd.Index(idx, name='date'),
    )


def write_stock_parquet(data_folder, ticker, df):
    tdir = os.path.join(data_folder, ticker)
    os.makedirs(tdir, exist_ok=True)
    path = os.path.join(tdir, f"{ticker}.parquet")
    df.to_parquet(path)
    return path


@pytest.fixture
def tmp_data_folder(tmp_path, monkeypatch):
    folder = tmp_path / "data"
    folder.mkdir()
    monkeypatch.setattr(mu, 'DATA_FOLDER', str(folder))
    return str(folder)


@pytest.fixture
def metadata_xlsx(tmp_path):
    """Синтетический metadata-файл: листы с датами, шапка на 4-й строке (skiprows=3)."""
    path = tmp_path / "stock-index-base.xlsx"
    sheets = {
        '05.01.2025': pd.DataFrame({'Code': ['TEST', 'OTHER'], 'Number of issued shares': [1000, 50]}),
        '08.01.2025': pd.DataFrame({'Code': ['TEST'], 'Number of issued shares': [2000]}),
        'Info': pd.DataFrame({'Code': ['JUNK'], 'Number of issued shares': [999]}),  # не дата — должен игнорироваться
    }
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, startrow=3, index=False)
    return str(path)


# ---------------------------------------------------------------- stocks: API parsing

class TestGetMoexStock:
    def test_daily_uses_official_history(self, monkeypatch):
        """Дневные данные (frequency=24) идут из /history: CLOSE = закрытие
        основной сессии, единая методика с индексами; дубли по доскам схлопываются;
        запрашиваются OHLC и WAPRICE."""
        history = [
            {'BOARDID': 'SMAL', 'TRADEDATE': '2025-01-01', 'OPEN': 98.0, 'LOW': 97.0,
             'HIGH': 100.0, 'CLOSE': 99.0, 'WAPRICE': 98.5, 'VOLUME': 5, 'VALUE': 100.0},
            {'BOARDID': 'TQBR', 'TRADEDATE': '2025-01-01', 'OPEN': 99.0, 'LOW': 98.0,
             'HIGH': 101.0, 'CLOSE': 100.5, 'WAPRICE': 100.0, 'VOLUME': 10, 'VALUE': 1000.0},
            {'BOARDID': 'TQBR', 'TRADEDATE': '2025-01-02', 'OPEN': 100.5, 'LOW': 100.0,
             'HIGH': 102.0, 'CLOSE': 101.0, 'WAPRICE': 100.8, 'VOLUME': 20, 'VALUE': 2000.0},
        ]
        requested = {}

        def fake_history(session, security, start, end, columns, market, engine):
            requested['columns'] = columns
            return history

        monkeypatch.setattr(mu.apimoex, 'get_market_history', fake_history)

        df = mu.get_moex_stock('SBER', start='2025-01-01', end='2025-01-02')

        assert isinstance(df.index, pd.DatetimeIndex)
        assert len(df) == 2                          # дубль по доске SMAL отброшен
        assert df.loc['2025-01-01', 'close'] == 100.5  # взята главная доска (max VALUE)
        assert df.loc['2025-01-01', 'value_rub'] == 1000.0
        assert df.loc['2025-01-01', 'open'] == 99.0
        assert df.loc['2025-01-01', 'high'] == 101.0
        assert df.loc['2025-01-01', 'low'] == 98.0
        assert df.loc['2025-01-01', 'waprice'] == 100.0
        assert df['volume'].dtype == 'float64'
        assert (df['ticker'] == 'SBER').all()
        assert 'BOARDID' not in df.columns
        assert {'OPEN', 'LOW', 'HIGH', 'WAPRICE'} <= set(requested['columns'])

    def test_intraday_uses_candles(self, monkeypatch):
        candles = [
            {'begin': '2025-01-01 10:00:00', 'open': 99.0, 'close': 100.5,
             'high': 101.0, 'low': 98.0, 'value': 1000.0, 'volume': 10},
            {'begin': '2025-01-01 11:00:00', 'open': 100.5, 'close': 101.0,
             'high': 102.0, 'low': 100.0, 'value': 2000.0, 'volume': 20},
        ]
        monkeypatch.setattr(mu.apimoex, 'get_market_candles',
                            lambda session, security, start, end, interval: candles)

        df = mu.get_moex_stock('SBER', start='2025-01-01', end='2025-01-02', frequency=60)

        assert isinstance(df.index, pd.DatetimeIndex)
        assert 'value_rub' in df.columns          # value переименован
        assert df['volume'].dtype == 'float64'    # int приведен к float
        assert (df['ticker'] == 'SBER').all()
        assert df['close'].iloc[-1] == 101.0

    def test_start_after_end_raises(self):
        with pytest.raises(ValueError):
            mu.get_moex_stock('SBER', start='2025-02-01', end='2025-01-01')

    def test_invalid_date_raises(self):
        with pytest.raises(ValueError):
            mu.get_moex_stock('SBER', start='2025-13-45')

    def test_empty_response_raises(self, monkeypatch):
        monkeypatch.setattr(mu.apimoex, 'get_market_history',
                            lambda **kwargs: [])
        with pytest.raises(RuntimeError, match='empty'):
            mu.get_moex_stock('SBER', start='2025-01-01', end='2025-01-02')


class TestGetMoexIndex:
    HISTORY = [
        {'TRADEDATE': '2025-01-01', 'VALUE': 1e9, 'CLOSE': 3000.0},
        {'TRADEDATE': '2025-01-02', 'VALUE': 2e9, 'CLOSE': 3050.0},
    ]

    def test_parses_history(self, monkeypatch):
        monkeypatch.setattr(mu.apimoex, 'get_market_history',
                            lambda **kwargs: self.HISTORY)
        df = mu.get_moex_index('IMOEX', start='2025-01-01', end='2025-01-02')
        assert list(df.columns) == ['volume', 'close']
        assert df['close'].iloc[-1] == 3050.0

    def test_works_with_provided_session(self, monkeypatch):
        """Регрессия: раньше передача session приводила к UnboundLocalError."""
        monkeypatch.setattr(mu.apimoex, 'get_market_history',
                            lambda **kwargs: self.HISTORY)
        df = mu.get_moex_index('IMOEX', start='2025-01-01', end='2025-01-02',
                               session=object())
        assert len(df) == 2

    def test_missing_columns_raises(self, monkeypatch):
        monkeypatch.setattr(mu.apimoex, 'get_market_history',
                            lambda **kwargs: [{'TRADEDATE': '2025-01-01'}])
        with pytest.raises(RuntimeError):
            mu.get_moex_index('IMOEX', start='2025-01-01', end='2025-01-02')


# ---------------------------------------------------------------- stocks: storage

class TestSaveReadUpdateStock:
    def test_save_writes_parquet_uppercase_dir(self, tmp_data_folder, monkeypatch):
        sample = make_stock_df(['2025-01-01', '2025-01-02'], [100, 101])
        monkeypatch.setattr(mu, 'get_moex_stock', lambda **kwargs: sample)

        out_path = mu.save_moex_stock('test', start='2025-01-01', end='2025-01-02',
                                      calculate_market_cap_flag=False)

        assert out_path == os.path.join(tmp_data_folder, 'TEST', 'TEST.parquet')
        assert os.path.exists(out_path)
        assert not os.path.exists(out_path + '.tmp')  # атомарная запись подчистила tmp
        loaded = pd.read_parquet(out_path)
        assert len(loaded) == 2

    def test_save_with_market_cap_from_metadata(self, tmp_data_folder, metadata_xlsx, monkeypatch):
        sample = make_stock_df(['2025-01-06', '2025-01-09'], [100, 100])
        monkeypatch.setattr(mu, 'get_moex_stock', lambda **kwargs: sample)

        out_path = mu.save_moex_stock('TEST', start='2025-01-06', end='2025-01-09',
                                      metadata_file=metadata_xlsx)

        loaded = pd.read_parquet(out_path)
        assert 'market_cap' in loaded.columns
        # 06.01 действует срез от 05.01 (1000 акций), 09.01 — от 08.01 (2000 акций)
        assert loaded['market_cap'].iloc[0] == pytest.approx(100 * 1000)
        assert loaded['market_cap'].iloc[1] == pytest.approx(100 * 2000)

    def test_save_empty_returns_none(self, tmp_data_folder, monkeypatch):
        monkeypatch.setattr(mu, 'get_moex_stock', lambda **kwargs: pd.DataFrame())
        assert mu.save_moex_stock('TEST') is None

    def test_read_existing_file(self, tmp_data_folder):
        sample = make_stock_df(['2025-01-01'], [100])
        write_stock_parquet(tmp_data_folder, 'TEST', sample)

        df = mu.read_moex_stock('TEST')
        assert len(df) == 1
        assert df['close'].iloc[0] == 100

    def test_update_appends_and_dedupes(self, tmp_data_folder, monkeypatch):
        existing = make_stock_df(['2025-01-01', '2025-01-02', '2025-01-03'], [100, 101, 102])
        write_stock_parquet(tmp_data_folder, 'TEST', existing)

        # Обновление перезапрашивает с последней даты: перекрытие по 03.01 (новая цена 999)
        new = make_stock_df(['2025-01-03', '2025-01-04', '2025-01-05'], [999, 103, 104])
        monkeypatch.setattr(mu, 'get_moex_stock',
                            lambda ticker, start, session=None, frequency=24: new)

        mu.update_moex_stock('TEST', calculate_market_cap_flag=False)

        path = os.path.join(tmp_data_folder, 'TEST', 'TEST.parquet')
        df = pd.read_parquet(path)
        assert len(df) == 5                                   # дубликат схлопнут
        assert df.index.is_monotonic_increasing
        assert df.loc['2025-01-03', 'close'] == 999           # keep='last'
        assert not os.path.exists(path + '.tmp')              # атомарная запись

    def test_ticker_case_insensitive(self, tmp_data_folder, monkeypatch):
        """save/read/update нормализуют тикер к верхнему регистру."""
        sample = make_stock_df(['2025-01-01'], [100])
        monkeypatch.setattr(mu, 'get_moex_stock',
                            lambda ticker=None, start=None, session=None, frequency=24, **kw: sample)

        out_path = mu.save_moex_stock('sber', calculate_market_cap_flag=False)
        assert out_path == os.path.join(tmp_data_folder, 'SBER', 'SBER.parquet')

        df = mu.read_moex_stock('sber')          # нижний регистр находит тот же файл
        assert len(df) == 1

        mu.update_moex_stock('sber', calculate_market_cap_flag=False)
        assert os.path.exists(out_path)

    def test_update_missing_file_is_noop(self, tmp_data_folder, caplog):
        import logging
        with caplog.at_level(logging.INFO, logger='moex_utils'):
            mu.update_moex_stock('NOFILE', calculate_market_cap_flag=False)
        assert 'No existing data' in caplog.text

    def test_update_all_stocks_discovers_tickers(self, tmp_data_folder, monkeypatch):
        for ticker in ('AAA', 'BBB'):
            write_stock_parquet(tmp_data_folder, ticker, make_stock_df(['2025-01-01'], [1]))
        os.makedirs(os.path.join(tmp_data_folder, 'EMPTY_DIR'))  # без parquet — должен игнорироваться

        updated = []
        sessions = []
        mc_flags = []

        div_folders = []

        def fake_update(ticker, session=None, calculate_market_cap_flag=True, div_folder=None):
            updated.append(ticker)
            sessions.append(session)
            mc_flags.append(calculate_market_cap_flag)
            div_folders.append(div_folder)

        monkeypatch.setattr(mu, 'update_moex_stock', fake_update)

        mu.update_all_stocks()
        assert sorted(updated) == ['AAA', 'BBB']
        # одна общая HTTP-сессия на весь прогон
        assert sessions[0] is not None
        assert all(s is sessions[0] for s in sessions)
        assert mc_flags == [True, True]        # дефолт сохранен

        mu.update_all_stocks(calculate_market_cap_flag=False, div_folder='DIVS')
        assert mc_flags[-2:] == [False, False]  # флаг доходит до каждого тикера
        assert div_folders[-2:] == ['DIVS', 'DIVS']

    def test_update_keeps_derived_columns_on_refetched_date(self, tmp_data_folder, monkeypatch):
        """Перекачанная последняя дата не обнуляет adj_close/market_cap из файла."""
        existing = make_stock_df(['2025-01-02', '2025-01-03'], [100, 101])
        existing['adj_close'] = [90.0, 91.0]
        existing['market_cap'] = [1e6, 1.01e6]
        path = write_stock_parquet(tmp_data_folder, 'TEST', existing)
        new = make_stock_df(['2025-01-03', '2025-01-04'], [101, 102])
        monkeypatch.setattr(mu, 'get_moex_stock',
                            lambda ticker, start, session=None, frequency=24: new)

        mu.update_moex_stock('TEST', calculate_market_cap_flag=False)

        df = pd.read_parquet(path)
        assert list(df.columns) == list(existing.columns)
        assert df.loc['2025-01-03', 'adj_close'] == 91.0
        assert df.loc['2025-01-03', 'market_cap'] == 1.01e6
        assert pd.isna(df.loc['2025-01-04', 'adj_close'])  # новая дата — до пересчета

    def test_update_without_changes_does_not_rewrite(self, tmp_data_folder, monkeypatch):
        existing = make_stock_df(['2025-01-02', '2025-01-03'], [100, 101])
        path = write_stock_parquet(tmp_data_folder, 'TEST', existing)
        before = os.path.getmtime(path)
        monkeypatch.setattr(mu, 'get_moex_stock',
                            lambda ticker, start, session=None, frequency=24: existing.iloc[-1:])
        monkeypatch.setattr(mu, '_atomic_to_parquet',
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError('лишняя запись')))

        mu.update_moex_stock('TEST', calculate_market_cap_flag=False)
        assert os.path.getmtime(path) == before

    def test_update_with_div_folder_computes_adj_close(self, tmp_data_folder, tmp_path, monkeypatch):
        existing = make_stock_df(['2025-01-02', '2025-01-03'], [100, 100])
        path = write_stock_parquet(tmp_data_folder, 'TEST', existing)
        new = make_stock_df(['2025-01-03', '2025-01-06'], [100, 90])
        monkeypatch.setattr(mu, 'get_moex_stock',
                            lambda ticker, start, session=None, frequency=24: new)
        divs = tmp_path / 'divs'
        divs.mkdir()
        pd.DataFrame({'closing_date': ['2025-01-06'], 'dividend_value': [10.0]}).to_csv(
            divs / 'TEST.csv', index=False)

        mu.update_moex_stock('TEST', calculate_market_cap_flag=False, div_folder=str(divs))

        df = pd.read_parquet(path)
        assert df['adj_close'].tolist() == pytest.approx([90.0, 90.0, 90.0])

    def test_update_all_stocks_rebuild_redownloads(self, tmp_data_folder, monkeypatch):
        for ticker in ('AAA', 'BBB'):
            write_stock_parquet(tmp_data_folder, ticker, make_stock_df(['2025-01-01'], [1]))

        saved = []
        monkeypatch.setattr(mu, 'save_moex_stock',
                            lambda ticker, **kwargs: saved.append((ticker, kwargs.get('start'))))
        monkeypatch.setattr(mu, 'update_moex_stock',
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError('должен быть rebuild')))

        mu.update_all_stocks(rebuild=True)
        assert sorted(t for t, _ in saved) == ['AAA', 'BBB']
        assert all(s == '2002-01-01' for _, s in saved)


class TestCombineStocks:
    def test_combines_all_tickers(self, tmp_path):
        folder = str(tmp_path)
        write_stock_parquet(folder, 'AAA', make_stock_df(['2025-01-01', '2025-01-02'], [1, 2], ticker='AAA'))
        write_stock_parquet(folder, 'BBB', make_stock_df(['2025-01-01'], [3], ticker='BBB'))

        df = mu.combine_moex_stocks(data_folder=folder)
        assert len(df) == 3
        assert set(df['ticker']) == {'AAA', 'BBB'}

    def test_empty_folder_raises(self, tmp_path):
        with pytest.raises(ValueError):
            mu.combine_moex_stocks(data_folder=str(tmp_path))


# ---------------------------------------------------------------- market cap

class TestSharesAndMarketCap:
    def test_load_shares_data(self, metadata_xlsx):
        shares = mu.load_shares_data(metadata_xlsx)

        assert list(shares.columns) == ['Code', 'date', 'Number of issued shares']
        assert set(shares['date']) == {pd.Timestamp('2025-01-05'), pd.Timestamp('2025-01-08')}
        assert 'JUNK' not in set(shares['Code'])  # лист 'Info' отброшен

    def test_load_shares_data_missing_file(self, tmp_path):
        assert mu.load_shares_data(str(tmp_path / 'nope.xlsx')).empty

    def test_load_shares_data_cached_until_file_changes(self, metadata_xlsx, monkeypatch):
        mu._shares_cache.clear()
        reads = {'count': 0}
        orig_excel_file = pd.ExcelFile

        def counting_excel_file(*args, **kwargs):
            reads['count'] += 1
            return orig_excel_file(*args, **kwargs)

        monkeypatch.setattr(pd, 'ExcelFile', counting_excel_file)

        first = mu.load_shares_data(metadata_xlsx)
        second = mu.load_shares_data(metadata_xlsx)
        assert reads['count'] == 1                      # второй вызов — из кэша
        pd.testing.assert_frame_equal(first, second)

        # меняем mtime — кэш должен инвалидироваться
        st = os.stat(metadata_xlsx)
        os.utime(metadata_xlsx, (st.st_atime, st.st_mtime + 10))
        mu.load_shares_data(metadata_xlsx)
        assert reads['count'] == 2

    def test_load_shares_data_cache_returns_copy(self, metadata_xlsx):
        mu._shares_cache.clear()
        first = mu.load_shares_data(metadata_xlsx)
        first['Code'] = 'MUTATED'                       # портим полученный DataFrame
        second = mu.load_shares_data(metadata_xlsx)
        assert 'MUTATED' not in set(second['Code'])     # кэш не задет

    def test_market_cap_ffill_bfill(self, metadata_xlsx):
        df = make_stock_df(
            ['2025-01-03', '2025-01-06', '2025-01-08', '2025-01-10'],
            [100, 100, 100, 100],
        )
        result = mu.calculate_market_cap(df, 'TEST', metadata_file=metadata_xlsx)

        # 03.01 — до первого среза, bfill от 05.01 → 1000 акций
        # 06.01 — ffill от 05.01 → 1000; 08.01 и 10.01 — срез 08.01 → 2000
        assert result['shares'].tolist() == [1000.0, 1000.0, 2000.0, 2000.0]
        assert result['market_cap'].tolist() == [100000.0, 100000.0, 200000.0, 200000.0]

    def test_market_cap_unknown_ticker_unchanged(self, metadata_xlsx):
        df = make_stock_df(['2025-01-06'], [100])
        result = mu.calculate_market_cap(df, 'UNKNOWN', metadata_file=metadata_xlsx)
        assert 'market_cap' not in result.columns

    def test_add_market_cap_to_all_stocks(self, tmp_data_folder, metadata_xlsx):
        write_stock_parquet(tmp_data_folder, 'TEST', make_stock_df(['2025-01-06'], [100]))

        mu.add_market_cap_to_all_stocks(metadata_file=metadata_xlsx)

        df = pd.read_parquet(os.path.join(tmp_data_folder, 'TEST', 'TEST.parquet'))
        assert df['market_cap'].iloc[0] == pytest.approx(100 * 1000)


# ---------------------------------------------------------------- adjusted close

class TestAdjClose:
    @staticmethod
    def write_dividends(folder, ticker, rows):
        path = os.path.join(str(folder), f"{ticker}.csv")
        pd.DataFrame(rows, columns=['closing_date', 'dividend_value']).to_csv(path, index=False)
        return path

    def test_single_dividend_math(self, tmp_path):
        df = make_stock_df(['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04'],
                           [100, 100, 100, 100])
        self.write_dividends(tmp_path, 'TEST', [('2025-01-03', 10.0)])

        result = mu.calculate_adj_close(df, div_folder=str(tmp_path))

        # T+1: экс-дата = дата отсечки 03.01; дивиденд 10 при цене 100 → фактор 0.9
        # ко всем датам строго до экс-даты
        assert result['adj_close'].tolist() == pytest.approx([90.0, 90.0, 100.0, 100.0])

    def test_two_dividends_compound(self, tmp_path):
        df = make_stock_df(['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04'],
                           [100, 100, 100, 100])
        self.write_dividends(tmp_path, 'TEST', [('2025-01-02', 10.0), ('2025-01-03', 10.0)])

        result = mu.calculate_adj_close(df, div_folder=str(tmp_path))

        # Поздний дивиденд: [90, 90, 100, 100]; ранний: фактор 0.9 к первой дате → [81, 90, 100, 100]
        assert result['adj_close'].tolist() == pytest.approx([81.0, 90.0, 100.0, 100.0])

    def test_ex_date_t2_before_july_2023(self, tmp_path):
        """До 31.07.2023 (T+2) гэп — за торговый день до отсечки, после (T+1) — в день отсечки."""
        df = make_stock_df(['2023-05-08', '2023-05-10', '2023-05-11', '2023-05-12'],
                           [100, 100, 95, 95])
        self.write_dividends(tmp_path, 'TEST', [('2023-05-11', 5.0)])
        result = mu.calculate_adj_close(df, div_folder=str(tmp_path))
        # экс-дата 10.05: гэп цены на 11.05 здесь условный, важна позиция корректировки
        assert result['adj_close'].tolist() == pytest.approx([95.0, 100.0, 95.0, 95.0])

    def test_large_dividend_uses_cum_price(self, tmp_path):
        """Спецдивиденд ~50% (SFIN, 12.2025): доходность — от цены ДО гэпа, а не после."""
        df = make_stock_df(['2025-12-23', '2025-12-24', '2025-12-25', '2025-12-26'],
                           [1799.0, 1828.2, 946.8, 961.2])
        self.write_dividends(tmp_path, 'TEST', [('2025-12-25', 902.0)])
        result = mu.calculate_adj_close(df, div_folder=str(tmp_path))
        f = 1 - 902.0 / 1828.2
        assert result['adj_close'].tolist() == pytest.approx([1799.0 * f, 1828.2 * f, 946.8, 961.2])

    def test_dividend_before_data_ignored(self, tmp_path):
        df = make_stock_df(['2025-01-10', '2025-01-11'], [100, 100])
        self.write_dividends(tmp_path, 'TEST', [('2025-01-01', 10.0)])

        result = mu.calculate_adj_close(df, div_folder=str(tmp_path))
        assert result['adj_close'].tolist() == [100.0, 100.0]

    def test_no_dividend_file(self, tmp_path):
        df = make_stock_df(['2025-01-01'], [100])
        result = mu.calculate_adj_close(df, div_folder=str(tmp_path))
        assert (result['adj_close'] == result['close']).all()

    def test_adj_close_is_split_adjusted(self, tmp_path, monkeypatch):
        """adj_close строится на сплит-скорректированной базе: без дивидендов
        это непрерывный ряд без разрыва на дате сплита."""
        splits = tmp_path / 'splits.csv'
        pd.DataFrame([('TEST', '2025-01-03', 10, 'price')],
                     columns=['ticker', 'date', 'ratio', 'kind']).to_csv(splits, index=False)
        monkeypatch.setattr(mu, 'SPLITS_FILE', str(splits))
        monkeypatch.setattr(mu, 'EXTERNAL_SPLITS_FILE', str(tmp_path / 'nope.json'))

        df = make_stock_df(['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04'],
                           [1000, 1010, 101, 102])
        result = mu.calculate_adj_close(df, div_folder=str(tmp_path))

        assert result['adj_close'].tolist() == pytest.approx([100.0, 101.0, 101.0, 102.0])
        assert result['close'].tolist() == [1000.0, 1010.0, 101.0, 102.0]  # сырой close не тронут

    def test_adj_close_dividend_across_split(self, tmp_path, monkeypatch):
        """Дивиденд в старой ценовой базе: фактор считается по сырой цене,
        применяется к сплит-скорректированной базе."""
        splits = tmp_path / 'splits.csv'
        pd.DataFrame([('TEST', '2025-01-03', 10, 'price')],
                     columns=['ticker', 'date', 'ratio', 'kind']).to_csv(splits, index=False)
        monkeypatch.setattr(mu, 'SPLITS_FILE', str(splits))
        monkeypatch.setattr(mu, 'EXTERNAL_SPLITS_FILE', str(tmp_path / 'nope.json'))

        # экс-дата 02.01: дивиденд 100 руб при сырой цене 1000 (старая база) → фактор 0.9
        self.write_dividends(tmp_path, 'TEST', [('2025-01-02', 100.0)])
        df = make_stock_df(['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04'],
                           [1000, 1010, 101, 102])
        result = mu.calculate_adj_close(df, div_folder=str(tmp_path))

        # база [100, 101, 101, 102], фактор 0.9 к первой дате
        assert result['adj_close'].tolist() == pytest.approx([90.0, 101.0, 101.0, 102.0])

    def test_adj_close_restated_dividend_basis(self, tmp_path, monkeypatch):
        """Дивиденды ВТБ-стиля: файл рестейтнут в новую базу (×5000), сырые цены
        старых дат — в старой. Доходность по сырой базе абсурдна (×350) —
        берется рестейтнутая."""
        splits = tmp_path / 'splits.csv'
        pd.DataFrame([('TEST', '2025-01-03', 0.0002, 'price')],
                     columns=['ticker', 'date', 'ratio', 'kind']).to_csv(splits, index=False)
        monkeypatch.setattr(mu, 'SPLITS_FILE', str(splits))
        monkeypatch.setattr(mu, 'EXTERNAL_SPLITS_FILE', str(tmp_path / 'nope.json'))

        # консолидация 5000:1: сырые цены 0.02 → 100; дивиденд 7 в НОВОЙ базе
        self.write_dividends(tmp_path, 'TEST', [('2025-01-02', 7.0)])
        df = make_stock_df(['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04'],
                           [0.02, 0.02, 100.0, 101.0])
        result = mu.calculate_adj_close(df, div_folder=str(tmp_path))

        # база [100, 100, 100, 101]; доходность 7/100 = 7% → фактор 0.93
        assert result['adj_close'].tolist() == pytest.approx([93.0, 100.0, 100.0, 101.0])

    def test_adjust_for_splits_leaves_adj_close(self, tmp_path):
        """adjust_for_splits не трогает adj_close — он уже в единой базе."""
        path = tmp_path / "splits.csv"
        pd.DataFrame([('TEST', '2025-01-02', 10, 'price')],
                     columns=['ticker', 'date', 'ratio', 'kind']).to_csv(path, index=False)
        df = make_stock_df(['2025-01-01', '2025-01-02'], [1000, 101])
        df['adj_close'] = [100.0, 101.0]  # уже непрерывный

        result = mu.adjust_for_splits(df, splits_file=str(path))
        assert result['close'].tolist() == pytest.approx([100.0, 101.0])
        assert result['adj_close'].tolist() == [100.0, 101.0]  # не изменился

    def test_add_adj_close_to_all_stocks(self, tmp_data_folder, tmp_path):
        write_stock_parquet(tmp_data_folder, 'TEST',
                            make_stock_df(['2025-01-01', '2025-01-02'], [100, 100]))
        div_folder = tmp_path / "divs"
        div_folder.mkdir()
        self.write_dividends(div_folder, 'TEST', [('2025-01-02', 10.0)])

        mu.add_adj_close_to_all_stocks(str(div_folder))

        df = pd.read_parquet(os.path.join(tmp_data_folder, 'TEST', 'TEST.parquet'))
        assert df['adj_close'].tolist() == pytest.approx([90.0, 100.0])


# ---------------------------------------------------------------- splits

class TestSplits:
    @staticmethod
    def write_splits(tmp_path, rows):
        path = tmp_path / "splits.csv"
        pd.DataFrame(rows, columns=['ticker', 'date', 'ratio']).to_csv(path, index=False)
        return str(path)

    def test_prices_divided_before_split_only(self, tmp_path):
        splits = self.write_splits(tmp_path, [('TEST', '2025-01-03', 10)])
        df = make_stock_df(['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04'],
                           [1000, 1010, 101, 102])
        df['waprice'] = [990.0, 1000.0, 100.5, 101.5]

        result = mu.adjust_for_splits(df, splits_file=splits)

        assert result['close'].tolist() == pytest.approx([100.0, 101.0, 101.0, 102.0])
        # waprice — тоже ценовая колонка, корректируется
        assert result['waprice'].tolist() == pytest.approx([99.0, 100.0, 100.5, 101.5])
        # объем до сплита умножается на ratio
        assert result['volume'].tolist() == pytest.approx([100.0, 100.0, 10.0, 10.0])
        # исходный DataFrame не изменен
        assert df['close'].iloc[0] == 1000

    def test_reverse_split(self, tmp_path):
        # Консолидация 100:1 → ratio=0.01: старые цены умножаются на 100
        splits = self.write_splits(tmp_path, [('TEST', '2025-01-02', 0.01)])
        df = make_stock_df(['2025-01-01', '2025-01-02'], [0.01, 1.0])

        result = mu.adjust_for_splits(df, splits_file=splits)
        assert result['close'].tolist() == pytest.approx([1.0, 1.0])

    def test_other_tickers_untouched(self, tmp_path):
        splits = self.write_splits(tmp_path, [('OTHER', '2025-01-02', 10)])
        df = make_stock_df(['2025-01-01', '2025-01-02'], [100, 101])

        result = mu.adjust_for_splits(df, splits_file=splits)
        assert result['close'].tolist() == [100.0, 101.0]

    def test_missing_registry_is_noop(self, tmp_path):
        df = make_stock_df(['2025-01-01'], [100])
        result = mu.adjust_for_splits(df, splits_file=str(tmp_path / 'nope.csv'))
        assert result['close'].tolist() == [100.0]

    def test_real_registry_contains_t_split(self):
        splits = mu.load_splits()
        row = splits[splits['ticker'] == 'T']
        assert len(row) == 1
        assert float(row['ratio'].iloc[0]) == 10.0
        assert row['kind'].iloc[0] == 'auto'

    def test_real_registry_contains_vtbr_consolidation(self):
        splits = mu.load_splits()
        row = splits[splits['ticker'] == 'VTBR']
        assert len(row) == 1
        assert row['kind'].iloc[0] == 'auto'
        # ценовая семантика: консолидация 5000:1 → делитель 1/5000
        assert float(row['ratio'].iloc[0]) == pytest.approx(0.0002)

    def test_load_splits_kind_defaults_to_price(self, tmp_path):
        # Старый формат файла без колонки kind (без внешнего реестра)
        path = self.write_splits(tmp_path, [('TEST', '2025-01-03', 10)])
        splits = mu.load_splits(path, external_file=str(tmp_path / 'nope.json'))
        assert (splits['kind'] == 'price').all()

    def test_adjust_for_splits_ignores_shares_kind(self, tmp_path):
        path = tmp_path / "splits.csv"
        pd.DataFrame([('TEST', '2025-01-02', 100, 'shares')],
                     columns=['ticker', 'date', 'ratio', 'kind']).to_csv(path, index=False)
        df = make_stock_df(['2025-01-01', '2025-01-02'], [100, 101])

        result = mu.adjust_for_splits(df, splits_file=str(path))
        assert result['close'].tolist() == [100.0, 101.0]  # цены не тронуты

    def test_load_splits_merges_external_json(self, tmp_path):
        csv_path = tmp_path / 'splits.csv'
        pd.DataFrame([('AAA', '2025-01-05', 10, 'price')],
                     columns=['ticker', 'date', 'ratio', 'kind']).to_csv(csv_path, index=False)
        json_path = tmp_path / 'splits.json'
        json_path.write_text(
            '{"AAA": [{"date": "2025-01-02", "ratio": 10, "kind": "split"}],'
            ' "BBB": [{"date": "2025-06-01", "ratio": 100, "kind": "reverse"}]}',
            encoding='utf-8')

        splits = mu.load_splits(str(csv_path), external_file=str(json_path))

        # AAA: явная запись из csv побеждает дубликат из json (±45 дней)
        aaa = splits[splits['ticker'] == 'AAA']
        assert len(aaa) == 1 and aaa['kind'].iloc[0] == 'price'
        # BBB: консолидация 100:1 из json → ценовой делитель 0.01, kind=auto
        bbb = splits[splits['ticker'] == 'BBB']
        assert len(bbb) == 1
        assert bbb['kind'].iloc[0] == 'auto'
        assert float(bbb['ratio'].iloc[0]) == pytest.approx(0.01)

    def test_auto_adjusts_prices_when_jump_present(self, tmp_path):
        path = tmp_path / "splits.csv"
        pd.DataFrame([('TEST', '2025-01-03', 10, 'auto')],
                     columns=['ticker', 'date', 'ratio', 'kind']).to_csv(path, index=False)
        df = make_stock_df(['2025-01-01', '2025-01-02', '2025-01-03'], [1000, 1010, 101])

        result = mu.adjust_for_splits(df, splits_file=str(path))
        assert result['close'].tolist() == pytest.approx([100.0, 101.0, 101.0])

    def test_auto_leaves_restated_prices_untouched(self, tmp_path):
        path = tmp_path / "splits.csv"
        pd.DataFrame([('TEST', '2025-01-03', 10, 'auto')],
                     columns=['ticker', 'date', 'ratio', 'kind']).to_csv(path, index=False)
        # ряд без разрыва — история уже рестейтнута источником
        df = make_stock_df(['2025-01-01', '2025-01-02', '2025-01-03'], [100, 101, 102])

        result = mu.adjust_for_splits(df, splits_file=str(path))
        assert result['close'].tolist() == [100.0, 101.0, 102.0]

    def test_market_cap_auto_shares_adjustment(self, tmp_path, monkeypatch):
        """auto + рестейтнутые цены: дробление 1:100 корректирует старые листы акций ×100."""
        splits = tmp_path / 'splits.csv'
        pd.DataFrame([('TEST', '2025-01-07', 100, 'auto')],
                     columns=['ticker', 'date', 'ratio', 'kind']).to_csv(splits, index=False)
        monkeypatch.setattr(mu, 'SPLITS_FILE', str(splits))
        monkeypatch.setattr(mu, 'EXTERNAL_SPLITS_FILE', str(tmp_path / 'nope.json'))

        meta = tmp_path / 'meta.xlsx'
        with pd.ExcelWriter(meta, engine='openpyxl') as writer:
            pd.DataFrame({'Code': ['TEST'], 'Number of issued shares': [1000]}).to_excel(
                writer, sheet_name='05.01.2025', startrow=3, index=False)
            pd.DataFrame({'Code': ['TEST'], 'Number of issued shares': [100000]}).to_excel(
                writer, sheet_name='08.01.2025', startrow=3, index=False)

        # цены гладкие (рестейтнуты) → поправка идет в акции
        df = make_stock_df(['2025-01-06', '2025-01-09'], [100, 100])
        result = mu.calculate_market_cap(df, 'TEST', metadata_file=str(meta))
        assert result['market_cap'].tolist() == pytest.approx([10000000.0, 10000000.0])

    def test_market_cap_shares_adjustment(self, tmp_path, monkeypatch):
        """Консолидация 100:1: старые листы метаданных в старых акциях, цены ISS
        рестейтнуты — без поправки market_cap до события завышен в 100 раз."""
        splits = tmp_path / 'splits.csv'
        pd.DataFrame([('TEST', '2025-01-07', 100, 'shares')],
                     columns=['ticker', 'date', 'ratio', 'kind']).to_csv(splits, index=False)
        monkeypatch.setattr(mu, 'SPLITS_FILE', str(splits))

        meta = tmp_path / 'meta.xlsx'
        with pd.ExcelWriter(meta, engine='openpyxl') as writer:
            pd.DataFrame({'Code': ['TEST'], 'Number of issued shares': [100000]}).to_excel(
                writer, sheet_name='05.01.2025', startrow=3, index=False)
            pd.DataFrame({'Code': ['TEST'], 'Number of issued shares': [1000]}).to_excel(
                writer, sheet_name='08.01.2025', startrow=3, index=False)

        df = make_stock_df(['2025-01-06', '2025-01-09'], [100, 100])
        result = mu.calculate_market_cap(df, 'TEST', metadata_file=str(meta))

        # Обе даты в одной (новой) базе: 100 руб × 1000 акций
        assert result['market_cap'].tolist() == pytest.approx([100000.0, 100000.0])


# ---------------------------------------------------------------- renames

class TestRenames:
    @staticmethod
    def write_renames(tmp_path, rows):
        path = tmp_path / "renames.csv"
        pd.DataFrame(rows, columns=['old', 'new', 'date']).to_csv(path, index=False)
        return str(path)

    def test_missing_registry_is_noop(self, tmp_path):
        df = make_stock_df(['2025-01-01'], [100], ticker='OLD')
        result = mu.apply_renames(df, renames_file=str(tmp_path / 'nope.csv'))
        assert result['ticker'].tolist() == ['OLD']
        # колонка прозрачности добавляется всегда
        assert result['source_ticker'].tolist() == ['OLD']

    def test_old_history_merged_with_source_marker(self, tmp_path):
        renames = self.write_renames(tmp_path, [('OLD', 'NEW', '2025-01-03')])
        df = pd.concat([
            make_stock_df(['2025-01-01', '2025-01-02'], [100, 101], ticker='OLD'),
            make_stock_df(['2025-01-03', '2025-01-04'], [102, 103], ticker='NEW'),
        ])

        result = mu.apply_renames(df, renames_file=renames).sort_index()

        assert (result['ticker'] == 'NEW').all()          # единый тикер
        assert result['source_ticker'].tolist() == ['OLD', 'OLD', 'NEW', 'NEW']
        assert result['close'].tolist() == [100.0, 101.0, 102.0, 103.0]

    def test_old_rows_after_rename_date_dropped(self, tmp_path):
        renames = self.write_renames(tmp_path, [('OLD', 'NEW', '2025-01-03')])
        df = pd.concat([
            make_stock_df(['2025-01-02', '2025-01-03'], [100, 999], ticker='OLD'),  # 03.01 — мусор
            make_stock_df(['2025-01-03'], [102], ticker='NEW'),
        ])

        result = mu.apply_renames(df, renames_file=renames).sort_index()

        assert len(result) == 2                            # строка OLD за 03.01 отброшена
        assert result['close'].tolist() == [100.0, 102.0]

    def test_combine_merges_renames(self, tmp_path, monkeypatch):
        folder = tmp_path / "data"
        folder.mkdir()
        write_stock_parquet(str(folder), 'OLD', make_stock_df(['2025-01-01'], [100], ticker='OLD'))
        write_stock_parquet(str(folder), 'NEW', make_stock_df(['2025-01-03'], [102], ticker='NEW'))
        monkeypatch.setattr(mu, 'RENAMES_FILE',
                            self.write_renames(tmp_path, [('OLD', 'NEW', '2025-01-03')]))

        df = mu.combine_moex_stocks(data_folder=str(folder))
        assert set(df['ticker']) == {'NEW'}
        assert set(df['source_ticker']) == {'OLD', 'NEW'}

        # склейку можно отключить
        df_raw = mu.combine_moex_stocks(data_folder=str(folder), merge_renames=False)
        assert set(df_raw['ticker']) == {'OLD', 'NEW'}

    def test_real_registry_contains_tcsg_to_t(self):
        renames = mu.load_renames()
        row = renames[(renames['old'] == 'TCSG') & (renames['new'] == 'T')]
        assert len(row) == 1


# ---------------------------------------------------------------- key rate

class TestKeyRate:
    def test_missing_file_gives_zero_rate(self, tmp_path):
        dates = pd.to_datetime(['2025-01-31', '2025-02-28'])
        rf = mu.risk_free_monthly(dates, key_rate_file=str(tmp_path / 'nope.csv'))
        assert rf.tolist() == [0.0, 0.0]

    def test_ffill_between_changes_and_bfill_before_start(self, tmp_path):
        path = tmp_path / 'key_rate.csv'
        pd.DataFrame([('2025-02-15', 12.0), ('2025-04-10', 24.0)],
                     columns=['date', 'rate']).to_csv(path, index=False)

        dates = pd.to_datetime(['2025-01-31', '2025-02-28', '2025-03-31', '2025-04-30'])
        rf = mu.risk_free_monthly(dates, key_rate_file=str(path))

        # до первой даты — bfill (12%), между изменениями — ffill,
        # после второго изменения — 24%; всё в месячных долях (/12/100)
        assert rf.tolist() == pytest.approx([0.01, 0.01, 0.01, 0.02])

    def test_real_registry_sane(self):
        kr = mu.load_key_rate()
        assert len(kr) > 50
        assert kr['rate'].between(3, 25).all()
        # известная точка: заморозка 28.02.2022 — ставка 20%
        row = kr[kr['date'] == '2022-02-28']
        assert len(row) == 1 and float(row['rate'].iloc[0]) == 20.0


# ---------------------------------------------------------------- indexes: storage

class TestIndexStorage:
    @pytest.fixture(autouse=True)
    def idx_folder(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mu, 'INDEXES_FOLDER', str(tmp_path))
        return str(tmp_path)

    @staticmethod
    def make_index_df(dates, closes):
        # get_moex_index возвращает индекс из date-объектов — воспроизводим это
        idx = pd.Index([pd.Timestamp(d).date() for d in dates], name='date')
        return pd.DataFrame({'volume': [1e9] * len(idx), 'close': closes}, index=idx)

    def test_save_and_read(self, monkeypatch):
        sample = self.make_index_df(['2025-01-01', '2025-01-02'], [3000.0, 3050.0])
        monkeypatch.setattr(mu, 'get_moex_index',
                            lambda ticker, start=None, end=None, session=None: sample)

        path = mu.save_moex_index('imoex')
        assert path.endswith('IMOEX.parquet')

        df = mu.read_moex_index('IMOEX')
        assert len(df) == 2
        assert isinstance(df.index, pd.DatetimeIndex)   # даты нормализованы
        assert (df['ticker'] == 'IMOEX').all()

    def test_read_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            mu.read_moex_index('NOFILE')

    def test_update_appends_and_dedupes(self, monkeypatch):
        first = self.make_index_df(['2025-01-01', '2025-01-02'], [3000.0, 3050.0])
        monkeypatch.setattr(mu, 'get_moex_index',
                            lambda ticker, start=None, end=None, session=None: first)
        mu.save_moex_index('IMOEX')

        # Обновление перезапрашивает с последней даты: перекрытие по 02.01
        new = self.make_index_df(['2025-01-02', '2025-01-03'], [3055.0, 3100.0])
        monkeypatch.setattr(mu, 'get_moex_index',
                            lambda ticker, start=None, end=None, session=None: new)
        mu.update_moex_index('IMOEX')

        df = mu.read_moex_index('IMOEX')
        assert len(df) == 3
        assert df.loc['2025-01-02', 'close'] == 3055.0  # keep='last'
        assert df.index.is_monotonic_increasing

    def test_update_missing_file_does_initial_save(self, monkeypatch):
        sample = self.make_index_df(['2025-01-01'], [3000.0])
        monkeypatch.setattr(mu, 'get_moex_index',
                            lambda ticker, start=None, end=None, session=None: sample)
        mu.update_moex_index('IMOEX')
        assert len(mu.read_moex_index('IMOEX')) == 1


# ---------------------------------------------------------------- bonds: API

class TestBondsApi:
    def test_get_bonds_list_filters_by_board_path(self):
        # Реальный формат ISS: {'columns': [...], 'data': [...]}
        expected = {'securities': {'columns': ['SECID', 'SHORTNAME'],
                                   'data': [['BOND1', 'Test Bond']]}}

        class FakeSession:
            def get(self, url, params=None):
                # фильтрация по доске должна идти через путь /boards/<board>/
                assert 'boards/TQCB/securities.json' in url
                return DummyResponse(expected)

        df = mu.get_moex_bonds_list(segment='TQCB', session=FakeSession())
        assert df.loc[0, 'SECID'] == 'BOND1'

    def test_get_bonds_list_legacy_formats(self):
        # Совместимость: список списков с шапкой и список словарей
        for payload in (
            {'securities': [['SECID', 'SHORTNAME'], ['BOND1', 'Test Bond']]},
            {'securities': [{'SECID': 'BOND1', 'SHORTNAME': 'Test Bond'}]},
        ):
            class FakeSession:
                def __init__(self, data):
                    self.data = data

                def get(self, url, params=None):
                    return DummyResponse(self.data)

            df = mu.get_moex_bonds_list(session=FakeSession(payload))
            assert df.loc[0, 'SECID'] == 'BOND1'

    def test_get_bond_params(self):
        data = {'securities': [['SECID', 'COUPONPERCENT'], ['BOND1', '10.0']]}

        class FakeSession:
            def get(self, url, params=None):
                assert 'bonds/securities/BOND1.json' in url
                return DummyResponse(data)

        df = mu.get_moex_bond_params('BOND1', session=FakeSession())
        assert df.loc[0, 'SECID'] == 'BOND1'

    def test_get_bond_prices(self):
        data = {
            'history': [
                ['TRADEDATE', 'CLOSE', 'WAPRICE'],
                ['2025-01-01', '101', '100'],
                ['2025-01-02', '102', '101'],
            ]
        }

        class FakeSession:
            def get(self, url, params=None):
                assert 'history/engines/stock/markets/bonds/securities/' in url
                return DummyResponse(data)

        df = mu.get_moex_bond_prices('BOND1', start='2025-01-01', end='2025-01-02',
                                     session=FakeSession())
        assert 'CLOSE' in df.columns
        assert df.index[0] == pd.Timestamp('2025-01-01')
        assert (df['secid'] == 'BOND1').all()

    def test_get_bond_prices_empty_history(self):
        class FakeSession:
            def get(self, url, params=None):
                return DummyResponse({'history': []})

        df = mu.get_moex_bond_prices('BOND1', session=FakeSession())
        assert df.empty

    def test_get_bond_prices_paginated(self):
        """ISS отдаёт history страницами — все страницы должны склеиваться."""
        all_rows = [[f'2025-01-{d:02d}', 100.0 + d] for d in range(1, 6)]  # 5 строк
        page_size = 2

        class FakeSession:
            def __init__(self):
                self.calls = []

            def get(self, url, params=None):
                offset = params['start']
                self.calls.append(offset)
                rows = all_rows[offset:offset + page_size]
                return DummyResponse({
                    'history': {'columns': ['TRADEDATE', 'CLOSE'], 'data': rows},
                    'history.cursor': {'columns': ['INDEX', 'TOTAL', 'PAGESIZE'],
                                       'data': [[offset, len(all_rows), page_size]]},
                })

        session = FakeSession()
        df = mu.get_moex_bond_prices('BOND1', start='2025-01-01', end='2025-01-05',
                                     session=session)

        assert len(df) == 5                       # все страницы, а не первая
        assert session.calls == [0, 2, 4]         # листали по offset
        assert df['CLOSE'].iloc[-1] == 105.0

    def test_http_error_raises(self):
        class FakeSession:
            def get(self, url, params=None):
                return DummyResponse({}, status_code=500)

        with pytest.raises(RuntimeError):
            mu.get_moex_bonds_list(session=FakeSession())


# ---------------------------------------------------------------- bonds: storage

class TestBondsStorage:
    @pytest.fixture(autouse=True)
    def bonds_folder(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mu, 'BONDS_FOLDER', str(tmp_path))
        return str(tmp_path)

    @staticmethod
    def make_bond_df(dates, closes):
        idx = pd.to_datetime(dates)
        return pd.DataFrame({'CLOSE': closes, 'WAPRICE': closes},
                            index=pd.Index(idx, name='TRADEDATE'))

    def test_save_and_read(self, monkeypatch):
        prices = self.make_bond_df(['2025-01-01', '2025-01-02'], [100, 101])
        monkeypatch.setattr(mu, 'get_moex_bond_prices',
                            lambda secid, start, end, session=None: prices)

        mu.save_moex_bond('BOND1', start='2025-01-01', end='2025-01-02')
        df = mu.read_moex_bond('BOND1')
        assert len(df) == 2

    def test_read_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            mu.read_moex_bond('NOFILE')

    def test_update_appends_new_rows(self, bonds_folder, monkeypatch):
        self.make_bond_df(['2025-01-01', '2025-01-02'], [100, 101]).to_parquet(
            os.path.join(bonds_folder, 'BOND1.parquet'))

        new = self.make_bond_df(['2025-01-03', '2025-01-04'], [102, 103])
        calls = {}

        def fake_prices(secid, start, end, session=None):
            calls['start'] = start
            return new

        monkeypatch.setattr(mu, 'get_moex_bond_prices', fake_prices)

        mu.update_moex_bond('BOND1')

        df = mu.read_moex_bond('BOND1')
        assert len(df) == 4
        assert calls['start'] == '2025-01-03'  # запрошено со следующего дня после последней даты

    def test_update_up_to_date_skips_fetch(self, bonds_folder, monkeypatch):
        today = pd.Timestamp.today().normalize()
        self.make_bond_df([today], [100]).to_parquet(
            os.path.join(bonds_folder, 'BOND1.parquet'))

        def fail(*args, **kwargs):
            raise AssertionError('fetch должен быть пропущен')

        monkeypatch.setattr(mu, 'get_moex_bond_prices', fail)
        mu.update_moex_bond('BOND1')  # не должно упасть


# ---------------------------------------------------------------- bonds: universe

class TestBondsUniverse:
    @pytest.fixture(autouse=True)
    def bonds_folder(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mu, 'BONDS_FOLDER', str(tmp_path))
        return str(tmp_path)

    LIST_TQOB = pd.DataFrame({'SECID': ['SU26238RMFS4', 'SU26240RMFS0'],
                              'SHORTNAME': ['ОФЗ 26238', 'ОФЗ 26240'],
                              'COUPONPERCENT': [7.1, 7.0]})
    LIST_TQCB = pd.DataFrame({'SECID': ['RU000A0001'],
                              'SHORTNAME': ['Корп 1'],
                              'COUPONPERCENT': [12.0]})

    def test_save_bonds_params_merges_segments(self, bonds_folder, monkeypatch):
        _lists = {'TQOB': self.LIST_TQOB, 'TQCB': self.LIST_TQCB}
        monkeypatch.setattr(mu, 'get_moex_bonds_list',
                            lambda segment, session=None: _lists[segment].copy())

        mu.save_bonds_params('TQOB')
        mu.save_bonds_params('TQCB')

        params = mu.read_bonds_params()
        assert len(params) == 3
        assert set(params['segment']) == {'TQOB', 'TQCB'}

        # повторный снапшот той же доски заменяет её записи, чужие не трогает
        mu.save_bonds_params('TQOB')
        params2 = mu.read_bonds_params()
        assert len(params2) == 3

    def test_download_bonds_universe(self, bonds_folder, monkeypatch):
        monkeypatch.setattr(mu, 'get_moex_bonds_list',
                            lambda segment, session=None: self.LIST_TQOB.copy())
        saved = []
        monkeypatch.setattr(mu, 'save_moex_bond',
                            lambda secid, start=None, session=None: saved.append(secid))

        n = mu.download_bonds_universe('TQOB', start='2020-01-01')
        assert n == 2
        assert sorted(saved) == ['SU26238RMFS4', 'SU26240RMFS0']
        assert len(mu.read_bonds_params()) == 2

    def test_download_universe_liquidity_and_maturity_filters(self, bonds_folder, monkeypatch):
        _list = pd.DataFrame({
            'SECID': ['BIG', 'SMALL', 'MATURED', 'BIG2'],
            'SHORTNAME': ['Большой', 'Мелкий', 'Погашенный', 'Большой2'],
            'ISSUESIZE': [3e7, 1e5, 3e7, 2e7],        # × FACEVALUE 1000 → руб
            'FACEVALUE': [1000, 1000, 1000, 1000],
            'MATDATE': ['2030-01-01', '2030-01-01', '2020-01-01', '2030-01-01'],
        })
        monkeypatch.setattr(mu, 'get_moex_bonds_list',
                            lambda segment, session=None: _list.copy())
        saved = []
        monkeypatch.setattr(mu, 'save_moex_bond',
                            lambda secid, start=None, session=None: saved.append(secid))

        # мин. объем 10 млрд руб: SMALL отсеян; MATURED погашен; max_issues=1 → BIG
        n = mu.download_bonds_universe('TQCB', min_issue_size=10e9, max_issues=1)
        assert n == 1
        assert saved == ['BIG']
        # реестр параметров при этом полный (вся доска)
        assert len(mu.read_bonds_params()) == 4

    def test_update_all_bonds_skips_params_file(self, bonds_folder, monkeypatch):
        # два выпуска + params.parquet, который не является выпуском
        idx = pd.to_datetime(['2025-01-01'])
        for secid in ('BOND1', 'BOND2'):
            pd.DataFrame({'CLOSE': [100.0]}, index=idx).to_parquet(
                f"{bonds_folder}/{secid}.parquet")
        pd.DataFrame({'SECID': ['BOND1'], 'segment': ['TQOB']}).to_parquet(
            f"{bonds_folder}/params.parquet")
        # мониторинг доски — тоже не выпуск
        pd.DataFrame({'date': idx, 'SECID': ['BOND1']}).to_parquet(
            f"{bonds_folder}/market_TQOB.parquet")

        updated = []
        monkeypatch.setattr(mu, 'update_moex_bond',
                            lambda secid, session=None: updated.append(secid))
        monkeypatch.setattr(mu, 'save_bonds_params',
                            lambda segment, session=None: pd.DataFrame())

        mu.update_all_bonds()
        assert sorted(updated) == ['BOND1', 'BOND2']

    def test_read_bonds_params_missing(self):
        with pytest.raises(FileNotFoundError):
            mu.read_bonds_params()


# ------------------------------------------------- bonds: мониторинг досок по датам

class TestBondsMarket:
    @pytest.fixture(autouse=True)
    def bonds_folder(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mu, 'BONDS_FOLDER', str(tmp_path))
        return str(tmp_path)

    def test_update_fetches_weekdays_and_appends(self, bonds_folder, monkeypatch):
        requested = []

        def fake_fetch(segment, date, session):
            requested.append(pd.Timestamp(date))
            return pd.DataFrame({
                'TRADEDATE': [pd.Timestamp(date).strftime('%Y-%m-%d')] * 2,
                'SECID': ['B1', 'B2'],
                'CLOSE': [100.0, 99.5],
                'YIELDCLOSE': [15.0, 16.0],
            })

        monkeypatch.setattr(mu, '_fetch_bonds_board_date', fake_fetch)

        start = (pd.Timestamp.today().normalize() - pd.Timedelta(days=6)).strftime('%Y-%m-%d')
        n = mu.update_bonds_market('TQOB', start=start)

        assert all(d.weekday() < 5 for d in requested)  # выходные не запрашиваются
        assert n == 2 * len(requested)
        df = mu.read_bonds_market('TQOB')
        assert set(['date', 'SECID', 'CLOSE', 'segment']).issubset(df.columns)
        assert (df['segment'] == 'TQOB').all()
        assert df['date'].max() == max(requested)

        # повторный запуск в тот же день — данные актуальны, запросов нет
        requested.clear()
        assert mu.update_bonds_market('TQOB', start=start) == 0
        assert requested == []

    def test_update_backfills_earlier_history(self, bonds_folder, monkeypatch):
        # уже сохранено [today-10 .. today]; просим start=today-30 → докачка начала
        today = pd.Timestamp.today().normalize()
        emin, emax = today - pd.Timedelta(days=10), today
        pd.DataFrame({'date': [emin, emax], 'SECID': ['B1', 'B1'],
                      'CLOSE': [100.0, 100.0], 'segment': ['TQOB', 'TQOB']}).to_parquet(
            os.path.join(bonds_folder, 'market_TQOB.parquet'))

        requested = []

        def fake_fetch(segment, date, session):
            requested.append(pd.Timestamp(date))
            return pd.DataFrame({'TRADEDATE': [pd.Timestamp(date).strftime('%Y-%m-%d')],
                                 'SECID': ['B2'], 'CLOSE': [99.0]})

        monkeypatch.setattr(mu, '_fetch_bonds_board_date', fake_fetch)
        n = mu.update_bonds_market('TQOB', start=(today - pd.Timedelta(days=30)).strftime('%Y-%m-%d'))

        assert n == len(requested) > 0
        assert all(d < emin for d in requested)  # уже покрытый диапазон не перекачивается
        df = mu.read_bonds_market('TQOB')
        assert df['date'].min() == min(requested)
        assert len(df) == 2 + n

        # малый зазор в начале (праздники) не вызывает вечную докачку
        requested.clear()
        assert mu.update_bonds_market(
            'TQOB', start=(min(requested, default=df['date'].min())
                           - pd.Timedelta(days=3)).strftime('%Y-%m-%d')) == 0
        assert requested == []

    def test_update_stops_on_failure_without_gaps(self, bonds_folder, monkeypatch):
        # сбой на середине хвоста: даты после сбоя не пишутся, чтобы не оставить дыру
        today = pd.Timestamp.today().normalize()
        requested = []

        def fake_fetch(segment, date, session):
            requested.append(pd.Timestamp(date))
            if len(requested) == 3:
                raise ConnectionError('ISS down')
            return pd.DataFrame({'TRADEDATE': [pd.Timestamp(date).strftime('%Y-%m-%d')],
                                 'SECID': ['B1'], 'CLOSE': [100.0]})

        monkeypatch.setattr(mu, '_fetch_bonds_board_date', fake_fetch)
        n = mu.update_bonds_market('TQOB', start=(today - pd.Timedelta(days=14)).strftime('%Y-%m-%d'))

        assert len(requested) == 3 and n == 2
        df = mu.read_bonds_market('TQOB')
        assert df['date'].max() == requested[1]  # следующий запуск продолжит с requested[2]

    def test_backfill_goes_backwards_from_history(self, bonds_folder, monkeypatch):
        today = pd.Timestamp.today().normalize()
        emin = today - pd.Timedelta(days=5)
        pd.DataFrame({'date': [emin, today], 'SECID': ['B1', 'B1'], 'CLOSE': [100.0, 100.0],
                      'segment': ['TQOB', 'TQOB']}).to_parquet(
            os.path.join(bonds_folder, 'market_TQOB.parquet'))
        requested = []

        def fake_fetch(segment, date, session):
            requested.append(pd.Timestamp(date))
            if len(requested) == 4:
                raise ConnectionError('ISS down')
            return pd.DataFrame({'TRADEDATE': [pd.Timestamp(date).strftime('%Y-%m-%d')],
                                 'SECID': ['B1'], 'CLOSE': [99.0]})

        monkeypatch.setattr(mu, '_fetch_bonds_board_date', fake_fetch)
        mu.update_bonds_market('TQOB', start=(today - pd.Timedelta(days=40)).strftime('%Y-%m-%d'))

        head = requested[:3]
        assert head == sorted(head, reverse=True) and head[0] < emin
        # скачанный кусок примыкает к истории: между ним и emin нет пропущенных будней
        assert not [d for d in pd.date_range(head[0] + pd.Timedelta(days=1), emin - pd.Timedelta(days=1))
                    if d.weekday() < 5]

    def test_legacy_file_migrates_to_years(self, bonds_folder):
        legacy = os.path.join(bonds_folder, 'market_TQCB.parquet')
        pd.DataFrame({'date': pd.to_datetime(['2024-12-30', '2025-01-03', '2025-01-03']),
                      'SECID': ['B1', 'B1', 'B2'], 'CLOSE': [100.0, 101.0, 99.0],
                      'segment': ['TQCB'] * 3}).to_parquet(legacy)

        df = mu.read_bonds_market('TQCB')

        assert not os.path.exists(legacy)
        assert sorted(os.listdir(os.path.join(bonds_folder, 'market_TQCB'))) == \
            ['2024.parquet', '2025.parquet']
        assert len(df) == 3 and set(df['SECID']) == {'B1', 'B2'}

    def test_update_rewrites_only_touched_year(self, bonds_folder, monkeypatch):
        today = pd.Timestamp.today().normalize()
        old_year = today.year - 1
        folder = os.path.join(bonds_folder, 'market_TQOB')
        os.makedirs(folder)
        old_path = os.path.join(folder, f'{old_year}.parquet')
        pd.DataFrame({'date': [pd.Timestamp(f'{old_year}-06-03')], 'SECID': ['B1'],
                      'CLOSE': [100.0], 'segment': ['TQOB']}).to_parquet(old_path)
        cur_path = os.path.join(folder, f'{today.year}.parquet')
        last = today - pd.Timedelta(days=7)
        pd.DataFrame({'date': [last], 'SECID': ['B1'], 'CLOSE': [100.0],
                      'segment': ['TQOB']}).to_parquet(cur_path)
        before = os.path.getmtime(old_path)

        def fake_fetch(segment, date, session):
            return pd.DataFrame({'TRADEDATE': [pd.Timestamp(date).strftime('%Y-%m-%d')],
                                 'SECID': ['B1'], 'CLOSE': [101.0]})

        monkeypatch.setattr(mu, '_fetch_bonds_board_date', fake_fetch)
        if last.year != today.year:  # первые дни января: хвост задел бы прошлый год
            pytest.skip('граница года')
        n = mu.update_bonds_market('TQOB', start=f'{old_year}-06-03')

        assert n > 0
        assert os.path.getmtime(old_path) == before  # прошлый год не перезаписан
        assert len(pd.read_parquet(cur_path)) == 1 + n

    def test_read_period_filters_years_and_dates(self, bonds_folder):
        folder = os.path.join(bonds_folder, 'market_TQOB')
        os.makedirs(folder)
        for y in (2023, 2024, 2025):
            pd.DataFrame({'date': pd.to_datetime([f'{y}-03-01', f'{y}-09-01']),
                          'SECID': ['B1', 'B1'], 'segment': ['TQOB', 'TQOB']}).to_parquet(
                os.path.join(folder, f'{y}.parquet'))
        df = mu.read_bonds_market('TQOB', start='2024-06-01', end='2025-06-01')
        assert list(df['date'].dt.strftime('%Y-%m-%d')) == ['2024-09-01', '2025-03-01']
        assert len(mu.read_bonds_market()) == 6  # без аргументов — вся история всех досок

    def test_fetch_board_date_paginates(self, monkeypatch):
        cols = ['TRADEDATE', 'SECID', 'CLOSE', 'BOARDID']

        class _Resp:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                pass

            def json(self):
                return self._payload

        class _Session:
            def get(self, url, params=None):
                offset = int(params.get('start', 0))
                rows = {0: [['2025-06-02', 'B1', 100.0, 'TQOB'],
                            ['2025-06-02', 'B2', 99.0, 'TQOB']],
                        2: [['2025-06-02', 'B3', 98.0, 'TQOB']]}.get(offset, [])
                return _Resp({
                    'history': {'columns': cols, 'data': rows},
                    'history.cursor': {'columns': ['INDEX', 'TOTAL', 'PAGESIZE'],
                                       'data': [[offset, 3, 2]]},
                })

        df = mu._fetch_bonds_board_date('TQOB', '2025-06-02', _Session())
        assert sorted(df['SECID']) == ['B1', 'B2', 'B3']
        assert 'BOARDID' not in df.columns  # сохраняется только рабочий набор колонок

    def test_read_all_segments_concat(self, bonds_folder):
        for seg in ('TQOB', 'TQCB'):
            pd.DataFrame({'date': pd.to_datetime(['2025-06-02']),
                          'SECID': [f'{seg}-BOND'], 'segment': [seg]}).to_parquet(
                os.path.join(bonds_folder, f"market_{seg}.parquet"))
        df = mu.read_bonds_market()
        assert len(df) == 2
        assert set(df['segment']) == {'TQOB', 'TQCB'}

    def test_read_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            mu.read_bonds_market()
        with pytest.raises(FileNotFoundError):
            mu.read_bonds_market('TQOB')

    def test_update_all_discovers_segments(self, bonds_folder, monkeypatch):
        for seg in ('TQOB', 'TQCB'):
            pd.DataFrame({'date': pd.to_datetime(['2025-06-02']),
                          'SECID': ['X'], 'segment': [seg]}).to_parquet(
                os.path.join(bonds_folder, f"market_{seg}.parquet"))
        called = []
        monkeypatch.setattr(mu, 'update_bonds_market',
                            lambda seg, session=None: called.append(seg))
        repaired = []
        monkeypatch.setattr(mu, 'repair_bonds_market',
                            lambda seg, session=None: repaired.append(seg))
        mu.update_bonds_market_all()
        assert sorted(called) == ['TQCB', 'TQOB']
        assert sorted(repaired) == ['TQCB', 'TQOB']

    def test_repair_fills_inner_gaps(self, bonds_folder, monkeypatch):
        saved = pd.to_datetime(['2025-06-02', '2025-06-05', '2025-06-06'])
        pd.DataFrame({'date': saved, 'SECID': ['B1'] * 3, 'CLOSE': [100.0] * 3,
                      'segment': ['TQOB'] * 3}).to_parquet(
            os.path.join(bonds_folder, 'market_TQOB.parquet'))
        requested = []

        def fake_fetch(segment, date, session):
            requested.append(pd.Timestamp(date))
            return pd.DataFrame({'TRADEDATE': [pd.Timestamp(date).strftime('%Y-%m-%d')],
                                 'SECID': ['B1'], 'CLOSE': [101.0]})

        monkeypatch.setattr(mu, '_fetch_bonds_board_date', fake_fetch)
        # календарь шире истории: даты вне [min, max] и выходные не докачиваются
        calendar = pd.to_datetime(['2025-05-30', '2025-06-02', '2025-06-03', '2025-06-04',
                                   '2025-06-05', '2025-06-06', '2025-06-07', '2025-06-09'])
        assert mu.repair_bonds_market('TQOB', calendar=calendar) == 2
        assert requested == list(pd.to_datetime(['2025-06-03', '2025-06-04']))
        assert len(mu.read_bonds_market('TQOB')) == 5
        requested.clear()
        assert mu.repair_bonds_market('TQOB', calendar=calendar) == 0 and requested == []


# ---------------------------------------------------------------- bond metrics

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

    def test_add_bond_metrics(self):
        dates = pd.date_range('2025-01-01', periods=3, freq='D')
        df = pd.DataFrame({'CLOSE': [102, 103, 104]}, index=dates)
        params = pd.Series({'FACEVALUE': 1000, 'COUPONPERCENT': 10, 'MATDATE': '2026-01-01'})

        result = mu.add_bond_metrics(df, params)

        assert {'ytm', 'duration', 'convexity', 'years_to_maturity'} <= set(result.columns)
        assert len(result) == 3
        # срок до погашения убывает с каждым днем
        assert result['years_to_maturity'].is_monotonic_decreasing

    def test_add_bond_metrics_missing_matdate(self):
        dates = pd.date_range('2025-01-01', periods=2, freq='D')
        df = pd.DataFrame({'CLOSE': [100, 100]}, index=dates)
        params = pd.Series({'FACEVALUE': 1000, 'COUPONPERCENT': 10})  # без MATDATE

        result = mu.add_bond_metrics(df, params)  # не должно падать
        assert result['ytm'].isna().all()
        assert result['duration'].isna().all()

    def test_add_bond_metrics_waprice_fallback(self):
        dates = pd.date_range('2025-01-01', periods=2, freq='D')
        df = pd.DataFrame({'WAPRICE': [100, 100]}, index=dates)
        params = pd.Series({'FACEVALUE': 1000, 'COUPONPERCENT': 10, 'MATDATE': '2026-01-01'})

        result = mu.add_bond_metrics(df, params)
        assert 'ytm' in result.columns


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

    def test_atomic_write_leaves_no_tmp_on_failure(self, tmp_path):
        path = str(tmp_path / 'x.parquet')

        class _Bad(pd.DataFrame):
            def to_parquet(self, *a, **k):
                open(a[0], 'w').close()
                raise OSError('disk full')

        with pytest.raises(OSError):
            mu._atomic_to_parquet(_Bad({'a': [1]}), path)
        assert os.listdir(tmp_path) == []

    def test_local_tickers(self, tmp_data_folder):
        write_stock_parquet(tmp_data_folder, 'AAA', make_stock_df(['2025-01-01'], [1.0], 'AAA'))
        os.makedirs(os.path.join(tmp_data_folder, 'EMPTY'))
        assert mu._local_tickers() == ['AAA']


class TestKeyRateUpdate:
    HTML = """<table><tr><th>Дата</th><th>Ставка</th></tr>
    <tr><td>23.03.2026</td><td>15,00</td></tr>
    <tr><td>20.03.2026</td><td>15,50</td></tr>
    <tr><td>16.02.2026</td><td>15,50</td></tr>
    <tr><td>13.02.2026</td><td>16,00</td></tr></table>"""

    def _session(self, html):
        class _Resp:
            text = html

            def raise_for_status(self):
                pass

        class _Session:
            def get(self, url, **kwargs):
                return _Resp()

        return _Session()

    def test_appends_only_changes(self, tmp_path):
        f = tmp_path / 'key_rate.csv'
        f.write_text('date,rate\n2025-12-22,16.00\n', encoding='utf-8')
        assert mu.update_key_rate(str(f), session=self._session(self.HTML)) == 2
        kr = mu.load_key_rate(str(f))
        assert list(kr['date'].dt.strftime('%Y-%m-%d')) == ['2025-12-22', '2026-02-16', '2026-03-23']
        assert list(kr['rate']) == [16.0, 15.5, 15.0]
        # повторный запуск — дубликатов нет
        assert mu.update_key_rate(str(f), session=self._session(self.HTML)) == 0
        assert len(mu.load_key_rate(str(f))) == 3


# ---------------------------------------------------------------- delisted, quality report

class TestDelisted:
    def test_update_all_skips_delisted(self, tmp_data_folder, tmp_path, monkeypatch):
        for ticker in ('LIVE', 'GONE'):
            write_stock_parquet(tmp_data_folder, ticker, make_stock_df(['2025-01-01'], [1], ticker))
        reg = tmp_path / 'delisted.csv'
        reg.write_text('ticker,last_date,note\nGONE,2025-01-01,test\n', encoding='utf-8')
        monkeypatch.setattr(mu, 'DELISTED_FILE', str(reg))
        updated = []
        monkeypatch.setattr(mu, 'update_moex_stock', lambda t, **k: updated.append(t))

        mu.update_all_stocks()
        assert updated == ['LIVE']
        updated.clear()
        mu.update_all_stocks(include_delisted=True)
        assert sorted(updated) == ['GONE', 'LIVE']

    def test_load_delisted_missing_file(self, tmp_path):
        assert mu.load_delisted(str(tmp_path / 'nope.csv')).empty

    def test_iss_is_traded(self):
        class _Resp:
            def __init__(self, rows):
                self._rows = rows

            def raise_for_status(self):
                pass

            def json(self):
                return {'boards': {'columns': ['boardid', 'market', 'is_traded'], 'data': self._rows}}

        class _Session:
            def __init__(self, rows):
                self.rows = rows

            def get(self, url, **kw):
                return _Resp(self.rows)

        assert mu.iss_is_traded('X', session=_Session([['TQBR', 'shares', 1]])) is True
        # торгуется только на внебиржевой доске другого рынка — не считается
        assert mu.iss_is_traded('X', session=_Session([['TQBR', 'shares', 0],
                                                      ['XXXX', 'ndm', 1]])) is False
        assert mu.iss_is_traded('X', session=_Session([])) is None


class TestAdjCloseEdgeCases:
    @staticmethod
    def write_dividends(folder, ticker, rows):
        pd.DataFrame(rows, columns=['closing_date', 'dividend_value']).to_csv(
            os.path.join(str(folder), f"{ticker}.csv"), index=False)

    def test_future_record_date_ignored(self, tmp_path):
        """Объявленный дивиденд с отсечкой после последней даты данных не корректирует историю."""
        df = make_stock_df(['2025-01-01', '2025-01-02', '2025-01-03'], [100, 100, 100])
        self.write_dividends(tmp_path, 'TEST', [('2025-01-20', 10.0)])
        result = mu.calculate_adj_close(df, div_folder=str(tmp_path))
        assert result['adj_close'].tolist() == [100.0, 100.0, 100.0]

    def test_skipped_dividends_reported(self, tmp_path):
        df = make_stock_df(['2025-01-01', '2025-01-02', '2025-01-03'], [100, 100, 100])
        self.write_dividends(tmp_path, 'TEST', [('2025-01-02', 80.0)])  # 80% — неправдоподобно
        result = mu.calculate_adj_close(df, div_folder=str(tmp_path))
        assert result.attrs['skipped_dividends'] == [(pd.Timestamp('2025-01-02'), 80.0)]
        assert result['adj_close'].tolist() == [100.0, 100.0, 100.0]


def _gap_df(ticker='TEST'):
    """Бумага с гэпом открытия -10% 2025-01-06 (понедельник)."""
    dates = pd.to_datetime(['2025-01-02', '2025-01-03', '2025-01-06', '2025-01-07'])
    df = pd.DataFrame({'open': [100.0, 100.0, 90.0, 90.0], 'close': [100.0, 100.0, 90.0, 90.0],
                       'volume': 1.0, 'ticker': ticker}, index=pd.Index(dates, name='date'))
    return df


class TestDividendGapCandidates:
    @pytest.fixture(autouse=True)
    def no_splits(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mu, 'SPLITS_FILE', str(tmp_path / 'no_splits.csv'))
        monkeypatch.setattr(mu, 'EXTERNAL_SPLITS_FILE', str(tmp_path / 'no_ext.json'))

    def test_unexplained_gap_is_candidate(self, tmp_path):
        flat = pd.Series(0.0, index=_gap_df().index)
        res = mu.find_dividend_gap_candidates(_gap_df(), str(tmp_path), market_returns=flat)
        assert list(res['date']) == [pd.Timestamp('2025-01-06')]
        assert res['gap'].iloc[0] == pytest.approx(-0.10)

    def test_gap_explained_by_dividend(self, tmp_path):
        pd.DataFrame({'closing_date': ['2025-01-06'], 'dividend_value': [10.0]}).to_csv(
            tmp_path / 'TEST.csv', index=False)
        flat = pd.Series(0.0, index=_gap_df().index)
        assert mu.find_dividend_gap_candidates(_gap_df(), str(tmp_path), market_returns=flat).empty

    def test_gap_explained_by_market(self, tmp_path):
        market = pd.Series([0.0, 0.0, -0.09, 0.0], index=_gap_df().index)
        assert mu.find_dividend_gap_candidates(_gap_df(), str(tmp_path), market_returns=market).empty


class TestQualityReport:
    @pytest.fixture
    def env(self, tmp_path, monkeypatch):
        data = tmp_path / 'data'
        data.mkdir()
        idx = tmp_path / 'indexes'
        idx.mkdir()
        bonds = tmp_path / 'bonds'
        bonds.mkdir()
        divs = tmp_path / 'divs'
        divs.mkdir()
        for name, val in (('DATA_FOLDER', data), ('INDEXES_FOLDER', idx), ('BONDS_FOLDER', bonds)):
            monkeypatch.setattr(mu, name, str(val))
        monkeypatch.setattr(mu, 'DELISTED_FILE', str(tmp_path / 'delisted.csv'))
        monkeypatch.setattr(mu, 'FUTURES_FOLDER', str(tmp_path / 'futures'))
        monkeypatch.setattr(mu, 'SPLITS_FILE', str(tmp_path / 'no_splits.csv'))
        monkeypatch.setattr(mu, 'EXTERNAL_SPLITS_FILE', str(tmp_path / 'no_ext.json'))
        cal = pd.bdate_range('2025-01-01', periods=60)
        pd.DataFrame({'close': 1000.0, 'ticker': 'IMOEX'},
                     index=pd.Index(cal, name='date')).to_parquet(idx / 'IMOEX.parquet')
        return {'data': str(data), 'cal': cal, 'divs': str(divs), 'bonds': str(bonds)}

    @staticmethod
    def stock(folder, ticker, dates, closes, adj=None):
        df = pd.DataFrame({'open': closes, 'close': closes, 'volume': 1.0, 'ticker': ticker},
                          index=pd.Index(pd.DatetimeIndex(dates), name='date'))
        df['adj_close'] = adj if adj is not None else df['close']
        df['market_cap'] = df['close'] * 10
        write_stock_parquet(folder, ticker, df)

    def test_clean_data_has_no_issues(self, env):
        self.stock(env['data'], 'OK', env['cal'], [100.0] * 60)
        issues = mu.data_quality_report(days=30, div_folder=env['divs'])
        assert issues.empty
        assert mu.quality_summary(issues) == 'Проверка данных: замечаний нет'

    def test_detects_stale_gaps_and_adj_artefact(self, env):
        cal = env['cal']
        self.stock(env['data'], 'LAG', cal[:-3], [100.0] * 57)            # отстает на 3 дня
        self.stock(env['data'], 'OLD', cal[:30], [100.0] * 30)            # 30 дней без данных
        self.stock(env['data'], 'HOLE', cal.delete([50, 51]), [100.0] * 58)
        adj = [100.0] * 60
        adj[55] = 150.0                                                    # скачок только в adj_close
        self.stock(env['data'], 'ADJ', cal, [100.0] * 60, adj=adj)

        issues = mu.data_quality_report(days=30, div_folder=env['divs'])
        got = {(r.check, r.object) for r in issues.itertuples()}
        assert ('stock_stale', 'LAG') in got
        assert ('stock_stale', 'OLD') in got and 'delisted.csv' in issues.loc[
            issues['object'] == 'OLD', 'detail'].iloc[0]
        assert ('stock_gaps', 'HOLE') in got
        assert ('adj_jump', 'ADJ') in got
        assert 'замечаний' in mu.quality_summary(issues)

    def test_delisted_and_real_moves_not_reported(self, env, tmp_path):
        cal = env['cal']
        self.stock(env['data'], 'GONE', cal[:10], [100.0] * 10)
        (tmp_path / 'delisted.csv').write_text('ticker,last_date,note\nGONE,2025-01-14,x\n',
                                               encoding='utf-8')
        closes = [100.0] * 50 + [150.0] * 10                              # реальный рост +50%, без разворота
        self.stock(env['data'], 'MOVE', cal, closes)
        assert mu.data_quality_report(days=30, div_folder=env['divs']).empty

    def test_price_spike_detected(self, env):
        closes = [100.0] * 60
        closes[55] = 200.0                                                 # +100% и обратно
        self.stock(env['data'], 'SPIKE', env['cal'], closes)
        issues = mu.data_quality_report(days=30, div_folder=env['divs'])
        assert list(issues['check']) == ['price_spike']

    def test_bonds_stale(self, env):
        self.stock(env['data'], 'OK', env['cal'], [100.0] * 60)
        folder = os.path.join(env['bonds'], 'market_TQOB')
        os.makedirs(folder)
        pd.DataFrame({'date': env['cal'][:-2], 'SECID': 'B1', 'segment': 'TQOB'}).to_parquet(
            os.path.join(folder, '2025.parquet'))
        issues = mu.data_quality_report(days=30, div_folder=env['divs'])
        assert list(issues['check']) == ['bonds_stale']


# ---------------------------------------------------------------- history store, ALL bonds, futures

class _PagedSession:
    """Фейковая ISS-сессия: history по страницам из rows, курсор с TOTAL."""

    def __init__(self, columns, rows, page=2):
        self.columns, self.rows, self.page = columns, rows, page
        self.urls = []

    def get(self, url, params=None):
        self.urls.append(url)
        start = int(params.get('start', 0))
        chunk = self.rows[start:start + self.page]

        class _R:
            def raise_for_status(_):
                pass

            def json(_):
                return {'history': {'columns': self.columns, 'data': chunk},
                        'history.cursor': {'columns': ['INDEX', 'TOTAL', 'PAGESIZE'],
                                           'data': [[start, len(self.rows), self.page]]}}
        return _R()


class TestHistoryStore:
    def test_normalize_iss_frame(self):
        df = pd.DataFrame({'num': [None, '1.5', 2], 'txt': ['a', None, 'b'], 'mix': ['1', 'x', None]})
        out = mu._normalize_iss_frame(df)
        assert out['num'].dtype == 'float64'
        assert str(out['txt'].dtype) == 'string' and str(out['mix'].dtype) == 'string'

    def test_all_segment_uses_market_url_and_keeps_boards(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mu, 'BONDS_FOLDER', str(tmp_path))
        cols = ['BOARDID', 'TRADEDATE', 'SECID', 'CLOSE', 'ZSPREAD']
        rows = [['TQCB', '2025-06-02', 'B1', 100.0, 50.0],
                ['PSOB', '2025-06-02', 'B1', 99.0, None],      # тот же выпуск на другой доске
                ['TQOB', '2025-06-02', 'B2', 98.0, 10.0]]
        sess = _PagedSession(cols, rows)
        df = mu._fetch_market_rows('ALL', '2025-06-02', sess)
        assert sess.urls[0].endswith('/markets/bonds/securities.json')  # весь рынок, не доска
        assert 'ZSPREAD' in df.columns and len(df) == 3
        mu._merge_market_rows('ALL', df)
        stored = mu.read_bonds_market('ALL')
        assert len(stored) == 3                     # ключ включает BOARDID — дубли по SECID не схлопнуты
        assert mu._market_segments() == ['ALL']

    def test_read_without_segment_skips_all(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mu, 'BONDS_FOLDER', str(tmp_path))
        for seg in ('TQOB', 'ALL'):
            mu._merge_market_rows(seg, pd.DataFrame({'date': pd.to_datetime(['2025-06-02']),
                                                     'SECID': ['B1'], 'segment': [seg]}))
        assert set(mu.read_bonds_market()['segment']) == {'TQOB'}
        assert set(mu.read_bonds_market('ALL')['segment']) == {'ALL'}

    def test_lock_blocks_concurrent_update(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mu, 'BONDS_FOLDER', str(tmp_path))
        folder = mu._market_dir('TQOB')
        calls = []
        monkeypatch.setattr(mu, '_fetch_bonds_board_date', lambda *a: calls.append(1) or pd.DataFrame())
        with mu._store_lock(folder):
            assert mu.update_bonds_market('TQOB', start='2025-06-02') == 0
        assert calls == []                          # занято — ничего не запрашивали
        assert not os.path.exists(os.path.join(folder, '.lock'))

    def test_stale_lock_is_taken_over(self, tmp_path):
        folder = str(tmp_path / 'store')
        os.makedirs(folder)
        lock = os.path.join(folder, '.lock')
        open(lock, 'w').close()
        import time
        old = time.time() - 13 * 3600
        os.utime(lock, (old, old))
        with mu._store_lock(folder):
            pass
        assert not os.path.exists(lock)

    def test_flush_keeps_progress_on_failure(self, tmp_path):
        folder = str(tmp_path / 'store')
        n = {'i': 0}

        def fetch(d):
            n['i'] += 1
            if n['i'] == 5:
                raise ConnectionError('down')
            return pd.DataFrame({'date': [d], 'SECID': ['X']})

        added = mu._update_store(folder, fetch, start='2025-01-01', max_days=10,
                                 label='test', flush_every=2)
        assert added == 4 and len(mu._store_dates(folder)) == 4

    def test_repair_remembers_empty_dates(self, tmp_path):
        folder = str(tmp_path / 'store')
        mu._store_merge(folder, pd.DataFrame({'date': pd.to_datetime(['2025-06-02', '2025-06-05']),
                                              'SECID': ['X', 'X']}))
        asked = []

        def fetch(d):
            asked.append(d)
            return pd.DataFrame()                   # ISS пуст за эти даты

        cal = pd.bdate_range('2025-06-02', '2025-06-05')
        assert mu._repair_store(folder, fetch, cal, 'test') == 0
        assert len(asked) == 2
        asked.clear()
        assert mu._repair_store(folder, fetch, cal, 'test') == 0
        assert asked == []                          # подтвержденно пустые больше не запрашиваются


class TestFutures:
    def test_update_and_read_futures(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mu, 'FUTURES_FOLDER', str(tmp_path))
        cols = ['BOARDID', 'TRADEDATE', 'SECID', 'SHORTNAME', 'ASSETCODE', 'SETTLEPRICE', 'OPENPOSITION']

        def fake_pages(url, date, session, max_pages=1000):
            d = pd.Timestamp(date).strftime('%Y-%m-%d')
            return pd.DataFrame([['RFUD', d, 'SiZ5', 'Si-12.25', 'Si', 80000.0, 100.0],
                                 ['RFUD', d, 'BRX5', 'BR-11.25', 'BR', 65.0, 50.0]], columns=cols)

        monkeypatch.setattr(mu, '_iss_history_pages', fake_pages)
        start = (pd.Timestamp.today().normalize() - pd.Timedelta(days=6)).strftime('%Y-%m-%d')
        n = mu.update_futures_history(start=start)
        assert n > 0
        si = mu.read_futures_history(assets='Si')
        assert set(si['SECID']) == {'SiZ5'} and 'TRADEDATE' not in si.columns
        assert si['date'].dtype.kind == 'M'

    def test_read_futures_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mu, 'FUTURES_FOLDER', str(tmp_path))
        with pytest.raises(FileNotFoundError):
            mu.read_futures_history()


class TestBondsSecurities:
    def test_registry_adds_only_new_secids(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mu, 'BONDS_FOLDER', str(tmp_path))
        mu._merge_market_rows('ALL', pd.DataFrame({
            'date': pd.to_datetime(['2005-06-01', '2005-06-01', '2026-09-25']),
            'SECID': ['OLD1', 'OLD2', 'NEW1'], 'BOARDID': ['EQOB', 'EQNB', 'TQCB']}))
        asked = []

        def fake_desc(secid, session=None):
            asked.append(secid)
            return {'SECID': secid, 'MATDATE': '2005-10-21', 'ISSUESIZE': '3000000',
                    'HASDEFAULT': '0'}

        monkeypatch.setattr(mu, 'get_security_description', fake_desc)
        assert mu.update_bonds_securities(max_new=2) == 2
        assert mu.update_bonds_securities() == 1             # остаток — в следующий прогон
        reg = mu.read_bonds_securities()
        assert sorted(reg['SECID']) == ['NEW1', 'OLD1', 'OLD2']
        assert reg['ISSUESIZE'].dtype == 'float64'
        asked.clear()
        assert mu.update_bonds_securities() == 0 and asked == []

    def test_registry_without_full_history_is_noop(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mu, 'BONDS_FOLDER', str(tmp_path))
        assert mu.update_bonds_securities() == 0


class TestAtomicWriteRetry:
    def test_retries_replace_while_file_is_busy(self, tmp_path, monkeypatch):
        path = str(tmp_path / 'x.parquet')
        real_replace = os.replace
        calls = {'n': 0}

        def flaky_replace(src, dst):
            calls['n'] += 1
            if calls['n'] < 3:
                raise PermissionError('busy')
            return real_replace(src, dst)

        monkeypatch.setattr(mu.os, 'replace', flaky_replace)
        monkeypatch.setattr(mu.time, 'sleep', lambda s: None)
        mu._atomic_to_parquet(pd.DataFrame({'a': [1]}), path)
        assert calls['n'] == 3 and os.path.exists(path)
        assert not os.path.exists(path + '.tmp')


class TestDataRoot:
    def test_env_var_moves_data_folders(self, tmp_path, monkeypatch):
        import importlib
        monkeypatch.setenv('MOEX_DATA_ROOT', str(tmp_path))
        try:
            m = importlib.reload(mu)
            assert m.DATA_ROOT == str(tmp_path)
            for folder, name in ((m.DATA_FOLDER, 'data'), (m.INDEXES_FOLDER, 'indexes'),
                                 (m.BONDS_FOLDER, 'bonds'), (m.FUTURES_FOLDER, 'futures')):
                assert folder == os.path.join(str(tmp_path), name)
            # реестры остаются в проекте
            assert m.SPLITS_FILE.startswith(m.BASE_DIR)
        finally:
            monkeypatch.delenv('MOEX_DATA_ROOT')
            importlib.reload(mu)
