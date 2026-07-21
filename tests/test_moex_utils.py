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
    def test_parses_candles(self, monkeypatch):
        candles = [
            {'begin': '2025-01-01 00:00:00', 'open': 99.0, 'close': 100.5,
             'high': 101.0, 'low': 98.0, 'value': 1000.0, 'volume': 10},
            {'begin': '2025-01-02 00:00:00', 'open': 100.5, 'close': 101.0,
             'high': 102.0, 'low': 100.0, 'value': 2000.0, 'volume': 20},
        ]
        monkeypatch.setattr(mu.apimoex, 'get_market_candles',
                            lambda session, security, start, end, interval: candles)

        df = mu.get_moex_stock('SBER', start='2025-01-01', end='2025-01-02')

        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index[0] == pd.Timestamp('2025-01-01')
        assert 'value_rub' in df.columns          # value переименован
        assert df['volume'].dtype == 'float64'    # int приведен к float
        assert (df['ticker'] == 'SBER').all()
        assert df.loc['2025-01-02', 'close'] == 101.0

    def test_start_after_end_raises(self):
        with pytest.raises(ValueError):
            mu.get_moex_stock('SBER', start='2025-02-01', end='2025-01-01')

    def test_invalid_date_raises(self):
        with pytest.raises(ValueError):
            mu.get_moex_stock('SBER', start='2025-13-45')

    def test_empty_response_raises(self, monkeypatch):
        monkeypatch.setattr(mu.apimoex, 'get_market_candles',
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

        def fake_update(ticker, session=None, calculate_market_cap_flag=True):
            updated.append(ticker)
            sessions.append(session)
            mc_flags.append(calculate_market_cap_flag)

        monkeypatch.setattr(mu, 'update_moex_stock', fake_update)

        mu.update_all_stocks()
        assert sorted(updated) == ['AAA', 'BBB']
        # одна общая HTTP-сессия на весь прогон
        assert sessions[0] is not None
        assert all(s is sessions[0] for s in sessions)
        assert mc_flags == [True, True]        # дефолт сохранен

        mu.update_all_stocks(calculate_market_cap_flag=False)
        assert mc_flags[-2:] == [False, False]  # флаг доходит до каждого тикера


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

        # Дивиденд 10 при цене 100 → фактор 0.9 ко всем датам до экс-даты включительно
        assert result['adj_close'].tolist() == pytest.approx([90.0, 90.0, 90.0, 100.0])

    def test_two_dividends_compound(self, tmp_path):
        df = make_stock_df(['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04'],
                           [100, 100, 100, 100])
        self.write_dividends(tmp_path, 'TEST', [('2025-01-02', 10.0), ('2025-01-03', 10.0)])

        result = mu.calculate_adj_close(df, div_folder=str(tmp_path))

        # Поздний дивиденд: [90, 90, 90, 100]; ранний: фактор 0.9 к первым двум → [81, 81, 90, 100]
        assert result['adj_close'].tolist() == pytest.approx([81.0, 81.0, 90.0, 100.0])

    def test_dividend_before_data_ignored(self, tmp_path):
        df = make_stock_df(['2025-01-10', '2025-01-11'], [100, 100])
        self.write_dividends(tmp_path, 'TEST', [('2025-01-01', 10.0)])

        result = mu.calculate_adj_close(df, div_folder=str(tmp_path))
        assert result['adj_close'].tolist() == [100.0, 100.0]

    def test_no_dividend_file(self, tmp_path):
        df = make_stock_df(['2025-01-01'], [100])
        result = mu.calculate_adj_close(df, div_folder=str(tmp_path))
        assert (result['adj_close'] == result['close']).all()

    def test_add_adj_close_to_all_stocks(self, tmp_data_folder, tmp_path):
        write_stock_parquet(tmp_data_folder, 'TEST',
                            make_stock_df(['2025-01-01', '2025-01-02'], [100, 100]))
        div_folder = tmp_path / "divs"
        div_folder.mkdir()
        self.write_dividends(div_folder, 'TEST', [('2025-01-02', 10.0)])

        mu.add_adj_close_to_all_stocks(str(div_folder))

        df = pd.read_parquet(os.path.join(tmp_data_folder, 'TEST', 'TEST.parquet'))
        assert df['adj_close'].tolist() == pytest.approx([90.0, 90.0])


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

        result = mu.adjust_for_splits(df, splits_file=splits)

        assert result['close'].tolist() == pytest.approx([100.0, 101.0, 101.0, 102.0])
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

    def test_add_bond_metrics(self):
        dates = pd.date_range('2025-01-01', periods=3, freq='D')
        df = pd.DataFrame({'CLOSE': [102, 103, 104]}, index=dates)
        params = pd.Series({'FACEVALUE': 1000, 'COUPONPERCENT': 10, 'MATDATE': '2026-01-01'})

        result = mu.add_bond_metrics(df, params)

        assert {'ytm', 'duration', 'years_to_maturity'} <= set(result.columns)
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
