"""
Тесты пакета: математика облигаций, HTTP-сессия, корень данных и реестров,
отсутствие pandas.
Логика акций, индексов и качества — в test_stocks.py / test_quality.py,
облигаций и фьючерсов — в test_history.py, хранилища — в test_lake.py.
"""
import os

import pytest

from moexutils import lake
from moexutils import bondmath, iss
from moexutils import stocks


class TestBondMetrics:
    def test_ytm_par_bond_equals_coupon(self):
        # Облигация по номиналу: YTM = купонной ставке
        ytm = bondmath.calculate_ytm(price=100, face_value=1000, coupon_rate=10,
                               years_to_maturity=1, coupon_freq=2)
        assert ytm == pytest.approx(10.0, abs=0.1)

    def test_ytm_premium_bond_below_coupon(self):
        ytm = bondmath.calculate_ytm(price=105, face_value=1000, coupon_rate=10,
                               years_to_maturity=1, coupon_freq=2)
        assert 0 < ytm < 10

    def test_ytm_discount_bond_above_coupon(self):
        ytm = bondmath.calculate_ytm(price=95, face_value=1000, coupon_rate=10,
                               years_to_maturity=1, coupon_freq=2)
        assert ytm > 10

    def test_ytm_zero_periods(self):
        assert bondmath.calculate_ytm(price=100, face_value=1000, coupon_rate=10,
                                years_to_maturity=0) == 0

    def test_ytm_independent_of_face_value_scale(self):
        """Регрессия: старый солвер сходился только при номинале ~1000."""
        for face in (1, 100, 1000, 100000):
            ytm = bondmath.calculate_ytm(price=100, face_value=face, coupon_rate=10,
                                   years_to_maturity=1, coupon_freq=2)
            assert ytm == pytest.approx(10.0, abs=1e-4), f"face_value={face}"

    def test_ytm_long_maturity_converges(self):
        ytm = bondmath.calculate_ytm(price=100, face_value=1000, coupon_rate=8,
                               years_to_maturity=30, coupon_freq=2)
        assert ytm == pytest.approx(8.0, abs=1e-4)

    def test_duration_zero_coupon_bond(self):
        # Для бескупонной облигации Маколей = сроку, модифицированная = срок / (1 + ytm/freq)
        duration = bondmath.calculate_duration(price=90, face_value=1000, coupon_rate=0,
                                         years_to_maturity=1, ytm=10, coupon_freq=2)
        assert duration == pytest.approx(1 / 1.05, abs=1e-6)

    def test_duration_below_maturity_for_coupon_bond(self):
        duration = bondmath.calculate_duration(price=100, face_value=1000, coupon_rate=10,
                                         years_to_maturity=5, ytm=10, coupon_freq=2)
        assert 0 < duration < 5

    def test_convexity_zero_coupon(self):
        # Бескупонная: единственный CF в периоде n → C = n(n+1)/(f²(1+y_p)²)
        n, f, ytm = 2, 2, 10.0
        y_p = ytm / 100 / f
        expected = n * (n + 1) / (f ** 2 * (1 + y_p) ** 2)
        convexity = bondmath.calculate_convexity(price=90, face_value=1000, coupon_rate=0,
                                           years_to_maturity=1, ytm=ytm, coupon_freq=f)
        assert convexity == pytest.approx(expected, rel=1e-9)

    def test_convexity_properties(self):
        # Положительна и растет со сроком
        c5 = bondmath.calculate_convexity(100, 1000, 10, 5, 10)
        c10 = bondmath.calculate_convexity(100, 1000, 10, 10, 10)
        assert 0 < c5 < c10
        assert bondmath.calculate_convexity(100, 1000, 10, 0, 10) == 0.0

# ---------------------------------------------------------------- infrastructure


class TestInfrastructure:
    def test_session_has_default_timeout(self, monkeypatch):
        seen = {}

        def fake_request(self, method, url, **kwargs):
            seen.update(kwargs)
            return None

        monkeypatch.setattr(iss.requests.Session, 'request', fake_request)
        session = iss.make_session()
        session.get('https://iss.moex.com/iss/x.json')
        assert seen['timeout'] == iss.ISS_TIMEOUT
        session.get('https://iss.moex.com/iss/x.json', timeout=5)
        assert seen['timeout'] == 5  # явный таймаут не перекрывается


class TestDataRoot:
    def test_env_var_moves_data_root(self, tmp_path, monkeypatch):
        import importlib
        monkeypatch.setenv('MOEX_DATA_ROOT', str(tmp_path))
        try:
            m = importlib.reload(lake)
            assert m.DATA_ROOT == str(tmp_path)
            assert m.LAKE_DATA_PATH == os.path.join(str(tmp_path), 'lake')
            assert stocks.SPLITS_FILE.startswith(stocks.BASE_DIR)       # реестры остаются в проекте
        finally:
            monkeypatch.delenv('MOEX_DATA_ROOT')
            importlib.reload(lake)


class TestPackage:
    def test_project_root_holds_registries(self):
        # пакет лежит в moexutils/, реестры metadata/ — в корне проекта уровнем выше
        assert os.path.isdir(os.path.join(stocks.BASE_DIR, 'metadata'))
        assert stocks.BASE_DIR == lake.BASE_DIR == os.path.dirname(os.path.dirname(stocks.__file__))

    def test_no_pandas_in_project_modules(self):
        import subprocess
        import sys
        code = ('import sys, update_data; '
                'from moexutils import backup, bondmath, history, iss, lake, notify, quality, stocks; '
                'print("pandas" in sys.modules)')
        out = subprocess.run([sys.executable, '-c', code], cwd=lake.BASE_DIR,
                             capture_output=True, text=True, check=True).stdout
        assert out.strip() == 'False'
