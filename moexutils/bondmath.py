"""
Математика облигаций по упрощенной модели (равные купоны, погашение по номиналу):
доходность к погашению, дюрация Маколея, выпуклость.
"""


def calculate_ytm(price: float, face_value: float, coupon_rate: float, years_to_maturity: float, coupon_freq: int = 2) -> float:
    """
    Calculates Yield to Maturity (YTM) for a bond.

    Parameters:
    price (float): Current price (% of face value).
    face_value (float): Face value.
    coupon_rate (float): Annual coupon rate (%).
    years_to_maturity (float): Years to maturity.
    coupon_freq (int): Coupons per year.

    Returns:
    float: YTM (%).
    """
    coupon = face_value * (coupon_rate / 100) / coupon_freq
    periods = int(years_to_maturity * coupon_freq)

    if periods == 0:
        return 0

    target = price / 100 * face_value

    def _pv(annual_rate: float) -> float:
        r = annual_rate / coupon_freq
        pv_coupons = sum(coupon / (1 + r) ** i for i in range(1, periods + 1))
        return pv_coupons + face_value / (1 + r) ** periods

    # Бисекция: PV монотонно убывает по ставке, ищем ставку в [-50%, 500%]
    lo, hi = -0.5, 5.0
    if target >= _pv(lo):
        return lo * 100
    if target <= _pv(hi):
        return hi * 100

    for _ in range(200):
        mid = (lo + hi) / 2
        if _pv(mid) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break

    return (lo + hi) / 2 * 100

def calculate_duration(price: float, face_value: float, coupon_rate: float, years_to_maturity: float, ytm: float, coupon_freq: int = 2) -> float:
    """
    Calculates modified duration.

    Returns:
    float: Duration in years.
    """
    coupon = face_value * (coupon_rate / 100) / coupon_freq
    periods = int(years_to_maturity * coupon_freq)
    ytm_period = ytm / 100 / coupon_freq

    if periods == 0:
        return 0

    pv_coupons = sum((i * coupon) / (1 + ytm_period)**i for i in range(1, periods+1))
    pv_face = (periods * face_value) / (1 + ytm_period)**periods
    total_pv_weighted = pv_coupons + pv_face

    total_pv = sum(coupon / (1 + ytm_period)**i for i in range(1, periods+1)) + face_value / (1 + ytm_period)**periods

    macaulay_duration = total_pv_weighted / total_pv / coupon_freq
    modified_duration = macaulay_duration / (1 + ytm_period)

    return modified_duration

def calculate_convexity(price: float, face_value: float, coupon_rate: float,
                        years_to_maturity: float, ytm: float, coupon_freq: int = 2) -> float:
    """
    Модифицированная выпуклость облигации (в годах²).

    Вторая производная цены по ставке: dP/P ≈ -D·dy + 0.5·C·dy².
    Параметр price не используется (PV восстанавливается из ytm) —
    сигнатура симметрична calculate_duration.
    """
    coupon = face_value * (coupon_rate / 100) / coupon_freq
    periods = int(years_to_maturity * coupon_freq)
    y = ytm / 100 / coupon_freq

    if periods == 0:
        return 0.0

    cash_flows = [(i, coupon + (face_value if i == periods else 0.0))
                  for i in range(1, periods + 1)]
    pv = sum(cf / (1 + y) ** i for i, cf in cash_flows)
    if pv <= 0:
        return float('nan')
    weighted = sum(cf * i * (i + 1) / (1 + y) ** i for i, cf in cash_flows)
    return weighted / (pv * (coupon_freq ** 2) * (1 + y) ** 2)
