import pandas as pd
import numpy as np

import moex_utils as moex  # ваш пакет

# ----------------------------
# Конфигурация
# ----------------------------
CSV_PATH = "combined_stocks.csv"

DATE_COL = "date"
TICKER_COL = "ticker"
PRICE_COL = "adj_close"
LIQ_COL_PREF = ["value_rub", "value", "turnover", "volume"]

# Издержки в bps: 10 bps = 0.10%
TC_BPS = 15

# Ликвидность: брать top N по обороту за месяц (если есть value_rub)
TOP_N_LIQ = 50

# Бенчмарк IMOEX Total Return
BENCH_TICKER = "MCFTR"
BENCH_START = "2000-01-01"

# Делистинг и пропуски цен:
# mode="exit" означает: если по акции нет цены в следующем месяце, считаем,
# что позиция закрыта по последней цене (доходность по этой бумаге за месяц = 0),
# и в следующем месяце держать ее нельзя.
# mode="penalize" означает: если цена пропала, доходность = -100% (жестко и консервативно).
MISSING_PRICE_MODE = "exit"  # "exit" или "penalize"

# ----------------------------
# Вспомогательные функции
# ----------------------------
def find_liq_col(df: pd.DataFrame):
    for c in LIQ_COL_PREF:
        if c in df.columns:
            return c
    return None

def safe_cagr(ret: pd.Series, freq=12):
    ret = ret.dropna()
    if ret.empty:
        return np.nan
    eq = (1 + ret).cumprod()
    if (eq <= 0).any():
        return np.nan
    return float(np.exp(np.log(eq.iloc[-1]) * (freq / len(ret))) - 1)

def max_drawdown(eq: pd.Series) -> float:
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min())

def perf_stats(ret: pd.Series, bench_ret: pd.Series | None = None, freq=12) -> dict:
    ret = ret.dropna()
    if ret.empty:
        return {}

    eq = (1 + ret).cumprod()
    cagr = safe_cagr(ret, freq=freq)
    vol = ret.std(ddof=0) * np.sqrt(freq)
    sharpe = np.nan if vol == 0 else (ret.mean() * freq) / vol
    mdd = max_drawdown(eq)

    out = {
        "CAGR": float(cagr) if np.isfinite(cagr) else np.nan,
        "Vol": float(vol),
        "Sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "MaxDD": float(mdd),
        "Months": int(len(ret)),
    }

    if bench_ret is not None:
        aligned = pd.concat([ret, bench_ret], axis=1).dropna()
        if not aligned.empty:
            a = aligned.iloc[:, 0] - aligned.iloc[:, 1]
            te = a.std(ddof=0) * np.sqrt(freq)
            ir = np.nan if te == 0 else (a.mean() * freq) / te
            out.update({
                "IR": float(ir) if np.isfinite(ir) else np.nan,
                "TE": float(te),
                "ActiveMeanMonthly": float(a.mean())
            })

    return out

# ----------------------------
# Загрузка и подготовка данных по акциям
# ----------------------------
def load_stocks(path: str) -> tuple[pd.DataFrame, str | None]:
    df = pd.read_csv(path)
    print("Stocks columns:", df.columns.tolist())
    print("Stocks rows:", len(df))

    missing = [c for c in [DATE_COL, TICKER_COL, PRICE_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"В CSV не хватает колонок: {missing}. Есть: {df.columns.tolist()}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    bad_dates = int(df[DATE_COL].isna().sum())
    if bad_dates > 0:
        raise ValueError(f"Не удалось распарсить {bad_dates} дат в колонке '{DATE_COL}'.")

    liq_col = find_liq_col(df)

    keep = [DATE_COL, TICKER_COL, PRICE_COL] + ([liq_col] if liq_col else [])
    df = df[keep].copy()

    # Чистка: цены должны быть > 0
    df[PRICE_COL] = pd.to_numeric(df[PRICE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, TICKER_COL, PRICE_COL])
    df = df[df[PRICE_COL] > 0]

    # Дубли по (date, ticker)
    df = df.sort_values([TICKER_COL, DATE_COL])
    df = df.drop_duplicates(subset=[DATE_COL, TICKER_COL], keep="last")

    print("Tickers:", df[TICKER_COL].nunique())
    print("Date range:", df[DATE_COL].min(), "to", df[DATE_COL].max())
    print("Liquidity column:", liq_col)

    return df, liq_col

def to_monthly_panels(df: pd.DataFrame, liq_col: str | None):
    df = df.copy()
    df["month"] = df[DATE_COL].dt.to_period("M").dt.to_timestamp("M")

    # Месячная цена: последняя в месяце
    px_m = (
        df.sort_values([TICKER_COL, DATE_COL])
          .groupby([TICKER_COL, "month"])[PRICE_COL]
          .last()
          .unstack(TICKER_COL)
          .sort_index()
    )
    px_m = px_m.where(px_m > 0)

    # Месячная ликвидность: сумма за месяц
    liq_m = None
    if liq_col:
        liq_m = (
            df.groupby([TICKER_COL, "month"])[liq_col]
              .sum()
              .unstack(TICKER_COL)
              .sort_index()
        )

    # Доходности без fill_method='pad'
    ret_m = px_m.pct_change(fill_method=None)

    # Диагностика аномалий доходностей
    bad_count = int((ret_m <= -1).sum().sum())
    min_ret = float(np.nanmin(ret_m.values)) if np.isfinite(np.nanmin(ret_m.values)) else np.nan
    print("Monthly returns <= -100%:", bad_count)
    print("Min monthly return:", min_ret)

    return px_m, ret_m, liq_m

# ----------------------------
# Бенчмарк MCFTR (IMOEX TR)
# ----------------------------
def load_benchmark_monthly(ticker: str, start: str) -> pd.Series:
    bench_df = moex.get_moex_index(ticker, start=start)
    if bench_df is None or len(bench_df) == 0:
        raise ValueError("moex.get_moex_index вернул пустой датафрейм.")

    bench_df = bench_df.copy()

    # 1) Найти дату: колонка или индекс
    if "date" in bench_df.columns:
        dcol = "date"
    elif "TRADEDATE" in bench_df.columns:
        dcol = "TRADEDATE"
    else:
        # возможно, дата в индексе
        idx_name = bench_df.index.name
        bench_df = bench_df.reset_index()
        if "date" in bench_df.columns:
            dcol = "date"
        elif "TRADEDATE" in bench_df.columns:
            dcol = "TRADEDATE"
        elif idx_name and idx_name in bench_df.columns:
            dcol = idx_name
        else:
            raise ValueError(f"Не нашел дату ни в колонках, ни в индексе. Колонки: {bench_df.columns.tolist()}")

    # 2) Принудительно привести к datetime
    bench_df[dcol] = pd.to_datetime(bench_df[dcol], errors="coerce")

    bad = int(bench_df[dcol].isna().sum())
    if bad > 0:
        # это нормально, но лучше видеть сколько строк потеряли
        bench_df = bench_df.dropna(subset=[dcol])
        print(f"[BENCH] Dropped {bad} rows with unparsed dates in '{dcol}'.")

    # 3) Найти колонку цены
    pcol = None
    for cand in ["adj_close", "close", "value", "index_value"]:
        if cand in bench_df.columns:
            pcol = cand
            break
    if pcol is None:
        raise ValueError(f"Не нашел колонку цены в данных индекса. Колонки: {bench_df.columns.tolist()}")

    bench_df[pcol] = pd.to_numeric(bench_df[pcol], errors="coerce")
    bench_df = bench_df.dropna(subset=[pcol])
    bench_df = bench_df[bench_df[pcol] > 0].copy()

    # 4) Месячная серия (конец месяца)
    bench_df["month"] = bench_df[dcol].dt.to_period("M").dt.to_timestamp("M")
    bench_px_m = bench_df.sort_values(dcol).groupby("month")[pcol].last().sort_index()

    bench_ret_m = bench_px_m.pct_change(fill_method=None)
    return bench_ret_m.rename("bench_ret")

def to_monthly_panels(df: pd.DataFrame, liq_col: str | None):
    df = df.copy()

    # страховка: вдруг дата после merge/прочих операций стала object
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL])

    df["month"] = df[DATE_COL].dt.to_period("M").dt.to_timestamp("M")

    px_m = (
        df.sort_values([TICKER_COL, DATE_COL])
          .groupby([TICKER_COL, "month"])[PRICE_COL]
          .last()
          .unstack(TICKER_COL)
          .sort_index()
    )
    px_m = px_m.where(px_m > 0)

    liq_m = None
    if liq_col:
        liq_m = (
            df.groupby([TICKER_COL, "month"])[liq_col]
              .sum()
              .unstack(TICKER_COL)
              .sort_index()
        )

    ret_m = px_m.pct_change(fill_method=None)

    bad_count = int((ret_m <= -1).sum().sum())
    min_ret = float(np.nanmin(ret_m.values)) if np.isfinite(np.nanmin(ret_m.values)) else np.nan
    print("Monthly returns <= -100%:", bad_count)
    print("Min monthly return:", min_ret)

    return px_m, ret_m, liq_m

# ----------------------------
# Моментум сигнал и портфель
# ----------------------------
def momentum_signal(px_m: pd.DataFrame, lookback: int, skip: int) -> pd.DataFrame:
    # P(t-skip) / P(t-lookback-skip) - 1
    p1 = px_m.shift(skip)
    p0 = px_m.shift(lookback + skip)
    return p1 / p0 - 1.0

def build_rebal_weights(sig: pd.DataFrame,
                        liq_m: pd.DataFrame | None,
                        q: float,
                        long_short: bool,
                        top_n_liq: int | None) -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=sig.index, columns=sig.columns)

    for t in sig.index:
        s = sig.loc[t].dropna()
        if s.empty:
            continue

        if liq_m is not None and top_n_liq is not None:
            liq = liq_m.loc[t].dropna()
            if not liq.empty:
                liquid = liq.sort_values(ascending=False).head(top_n_liq).index
                s = s.loc[s.index.intersection(liquid)]
                if s.empty:
                    continue

        s = s.sort_values()
        n = len(s)
        k = max(1, int(np.floor(n * q)))

        if long_short:
            losers = s.index[:k]
            winners = s.index[-k:]
            w.loc[t, winners] = 1.0 / len(winners)
            w.loc[t, losers] = -1.0 / len(losers)
        else:
            winners = s.index[-k:]
            w.loc[t, winners] = 1.0 / len(winners)

    return w

def apply_holding(w_rebal: pd.DataFrame, hold: int) -> pd.DataFrame:
    if hold <= 1:
        return w_rebal
    w = pd.DataFrame(0.0, index=w_rebal.index, columns=w_rebal.columns)
    for i in range(hold):
        w = w.add(w_rebal.shift(i), fill_value=0.0)
    return w / hold

def backtest_monthly(ret_m: pd.DataFrame, w_m: pd.DataFrame, tc_bps: float, missing_mode: str) -> pd.DataFrame:
    w = w_m.fillna(0.0).copy()
    r = ret_m.copy()

    # Обработка пропусков цен в следующем месяце для удерживаемых позиций
    # Доходность по активу на месяце t применяется к весам t-1.
    # Значит пропуск в r[t, asset] влияет на портфель на t, если w[t-1, asset] != 0.
    w_prev = w.shift(1).fillna(0.0)

    if missing_mode not in ["exit", "penalize"]:
        raise ValueError("missing_mode должен быть 'exit' или 'penalize'.")

    if missing_mode == "exit":
        # где доходность NaN и позиция была, ставим 0 доходность
        r_eff = r.copy()
        mask = r_eff.isna() & (w_prev != 0)
        if mask.values.any():
            r_eff = r_eff.mask(mask, 0.0)
        # остальное NaN можно оставить, позже заполним 0, так как веса там 0
        r_eff = r_eff.fillna(0.0)
    else:
        # penalize: пропуск цены при открытой позиции = -100%
        r_eff = r.copy()
        mask = r_eff.isna() & (w_prev != 0)
        if mask.values.any():
            r_eff = r_eff.mask(mask, -1.0)
        r_eff = r_eff.fillna(0.0)

    # Дополнительная защита: если в данных есть ret <= -100%, считаем это невалидным
    # и зануляем вклад (exit) или штрафуем (penalize) только если позиция была.
    # Это не "улучшение" результата, а защита от артефактов данных.
    bad = (r_eff <= -1) & (w_prev != 0)
    if bad.values.any():
        if missing_mode == "exit":
            r_eff = r_eff.mask(bad, 0.0)
        else:
            r_eff = r_eff.mask(bad, -1.0)

    gross = (w_prev * r_eff).sum(axis=1)

    # turnover и издержки
    dw = (w - w.shift(1).fillna(0.0)).abs().sum(axis=1)
    turnover = 0.5 * dw
    tc = (tc_bps / 10000.0) * turnover
    net = gross - tc

    return pd.DataFrame({
        "ret_gross": gross,
        "ret_net": net,
        "turnover": turnover,
        "tc": tc
    })

# ----------------------------
# Подбор параметров (walk-forward)
# ----------------------------
def walk_forward_select(px_m, ret_m, liq_m, bench_ret_m: pd.Series,
                        grid: list[dict],
                        train_frac=0.7,
                        objective="Sharpe",
                        top_n_liq=50,
                        tc_bps=15,
                        missing_mode="exit") -> tuple[pd.DataFrame, dict]:

    # Синхронизируем по месяцам: только пересечение акций и бенчмарка
    common_idx = ret_m.index.intersection(bench_ret_m.index)
    ret_m = ret_m.loc[common_idx]
    px_m = px_m.loc[common_idx]
    if liq_m is not None:
        liq_m = liq_m.loc[common_idx]
    bench = bench_ret_m.loc[common_idx]

    idx = common_idx
    if len(idx) < 60:
        raise ValueError(f"Слишком мало месяцев пересечения с бенчмарком: {len(idx)}")

    split = int(np.floor(len(idx) * train_frac))
    train_idx = idx[:split]
    test_idx = idx[split:]

    rows = []
    best = None
    best_score = -np.inf

    for p in grid:
        sig = momentum_signal(px_m, p["lookback"], p["skip"])
        w0 = build_rebal_weights(sig, liq_m, p["q"], p["long_short"], top_n_liq=top_n_liq)
        w = apply_holding(w0, p["hold"])

        bt = backtest_monthly(ret_m, w, tc_bps=tc_bps, missing_mode=missing_mode)

        tr = bt.loc[train_idx, "ret_net"]
        te = bt.loc[test_idx, "ret_net"]

        st_tr = perf_stats(tr, bench.loc[train_idx])
        st_te = perf_stats(te, bench.loc[test_idx])

        score = st_tr.get("Sharpe") if objective == "Sharpe" else st_tr.get("IR")
        if score is None or not np.isfinite(score):
            score = -np.inf

        row = {
            **p,
            "train_Sharpe": st_tr.get("Sharpe"),
            "train_CAGR": st_tr.get("CAGR"),
            "train_MaxDD": st_tr.get("MaxDD"),
            "train_IR": st_tr.get("IR"),
            "test_Sharpe": st_te.get("Sharpe"),
            "test_CAGR": st_te.get("CAGR"),
            "test_MaxDD": st_te.get("MaxDD"),
            "test_IR": st_te.get("IR"),
            "avg_turnover": float(bt["turnover"].mean()),
        }
        rows.append(row)

        if score > best_score:
            best_score = score
            best = {
                "params": p,
                "bt": bt,
                "weights": w,
                "bench": bench,
                "train_stats": st_tr,
                "test_stats": st_te
            }

    res = pd.DataFrame(rows).sort_values(["train_Sharpe", "train_CAGR"], ascending=False)
    return res, best

# ----------------------------
# Main
# ----------------------------
def main():
    # 1) Акции
    stocks_df, liq_col = load_stocks(CSV_PATH)
    px_m, ret_m, liq_m = to_monthly_panels(stocks_df, liq_col)

    # 2) Бенчмарк IMOEX TR
    bench_ret_m = load_benchmark_monthly(BENCH_TICKER, start=BENCH_START)
    print("Benchmark months:", len(bench_ret_m), "range:", bench_ret_m.index.min(), "to", bench_ret_m.index.max())

    # 3) Сетка параметров (умеренная)
    grid = []
    for lookback in [3, 6, 9, 12]:
        for skip in [0, 1]:
            for hold in [1, 3, 6]:
                for q in [0.1, 0.2]:
                    for long_short in [False, True]:
                        grid.append({
                            "lookback": lookback,
                            "skip": skip,
                            "hold": hold,
                            "q": q,
                            "long_short": long_short
                        })

    # 4) Подбор
    results, best = walk_forward_select(
        px_m, ret_m, liq_m, bench_ret_m,
        grid=grid,
        train_frac=0.7,
        objective="Sharpe",   # можно "IR"
        top_n_liq=TOP_N_LIQ if liq_m is not None else None,
        tc_bps=TC_BPS,
        missing_mode=MISSING_PRICE_MODE
    )

    print("\nTop 10 strategies (sorted by train Sharpe, net, benchmark MCFTR):")
    print(results.head(10).to_string(index=False))

    print("\nBest params:", best["params"])
    print("Train stats vs MCFTR:", best["train_stats"])
    print("Test stats vs MCFTR:", best["test_stats"])

    # 5) Дополнительная диагностика: активность портфеля
    w = best["weights"].fillna(0.0)
    active_months = float((w.abs().sum(axis=1) > 0).mean())
    avg_names = float((w.abs() > 0).sum(axis=1).mean())
    print("\nPortfolio diagnostics:")
    print("Active months share:", active_months)
    print("Average number of positions:", avg_names)

if __name__ == "__main__":
    main()
