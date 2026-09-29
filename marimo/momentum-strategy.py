"""
marimo notebook: Моментум-стратегия на акциях MOEX — бэктест с издержками

Использование:
1. pip install marimo plotly
2. marimo edit momentum-strategy.py

Методика:
- Сигнал: momentum = P(t-skip) / P(t-lookback-skip) - 1 (skip-месяц отсекает
  краткосрочный разворот)
- Ребалансировка ежемесячная, равные веса в top-q квантиле победителей
  (опционально long-short: шорт проигравших)
- Фильтр ликвидности: top-N бумаг по обороту за месяц
- Издержки: tc_bps × turnover; делистинги: 'exit' (выход по нулевой
  доходности) или 'penalize' (-100%)
- Данные: adj_close (полная доходность, сплиты и склейка переименований
  учтены); бенчмарк — MCFTR из локального кэша (фоллбэк IMOEX)
- Walk-forward: подбор параметров на train, честная оценка на test
"""

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium", app_title="Моментум-стратегия", css_file="styles.css")


@app.cell(hide_code=True)
def _():
    # stocks.py лежит в корне проекта (родительская папка от marimo/)
    import sys as _sys
    import os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    import stocks
    import polars as pl
    import numpy as np
    import marimo as mo
    try:
        import plotly.graph_objects as go
        plotly_available = True
    except ImportError:
        go = None
        plotly_available = False
    return go, mo, np, pl, plotly_available, stocks


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Моментум-стратегия на российских акциях

    Классическая кросс-секционная стратегия (Jegadeesh & Titman, 1993): покупаем
    бумаги, которые росли сильнее остальных последние `lookback` месяцев,
    пропуская последний `skip`-месяц (краткосрочный разворот). Ребалансировка
    ежемесячная, издержки учитываются через оборот портфеля.

    Ниже два режима:

    1. **Одиночная стратегия** — задайте параметры и смотрите результат сразу.
    2. **Walk-forward подбор** — grid search по сетке параметров: выбор лучшей
       на train-периоде и честная проверка на test (за кнопкой — расчет долгий).
    """)
    return


@app.cell(hide_code=True)
def _(mo, np, pl, stocks):
    # Данные: месячные панели цен (полная доходность) и ликвидности.
    # Панели — numpy-матрицы «месяцы × тикеры»: строки — months (конец
    # календарного месяца), столбцы — tickers (по алфавиту), нет данных — NaN
    _c = stocks.read_stocks(split_adjusted=True)
    if 'adj_close' not in _c.columns:
        _c = _c.with_columns(adj_close=pl.col('close'))
    _c = (_c.with_columns(pl.col('adj_close', 'close', 'value_rub').fill_nan(None))
          .filter(pl.col('adj_close').is_not_null() & (pl.col('adj_close') > 0))
          .with_columns(month=pl.col('date').dt.month_end())
          .sort('ticker', 'date'))
    tickers = sorted(_c['ticker'].unique().to_list())
    months = _c['month'].unique().sort().to_list()

    def _panel(_df, _key, _keys, _val):
        """Длинная таблица -> матрица (_keys × tickers), пропуски — NaN"""
        _w = pl.DataFrame({_key: _keys}).join(
            _df.pivot(on='ticker', index=_key, values=_val, aggregate_function='last'),
            on=_key, how='left')
        return _w.select([pl.col(_t).cast(pl.Float64) if _t in _w.columns
                          else pl.lit(None, pl.Float64).alias(_t)
                          for _t in tickers]).to_numpy()

    def _lag(_a, _k):
        """Сдвиг матрицы на _k строк вниз (как shift), сверху NaN"""
        _out = np.full(_a.shape, np.nan)
        if _k < len(_a):
            _out[_k:] = _a[:len(_a) - _k]
        return _out

    # Месячные агрегаты по бумаге: последняя цена месяца; оборот — сумма
    # за торговые дни бумаги в месяце (пропуск оборота в дне = 0)
    _m = (_c.group_by('ticker', 'month', maintain_order=True)
          .agg(pl.col('adj_close').last(),
               pl.col('close').drop_nulls().last(),
               pl.col('value_rub').fill_null(0.0).sum()))
    px_m = _panel(_m, 'month', months, 'adj_close')
    px_m[~(px_m > 0)] = np.nan
    liq_m = _panel(_m, 'month', months, 'value_rub')
    ret_m = px_m / _lag(px_m, 1) - 1

    # Волатильность бумаг для inverse-vol взвешивания: дневные доходности,
    # окно 63 торговых дня (квартал), аннуализация, срез на конец месяца.
    # Ex-ante: оценка на конец месяца t применяется к портфелю месяца t+1.
    # Доходности — по общей сетке торговых дней рынка (пропуск дня у бумаги
    # дает пропуск доходности), в окне нужно ≥ 21 наблюдения
    _px_d = pl.DataFrame({'date': _c['date'].unique().sort()}).join(
        _c.pivot(on='ticker', index='date', values='adj_close', aggregate_function='last'),
        on='date', how='left')
    vol_m = (_px_d.select(
                pl.col('date').dt.month_end().alias('month'),
                *[((pl.col(_t) / pl.col(_t).shift(1) - 1)
                   .rolling_std(63, min_samples=21) * np.sqrt(252)).alias(_t)
                  for _t in tickers])
             .group_by('month', maintain_order=True)
             .agg(pl.all().drop_nulls().last())
             .select(tickers).to_numpy())

    # Дивидендная доходность за 12 мес (фактор для композитного сигнала):
    # разница полной (adj_close) и ценовой (close) 12-месячной доходности.
    # Обе серии в единой пост-сплитовой базе — никаких проблем с базой
    # дивидендов (рестейт ВТБ и т.п.) этот способ не имеет.
    _px_close_m = _panel(_m, 'month', months, 'close')
    div12_m = (px_m / _lag(px_m, 12) - 1) - (_px_close_m / _lag(_px_close_m, 12) - 1)

    # Фактическая последняя дата дневных данных: месячная панель лейблится
    # концом периода, но последний месяц может быть незавершенным
    last_data_date = _c['date'].max()
    _partial = months[-1] > last_data_date
    _bad = int(np.sum(ret_m <= -1))
    data_status = mo.md(
        f"**Данные:** {px_m.shape[1]} тикеров · {px_m.shape[0]} месяцев "
        f"({months[0]:%Y-%m} — {months[-1]:%Y-%m}) · "
        f"цены — `adj_close` (дивиденды + сплиты) · "
        f"месячных доходностей ≤ -100%: {_bad}"
        + (f" · ⏳ последний месяц не завершен (данные до {last_data_date:%d.%m.%Y})"
           if _partial else "")
    )
    data_status
    return div12_m, last_data_date, liq_m, months, px_m, ret_m, tickers, vol_m


@app.cell(hide_code=True)
def _(mo, pl, stocks):
    # Бенчмарк: MCFTR (полная доходность) из кэша, фоллбэк IMOEX.
    # Месячный ряд — DataFrame (date — конец месяца, bench_ret)
    def _monthly_bench(_name):
        try:
            _pm = (stocks.read_index(_name)
                   .select('date', pl.col('close').cast(pl.Float64).fill_nan(None))
                   .sort('date')
                   .group_by(pl.col('date').dt.month_end(), maintain_order=True)
                   .agg(pl.col('close').drop_nulls().last()))
            # drop_nulls: первый месяц доходности — пропуск, он ломал бы ребейз графика
            return (_pm.select('date', bench_ret=pl.col('close') / pl.col('close').shift(1) - 1)
                    .drop_nulls())
        except Exception:
            return pl.DataFrame(schema={'date': pl.Date, 'bench_ret': pl.Float64})

    bench_ret_m = _monthly_bench('MCFTR')
    bench_name = 'MCFTR'
    if len(bench_ret_m) < 12:
        bench_ret_m = _monthly_bench('IMOEX')
        bench_name = 'IMOEX (ценовой — кэш MCFTR не найден)'

    # Трендовые сигналы по дневному IMOEX (для trend filter):
    # ex-ante — состояние на конец месяца t управляет портфелем месяца t+1
    # (DataFrame: date — конец месяца, ewmac, ma200 — булевы)
    try:
        _imx = stocks.read_index('IMOEX').sort('date')
        if _imx.is_empty():
            raise FileNotFoundError('IMOEX')
        _cl = pl.col('close').cast(pl.Float64)
        _e16 = _cl.ewm_mean(span=16, adjust=False)
        _e64 = _cl.ewm_mean(span=64, adjust=False)
        _ma200 = _cl.rolling_mean(200, min_samples=100)
        trend_signals = (_imx
                         .select('date', ewmac=(_e16 > _e64).fill_null(False),
                                 ma200=(_cl > _ma200).fill_null(False))
                         .group_by(pl.col('date').dt.month_end(), maintain_order=True)
                         .agg(pl.col('ewmac', 'ma200').last()))
    except Exception:
        trend_signals = None

    bench_status = mo.md(
        f"**Бенчмарк:** {bench_name} · {len(bench_ret_m)} месяцев"
        if len(bench_ret_m) else
        "**Бенчмарк недоступен** — выполните `python update_data.py` (шаг 1b)"
    )
    bench_status
    return bench_name, bench_ret_m, trend_signals


@app.cell(hide_code=True)
def _(np, pl):
    # Ядро бэктеста. Панели — numpy-матрицы «месяцы × тикеры» (NaN — нет
    # данных), результаты бэктеста — polars DataFrame с колонкой date
    def _lag(a, k):
        """Сдвиг на k строк вниз (как shift), сверху NaN"""
        out = np.full(a.shape, np.nan)
        if k < len(a):
            out[k:] = a[:len(a) - k]
        return out

    def _rowsum(a):
        """Сумма по строке с последовательным сложением столбцов"""
        return np.asfortranarray(a).sum(axis=1)

    def nargsort_desc(v):
        """Индексы сортировки по убыванию, NaN в конце. Порядок ничьих —
        quicksort по развернутому ряду (как в прежней версии ноутбука)"""
        v = np.asarray(v, dtype=float)
        _nan = np.isnan(v)
        _idx = np.flatnonzero(~_nan)[::-1]
        _order = _idx[v[_idx].argsort(kind='quicksort')][::-1]
        return np.concatenate([_order, np.flatnonzero(_nan)]).astype(int)

    def _rank_pct(a):
        """Кросс-секционный процентильный ранг по строке: средний ранг при
        ничьих / число наблюдений; NaN остаются NaN"""
        out = np.full(a.shape, np.nan)
        for i in range(a.shape[0]):
            m = ~np.isnan(a[i])
            n = int(m.sum())
            if n == 0:
                continue
            _, inv, cnt = np.unique(a[i, m], return_inverse=True, return_counts=True)
            start = np.cumsum(cnt) - cnt
            out[i, m] = (start + (cnt + 1) / 2.0)[inv] / n
        return out

    def rolling_std(x, window, min_periods, ddof=1):
        """Скользящее std с пропуском NaN (≥ min_periods наблюдений в окне);
        окно из одинаковых значений — ровно 0"""
        x = np.asarray(x, dtype=float)
        out = np.full(len(x), np.nan)
        for i in range(len(x)):
            _v = x[max(0, i - window + 1):i + 1]
            _v = _v[~np.isnan(_v)]
            if len(_v) >= max(min_periods, ddof + 1):
                out[i] = 0.0 if np.all(_v == _v[0]) else float(np.std(_v, ddof=ddof))
        return out

    def momentum_signal(px, lookback, skip):
        """P(t-skip) / P(t-lookback-skip) - 1"""
        return _lag(px, skip) / _lag(px, lookback + skip) - 1.0

    def composite_signal(mom_sig, vol_panel, div_panel,
                         w_mom=1.0, w_lowvol=0.5, w_div=0.5):
        """Композитный скор: взвешенная сумма кросс-секционных процентильных
        рангов momentum, low-vol (ниже волатильность — выше ранг) и
        12-месячной дивидендной доходности. Недостающие факторы бумаги
        исключаются из знаменателя; momentum обязателен.
        Панели факторов выровнены с mom_sig по строкам и столбцам."""
        r_mom = _rank_pct(mom_sig)
        r_lowvol = _rank_pct(-vol_panel)
        r_div = _rank_pct(div_panel)

        num = (w_mom * np.nan_to_num(r_mom, nan=0.0)
               + w_lowvol * np.nan_to_num(r_lowvol, nan=0.0)
               + w_div * np.nan_to_num(r_div, nan=0.0))
        den = (w_mom * ~np.isnan(r_mom) + w_lowvol * ~np.isnan(r_lowvol)
               + w_div * ~np.isnan(r_div))
        with np.errstate(divide='ignore', invalid='ignore'):
            score = num / np.where(den == 0, np.nan, den)
        return np.where(np.isnan(r_mom), np.nan, score)

    def build_rebal_weights(sig, liq, q, long_short, top_n_liq,
                            weighting='equal', vol=None):
        """weighting='equal' — равные веса внутри квантиля;
        'invvol' — вес ∝ 1/σ бумаги (нормировка до 1 внутри лонгов/шортов).
        sig, liq, vol — матрицы, выровненные по строкам и столбцам."""

        def _leg_weights(t, names, sign):
            if weighting == 'invvol' and vol is not None:
                with np.errstate(divide='ignore'):
                    _iv = 1.0 / vol[t, names]
                _ok = np.isfinite(_iv)
                if _ok.any():
                    return names[_ok], sign * _iv[_ok] / _iv[_ok].sum()
            return names, np.full(len(names), sign / len(names))

        w = np.zeros(sig.shape)
        for t in range(sig.shape[0]):
            s = np.flatnonzero(~np.isnan(sig[t]))
            if s.size == 0:
                continue
            if liq is not None and top_n_liq:
                _l = np.flatnonzero(~np.isnan(liq[t]))
                if _l.size:
                    _liquid = _l[nargsort_desc(liq[t, _l])[:top_n_liq]]
                    s = s[np.isin(s, _liquid)]
                    if s.size == 0:
                        continue
            s = s[np.argsort(sig[t, s], kind='quicksort')]
            k = max(1, int(np.floor(len(s) * q)))
            _names, _vals = _leg_weights(t, s[-k:], 1.0)
            w[t, _names] = _vals
            if long_short:
                _names, _vals = _leg_weights(t, s[:k], -1.0)
                w[t, _names] = _vals
        return w

    def apply_holding(w_rebal, hold):
        """Перекрывающиеся портфели: средний вес hold последних ребалансов"""
        if hold <= 1:
            return w_rebal
        w = np.zeros(w_rebal.shape)
        for i in range(hold):
            w = w + np.nan_to_num(_lag(w_rebal, i), nan=0.0)
        return w / hold

    def backtest_monthly(dates, ret, w_m, tc_bps, missing_mode):
        """Доходность месяца t применяется к весам t-1; издержки = tc × turnover.
        Пропавшая цена при открытой позиции: exit → 0%, penalize → -100%.
        dates — месяцы (строки ret и w_m)."""
        w = np.nan_to_num(w_m, nan=0.0)
        w_prev = np.nan_to_num(_lag(w, 1), nan=0.0)

        _fill = 0.0 if missing_mode == 'exit' else -1.0
        r_eff = np.where(np.isnan(ret) & (w_prev != 0), _fill, ret)
        r_eff = np.nan_to_num(r_eff, nan=0.0)
        # защита от артефактов данных (доходность ≤ -100%)
        r_eff = np.where((r_eff <= -1) & (w_prev != 0), _fill, r_eff)

        gross = _rowsum(w_prev * r_eff)
        turnover = 0.5 * _rowsum(np.abs(w - w_prev))
        tc = (tc_bps / 10000.0) * turnover
        return pl.DataFrame({'date': list(dates), 'ret_gross': gross,
                             'ret_net': gross - tc, 'turnover': turnover, 'tc': tc})

    def max_drawdown(eq):
        eq = np.asarray(eq, dtype=float)
        return float((eq / np.maximum.accumulate(eq) - 1.0).min())

    def perf_stats(ret, bench=None, freq=12, rf=None):
        """ret, bench — DataFrame (date, значение); rf — месячная безрисковая
        ставка (DataFrame date, rf в долях): Sharpe считается по избыточной
        доходности ret - rf; CAGR/Vol/MaxDD — по сырой."""
        _r = (ret.select('date', pl.col(ret.columns[1]).alias('r'))
              .filter(pl.col('r').is_not_null() & pl.col('r').is_not_nan()))
        if _r.is_empty():
            return {}
        r = _r['r'].to_numpy()
        eq = np.cumprod(1 + r)
        _years = len(r) / freq
        cagr = float(eq[-1] ** (1 / _years) - 1) if eq[-1] > 0 and _years > 0 else np.nan
        vol = float(r.std(ddof=0) * np.sqrt(freq))
        if rf is not None:
            _ex = (_r.join(rf.select('date', 'rf'), on='date', how='left')
                   .select(pl.col('r') - pl.col('rf').fill_null(0.0))
                   .to_series().to_numpy())
        else:
            _ex = r
        _ex_vol = float(_ex.std(ddof=0) * np.sqrt(freq))
        out = {
            'CAGR': cagr,
            'Vol': vol,
            'Sharpe': float(_ex.mean() * freq / _ex_vol) if _ex_vol > 0 else np.nan,
            'MaxDD': max_drawdown(eq),
            'Months': int(len(r)),
        }
        if bench is not None:
            _al = (_r.join(bench.select('date', pl.col(bench.columns[1]).alias('b')),
                           on='date', how='inner')
                   .filter(pl.col('b').is_not_null() & pl.col('b').is_not_nan())
                   .sort('date'))
            if _al.height:
                _a = (_al['r'] - _al['b']).to_numpy()
                _te = float(_a.std(ddof=0) * np.sqrt(freq))
                out['IR'] = float(_a.mean() * freq / _te) if _te > 0 else np.nan
                out['TE'] = _te
        return out

    return (apply_holding, backtest_monthly, build_rebal_weights,
            composite_signal, max_drawdown, momentum_signal, nargsort_desc,
            perf_stats, rolling_std)


@app.cell(hide_code=True)
def _(mo):
    mo.md("---\n## 1. Одиночная стратегия")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Параметры коротко:**

    - **Lookback** — окно импульса: за сколько месяцев считается доходность-сигнал.
    - **Skip** — сколько последних месяцев отрезать из сигнала. `skip=1` исключает
      краткосрочный разворот (классика «12-1»); `skip=0` — сигнал по самой свежей цене.
    - **Holding** — период удержания через *перекрывающиеся портфели*: при `hold=6`
      капитал разбит на 6 «винтажей» по 1/6, каждый месяц обновляется только один
      из них (средний возраст позиции ~3 мес). Эффект: оборот и издержки в разы
      ниже, кривая глаже — но сигнал в среднем «старее».
    - **Квантиль отбора** — какая доля лучших по сигналу бумаг покупается
      (равными весами).
    - **Long-Short** — дополнительно шортится нижний квантиль. Ближе к рыночной
      нейтральности, но шорт на MOEX дорог и доступен не по всем бумагам.
    - **Издержки (bps)** — стоимость оборота: 15 bps = 0.15% от каждой полной
      замены позиции (комиссия + спред).
    - **Top-N по обороту** — вселенная ограничивается N самыми ликвидными
      бумагами месяца: сигнал в неликвидах на практике не реализуем.
    - **Пропуск цены** — судьба позиции при исчезновении котировок (делистинг):
      `exit` — выход по 0% за месяц, `penalize` — консервативный штраф -100%.
    - **Взвешивание** — внутри выбранного квантиля: равные веса или
      *inverse-vol* (вес ∝ 1/σ бумаги, σ — за 63 торговых дня): спокойные бумаги
      получают больше, буйные меньше — риск распределяется равномернее между
      позициями. Не путать с volatility scaling — тот масштабирует весь портфель.
    - **Volatility scaling** — см. пояснение ниже при включении опции.
    - **Trend filter** — портфель держится только при аптренде IMOEX
      (EWMAC 16/64 или цена выше MA200), иначе кэш; см. пояснение при включении.
    - **Сигнал отбора** — чистый momentum или композит из трех факторов
      (momentum + low-vol + дивидендная доходность 12м, взвешенные ранги);
      см. пояснение при выборе композита.

    **Метрики в сводке:**

    - **CAGR** — среднегодовой темп роста капитала (сложный процент).
    - **Sharpe** — избыточная доходность на единицу риска: средняя доходность
      *сверх безрисковой ставки* / волатильность. По умолчанию rf — ключевая
      ставка ЦБ (ряд `metadata/key_rate.csv`, до 09.2013 — ставка
      рефинансирования); переключается на rf=0 в контролах. При российских
      ставках разница огромна: rf=0 завышает Sharpe примерно вдвое.
      Выше 1 на длинном окне — редкость.
    - **MaxDD** — максимальная просадка: худшее падение от пика до дна.
    - **Tracking error (TE)** — волатильность *активной* доходности
      (стратегия − бенчмарк), годовая. Показывает, насколько результат
      «гуляет» вокруг бенчмарка: индексный фонд ~0-1%, концентрированная
      активная стратегия — 10%+.
    - **Information Ratio (IR)** — «Sharpe активного управляющего»:
      средняя активная доходность / TE. Расшифровка: IR × TE ≈ средняя
      добавка над бенчмарком в % годовых (IR 0.32 × TE 14% ≈ +4.5%).
      Калибровка (Grinold-Kahn): 0.25-0.5 — прилично, 0.5+ — хорошо,
      1.0 — исключительно. При высокой TE даже хороший IR означает,
      что в отдельный год легко отстать от индекса на 10-15 п.п.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # Определения контролов (компоновка — в следующей ячейке)
    lookback_slider = mo.ui.slider(start=1, stop=12, step=1, value=6,
                                   label="Lookback, мес:", show_value=True)
    skip_slider = mo.ui.slider(start=0, stop=2, step=1, value=1,
                               label="Skip, мес:", show_value=True)
    hold_slider = mo.ui.slider(start=1, stop=6, step=1, value=1,
                               label="Holding, мес:", show_value=True)
    q_dropdown = mo.ui.dropdown(options={"10%": 0.1, "20%": 0.2, "30%": 0.3},
                                value="20%", label="Доля лучших бумаг:")
    ls_checkbox = mo.ui.checkbox(value=False, label="Long-Short (шорт проигравших)")
    tc_slider = mo.ui.slider(start=0, stop=50, step=5, value=15,
                             label="Издержки, bps за оборот:", show_value=True)
    topn_slider = mo.ui.slider(start=20, stop=100, step=10, value=50,
                               label="Top-N по обороту:", show_value=True)
    missing_dropdown = mo.ui.dropdown(options={"exit (выход по 0%)": "exit",
                                               "penalize (-100%)": "penalize"},
                                      value="exit (выход по 0%)", label="Делистинг:")
    weighting_dropdown = mo.ui.dropdown(
        options={"Равные веса": "equal", "Inverse-vol (вес ∝ 1/σ бумаги)": "invvol"},
        value="Равные веса", label="Взвешивание:")
    vol_checkbox = mo.ui.checkbox(value=False, label="Volatility scaling")
    vol_target_slider = mo.ui.slider(start=10, stop=30, step=1, value=15,
                                     label="Целевая волатильность, % годовых:",
                                     show_value=True)
    lev_cap_dropdown = mo.ui.dropdown(
        options={"1.0× (без плеча)": 1.0, "1.5×": 1.5, "2.0×": 2.0},
        value="1.5×", label="Макс. плечо:")
    trend_checkbox = mo.ui.checkbox(value=False, label="Trend filter (по IMOEX)")
    trend_mode_dropdown = mo.ui.dropdown(
        options={"EWMAC 16/64": "ewmac", "Цена выше MA200": "ma200"},
        value="EWMAC 16/64", label="Сигнал тренда:")
    rf_dropdown = mo.ui.dropdown(
        options={"Ключевая ставка ЦБ": "key", "0 (без rf)": "zero"},
        value="Ключевая ставка ЦБ", label="Безрисковая для Sharpe:")
    signal_dropdown = mo.ui.dropdown(
        options={"Momentum": "mom",
                 "Композит: momentum + low-vol + дивиденды": "composite"},
        value="Momentum", label="Сигнал:")
    lowvol_w_slider = mo.ui.slider(start=0.0, stop=1.0, step=0.25, value=0.5,
                                   label="Вес low-vol (momentum = 1):", show_value=True)
    div_w_slider = mo.ui.slider(start=0.0, stop=1.0, step=0.25, value=0.5,
                                label="Вес дивидендов:", show_value=True)
    return (
        div_w_slider,
        hold_slider,
        lev_cap_dropdown,
        lookback_slider,
        lowvol_w_slider,
        ls_checkbox,
        missing_dropdown,
        q_dropdown,
        rf_dropdown,
        signal_dropdown,
        skip_slider,
        tc_slider,
        topn_slider,
        trend_checkbox,
        trend_mode_dropdown,
        vol_checkbox,
        vol_target_slider,
        weighting_dropdown,
    )


@app.cell(hide_code=True)
def _(
    div_w_slider,
    hold_slider,
    lev_cap_dropdown,
    lookback_slider,
    lowvol_w_slider,
    ls_checkbox,
    missing_dropdown,
    mo,
    q_dropdown,
    rf_dropdown,
    signal_dropdown,
    skip_slider,
    tc_slider,
    topn_slider,
    trend_checkbox,
    trend_mode_dropdown,
    vol_checkbox,
    vol_target_slider,
    weighting_dropdown,
):
    # Панель параметров: логические группы; зависимые контролы появляются
    # только при включении своей опции
    _sig_row = [signal_dropdown, lookback_slider, skip_slider]
    if signal_dropdown.value == 'composite':
        _sig_row += [lowvol_w_slider, div_w_slider]

    _risk_row = [vol_checkbox]
    if vol_checkbox.value:
        _risk_row += [vol_target_slider, lev_cap_dropdown]
    _risk_row += [trend_checkbox]
    if trend_checkbox.value:
        _risk_row += [trend_mode_dropdown]

    controls_panel = mo.vstack([
        mo.md("**🎯 Сигнал отбора** — что и за какой период измеряем"),
        mo.hstack(_sig_row, justify='start', wrap=True),
        mo.md("**📦 Портфель** — как из сигнала собираются позиции"),
        mo.hstack([q_dropdown, weighting_dropdown, hold_slider, ls_checkbox],
                  justify='start', wrap=True),
        mo.md("**🌐 Вселенная и издержки**"),
        mo.hstack([topn_slider, tc_slider, missing_dropdown],
                  justify='start', wrap=True),
        mo.md("**🛡️ Риск-менеджмент** — надстройки поверх стратегии"),
        mo.hstack(_risk_row, justify='start', wrap=True),
        mo.md("**⚙️ Метрики**"),
        mo.hstack([rf_dropdown], justify='start'),
    ], gap=0.4)
    controls_panel
    return


@app.cell(hide_code=True)
def _(mo, vol_checkbox):
    # Пояснение к volatility scaling (показывается, когда опция включена)
    if vol_checkbox.value:
        vol_explain = mo.md(r"""
    **Как работает volatility scaling.** Экспозиция портфеля масштабируется так,
    чтобы его *ожидаемая* волатильность равнялась целевой:

    $$lev_t = \min\!\left(\frac{\sigma_{target}}{\hat\sigma_t},\ cap\right),
    \qquad w^{scaled}_t = lev_t \cdot w_t$$

    где $\hat\sigma_t$ — реализованная волатильность стратегии за последние
    12 месяцев (только прошлые данные — без заглядывания в будущее; оценка
    на конец месяца $t$ применяется к портфелю, который держится в $t{+}1$).

    Зачем это нужно:

    - **Постоянный риск.** Без скейлинга портфель несет вдвое больше риска в
      кризис, чем в спокойный год, — хотя вы «держите ту же стратегию».
    - **Защита от momentum crash.** Крупнейшие провалы моментума (2009, 2020)
      случаются при высокой волатильности — скейлинг механически режет
      экспозицию именно в такие периоды.
    - Волатильность предсказуема (кластеризуется), в отличие от доходности, —
      поэтому такое масштабирование исторически улучшает Sharpe.

    Цена вопроса: дополнительный оборот (издержки растут), а $lev_t > 1$
    означает плечо — при «Макс. плечо = 1.0×» стратегия только снижает
    экспозицию в бурные периоды, никогда не занимая.
    """)
    else:
        vol_explain = mo.md("")
    vol_explain
    return


@app.cell(hide_code=True)
def _(mo, trend_checkbox):
    # Пояснение к trend filter (показывается, когда опция включена)
    if trend_checkbox.value:
        trend_explain = mo.md(r"""
    **Как работает trend filter.** Портфель держится только в те месяцы, когда
    рынок в аптренде; вне тренда — кэш (0%, консервативно, без ставки на остаток):

    $$w^{filtered}_t = w_t \cdot \mathbb{1}[\text{тренд}_t],\qquad
    \text{тренд}_t = \begin{cases}
    EWMA_{16} > EWMA_{64} & \text{(EWMAC)}\\
    P > MA_{200} & \text{(MA200)}
    \end{cases}$$

    Сигнал считается по дневному IMOEX на конец месяца $t$ и управляет портфелем
    месяца $t{+}1$ — заглядывания в будущее нет. Выход в кэш и возврат в рынок
    проходят через оборот и платят издержки.

    Зачем это нужно:

    - **Обрезание хвостов.** Крупнейшие потери рынка РФ (2008: -70%, 2022: -50%)
      разворачивались месяцами — трендовый сигнал успевает вывести в кэш.
      Избегание глубоких просадок компаундится сильнее, чем отбор бумаг.
    - **Защита моментума от самого себя**: momentum crash случается на резких
      разворотах вверх *после* обвала — фильтр в эти месяцы уже в кэше.
    - Это time-series momentum (Moskowitz-Ooi-Pedersen) поверх кросс-секционного.

    Цена вопроса: **пила (whipsaw)** на боковике — фильтр выходит/входит с опозданием
    и теряет на ложных сигналах; в устойчивый бычий год фильтр только мешает.
    Важно: кэш IMOEX начинается с 2010 года — до этой даты фильтр неактивен
    (портфель всегда в рынке); удлините кэш индекса для полной истории.
    """)
    else:
        trend_explain = mo.md("")
    trend_explain
    return


@app.cell(hide_code=True)
def _(mo, signal_dropdown):
    # Пояснение к композитному сигналу (показывается при выборе композита)
    if signal_dropdown.value == 'composite':
        composite_explain = mo.md(r"""
    **Как устроен композитный сигнал.** Вместо одного momentum бумаги
    ранжируются по трем факторам сразу; скор — взвешенная сумма
    кросс-секционных процентильных рангов:

    $$score_i = \frac{w_m \cdot rank(mom_i) + w_{lv} \cdot rank(-\sigma_i)
    + w_d \cdot rank(divyield_i)}{w_m + w_{lv} + w_d}$$

    - **Momentum** — как в базовой стратегии (lookback/skip из контролов выше).
    - **Low-vol** — ранг по *низкой* волатильности (σ за 63 торговых дня):
      аномалия низкого риска — спокойные бумаги исторически дают доходность
      не хуже буйных при меньшем риске.
    - **Дивиденды** — 12-месячная дивидендная доходность, вычисленная как
      разница полной (adj_close) и ценовой (close) годовой доходности —
      без парсинга дивидендных файлов и проблем с базой после сплитов.

    Ранги (а не сырые значения) делают факторы сопоставимыми по масштабу.
    Если у бумаги нет какого-то фактора (короткая история), он исключается
    из знаменателя; momentum обязателен. Дальше всё как обычно: top-квантиль
    по скору, фильтр ликвидности, взвешивание, издержки.

    Зачем: факторы слабо коррелированы, и их сумма дает более стабильный
    сигнал, чем каждый по отдельности (диверсификация источников альфы —
    практически единственный «бесплатный обед» при малом числе бумаг).
    Цена: композит «размывает» чистый momentum — в годы, когда моментум
    силен, композит отстанет от него.
    """)
    else:
        composite_explain = mo.md("")
    composite_explain
    return


@app.cell(hide_code=True)
def _(
    apply_holding,
    backtest_monthly,
    bench_ret_m,
    build_rebal_weights,
    composite_signal,
    div12_m,
    div_w_slider,
    hold_slider,
    lev_cap_dropdown,
    liq_m,
    lookback_slider,
    lowvol_w_slider,
    ls_checkbox,
    missing_dropdown,
    momentum_signal,
    months,
    np,
    perf_stats,
    pl,
    px_m,
    q_dropdown,
    ret_m,
    rf_dropdown,
    rolling_std,
    signal_dropdown,
    skip_slider,
    stocks,
    tc_slider,
    topn_slider,
    trend_checkbox,
    trend_mode_dropdown,
    trend_signals,
    vol_checkbox,
    vol_m,
    vol_target_slider,
    weighting_dropdown,
):
    # Бэктест одиночной стратегии
    _sig = momentum_signal(px_m, lookback_slider.value, skip_slider.value)
    if signal_dropdown.value == 'composite':
        _sig = composite_signal(_sig, vol_m, div12_m,
                                w_mom=1.0,
                                w_lowvol=lowvol_w_slider.value,
                                w_div=div_w_slider.value)
    _w0 = build_rebal_weights(_sig, liq_m, q_dropdown.value,
                              ls_checkbox.value, topn_slider.value,
                              weighting=weighting_dropdown.value, vol=vol_m)
    _w_base = apply_holding(_w0, hold_slider.value)

    # Trend filter: вне аптренда IMOEX — в кэш (веса обнуляются).
    # Где сигнала нет (до начала кэша индекса) — фильтр неактивен (в рынке)
    trend_share = None
    if trend_checkbox.value and trend_signals is not None:
        _gate = (pl.DataFrame({'date': months})
                 .join(trend_signals.select('date', trend_mode_dropdown.value),
                       on='date', how='left')[trend_mode_dropdown.value]
                 .cast(pl.Float64).fill_null(1.0).to_numpy())
        _w_base = _w_base * _gate[:, None]
        trend_share = float(_gate.mean())

    if vol_checkbox.value:
        # Volatility scaling: плечо = target / realized vol (12 мес, ex-ante).
        # Оценка волатильности на конец месяца t применяется к весам t,
        # которые работают в t+1 — заглядывания в будущее нет.
        _bt_raw = backtest_monthly(months, ret_m, _w_base, tc_bps=0,
                                   missing_mode=missing_dropdown.value)
        _sigma = rolling_std(_bt_raw['ret_gross'].to_numpy(), 12, 6, ddof=0) * np.sqrt(12)
        with np.errstate(divide='ignore'):
            _lev = (vol_target_slider.value / 100.0 / _sigma)
        _lev[np.isinf(_lev)] = np.nan
        _lev = np.nan_to_num(np.minimum(_lev, float(lev_cap_dropdown.value)), nan=1.0)
        leverage_series = pl.DataFrame({'date': months, 'lev': _lev})
        weights_single = _w_base * _lev[:, None]
    else:
        leverage_series = None
        weights_single = _w_base

    bt_single = backtest_monthly(months, ret_m, weights_single,
                                 tc_bps=tc_slider.value,
                                 missing_mode=missing_dropdown.value)
    # первые lookback+skip месяцев сигнала нет — отбрасываем разогрев
    _warmup = lookback_slider.value + skip_slider.value + 1
    bt_single = bt_single.slice(_warmup)
    if leverage_series is not None:
        leverage_series = leverage_series.slice(_warmup)

    # Безрисковая ставка для Sharpe: месячный ряд ключевой ставки ЦБ
    if rf_dropdown.value == 'key':
        rf_monthly = stocks.risk_free_monthly(bt_single['date'])
        rf_mean_ann = float(rf_monthly['rf'].mean() * 12 * 100)
    else:
        rf_monthly = None
        rf_mean_ann = None

    stats_net = perf_stats(bt_single.select('date', 'ret_net'), bench_ret_m, rf=rf_monthly)
    stats_gross = perf_stats(bt_single.select('date', 'ret_gross'), rf=rf_monthly)
    # Бенчмарк и сопоставимая статистика стратегии — на общем окне
    # (кэш MCFTR может начинаться позже старта стратегии)
    _common_idx = (bt_single.select('date')
                   .join(bench_ret_m.select('date'), on='date', how='semi')['date'])
    stats_bench = perf_stats(bench_ret_m.filter(pl.col('date').is_in(_common_idx.implode())),
                             rf=rf_monthly)
    stats_net_common = perf_stats(
        bt_single.filter(pl.col('date').is_in(_common_idx.implode())).select('date', 'ret_net'),
        rf=rf_monthly)
    common_start = _common_idx.min() if len(_common_idx) else None
    return (bt_single, common_start, leverage_series, rf_mean_ann, stats_bench,
            stats_gross, stats_net, stats_net_common, trend_share, weights_single)


@app.cell(hide_code=True)
def _(
    bench_name,
    bt_single,
    common_start,
    div_w_slider,
    leverage_series,
    lowvol_w_slider,
    mo,
    np,
    rf_mean_ann,
    signal_dropdown,
    stats_bench,
    stats_gross,
    stats_net,
    stats_net_common,
    trend_mode_dropdown,
    trend_share,
    vol_target_slider,
    weights_single,
):
    # Сводка одиночной стратегии
    def _sgn(_v, _suffix='%', _nd=1, _mult=100):
        if _v is None or _v != _v:
            return 'н/д'
        _x = _v * _mult
        _cls = 'pos' if _x >= 0 else 'neg'
        return f'<span class="{_cls}">{format(_x, f"+.{_nd}f")}{_suffix}</span>'

    if not stats_net:
        single_summary = mo.md("Недостаточно данных для бэктеста")
    else:
        _w_abs = np.abs(np.nan_to_num(weights_single, nan=0.0))
        _avg_pos = float((_w_abs > 0).sum(axis=1).mean())
        _avg_to = float(bt_single['turnover'].mean())
        _trend_label = {'ewmac': 'EWMAC 16/64', 'ma200': 'выше MA200'}.get(
            trend_mode_dropdown.value, '')
        single_summary = mo.md(
            f"### Результат (net, после издержек)\n\n"
            f"- **CAGR: {_sgn(stats_net['CAGR'])}** | волатильность {stats_net['Vol'] * 100:.0f}% | "
            f"Sharpe **{stats_net['Sharpe']:.2f}** | MaxDD {_sgn(stats_net['MaxDD'])}\n"
            f"- Gross (до издержек): CAGR {_sgn(stats_gross.get('CAGR'))}, "
            f"Sharpe {stats_gross.get('Sharpe', float('nan')):.2f}\n"
            + (f"- Сравнение с {bench_name} за общий период "
               f"(с {common_start:%Y-%m}): бенчмарк CAGR {_sgn(stats_bench.get('CAGR'))}, "
               f"Sharpe {stats_bench.get('Sharpe', float('nan')):.2f}, "
               f"MaxDD {_sgn(stats_bench.get('MaxDD'))} · "
               f"стратегия CAGR {_sgn(stats_net_common.get('CAGR'))}, "
               f"Sharpe {stats_net_common.get('Sharpe', float('nan')):.2f}, "
               f"MaxDD {_sgn(stats_net_common.get('MaxDD'))}\n"
               if common_start is not None else "- Бенчмарк недоступен\n")
            + f"- Information Ratio: **{stats_net.get('IR', float('nan')):.2f}** "
            f"(tracking error {stats_net.get('TE', float('nan')) * 100:.0f}%)\n"
            f"- Средний месячный оборот: {_avg_to * 100:.0f}% | "
            f"средне позиций: {_avg_pos:.0f} | месяцев: {stats_net['Months']}"
            + (f"\n- Vol scaling: цель {vol_target_slider.value}%, "
               f"реализовано {stats_net['Vol'] * 100:.0f}%; плечо: "
               f"среднее {float(leverage_series['lev'].mean()):.2f}×, "
               f"диапазон {float(leverage_series['lev'].min()):.2f}—{float(leverage_series['lev'].max()):.2f}×"
               if leverage_series is not None else "")
            + (f"\n- Trend filter ({_trend_label} по IMOEX): "
               f"в рынке {trend_share * 100:.0f}% месяцев"
               if trend_share is not None else "")
            + (f"\n- Sharpe — по избыточной доходности над ключевой ставкой ЦБ "
               f"(средняя за окно: {rf_mean_ann:.1f}% годовых)"
               if rf_mean_ann is not None else "\n- Sharpe считается при rf = 0")
            + (f"\n- Сигнал: композит (momentum 1.0 / low-vol {lowvol_w_slider.value} / "
               f"дивиденды {div_w_slider.value})"
               if signal_dropdown.value == 'composite' else "")
        )
    single_summary
    return


@app.cell(hide_code=True)
def _(bench_name, bench_ret_m, bt_single, go, mo, np, plotly_available):
    # График капитала: стратегия net/gross против бенчмарка (лог-шкала)
    if not plotly_available or len(bt_single) == 0:
        equity_block = mo.md("")
    else:
        _x = bt_single['date'].to_list()
        _eq_net = np.cumprod(1 + bt_single['ret_net'].to_numpy())
        _eq_gross = np.cumprod(1 + bt_single['ret_gross'].to_numpy())
        _b = (bench_ret_m.join(bt_single.select('date'), on='date', how='semi')
              .drop_nulls().drop_nans().sort('date'))
        _xb = _b['date'].to_list()
        _eq_bench = np.cumprod(1 + _b['bench_ret'].to_numpy())

        # Ребейз всех линий к первой ОБЩЕЙ дате: иначе «1» у стратегии и
        # бенчмарка приходится на разные моменты, и уровни несопоставимы
        _t0 = _xb[0] if len(_xb) else None
        if _t0 is not None:
            _i0 = _x.index(_t0)
            _eq_net = _eq_net / _eq_net[_i0]
            _eq_gross = _eq_gross / _eq_gross[_i0]
            _eq_bench = _eq_bench / _eq_bench[0]

        _fig = go.Figure()
        _fig.add_scatter(x=_x, y=_eq_net, name='Стратегия (net)',
                         line=dict(color='#1f77b4', width=2),
                         hovertemplate='%{y:.2f}<extra>net</extra>')
        _fig.add_scatter(x=_x, y=_eq_gross, name='Стратегия (gross)',
                         line=dict(color='#aec7e8', width=1.2, dash='dot'),
                         hovertemplate='%{y:.2f}<extra>gross</extra>')
        _fig.add_scatter(x=_xb, y=_eq_bench, name=bench_name.split(' ')[0],
                         line=dict(color='#7f7f7f', width=1.5, dash='dash'),
                         hovertemplate='%{y:.2f}<extra>бенчмарк</extra>')
        if _t0 is not None and _x[0] < _t0:
            _fig.add_vline(x=_t0, line_dash='dot', line_color='gray', line_width=1)
            _fig.add_annotation(x=_t0, y=1, yref='y', text='старт сравнения<br>(есть бенчмарк)',
                                showarrow=False, xanchor='left', xshift=6,
                                font=dict(size=10, color='gray'))
        _fig.update_layout(
            height=420, hovermode='x unified',
            title=dict(text='Рост капитала (1 = старт сравнения, лог-шкала)', font_size=14),
            yaxis=dict(type='log'),
            legend=dict(orientation='h', y=1.1, x=1, xanchor='right'),
            margin=dict(t=44, l=10, r=10, b=10),
        )
        equity_block = _fig
    equity_block
    return


@app.cell(hide_code=True)
def _(bt_single, go, leverage_series, mo, np, plotly_available):
    # Просадки стратегии (net), месячный оборот и плечо (если vol scaling включен)
    if not plotly_available or len(bt_single) == 0:
        dd_to_block = mo.md("")
    else:
        _x = bt_single['date'].to_list()
        _eq = np.cumprod(1 + bt_single['ret_net'].to_numpy())
        _dd = (_eq / np.maximum.accumulate(_eq) - 1) * 100
        _figd = go.Figure()
        _figd.add_scatter(x=_x, y=_dd, fill='tozeroy',
                          line=dict(color='#d62728', width=1),
                          fillcolor='rgba(214,39,40,0.25)',
                          name='просадка',
                          hovertemplate='%{y:.1f}%<extra>DD</extra>')
        _figd.add_bar(x=_x, y=bt_single['turnover'].to_numpy() * 100,
                      name='оборот', marker_color='rgba(31,119,180,0.4)', yaxis='y2',
                      hovertemplate='%{y:.0f}%<extra>оборот</extra>')
        if leverage_series is not None:
            _figd.add_scatter(x=leverage_series['date'].to_list(),
                              y=leverage_series['lev'].to_numpy() * 100,
                              name='плечо (экспозиция)', yaxis='y2',
                              line=dict(color='#ff7f0e', width=1.6),
                              hovertemplate='%{y:.0f}%<extra>плечо</extra>')
        _figd.update_layout(
            height=300,
            title=dict(text='Просадки (net) и месячный оборот', font_size=13),
            yaxis=dict(title='Просадка', ticksuffix='%'),
            yaxis2=dict(overlaying='y', side='right', title='Оборот',
                        ticksuffix='%', showgrid=False),
            legend=dict(orientation='h', y=1.14, x=1, xanchor='right'),
            margin=dict(t=44, l=10, r=10, b=10),
        )
        dd_to_block = _figd
    dd_to_block
    return


@app.cell(hide_code=True)
def _(bt_single, go, last_data_date, mo, np, pl, plotly_available):
    # Доходности по годам и месяцам (net): строки — годы, последний сверху
    if not plotly_available or len(bt_single) == 0:
        monthly_block = mo.md("")
    else:
        _partial_note = (f" · последняя ячейка — незавершенный месяц (до {last_data_date:%d.%m})"
                         if len(bt_single) and bt_single['date'].max() > last_data_date else "")
        _tbl = bt_single.select(y=pl.col('date').dt.year(), m=pl.col('date').dt.month(),
                                v=pl.col('ret_net'))
        _years = sorted(_tbl['y'].unique().to_list(), reverse=True)
        _pv = np.full((len(_years), 13), np.nan)
        for _row in _tbl.group_by('y', 'm').agg(pl.col('v').sum()).iter_rows(named=True):
            _pv[_years.index(_row['y']), _row['m'] - 1] = _row['v']
        for _row in (_tbl.group_by('y').agg(((1 + pl.col('v')).product() - 1).alias('v'))
                     .iter_rows(named=True)):
            _pv[_years.index(_row['y']), 12] = _row['v']

        _z = _pv * 100
        _xlabels = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 'Июл', 'Авг',
                    'Сен', 'Окт', 'Ноя', 'Дек', 'Год']
        _ylabels = [str(_y) for _y in _years]
        _text = [['' if np.isnan(_v) else f'{_v:+.1f}' for _v in _row] for _row in _z]
        _vmax = max(float(np.nanpercentile(np.abs(_z), 95)), 1e-9)

        _figm = go.Figure(go.Heatmap(
            z=_z, x=_xlabels, y=_ylabels,
            text=_text, texttemplate='%{text}', textfont_size=10,
            colorscale='RdYlGn', zmid=0, zmin=-_vmax, zmax=_vmax,
            showscale=False, xgap=1, ygap=1,
            hovertemplate='%{y} %{x}: %{z:+.2f}%<extra></extra>',
        ))
        _figm.update_layout(
            height=max(300, 26 * len(_ylabels) + 90),
            title=dict(text=f'Доходности стратегии по месяцам (net, %){_partial_note}', font_size=13),
            yaxis=dict(autorange='reversed', type='category'),
            xaxis=dict(side='top'),
            margin=dict(t=70, l=10, r=10, b=10),
        )
        monthly_block = _figm
    monthly_block
    return


@app.cell(hide_code=True)
def _(mo, months, np, weights_single):
    # Выбор месяца для просмотра состава портфеля (по умолчанию — последний)
    _active = [months[_i] for _i in np.flatnonzero(np.abs(weights_single).sum(axis=1) > 0)]
    _opts = {_i.strftime('%Y-%m'): _i for _i in reversed(_active)}
    if _opts:
        month_dropdown = mo.ui.dropdown(
            options=_opts, value=list(_opts)[0],
            label='Состав портфеля на месяц:', searchable=True,
        )
        _md_ctrl = month_dropdown
    else:
        month_dropdown = None
        _md_ctrl = mo.md("")
    _md_ctrl
    return (month_dropdown,)


@app.cell(hide_code=True)
def _(
    bt_single,
    go,
    last_data_date,
    mo,
    momentum_signal,
    month_dropdown,
    months,
    lookback_slider,
    np,
    pl,
    plotly_available,
    px_m,
    skip_slider,
    stocks,
    tickers,
    vol_m,
    weights_single,
):
    # Структура портфеля на выбранный месяц
    if month_dropdown is None or not plotly_available:
        portfolio_block = mo.md("")
    else:
        _t = month_dropdown.value
        _ti = months.index(_t)
        _row = weights_single[_ti]
        _nz = np.flatnonzero(_row != 0)
        _nz = _nz[np.argsort(_row[_nz], kind='quicksort')]
        _w_names = [tickers[_j] for _j in _nz]
        _w_vals = _row[_nz]

        if len(_w_vals) == 0:
            portfolio_block = mo.md("*Портфель на этот месяц пуст*")
        else:
            # Сигнал momentum, по которому портфель был сформирован
            _sig_row = dict(zip(tickers, momentum_signal(
                px_m, lookback_slider.value, skip_slider.value)[_ti]))

            # Сектора из справочника
            import os as _os3
            _sec_path = _os3.path.join(stocks.BASE_DIR, 'metadata', 'sectors.csv')
            _sec_map = (dict(pl.read_csv(_sec_path).select('ticker', 'sector').iter_rows())
                        if _os3.path.exists(_sec_path) else {})

            # Реализованная доходность портфеля в следующем месяце (если он уже прошел)
            _next = bt_single.filter(pl.col('date') > _t)
            _realized = (f" · реализовано в след. месяце: "
                         f"{_next['ret_net'][0] * 100:+.1f}% (net)"
                         if len(_next) else " · следующий месяц еще не завершен")

            # Метка месяца — конец периода; если месяц не завершен, честно
            # показываем фактическую дату данных и статус состава
            if _t > last_data_date:
                _asof = (f"по данным на {last_data_date:%d.%m.%Y} "
                         f"(месяц не завершен; состав "
                         + ("финальный: сигнал при skip≥1 не использует цены "
                            "текущего месяца" if skip_slider.value >= 1
                            else "предварительный: при skip=0 может измениться "
                                 "до конца месяца") + ")")
            else:
                _asof = f"по данным на {_t:%d.%m.%Y}"

            _vol_row = dict(zip(tickers, vol_m[_ti]))
            _hover = [
                f"{_tk}<br>вес: {_v * 100:+.1f}%<br>momentum: {_sig_row.get(_tk, float('nan')) * 100:+.1f}%"
                f"<br>σ 63д: {_vol_row.get(_tk, float('nan')) * 100:.0f}%"
                f"<br>{_sec_map.get(_tk, 'сектор н/д')}"
                for _tk, _v in zip(_w_names, _w_vals)
            ]
            _figp = go.Figure(go.Bar(
                x=_w_vals * 100, y=_w_names, orientation='h',
                marker_color=['green' if _v >= 0 else 'red' for _v in _w_vals],
                text=[f'{_v * 100:+.1f}%' for _v in _w_vals],
                textposition='outside',
                hovertext=_hover, hoverinfo='text',
            ))
            _figp.update_layout(
                height=max(260, 24 * len(_w_vals) + 90),
                title=dict(
                    text=f'Портфель, сформированный {_asof} '
                         f'(удерживается в следующем месяце) · позиций: {len(_w_vals)} · '
                         f'экспозиция Σ|w| = {np.abs(_w_vals).sum() * 100:.0f}%{_realized}',
                    font_size=12),
                xaxis=dict(title='Вес (%)', zeroline=True, zerolinecolor='black'),
                margin=dict(t=48, l=10, r=10, b=10),
            )
            portfolio_block = _figp
    portfolio_block
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 2. Walk-forward подбор параметров

    Grid search по сетке (lookback × skip × holding × квантиль × long/short).
    Параметры выбираются по **Sharpe на train-периоде**, затем стратегия
    оценивается на **test** — это защита от overfitting: красивый train
    ничего не гарантирует, смотреть надо на test-колонки.

    Расчет перебирает ~100+ комбинаций и занимает до минуты.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    train_frac_slider = mo.ui.slider(start=50, stop=90, step=5, value=70,
                                     label="% месяцев на train:", show_value=True)
    objective_dropdown = mo.ui.dropdown(options=["Sharpe", "IR"], value="Sharpe",
                                        label="Критерий отбора:")
    run_grid_button = mo.ui.run_button(label="Запустить grid search")
    mo.hstack([train_frac_slider, objective_dropdown, run_grid_button], justify='start')
    return objective_dropdown, run_grid_button, train_frac_slider


@app.cell(hide_code=True)
def _(
    apply_holding,
    backtest_monthly,
    bench_ret_m,
    build_rebal_weights,
    liq_m,
    composite_signal,
    div12_m,
    div_w_slider,
    lowvol_w_slider,
    missing_dropdown,
    mo,
    momentum_signal,
    months,
    nargsort_desc,
    np,
    objective_dropdown,
    perf_stats,
    pl,
    px_m,
    ret_m,
    rf_dropdown,
    run_grid_button,
    signal_dropdown,
    stocks,
    tc_slider,
    topn_slider,
    train_frac_slider,
    trend_checkbox,
    trend_mode_dropdown,
    trend_signals,
    vol_m,
    weighting_dropdown,
):
    # Walk-forward grid search (издержки/ликвидность/пропуски/взвешивание/тренд-фильтр — из контролов)
    if not run_grid_button.value:
        grid_results = pl.DataFrame()
        grid_best = None
        grid_status = mo.md("Нажмите **«Запустить grid search»** для расчета.")
    elif len(bench_ret_m) < 12:
        grid_results = pl.DataFrame()
        grid_best = None
        grid_status = mo.md("**Бенчмарк недоступен** — walk-forward требует бенчмарк.")
    else:
        # Общие месяцы панели и бенчмарка; строки панелей берутся по ним
        _bench_dates = set(bench_ret_m['date'].to_list())
        _ci = [_i for _i, _d in enumerate(months) if _d in _bench_dates]
        _common = [months[_i] for _i in _ci]
        _ret = ret_m[_ci]
        _px = px_m[_ci]
        _liq = liq_m[_ci] if liq_m is not None else None
        _vol = vol_m[_ci]
        _div = div12_m[_ci]
        _bench = (pl.DataFrame({'date': _common})
                  .join(bench_ret_m, on='date', how='left'))

        _split = int(np.floor(len(_common) * train_frac_slider.value / 100))
        _train_idx = _common[:_split]
        _test_idx = _common[_split:]

        # Trend filter применяется, если включен в контролах одиночной стратегии
        if trend_checkbox.value and trend_signals is not None:
            _gate_g = (pl.DataFrame({'date': _common})
                       .join(trend_signals.select('date', trend_mode_dropdown.value),
                             on='date', how='left')[trend_mode_dropdown.value]
                       .cast(pl.Float64).fill_null(1.0).to_numpy())
        else:
            _gate_g = None

        # Безрисковая ставка — как в контролах одиночной стратегии
        _rf_g = stocks.risk_free_monthly(_common) if rf_dropdown.value == 'key' else None

        _rows = []
        grid_best = None
        _best_score = -np.inf
        for _lb in (3, 6, 9, 12):
            for _sk in (0, 1):
                for _hd in (1, 3, 6):
                    for _qq in (0.1, 0.2):
                        for _ls in (False, True):
                            _sig_g = momentum_signal(_px, _lb, _sk)
                            if signal_dropdown.value == 'composite':
                                _sig_g = composite_signal(
                                    _sig_g, _vol, _div, w_mom=1.0,
                                    w_lowvol=lowvol_w_slider.value,
                                    w_div=div_w_slider.value)
                            _w_g = apply_holding(
                                build_rebal_weights(_sig_g, _liq, _qq, _ls, topn_slider.value,
                                                    weighting=weighting_dropdown.value,
                                                    vol=_vol),
                                _hd)
                            if _gate_g is not None:
                                _w_g = _w_g * _gate_g[:, None]
                            _bt_g = backtest_monthly(_common, _ret, _w_g,
                                                     tc_bps=tc_slider.value,
                                                     missing_mode=missing_dropdown.value)
                            _st_tr = perf_stats(_bt_g.slice(0, _split).select('date', 'ret_net'),
                                                _bench.slice(0, _split), rf=_rf_g)
                            _st_te = perf_stats(_bt_g.slice(_split).select('date', 'ret_net'),
                                                _bench.slice(_split), rf=_rf_g)
                            _score = _st_tr.get(objective_dropdown.value, np.nan)
                            if _score is None or not np.isfinite(_score):
                                _score = -np.inf
                            _rows.append({
                                'lookback': _lb, 'skip': _sk, 'hold': _hd,
                                'q': _qq, 'long_short': _ls,
                                'train_Sharpe': float(round(_st_tr.get('Sharpe', np.nan), 2)),
                                'train_CAGR_%': float(round(_st_tr.get('CAGR', np.nan) * 100, 1)),
                                'train_IR': float(round(_st_tr.get('IR', np.nan), 2)),
                                'test_Sharpe': float(round(_st_te.get('Sharpe', np.nan), 2)),
                                'test_CAGR_%': float(round(_st_te.get('CAGR', np.nan) * 100, 1)),
                                'test_MaxDD_%': float(round(_st_te.get('MaxDD', np.nan) * 100, 1)),
                                'test_IR': float(round(_st_te.get('IR', np.nan), 2)),
                                'turnover_%': float(round(float(_bt_g['turnover'].mean()) * 100, 0)),
                            })
                            if _score > _best_score:
                                _best_score = _score
                                grid_best = {'params': (_lb, _sk, _hd, _qq, _ls),
                                             'bt': _bt_g, 'test_idx': _test_idx,
                                             'bench': _bench,
                                             'st_tr': _st_tr, 'st_te': _st_te}

        # Сортировка по train-Sharpe по убыванию (NaN — в конце)
        grid_results = pl.DataFrame(_rows)
        grid_results = grid_results[nargsort_desc(grid_results['train_Sharpe'].to_numpy())]
        _p = grid_best['params']
        grid_status = mo.md(
            f"### Лучшая по train-{objective_dropdown.value}: "
            f"lookback={_p[0]}, skip={_p[1]}, hold={_p[2]}, q={_p[3]:.0%}, "
            f"{'long-short' if _p[4] else 'long-only'}\n\n"
            f"- Train: Sharpe **{grid_best['st_tr'].get('Sharpe', float('nan')):.2f}**, "
            f"CAGR {grid_best['st_tr'].get('CAGR', float('nan')) * 100:+.1f}%\n"
            f"- **Test (out-of-sample): Sharpe {grid_best['st_te'].get('Sharpe', float('nan')):.2f}, "
            f"CAGR {grid_best['st_te'].get('CAGR', float('nan')) * 100:+.1f}%, "
            f"IR {grid_best['st_te'].get('IR', float('nan')):.2f}**"
        )
    grid_status
    return grid_best, grid_results


@app.cell(hide_code=True)
def _(grid_results, mo):
    if len(grid_results) > 0:
        grid_table = mo.ui.table(grid_results.head(25), label="Top-25 стратегий (по train)")
    else:
        grid_table = mo.md("")
    grid_table
    return


@app.cell(hide_code=True)
def _(go, grid_results, mo, np, pl, plotly_available):
    # Карта устойчивости: средний test-Sharpe по lookback × hold.
    # Усреднение по остальным осям сетки (skip, квантиль, long/short) показывает,
    # где параметры образуют устойчивое плато, а где — одинокий пик (оверфиттинг).
    if not plotly_available or len(grid_results) == 0:
        robustness_block = mo.md("")
    else:
        # Среднее по ячейке без NaN; пустые ячейки/строки/столбцы не выводятся
        _agg = (grid_results.group_by('hold', 'lookback')
                .agg(pl.col('test_Sharpe').fill_nan(None).mean())
                .drop_nulls('test_Sharpe'))
        _pv_rows = sorted(_agg['hold'].unique().to_list())
        _pv_cols = sorted(_agg['lookback'].unique().to_list())
        _z = np.full((len(_pv_rows), len(_pv_cols)), np.nan)
        for _h, _lb, _v in _agg.iter_rows():
            _z[_pv_rows.index(_h), _pv_cols.index(_lb)] = _v
        _text = [['' if np.isnan(_v) else f'{_v:.2f}' for _v in _row] for _row in _z]
        _vmax = max(float(np.nanmax(np.abs(_z))), 1e-9)

        _figr2 = go.Figure(go.Heatmap(
            z=_z,
            x=[f'lookback {_c}' for _c in _pv_cols],
            y=[f'hold {_i}' for _i in _pv_rows],
            text=_text, texttemplate='%{text}', textfont_size=13,
            colorscale='RdYlGn', zmid=0, zmin=-_vmax, zmax=_vmax,
            showscale=False, xgap=2, ygap=2,
            hovertemplate='%{x}, %{y}: средний test-Sharpe %{z:.2f}<extra></extra>',
        ))
        _figr2.update_layout(
            height=90 + 60 * len(_pv_rows),
            title=dict(text='Карта устойчивости: средний test-Sharpe '
                            '(усреднение по skip, квантилю и long/short)',
                       font_size=13),
            margin=dict(t=44, l=10, r=10, b=10),
        )
        robustness_block = mo.vstack([
            _figr2,
            mo.md("*Как читать: доверять стоит области, где соседние ячейки "
                  "одинаково зеленые (плато). Если лучшая по train конфигурация "
                  "стоит в одинокой яркой ячейке среди бледных — это, скорее "
                  "всего, подгонка под train-период, а не сигнал.*"),
        ])
    robustness_block
    return


@app.cell(hide_code=True)
def _(bench_name, go, grid_best, mo, np, pl, plotly_available):
    # Out-of-sample: лучшая стратегия против бенчмарка на test-периоде
    if not plotly_available or grid_best is None:
        oos_block = mo.md("")
    else:
        _te_idx = grid_best['test_idx']
        _r_te = grid_best['bt'].filter(pl.col('date').is_in(_te_idx))
        _b_te = grid_best['bench'].filter(pl.col('date').is_in(_te_idx))
        _eq_s = np.cumprod(1 + _r_te['ret_net'].to_numpy())
        _eq_b = np.cumprod(1 + _b_te['bench_ret'].to_numpy())

        _figo = go.Figure()
        _figo.add_scatter(x=_r_te['date'].to_list(), y=_eq_s, name='Стратегия (net)',
                          line=dict(color='#1f77b4', width=2),
                          hovertemplate='%{y:.2f}<extra>стратегия</extra>')
        _figo.add_scatter(x=_b_te['date'].to_list(), y=_eq_b, name=bench_name.split(' ')[0],
                          line=dict(color='#7f7f7f', width=1.5, dash='dash'),
                          hovertemplate='%{y:.2f}<extra>бенчмарк</extra>')
        _figo.update_layout(
            height=380, hovermode='x unified',
            title=dict(text='Out-of-sample (test-период): лучшая стратегия vs бенчмарк',
                       font_size=14),
            legend=dict(orientation='h', y=1.12, x=1, xanchor='right'),
            margin=dict(t=44, l=10, r=10, b=10),
        )
        oos_block = _figo
    oos_block
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    **Что важно помнить об этом бэктесте**

    - Вселенная — бумаги, которые есть в `data/` сейчас: у выборки есть
      survivorship bias (часть делистингов 2000-х отсутствует). Обработка
      пропусков (`exit`/`penalize`) частично компенсирует его на имеющихся данных.
    - Издержки заданы константой в bps на оборот; для неликвидов реальный
      спред выше — фильтр top-N по обороту обязателен.
    - Walk-forward с одним split — минимальная защита от overfitting;
      несколько окон (rolling) были бы строже.
    """)
    return


if __name__ == "__main__":
    app.run()
