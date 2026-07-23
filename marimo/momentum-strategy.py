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
    # moex_utils лежит в корне проекта (родительская папка от marimo/)
    import sys as _sys
    import os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    import moex_utils as moex
    import pandas as pd
    import numpy as np
    import marimo as mo
    try:
        import plotly.graph_objects as go
        plotly_available = True
    except ImportError:
        go = None
        plotly_available = False
    return go, mo, moex, np, pd, plotly_available


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
def _(mo, moex, np, pd):
    # Данные: месячные панели цен (полная доходность) и ликвидности
    _c = moex.adjust_for_splits(moex.combine_moex_stocks())
    if 'adj_close' not in _c.columns:
        _c['adj_close'] = _c['close']
    _c = _c[_c['adj_close'].notna() & (_c['adj_close'] > 0)]

    _px_d = _c.pivot_table(index=_c.index, columns='ticker',
                           values='adj_close', aggfunc='last').sort_index()
    px_m = _px_d.groupby(_px_d.index.to_period('M')).last()
    px_m.index = px_m.index.to_timestamp('M')
    px_m = px_m.where(px_m > 0)

    _liq_d = _c.pivot_table(index=_c.index, columns='ticker',
                            values='value_rub', aggfunc='sum').sort_index()
    liq_m = _liq_d.groupby(_liq_d.index.to_period('M')).sum(min_count=1)
    liq_m.index = liq_m.index.to_timestamp('M')

    ret_m = px_m.pct_change(fill_method=None)

    _bad = int((ret_m <= -1).sum().sum())
    data_status = mo.md(
        f"**Данные:** {px_m.shape[1]} тикеров · {px_m.shape[0]} месяцев "
        f"({px_m.index.min():%Y-%m} — {px_m.index.max():%Y-%m}) · "
        f"цены — `adj_close` (дивиденды + сплиты) · "
        f"месячных доходностей ≤ -100%: {_bad}"
    )
    data_status
    return liq_m, px_m, ret_m


@app.cell(hide_code=True)
def _(mo, moex, pd):
    # Бенчмарк: MCFTR (полная доходность) из кэша, фоллбэк IMOEX
    def _monthly_bench(_name):
        try:
            _i = moex.read_moex_index(_name)
            _i.index = pd.to_datetime(_i.index)
            _s = _i['close'].astype(float).sort_index()
            _pm = _s.groupby(_s.index.to_period('M')).last()
            _pm.index = _pm.index.to_timestamp('M')
            return _pm.pct_change(fill_method=None).rename('bench_ret')
        except Exception:
            return pd.Series(dtype=float)

    bench_ret_m = _monthly_bench('MCFTR')
    bench_name = 'MCFTR'
    if len(bench_ret_m) < 12:
        bench_ret_m = _monthly_bench('IMOEX')
        bench_name = 'IMOEX (ценовой — кэш MCFTR не найден)'

    bench_status = mo.md(
        f"**Бенчмарк:** {bench_name} · {len(bench_ret_m)} месяцев"
        if len(bench_ret_m) else
        "**Бенчмарк недоступен** — выполните `python update_data.py` (шаг 1b)"
    )
    bench_status
    return bench_name, bench_ret_m


@app.cell(hide_code=True)
def _(np, pd):
    # Ядро бэктеста
    def momentum_signal(px, lookback, skip):
        """P(t-skip) / P(t-lookback-skip) - 1"""
        return px.shift(skip) / px.shift(lookback + skip) - 1.0

    def build_rebal_weights(sig, liq, q, long_short, top_n_liq):
        w = pd.DataFrame(0.0, index=sig.index, columns=sig.columns)
        for t in sig.index:
            s = sig.loc[t].dropna()
            if s.empty:
                continue
            if liq is not None and top_n_liq and t in liq.index:
                _l = liq.loc[t].dropna()
                if not _l.empty:
                    _liquid = _l.sort_values(ascending=False).head(top_n_liq).index
                    s = s.loc[s.index.intersection(_liquid)]
                    if s.empty:
                        continue
            s = s.sort_values()
            k = max(1, int(np.floor(len(s) * q)))
            if long_short:
                w.loc[t, s.index[-k:]] = 1.0 / k
                w.loc[t, s.index[:k]] = -1.0 / k
            else:
                w.loc[t, s.index[-k:]] = 1.0 / k
        return w

    def apply_holding(w_rebal, hold):
        """Перекрывающиеся портфели: средний вес hold последних ребалансов"""
        if hold <= 1:
            return w_rebal
        w = pd.DataFrame(0.0, index=w_rebal.index, columns=w_rebal.columns)
        for i in range(hold):
            w = w.add(w_rebal.shift(i), fill_value=0.0)
        return w / hold

    def backtest_monthly(ret, w_m, tc_bps, missing_mode):
        """Доходность месяца t применяется к весам t-1; издержки = tc × turnover.
        Пропавшая цена при открытой позиции: exit → 0%, penalize → -100%."""
        w = w_m.fillna(0.0)
        w_prev = w.shift(1).fillna(0.0)

        _fill = 0.0 if missing_mode == 'exit' else -1.0
        r_eff = ret.copy()
        _mask = r_eff.isna() & (w_prev != 0)
        if _mask.values.any():
            r_eff = r_eff.mask(_mask, _fill)
        r_eff = r_eff.fillna(0.0)
        # защита от артефактов данных (доходность ≤ -100%)
        _bad = (r_eff <= -1) & (w_prev != 0)
        if _bad.values.any():
            r_eff = r_eff.mask(_bad, _fill)

        gross = (w_prev * r_eff).sum(axis=1)
        turnover = 0.5 * (w - w.shift(1).fillna(0.0)).abs().sum(axis=1)
        tc = (tc_bps / 10000.0) * turnover
        return pd.DataFrame({'ret_gross': gross, 'ret_net': gross - tc,
                             'turnover': turnover, 'tc': tc})

    def max_drawdown(eq):
        return float((eq / eq.cummax() - 1.0).min())

    def perf_stats(ret, bench=None, freq=12):
        ret = ret.dropna()
        if ret.empty:
            return {}
        eq = (1 + ret).cumprod()
        _years = len(ret) / freq
        cagr = float(eq.iloc[-1] ** (1 / _years) - 1) if eq.iloc[-1] > 0 and _years > 0 else np.nan
        vol = float(ret.std(ddof=0) * np.sqrt(freq))
        out = {
            'CAGR': cagr,
            'Vol': vol,
            'Sharpe': float(ret.mean() * freq / vol) if vol > 0 else np.nan,
            'MaxDD': max_drawdown(eq),
            'Months': int(len(ret)),
        }
        if bench is not None:
            _al = pd.concat([ret, bench], axis=1).dropna()
            if len(_al):
                _a = _al.iloc[:, 0] - _al.iloc[:, 1]
                _te = float(_a.std(ddof=0) * np.sqrt(freq))
                out['IR'] = float(_a.mean() * freq / _te) if _te > 0 else np.nan
                out['TE'] = _te
        return out

    return apply_holding, backtest_monthly, build_rebal_weights, max_drawdown, momentum_signal, perf_stats


@app.cell(hide_code=True)
def _(mo):
    mo.md("---\n## 1. Одиночная стратегия")
    return


@app.cell(hide_code=True)
def _(mo):
    lookback_slider = mo.ui.slider(start=1, stop=12, step=1, value=6, label="Lookback (мес):")
    skip_slider = mo.ui.slider(start=0, stop=2, step=1, value=1, label="Skip (мес):")
    hold_slider = mo.ui.slider(start=1, stop=6, step=1, value=1, label="Holding (мес):")
    q_dropdown = mo.ui.dropdown(options={"10%": 0.1, "20%": 0.2, "30%": 0.3},
                                value="20%", label="Квантиль отбора:")
    ls_checkbox = mo.ui.checkbox(value=False, label="Long-Short (шорт проигравших)")
    tc_slider = mo.ui.slider(start=0, stop=50, step=5, value=15, label="Издержки (bps за оборот):")
    topn_slider = mo.ui.slider(start=20, stop=100, step=10, value=50,
                               label="Фильтр ликвидности (top-N по обороту):")
    missing_dropdown = mo.ui.dropdown(options={"exit (выход по 0%)": "exit",
                                               "penalize (-100%)": "penalize"},
                                      value="exit (выход по 0%)", label="Пропуск цены:")
    mo.vstack([
        mo.hstack([lookback_slider, skip_slider, hold_slider], justify='start'),
        mo.hstack([q_dropdown, ls_checkbox, tc_slider], justify='start'),
        mo.hstack([topn_slider, missing_dropdown], justify='start'),
    ])
    return (
        hold_slider,
        lookback_slider,
        ls_checkbox,
        missing_dropdown,
        q_dropdown,
        skip_slider,
        tc_slider,
        topn_slider,
    )


@app.cell(hide_code=True)
def _(
    apply_holding,
    backtest_monthly,
    bench_ret_m,
    build_rebal_weights,
    hold_slider,
    liq_m,
    lookback_slider,
    ls_checkbox,
    missing_dropdown,
    momentum_signal,
    perf_stats,
    px_m,
    q_dropdown,
    ret_m,
    skip_slider,
    tc_slider,
    topn_slider,
):
    # Бэктест одиночной стратегии
    _sig = momentum_signal(px_m, lookback_slider.value, skip_slider.value)
    _w0 = build_rebal_weights(_sig, liq_m, q_dropdown.value,
                              ls_checkbox.value, topn_slider.value)
    weights_single = apply_holding(_w0, hold_slider.value)
    bt_single = backtest_monthly(ret_m, weights_single,
                                 tc_bps=tc_slider.value,
                                 missing_mode=missing_dropdown.value)
    # первые lookback+skip месяцев сигнала нет — отбрасываем разогрев
    _warmup = lookback_slider.value + skip_slider.value + 1
    bt_single = bt_single.iloc[_warmup:]

    stats_net = perf_stats(bt_single['ret_net'], bench_ret_m)
    stats_gross = perf_stats(bt_single['ret_gross'])
    stats_bench = perf_stats(bench_ret_m.loc[bench_ret_m.index.intersection(bt_single.index)])
    return bt_single, stats_bench, stats_gross, stats_net, weights_single


@app.cell(hide_code=True)
def _(bench_name, bt_single, mo, stats_bench, stats_gross, stats_net, weights_single):
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
        _w_abs = weights_single.fillna(0.0).abs()
        _avg_pos = float((_w_abs > 0).sum(axis=1).mean())
        _avg_to = float(bt_single['turnover'].mean())
        single_summary = mo.md(
            f"### Результат (net, после издержек)\n\n"
            f"- **CAGR: {_sgn(stats_net['CAGR'])}** | волатильность {stats_net['Vol'] * 100:.0f}% | "
            f"Sharpe **{stats_net['Sharpe']:.2f}** | MaxDD {_sgn(stats_net['MaxDD'])}\n"
            f"- Gross (до издержек): CAGR {_sgn(stats_gross.get('CAGR'))}, "
            f"Sharpe {stats_gross.get('Sharpe', float('nan')):.2f}\n"
            f"- Бенчмарк {bench_name}: CAGR {_sgn(stats_bench.get('CAGR'))}, "
            f"Sharpe {stats_bench.get('Sharpe', float('nan')):.2f}\n"
            f"- Information Ratio: **{stats_net.get('IR', float('nan')):.2f}** "
            f"(tracking error {stats_net.get('TE', float('nan')) * 100:.0f}%)\n"
            f"- Средний месячный оборот: {_avg_to * 100:.0f}% | "
            f"средне позиций: {_avg_pos:.0f} | месяцев: {stats_net['Months']}"
        )
    single_summary
    return


@app.cell(hide_code=True)
def _(bench_name, bench_ret_m, bt_single, go, mo, plotly_available):
    # График капитала: стратегия net/gross против бенчмарка (лог-шкала)
    if not plotly_available or len(bt_single) == 0:
        equity_block = mo.md("")
    else:
        _eq_net = (1 + bt_single['ret_net']).cumprod()
        _eq_gross = (1 + bt_single['ret_gross']).cumprod()
        _b = bench_ret_m.loc[bench_ret_m.index.intersection(bt_single.index)]
        _eq_bench = (1 + _b).cumprod()

        _fig = go.Figure()
        _fig.add_scatter(x=_eq_net.index, y=_eq_net.values, name='Стратегия (net)',
                         line=dict(color='#1f77b4', width=2),
                         hovertemplate='%{y:.2f}<extra>net</extra>')
        _fig.add_scatter(x=_eq_gross.index, y=_eq_gross.values, name='Стратегия (gross)',
                         line=dict(color='#aec7e8', width=1.2, dash='dot'),
                         hovertemplate='%{y:.2f}<extra>gross</extra>')
        _fig.add_scatter(x=_eq_bench.index, y=_eq_bench.values, name=bench_name.split(' ')[0],
                         line=dict(color='#7f7f7f', width=1.5, dash='dash'),
                         hovertemplate='%{y:.2f}<extra>бенчмарк</extra>')
        _fig.update_layout(
            height=420, hovermode='x unified',
            title=dict(text='Рост капитала (1 = старт, лог-шкала)', font_size=14),
            yaxis=dict(type='log'),
            legend=dict(orientation='h', y=1.1, x=1, xanchor='right'),
            margin=dict(t=44, l=10, r=10, b=10),
        )
        equity_block = _fig
    equity_block
    return


@app.cell(hide_code=True)
def _(bt_single, go, mo, plotly_available):
    # Просадки стратегии (net) и месячный оборот
    if not plotly_available or len(bt_single) == 0:
        dd_to_block = mo.md("")
    else:
        _eq = (1 + bt_single['ret_net']).cumprod()
        _dd = (_eq / _eq.cummax() - 1) * 100
        _figd = go.Figure()
        _figd.add_scatter(x=_dd.index, y=_dd.values, fill='tozeroy',
                          line=dict(color='#d62728', width=1),
                          fillcolor='rgba(214,39,40,0.25)',
                          name='просадка',
                          hovertemplate='%{y:.1f}%<extra>DD</extra>')
        _figd.add_bar(x=bt_single.index, y=bt_single['turnover'] * 100,
                      name='оборот', marker_color='rgba(31,119,180,0.4)', yaxis='y2',
                      hovertemplate='%{y:.0f}%<extra>оборот</extra>')
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
                                     label="% месяцев на train:")
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
    missing_dropdown,
    mo,
    momentum_signal,
    np,
    objective_dropdown,
    pd,
    perf_stats,
    px_m,
    ret_m,
    run_grid_button,
    tc_slider,
    topn_slider,
    train_frac_slider,
):
    # Walk-forward grid search (издержки/ликвидность/пропуски — из контролов выше)
    if not run_grid_button.value:
        grid_results = pd.DataFrame()
        grid_best = None
        grid_status = mo.md("Нажмите **«Запустить grid search»** для расчета.")
    elif len(bench_ret_m) < 12:
        grid_results = pd.DataFrame()
        grid_best = None
        grid_status = mo.md("**Бенчмарк недоступен** — walk-forward требует бенчмарк.")
    else:
        _common = ret_m.index.intersection(bench_ret_m.index)
        _ret = ret_m.loc[_common]
        _px = px_m.loc[_common]
        _liq = liq_m.loc[_common] if liq_m is not None else None
        _bench = bench_ret_m.loc[_common]

        _split = int(np.floor(len(_common) * train_frac_slider.value / 100))
        _train_idx = _common[:_split]
        _test_idx = _common[_split:]

        _rows = []
        grid_best = None
        _best_score = -np.inf
        for _lb in (3, 6, 9, 12):
            for _sk in (0, 1):
                for _hd in (1, 3, 6):
                    for _qq in (0.1, 0.2):
                        for _ls in (False, True):
                            _sig_g = momentum_signal(_px, _lb, _sk)
                            _w_g = apply_holding(
                                build_rebal_weights(_sig_g, _liq, _qq, _ls, topn_slider.value),
                                _hd)
                            _bt_g = backtest_monthly(_ret, _w_g,
                                                     tc_bps=tc_slider.value,
                                                     missing_mode=missing_dropdown.value)
                            _st_tr = perf_stats(_bt_g.loc[_train_idx, 'ret_net'],
                                                _bench.loc[_train_idx])
                            _st_te = perf_stats(_bt_g.loc[_test_idx, 'ret_net'],
                                                _bench.loc[_test_idx])
                            _score = _st_tr.get(objective_dropdown.value, np.nan)
                            if _score is None or not np.isfinite(_score):
                                _score = -np.inf
                            _rows.append({
                                'lookback': _lb, 'skip': _sk, 'hold': _hd,
                                'q': _qq, 'long_short': _ls,
                                'train_Sharpe': round(_st_tr.get('Sharpe', np.nan), 2),
                                'train_CAGR_%': round(_st_tr.get('CAGR', np.nan) * 100, 1),
                                'train_IR': round(_st_tr.get('IR', np.nan), 2),
                                'test_Sharpe': round(_st_te.get('Sharpe', np.nan), 2),
                                'test_CAGR_%': round(_st_te.get('CAGR', np.nan) * 100, 1),
                                'test_MaxDD_%': round(_st_te.get('MaxDD', np.nan) * 100, 1),
                                'test_IR': round(_st_te.get('IR', np.nan), 2),
                                'turnover_%': round(float(_bt_g['turnover'].mean()) * 100, 0),
                            })
                            if _score > _best_score:
                                _best_score = _score
                                grid_best = {'params': (_lb, _sk, _hd, _qq, _ls),
                                             'bt': _bt_g, 'test_idx': _test_idx,
                                             'bench': _bench,
                                             'st_tr': _st_tr, 'st_te': _st_te}

        grid_results = pd.DataFrame(_rows).sort_values('train_Sharpe', ascending=False)
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
def _(bench_name, go, grid_best, mo, plotly_available):
    # Out-of-sample: лучшая стратегия против бенчмарка на test-периоде
    if not plotly_available or grid_best is None:
        oos_block = mo.md("")
    else:
        _te_idx = grid_best['test_idx']
        _r_te = grid_best['bt'].loc[_te_idx, 'ret_net']
        _b_te = grid_best['bench'].loc[_te_idx]
        _eq_s = (1 + _r_te).cumprod()
        _eq_b = (1 + _b_te).cumprod()

        _figo = go.Figure()
        _figo.add_scatter(x=_eq_s.index, y=_eq_s.values, name='Стратегия (net)',
                          line=dict(color='#1f77b4', width=2),
                          hovertemplate='%{y:.2f}<extra>стратегия</extra>')
        _figo.add_scatter(x=_eq_b.index, y=_eq_b.values, name=bench_name.split(' ')[0],
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
