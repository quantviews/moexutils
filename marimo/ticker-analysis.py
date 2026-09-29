"""
marimo notebook: Анализ тикера — доходность, риск и сопоставление с рынком (IMOEX)

Использование:
1. Установите зависимости: pip install marimo plotly scipy
2. Запустите ноутбук: marimo edit ticker-analysis.py
3. Выберите тикер и период

Функционал:
- Полная история бумаги: склейка переименований (TCSG→T и т.д.), сплит-коррекция
- Сводка: цена/полная доходность, сравнение с IMOEX, бета, альфа, дивдоходность
- Нормированный график бумаги против индекса, относительная сила
- Скользящие бета и корреляция к IMOEX, просадки, волатильность
- Распределение доходностей (гистограмма + Q-Q plot), объемы — в аккордеоне

Цены: close — закрытие основной сессии (сплит-скорректированный),
adj_close — полная доходность (дивиденды + сплиты). IMOEX — ценовой индекс.
"""

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium", app_title="Анализ тикера", css_file="styles.css")


@app.cell(hide_code=True)
def _():
    # stocks лежит в корне проекта (родительская папка от marimo/)
    import sys as _sys
    import os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    import stocks
    import polars as pl
    import numpy as np
    import marimo as mo
    from scipy import stats
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        plotly_available = True
    except ImportError:
        go, make_subplots = None, None
        plotly_available = False
    return go, make_subplots, mo, np, pl, plotly_available, stats, stocks


@app.cell(hide_code=True)
def _(stocks):
    # Список тикеров: бумаги из хранилища, кроме старых имен переименованных бумаг
    _old_names = set(stocks.load_renames()['old'].to_list())
    available_tickers = [_t for _t in stocks.list_tickers() if _t not in _old_names] or ["SBER"]
    return (available_tickers,)


@app.cell(hide_code=True)
def _(available_tickers, mo):
    ticker = mo.ui.dropdown(
        options=available_tickers,
        value="SBER" if "SBER" in available_tickers else available_tickers[0],
        label="Тикер:",
        searchable=True,
    )
    period_choice = mo.ui.dropdown(
        options={"1 год": "1y", "3 года": "3y", "5 лет": "5y",
                 "С 2022": "2022", "Вся история": "all"},
        value="3 года",
        label="Период:",
    )
    return period_choice, ticker


@app.cell(hide_code=True)
def _(pl, period_choice, stocks, ticker):
    # Полная история бумаги: основной тикер + старые имена (renames.csv),
    # склейка с source_ticker, затем сплит-коррекция цен
    df_full = stocks.read_stocks(ticker.value, split_adjusted=True)
    df_full = df_full.filter(pl.col('ticker') == ticker.value)
    df_full = (df_full.sort('date', maintain_order=True)
               .unique('date', keep='last', maintain_order=True))
    if 'adj_close' not in df_full.columns:
        df_full = df_full.with_columns(pl.col('close').alias('adj_close'))

    # Обрезка по выбранному периоду
    _end = df_full['date'].max()
    if period_choice.value == 'all':
        df = df_full
    elif _end is None:
        df = df_full.clear()
    elif period_choice.value == '2022':
        df = df_full.filter(pl.col('date') >= pl.date(2022, 1, 1))
    else:
        _n_years = {'1y': 1, '3y': 3, '5y': 5}[period_choice.value]
        _start = pl.Series([_end]).dt.offset_by(f'-{_n_years}y')[0]
        df = df_full.filter(pl.col('date') >= _start)
    return df, df_full


@app.cell(hide_code=True)
def _(df, pl, stocks):
    # Индексы за тот же период (хранилище, таблица indexes):
    # IMOEX — ценовой, MCFTR — полной доходности (брутто, с дивидендами).
    # Ряды — polars DataFrame (date, close)
    def _load_idx(_name):
        try:
            _s = (stocks.read_index(_name)
                  .select('date', pl.col('close').cast(pl.Float64))
                  .sort('date'))
            return _s.filter((pl.col('date') >= df['date'].min())
                             & (pl.col('date') <= df['date'].max()))
        except Exception:
            return pl.DataFrame(schema={'date': pl.Date, 'close': pl.Float64})

    index_close = _load_idx('IMOEX')
    mcftr_close = _load_idx('MCFTR')

    # Бенчмарк для полной доходности бумаги: MCFTR; пока данных нет — IMOEX
    if mcftr_close.height > 1:
        bench_close, bench_name = mcftr_close, 'MCFTR'
    else:
        bench_close, bench_name = index_close, 'IMOEX'
    return bench_close, bench_name, index_close, mcftr_close


@app.cell(hide_code=True)
def _(bench_close, df, df_full, index_close, np, pl):
    # Расчет всех метрик за выбранный период.
    # Бета/корреляция/альфа — против бенчмарка полной доходности (MCFTR,
    # при отсутствии данных — IMOEX): обе стороны включают дивиденды.
    # Ряды — polars DataFrame (date, <значение>) без пропусков
    def _valid(_c):
        return pl.col(_c).is_not_null() & pl.col(_c).is_not_nan()

    def _f(_v):
        return float('nan') if _v is None else float(_v)

    price = df.select('date', pl.col('close').cast(pl.Float64).alias('price')).filter(_valid('price'))
    adj = df.select('date', pl.col('adj_close').cast(pl.Float64).alias('adj')).filter(_valid('adj'))
    ret_d = (adj.select('date', (pl.col('adj') / pl.col('adj').shift(1)).log().alias('r'))
             .filter(_valid('r')))
    idx_ret_d = (bench_close.select('date', (pl.col('close') / pl.col('close').shift(1)).log().alias('m'))
                 .filter(_valid('m'))
                 if bench_close.height > 1
                 else pl.DataFrame(schema={'date': pl.Date, 'm': pl.Float64}))

    M = {}
    if price.height > 1 and adj.height > 1:
        _p, _a, _r = price['price'], adj['adj'], ret_d['r']
        _years = max((adj['date'].max() - adj['date'].min()).days / 365.25, 1e-9)
        M['years'] = _years
        M['last_close'] = float(_p[-1])
        M['px_total'] = (float(_p[-1]) / float(_p[0]) - 1) * 100
        M['tr_total'] = (float(_a[-1]) / float(_a[0]) - 1) * 100
        M['cagr_px'] = ((float(_p[-1]) / float(_p[0])) ** (1 / _years) - 1) * 100
        M['cagr_tr'] = ((float(_a[-1]) / float(_a[0])) ** (1 / _years) - 1) * 100
        M['div_yield_ann'] = M['cagr_tr'] - M['cagr_px']

        _std = _f(_r.std())
        M['vol_ann'] = _std * np.sqrt(252) * 100
        _down = _r.filter(_r < 0)
        M['downside_ann'] = (_f(_down.std()) * np.sqrt(252) * 100) if len(_down) > 1 else float('nan')
        M['sharpe'] = (_f(_r.mean()) / _std * np.sqrt(252)) if _std > 0 else float('nan')
        M['var5'] = float(np.percentile(_r.to_numpy(), 5)) * 100
        M['skew'] = _f(_r.skew(bias=False))
        M['kurt'] = _f(_r.kurtosis(fisher=True, bias=False))

        # Просадки — по полной доходности
        _cum = _a / _a[0]
        dd_series = adj.select('date', ((_cum / _cum.cum_max() - 1) * 100).alias('dd'))
        M['max_dd'] = float(dd_series['dd'].min())
        M['dd_now'] = float(dd_series['dd'][-1])

        # 52 недели — по полной истории, не зависит от выбранного периода
        _y = df_full.select('date', pl.col('close').cast(pl.Float64).alias('c')).filter(_valid('c'))
        _y = _y.filter(pl.col('date') >= pl.lit(_y['date'].max()).dt.offset_by('-1y'))['c']
        if len(_y) > 1:
            M['off_52w_high'] = (float(_y[-1]) / float(_y.max()) - 1) * 100
            M['above_52w_low'] = (float(_y[-1]) / float(_y.min()) - 1) * 100

        # Сопоставление с рынком: цена — против IMOEX (ценовой индекс),
        # полная доходность — против бенчмарка (MCFTR / фоллбэк IMOEX)
        if index_close.height > 1:
            _ic = index_close['close']
            M['idx_total'] = (float(_ic[-1]) / float(_ic[0]) - 1) * 100
            M['rel_px'] = M['px_total'] - M['idx_total']
        if bench_close.height > 1 and idx_ret_d.height > 30:
            _bc = bench_close['close']
            M['bench_total'] = (float(_bc[-1]) / float(_bc[0]) - 1) * 100
            M['rel_tr'] = M['tr_total'] - M['bench_total']
            _al = ret_d.join(idx_ret_d, on='date', how='inner')
            _mvar = _f(_al['m'].var()) if _al.height else float('nan')
            if _al.height > 30 and _mvar > 0:
                _st = _al.select(cov=pl.cov('r', 'm', ddof=1), corr=pl.corr('r', 'm'),
                                 s_mean=pl.col('r').mean(), m_mean=pl.col('m').mean()).row(0, named=True)
                M['beta'] = float(_st['cov']) / _mvar
                M['corr'] = float(_st['corr'])
                # альфа (годовая): доходность бумаги минус бета × доходность бенчмарка
                M['alpha_ann'] = (float(_st['s_mean']) - M['beta'] * float(_st['m_mean'])) * 252 * 100
    else:
        dd_series = pl.DataFrame(schema={'date': pl.Date, 'dd': pl.Float64})
    return M, adj, dd_series, idx_ret_d, price, ret_d


@app.cell(hide_code=True)
def _(M, bench_name, df, mo, period_choice, pl, ticker):
    # Сводка
    def _sgn(_v, _suffix='%', _nd=1):
        if _v != _v:  # NaN
            return 'н/д'
        _cls = 'pos' if _v >= 0 else 'neg'
        _txt = format(_v, f'+,.{_nd}f').replace(',', ' ')
        return f'<span class="{_cls}">{_txt}{_suffix}</span>'

    if not M:
        summary_md = mo.md("Нет данных за выбранный период")
    else:
        _src = ""
        if 'source_ticker' in df.columns and df['source_ticker'].drop_nulls().n_unique() > 1:
            _src = (" · история склеена из: "
                    + " → ".join(df.sort('date', maintain_order=True)['source_ticker']
                                 .drop_nulls().unique(maintain_order=True).to_list()))
        _vs_parts = []
        if 'idx_total' in M:
            _vs_parts.append(f"цена vs IMOEX ({_sgn(M['idx_total'])}): {_sgn(M['rel_px'])}")
        if 'bench_total' in M:
            _vs_parts.append(f"полная vs {bench_name} ({_sgn(M['bench_total'])}): {_sgn(M['rel_tr'])}")
        if 'beta' in M:
            _vs_parts.append(f"бета **{M['beta']:.2f}**")
            _vs_parts.append(f"альфа {_sgn(M['alpha_ann'])} годовых")
        _vs = ("- **Против рынка:** " + " | ".join(_vs_parts) + "\n") if _vs_parts else ""
        _levels = ""
        if 'off_52w_high' in M:
            _levels = (f"- **Уровни (52 нед.):** от максимума {_sgn(M['off_52w_high'])}, "
                       f"от минимума {_sgn(M['above_52w_low'])}; текущая просадка {_sgn(M['dd_now'])}\n")

        _period_labels = {'1y': '1 год', '3y': '3 года', '5y': '5 лет',
                          '2022': 'с 2022', 'all': 'вся история'}
        summary_md = mo.md(
            f"## {ticker.value} — {M['last_close']:,.2f} руб".replace(',', ' ')
            + f" · период: {_period_labels.get(period_choice.value, period_choice.value)}"
            + f" ({df['date'].min().strftime('%d.%m.%Y')} — {df['date'].max().strftime('%d.%m.%Y')}){_src}\n\n"
            + f"- **Цена:** {_sgn(M['px_total'])} за период (CAGR {_sgn(M['cagr_px'])}) | "
            + f"**полная доходность:** {_sgn(M['tr_total'])} (CAGR {_sgn(M['cagr_tr'])}) | "
            + f"дивиденды ≈ {_sgn(M['div_yield_ann'])} годовых\n"
            + _vs + _levels
        )
    return (summary_md,)


@app.cell(hide_code=True)
def _(M, adj, go, index_close, mcftr_close, mo, plotly_available, price, ticker):
    # Нормированный график, старт = 100: цена ↔ IMOEX, полная доходность ↔ MCFTR
    if not plotly_available or not M:
        block_overview = mo.md("")
    else:
        _figo = go.Figure()
        _figo.add_scatter(x=price['date'].to_list(),
                          y=(price['price'] / price['price'][0] * 100).to_numpy(),
                          name=f'{ticker.value} (цена)',
                          line=dict(color='#1f77b4', width=1.8),
                          hovertemplate='%{y:.1f}<extra>цена</extra>')
        _figo.add_scatter(x=adj['date'].to_list(),
                          y=(adj['adj'] / adj['adj'][0] * 100).to_numpy(),
                          name=f'{ticker.value} (полная доходность)',
                          line=dict(color='#2ca02c', width=1.6),
                          hovertemplate='%{y:.1f}<extra>полная дох.</extra>')
        if index_close.height > 1:
            _figo.add_scatter(x=index_close['date'].to_list(),
                              y=(index_close['close'] / index_close['close'][0] * 100).to_numpy(),
                              name='IMOEX (цена)',
                              line=dict(color='#7f7f7f', width=1.4, dash='dot'),
                              hovertemplate='%{y:.1f}<extra>IMOEX</extra>')
        if mcftr_close.height > 1:
            _figo.add_scatter(x=mcftr_close['date'].to_list(),
                              y=(mcftr_close['close'] / mcftr_close['close'][0] * 100).to_numpy(),
                              name='MCFTR (полная дох.)',
                              line=dict(color='#8c564b', width=1.4, dash='dash'),
                              hovertemplate='%{y:.1f}<extra>MCFTR</extra>')
        _figo.add_hline(y=100, line_color='black', line_width=0.7)
        _figo.update_layout(
            height=420, hovermode='x unified',
            title=dict(text='Динамика, старт периода = 100', font_size=14),
            legend=dict(orientation='h', y=1.1, x=1, xanchor='right'),
            margin=dict(t=44, l=10, r=10, b=10),
        )
        block_overview = _figo
    return (block_overview,)


@app.cell(hide_code=True)
def _(M, adj, bench_close, bench_name, go, mo, pl, plotly_available, ticker):
    # Относительная сила: полная доходность бумаги / бенчмарк полной доходности
    # (нормировано, >100 — обгоняет рынок)
    if not plotly_available or not M or bench_close.height < 2:
        block_rs = mo.md("*Относительная сила: нет данных индекса за период*") if M else mo.md("")
    else:
        _joint = (adj.join(bench_close, on='date', how='inner')
                  .filter(pl.col('close').is_not_null() & pl.col('close').is_not_nan()))
        _rs = (_joint['adj'] / _joint['adj'][0]) / (_joint['close'] / _joint['close'][0]) * 100
        _figr = go.Figure()
        _figr.add_scatter(x=_joint['date'].to_list(), y=_rs.to_numpy(), name='RS',
                          line=dict(color='#9467bd', width=1.8),
                          hovertemplate='%{y:.1f}<extra>RS</extra>')
        _figr.add_hline(y=100, line_dash='dash', line_color='gray', line_width=1)
        _figr.update_layout(
            height=240,
            title=dict(text=f'Относительная сила {ticker.value} / {bench_name} '
                            f'(полная доходность; выше 100 — обгоняет рынок)', font_size=13),
            margin=dict(t=40, l=10, r=10, b=10),
        )
        block_rs = _figr
    return (block_rs,)


@app.cell(hide_code=True)
def _(M, bench_name, go, idx_ret_d, mo, pl, plotly_available, ret_d):
    # Скользящие бета и корреляция к бенчмарку (окно 126 торговых дней ≈ полгода)
    if not plotly_available or not M or idx_ret_d.height < 150:
        block_beta = mo.md("")
    else:
        _al2 = ret_d.join(idx_ret_d, on='date', how='inner').select(
            'date',
            (pl.rolling_cov('r', 'm', window_size=126, ddof=1)
             / pl.col('m').rolling_var(126, ddof=1)).alias('beta'),
            pl.rolling_corr('r', 'm', window_size=126).alias('corr'),
        )
        _dates2 = _al2['date'].to_list()
        _figb2 = go.Figure()
        _figb2.add_scatter(x=_dates2, y=_al2['beta'].to_numpy(), name='бета (126д)',
                           line=dict(color='#d62728', width=1.6),
                           hovertemplate='%{y:.2f}<extra>бета</extra>')
        _figb2.add_scatter(x=_dates2, y=_al2['corr'].to_numpy(), name='корреляция (126д)',
                           line=dict(color='#1f77b4', width=1.2, dash='dot'),
                           hovertemplate='%{y:.2f}<extra>корреляция</extra>')
        _figb2.add_hline(y=1, line_dash='dash', line_color='gray', line_width=0.8)
        _figb2.update_layout(
            height=260, hovermode='x unified',
            title=dict(text=f'Скользящие бета и корреляция к {bench_name}', font_size=13),
            legend=dict(orientation='h', y=1.15, x=1, xanchor='right'),
            margin=dict(t=44, l=10, r=10, b=10),
        )
        block_beta = _figb2
    return (block_beta,)


@app.cell(hide_code=True)
def _(M, dd_series, go, mo, plotly_available, ticker):
    # Просадки по полной доходности
    if not plotly_available or not M or dd_series.height == 0:
        block_dd = mo.md("")
    else:
        _figd = go.Figure()
        _figd.add_scatter(x=dd_series['date'].to_list(), y=dd_series['dd'].to_numpy(), fill='tozeroy',
                          line=dict(color='#d62728', width=1.2),
                          fillcolor='rgba(214,39,40,0.25)',
                          hovertemplate='%{y:.1f}%<extra>просадка</extra>')
        _dd_min = float(dd_series['dd'].min())
        _dd_min_date = dd_series['date'][dd_series['dd'].arg_min()]
        _figd.add_annotation(x=_dd_min_date, y=_dd_min,
                             text=f"max DD {_dd_min:.1f}%",
                             showarrow=True, arrowhead=1, yshift=-4)
        _figd.update_layout(
            height=260,
            title=dict(text=f'Просадки {ticker.value} (по полной доходности)', font_size=13),
            yaxis=dict(ticksuffix='%'),
            margin=dict(t=40, l=10, r=10, b=10),
        )
        block_dd = _figd
    return (block_dd,)


@app.cell(hide_code=True)
def _(M, bench_name, mo):
    # Метрики тремя колонками
    def _sgn(_v, _suffix='%', _nd=1):
        if _v != _v:
            return 'н/д'
        _cls = 'pos' if _v >= 0 else 'neg'
        _txt = format(_v, f'+,.{_nd}f').replace(',', ' ')
        return f'<span class="{_cls}">{_txt}{_suffix}</span>'

    if not M:
        metrics_md = mo.md("")
    else:
        _c1 = mo.md(
            "**Доходность**\n\n"
            f"- Цена за период: {_sgn(M['px_total'])}\n"
            f"- Полная за период: {_sgn(M['tr_total'])}\n"
            f"- CAGR (цена): {_sgn(M['cagr_px'])}\n"
            f"- CAGR (полная): {_sgn(M['cagr_tr'])}\n"
            f"- Дивиденды (годовых): {_sgn(M['div_yield_ann'])}"
        )
        _c2 = mo.md(
            "**Риск**\n\n"
            f"- Волатильность: {M['vol_ann']:.0f}%\n"
            f"- Downside-вола: {M['downside_ann']:.0f}%\n"
            f"- Max drawdown: {_sgn(M['max_dd'])}\n"
            f"- VaR 5% (день): {_sgn(M['var5'], '%', 2)}\n"
            f"- Sharpe (rf=0): {M['sharpe']:.2f}"
        )
        if 'beta' in M:
            _c3 = mo.md(
                "**Против рынка**\n\n"
                f"- IMOEX (цена): {_sgn(M.get('idx_total', float('nan')))}\n"
                f"- {bench_name} (полная): {_sgn(M.get('bench_total', float('nan')))}\n"
                f"- Опережение (полная): {_sgn(M.get('rel_tr', float('nan')))}\n"
                f"- Бета к {bench_name}: {M['beta']:.2f} (корр. {M['corr']:.2f})\n"
                f"- Альфа (годовых): {_sgn(M['alpha_ann'])}"
            )
        else:
            _c3 = mo.md("**Против рынка**\n\nнет данных индексов — обновите кэш (`python update_data.py`)")
        metrics_md = mo.hstack([_c1, _c2, _c3], justify='start', gap=3)
    return (metrics_md,)


@app.cell(hide_code=True)
def _(M, make_subplots, mo, np, plotly_available, ret_d, stats):
    # Распределение дневных доходностей: гистограмма + нормальная кривая, Q-Q plot
    if not plotly_available or not M or ret_d.height < 30:
        block_dist = mo.md("")
    else:
        _r = ret_d['r'].to_numpy() * 100
        _figh = make_subplots(rows=1, cols=2,
                              subplot_titles=('Распределение дневных доходностей', 'Q-Q plot'))
        _figh.add_histogram(x=_r, nbinsx=60, name='доходности',
                            marker_color='#1f77b4', opacity=0.75, row=1, col=1)
        _xs = np.linspace(float(_r.min()), float(_r.max()), 200)
        _pdf = stats.norm.pdf(_xs, float(_r.mean()), float(_r.std(ddof=1)))
        _binw = (float(_r.max()) - float(_r.min())) / 60
        _figh.add_scatter(x=_xs, y=_pdf * len(_r) * _binw, name='нормальное',
                          line=dict(color='#d62728', width=1.6), row=1, col=1)
        _figh.add_vline(x=M['var5'], line_dash='dash', line_color='black',
                        annotation_text=f"VaR5 {M['var5']:.1f}%", row=1, col=1)

        (_osm, _osr), (_sl, _ic, _rq) = stats.probplot(ret_d['r'].to_numpy(), dist='norm')
        _figh.add_scatter(x=_osm, y=_osr * 100, mode='markers', name='квантили',
                          marker=dict(size=3, color='#1f77b4'), row=1, col=2)
        _figh.add_scatter(x=_osm, y=(_sl * _osm + _ic) * 100, mode='lines', name='норм. линия',
                          line=dict(color='#d62728', width=1.4), row=1, col=2)
        _figh.update_layout(
            height=340, showlegend=False,
            title=dict(text=f"Асимметрия {M['skew']:.2f} · эксцесс {M['kurt']:.1f} "
                            f"(у нормального 0)", font_size=12),
            margin=dict(t=64, l=10, r=10, b=10),
        )
        block_dist = _figh
    return (block_dist,)


@app.cell(hide_code=True)
def _(M, go, mo, np, pl, plotly_available, ret_d):
    # Скользящая годовая волатильность
    if not plotly_available or not M or ret_d.height < 60:
        block_vol = mo.md("")
    else:
        _figv = go.Figure()
        _dates_v = ret_d['date'].to_list()
        for _w, _cl in ((30, '#1f77b4'), (90, '#ff7f0e'), (252, '#2ca02c')):
            if ret_d.height > _w:
                _rv = ret_d.select(pl.col('r').rolling_std(_w, ddof=1) * np.sqrt(252) * 100)['r']
                _figv.add_scatter(x=_dates_v, y=_rv.to_numpy(), name=f'{_w} дней',
                                  line=dict(color=_cl, width=1.4),
                                  hovertemplate='%{y:.0f}%<extra>' + f'{_w}д</extra>')
        _figv.update_layout(
            height=280, hovermode='x unified',
            title=dict(text='Скользящая годовая волатильность', font_size=13),
            yaxis=dict(ticksuffix='%'),
            legend=dict(orientation='h', y=1.12, x=1, xanchor='right'),
            margin=dict(t=42, l=10, r=10, b=10),
        )
        block_vol = _figv
    return (block_vol,)


@app.cell(hide_code=True)
def _(M, df, go, mo, pl, plotly_available):
    # Объем торгов (оборот, млн руб) со средним за 20 дней
    if not plotly_available or not M or 'value_rub' not in df.columns:
        block_volume = mo.md("")
    else:
        _val = (df.select('date', (pl.col('value_rub').cast(pl.Float64) / 1e6).alias('v'))
                .filter(pl.col('v').is_not_null() & pl.col('v').is_not_nan())
                .with_columns(ma=pl.col('v').rolling_mean(20)))
        _dates_val = _val['date'].to_list()
        _figvol = go.Figure()
        _figvol.add_bar(x=_dates_val, y=_val['v'].to_numpy(), name='оборот/день',
                        marker_color='rgba(31,119,180,0.45)',
                        hovertemplate='%{y:,.0f} млн<extra></extra>')
        _figvol.add_scatter(x=_dates_val, y=_val['ma'].to_numpy(), name='среднее 20д',
                            line=dict(color='#d62728', width=1.5),
                            hovertemplate='%{y:,.0f} млн<extra>MA20</extra>')
        _figvol.update_layout(
            height=280,
            title=dict(text='Оборот торгов, млн руб', font_size=13),
            legend=dict(orientation='h', y=1.12, x=1, xanchor='right'),
            margin=dict(t=42, l=10, r=10, b=10),
        )
        block_volume = _figvol
    return (block_volume,)


@app.cell(hide_code=True)
def _(
    block_beta,
    block_dd,
    block_dist,
    block_overview,
    block_rs,
    block_vol,
    block_volume,
    metrics_md,
    mo,
    period_choice,
    summary_md,
    ticker,
):
    # Основной layout
    mo.vstack([
        mo.hstack([ticker, period_choice], justify='start'),
        summary_md,
        block_overview,
        block_rs,
        block_beta,
        metrics_md,
        block_dd,
        mo.accordion({
            "📊 Распределение доходностей": block_dist,
            "📈 Скользящая волатильность": block_vol,
            "💹 Оборот торгов": block_volume,
        }),
    ])
    return


if __name__ == "__main__":
    app.run()
