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
    # moex_utils лежит в корне проекта (родительская папка от marimo/)
    import sys as _sys
    import os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    import moex_utils as moex
    import pandas as pd
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
    return go, make_subplots, mo, moex, np, pd, plotly_available, stats


@app.cell(hide_code=True)
def _(moex):
    # Список тикеров: файлы из data/, кроме старых имен переименованных бумаг
    import os as _os
    _renames_df = moex.load_renames()
    _old_names = set(_renames_df['old']) if len(_renames_df) else set()
    if _os.path.exists(moex.DATA_FOLDER):
        available_tickers = sorted(
            _d for _d in _os.listdir(moex.DATA_FOLDER)
            if _os.path.isdir(_os.path.join(moex.DATA_FOLDER, _d))
            and _os.path.exists(_os.path.join(moex.DATA_FOLDER, _d, f"{_d}.parquet"))
            and _d not in _old_names
        )
    else:
        available_tickers = ["SBER"]
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
def _(moex, pd, period_choice, ticker):
    # Полная история бумаги: основной файл + файлы старых имен (renames.csv),
    # склейка с source_ticker, затем сплит-коррекция цен
    _frames = [moex.read_moex_stock(ticker.value)]
    _ren = moex.load_renames()
    if len(_ren):
        for _old in _ren.loc[_ren['new'] == ticker.value, 'old']:
            try:
                _frames.append(moex.read_moex_stock(str(_old)))
            except Exception:
                pass

    df_full = pd.concat(_frames)
    if not isinstance(df_full.index, pd.DatetimeIndex):
        df_full.index = pd.to_datetime(df_full.index)
    df_full = moex.apply_renames(df_full)
    df_full = df_full[df_full['ticker'] == ticker.value]
    df_full = moex.adjust_for_splits(df_full).sort_index()
    df_full = df_full[~df_full.index.duplicated(keep='last')]
    if 'adj_close' not in df_full.columns:
        df_full['adj_close'] = df_full['close']

    # Обрезка по выбранному периоду
    _end = df_full.index.max()
    if period_choice.value == 'all':
        df = df_full
    elif period_choice.value == '2022':
        df = df_full[df_full.index >= pd.Timestamp('2022-01-01')]
    else:
        _n_years = {'1y': 1, '3y': 3, '5y': 5}[period_choice.value]
        df = df_full[df_full.index >= _end - pd.DateOffset(years=_n_years)]
    return df, df_full


@app.cell(hide_code=True)
def _(df, moex, pd):
    # IMOEX за тот же период (локальный кэш indexes/IMOEX.parquet)
    try:
        _idx = moex.read_moex_index('IMOEX')
        _idx.index = pd.to_datetime(_idx.index)
        index_close = _idx['close'].astype(float).sort_index()
        index_close = index_close[(index_close.index >= df.index.min()) &
                                  (index_close.index <= df.index.max())]
    except Exception:
        index_close = pd.Series(dtype=float)
    return (index_close,)


@app.cell(hide_code=True)
def _(df, df_full, index_close, np, pd):
    # Расчет всех метрик за выбранный период
    price = df['close'].astype(float).dropna()
    adj = df['adj_close'].astype(float).dropna()
    ret_d = np.log(adj / adj.shift(1)).dropna()
    idx_ret_d = (np.log(index_close / index_close.shift(1)).dropna()
                 if len(index_close) > 1 else pd.Series(dtype=float))

    M = {}
    if len(price) > 1 and len(adj) > 1:
        _years = max((adj.index.max() - adj.index.min()).days / 365.25, 1e-9)
        M['years'] = _years
        M['last_close'] = float(price.iloc[-1])
        M['px_total'] = (float(price.iloc[-1]) / float(price.iloc[0]) - 1) * 100
        M['tr_total'] = (float(adj.iloc[-1]) / float(adj.iloc[0]) - 1) * 100
        M['cagr_px'] = ((float(price.iloc[-1]) / float(price.iloc[0])) ** (1 / _years) - 1) * 100
        M['cagr_tr'] = ((float(adj.iloc[-1]) / float(adj.iloc[0])) ** (1 / _years) - 1) * 100
        M['div_yield_ann'] = M['cagr_tr'] - M['cagr_px']

        M['vol_ann'] = float(ret_d.std()) * np.sqrt(252) * 100
        _down = ret_d[ret_d < 0]
        M['downside_ann'] = (float(_down.std()) * np.sqrt(252) * 100) if len(_down) > 1 else float('nan')
        M['sharpe'] = (float(ret_d.mean()) / float(ret_d.std()) * np.sqrt(252)) if float(ret_d.std()) > 0 else float('nan')
        M['var5'] = float(np.percentile(ret_d, 5)) * 100
        M['skew'] = float(ret_d.skew())
        M['kurt'] = float(ret_d.kurt())

        # Просадки — по полной доходности
        _cum = adj / adj.iloc[0]
        dd_series = (_cum / _cum.cummax() - 1) * 100
        M['max_dd'] = float(dd_series.min())
        M['dd_now'] = float(dd_series.iloc[-1])

        # 52 недели — по полной истории, не зависит от выбранного периода
        _y = df_full['close'].astype(float).dropna()
        _y = _y[_y.index >= _y.index.max() - pd.DateOffset(years=1)]
        if len(_y) > 1:
            M['off_52w_high'] = (float(_y.iloc[-1]) / float(_y.max()) - 1) * 100
            M['above_52w_low'] = (float(_y.iloc[-1]) / float(_y.min()) - 1) * 100

        # Сопоставление с рынком (IMOEX — ценовой индекс)
        if len(index_close) > 1 and len(idx_ret_d) > 30:
            M['idx_total'] = (float(index_close.iloc[-1]) / float(index_close.iloc[0]) - 1) * 100
            M['rel_px'] = M['px_total'] - M['idx_total']
            M['rel_tr'] = M['tr_total'] - M['idx_total']
            _al = pd.concat([ret_d.rename('s'), idx_ret_d.rename('m')], axis=1, join='inner').dropna()
            if len(_al) > 30 and float(_al['m'].var()) > 0:
                M['beta'] = float(_al['s'].cov(_al['m'])) / float(_al['m'].var())
                M['corr'] = float(_al['s'].corr(_al['m']))
                # альфа (годовая): доходность бумаги минус бета × доходность рынка
                M['alpha_ann'] = (float(_al['s'].mean()) - M['beta'] * float(_al['m'].mean())) * 252 * 100
    else:
        dd_series = pd.Series(dtype=float)
    return M, adj, dd_series, idx_ret_d, price, ret_d


@app.cell(hide_code=True)
def _(M, df, mo, period_choice, ticker):
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
        if 'source_ticker' in df.columns and df['source_ticker'].nunique() > 1:
            _src = (" · история склеена из: "
                    + " → ".join(df.sort_index()['source_ticker'].unique()))
        _vs = ""
        if 'idx_total' in M:
            _vs = (f"- **Против IMOEX** ({_sgn(M['idx_total'])}): цена {_sgn(M['rel_px'])} "
                   f"| полная доходность {_sgn(M['rel_tr'])}"
                   + (f" | бета **{M['beta']:.2f}** | альфа {_sgn(M['alpha_ann'])} годовых"
                      if 'beta' in M else "") + "\n")
        _levels = ""
        if 'off_52w_high' in M:
            _levels = (f"- **Уровни (52 нед.):** от максимума {_sgn(M['off_52w_high'])}, "
                       f"от минимума {_sgn(M['above_52w_low'])}; текущая просадка {_sgn(M['dd_now'])}\n")

        _period_labels = {'1y': '1 год', '3y': '3 года', '5y': '5 лет',
                          '2022': 'с 2022', 'all': 'вся история'}
        summary_md = mo.md(
            f"## {ticker.value} — {M['last_close']:,.2f} руб".replace(',', ' ')
            + f" · период: {_period_labels.get(period_choice.value, period_choice.value)}"
            + f" ({df.index.min().strftime('%d.%m.%Y')} — {df.index.max().strftime('%d.%m.%Y')}){_src}\n\n"
            + f"- **Цена:** {_sgn(M['px_total'])} за период (CAGR {_sgn(M['cagr_px'])}) | "
            + f"**полная доходность:** {_sgn(M['tr_total'])} (CAGR {_sgn(M['cagr_tr'])}) | "
            + f"дивиденды ≈ {_sgn(M['div_yield_ann'])} годовых\n"
            + _vs + _levels
        )
    return (summary_md,)


@app.cell(hide_code=True)
def _(M, adj, go, index_close, mo, plotly_available, price, ticker):
    # Нормированный график: бумага (цена и полная доходность) против IMOEX, старт = 100
    if not plotly_available or not M:
        block_overview = mo.md("")
    else:
        _figo = go.Figure()
        _figo.add_scatter(x=price.index, y=price / price.iloc[0] * 100,
                          name=f'{ticker.value} (цена)',
                          line=dict(color='#1f77b4', width=1.8),
                          hovertemplate='%{y:.1f}<extra>цена</extra>')
        _figo.add_scatter(x=adj.index, y=adj / adj.iloc[0] * 100,
                          name=f'{ticker.value} (полная доходность)',
                          line=dict(color='#2ca02c', width=1.6),
                          hovertemplate='%{y:.1f}<extra>полная дох.</extra>')
        if len(index_close) > 1:
            _figo.add_scatter(x=index_close.index, y=index_close / index_close.iloc[0] * 100,
                              name='IMOEX', line=dict(color='#7f7f7f', width=1.4, dash='dot'),
                              hovertemplate='%{y:.1f}<extra>IMOEX</extra>')
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
def _(M, go, index_close, mo, plotly_available, price, ticker):
    # Относительная сила: цена бумаги / IMOEX (нормировано, >100 — обгоняет рынок)
    if not plotly_available or not M or len(index_close) < 2:
        block_rs = mo.md("*Относительная сила: нет данных IMOEX за период*") if M else mo.md("")
    else:
        _joint = price.to_frame('p').join(index_close.to_frame('i'), how='inner').dropna()
        _rs = (_joint['p'] / _joint['p'].iloc[0]) / (_joint['i'] / _joint['i'].iloc[0]) * 100
        _figr = go.Figure()
        _figr.add_scatter(x=_rs.index, y=_rs.values, name='RS',
                          line=dict(color='#9467bd', width=1.8),
                          hovertemplate='%{y:.1f}<extra>RS</extra>')
        _figr.add_hline(y=100, line_dash='dash', line_color='gray', line_width=1)
        _figr.update_layout(
            height=240,
            title=dict(text=f'Относительная сила {ticker.value} / IMOEX '
                            f'(выше 100 — обгоняет рынок)', font_size=13),
            margin=dict(t=40, l=10, r=10, b=10),
        )
        block_rs = _figr
    return (block_rs,)


@app.cell(hide_code=True)
def _(M, go, idx_ret_d, mo, pd, plotly_available, ret_d):
    # Скользящие бета и корреляция к IMOEX (окно 126 торговых дней ≈ полгода)
    if not plotly_available or not M or len(idx_ret_d) < 150:
        block_beta = mo.md("")
    else:
        _al2 = pd.concat([ret_d.rename('s'), idx_ret_d.rename('m')], axis=1, join='inner').dropna()
        _rbeta = _al2['s'].rolling(126).cov(_al2['m']) / _al2['m'].rolling(126).var()
        _rcorr = _al2['s'].rolling(126).corr(_al2['m'])
        _figb2 = go.Figure()
        _figb2.add_scatter(x=_rbeta.index, y=_rbeta.values, name='бета (126д)',
                           line=dict(color='#d62728', width=1.6),
                           hovertemplate='%{y:.2f}<extra>бета</extra>')
        _figb2.add_scatter(x=_rcorr.index, y=_rcorr.values, name='корреляция (126д)',
                           line=dict(color='#1f77b4', width=1.2, dash='dot'),
                           hovertemplate='%{y:.2f}<extra>корреляция</extra>')
        _figb2.add_hline(y=1, line_dash='dash', line_color='gray', line_width=0.8)
        _figb2.update_layout(
            height=260, hovermode='x unified',
            title=dict(text='Скользящие бета и корреляция к IMOEX', font_size=13),
            legend=dict(orientation='h', y=1.15, x=1, xanchor='right'),
            margin=dict(t=44, l=10, r=10, b=10),
        )
        block_beta = _figb2
    return (block_beta,)


@app.cell(hide_code=True)
def _(M, dd_series, go, mo, plotly_available, ticker):
    # Просадки по полной доходности
    if not plotly_available or not M or len(dd_series) == 0:
        block_dd = mo.md("")
    else:
        _figd = go.Figure()
        _figd.add_scatter(x=dd_series.index, y=dd_series.values, fill='tozeroy',
                          line=dict(color='#d62728', width=1.2),
                          fillcolor='rgba(214,39,40,0.25)',
                          hovertemplate='%{y:.1f}%<extra>просадка</extra>')
        _dd_min_date = dd_series.idxmin()
        _figd.add_annotation(x=_dd_min_date, y=float(dd_series.min()),
                             text=f"max DD {dd_series.min():.1f}%",
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
def _(M, mo):
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
                f"- IMOEX за период: {_sgn(M.get('idx_total', float('nan')))}\n"
                f"- Отставание/опережение (цена): {_sgn(M['rel_px'])}\n"
                f"- Бета: {M['beta']:.2f}\n"
                f"- Корреляция: {M['corr']:.2f}\n"
                f"- Альфа (годовых): {_sgn(M['alpha_ann'])}"
            )
        else:
            _c3 = mo.md("**Против рынка**\n\nнет данных IMOEX")
        metrics_md = mo.hstack([_c1, _c2, _c3], justify='start', gap=3)
    return (metrics_md,)


@app.cell(hide_code=True)
def _(M, make_subplots, mo, np, plotly_available, ret_d, stats):
    # Распределение дневных доходностей: гистограмма + нормальная кривая, Q-Q plot
    if not plotly_available or not M or len(ret_d) < 30:
        block_dist = mo.md("")
    else:
        _r = ret_d * 100
        _figh = make_subplots(rows=1, cols=2,
                              subplot_titles=('Распределение дневных доходностей', 'Q-Q plot'))
        _figh.add_histogram(x=_r, nbinsx=60, name='доходности',
                            marker_color='#1f77b4', opacity=0.75, row=1, col=1)
        _xs = np.linspace(float(_r.min()), float(_r.max()), 200)
        _pdf = stats.norm.pdf(_xs, float(_r.mean()), float(_r.std()))
        _binw = (float(_r.max()) - float(_r.min())) / 60
        _figh.add_scatter(x=_xs, y=_pdf * len(_r) * _binw, name='нормальное',
                          line=dict(color='#d62728', width=1.6), row=1, col=1)
        _figh.add_vline(x=M['var5'], line_dash='dash', line_color='black',
                        annotation_text=f"VaR5 {M['var5']:.1f}%", row=1, col=1)

        (_osm, _osr), (_sl, _ic, _rq) = stats.probplot(ret_d, dist='norm')
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
def _(M, go, mo, np, plotly_available, ret_d):
    # Скользящая годовая волатильность
    if not plotly_available or not M or len(ret_d) < 60:
        block_vol = mo.md("")
    else:
        _figv = go.Figure()
        for _w, _cl in ((30, '#1f77b4'), (90, '#ff7f0e'), (252, '#2ca02c')):
            if len(ret_d) > _w:
                _rv = ret_d.rolling(_w).std() * np.sqrt(252) * 100
                _figv.add_scatter(x=_rv.index, y=_rv.values, name=f'{_w} дней',
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
def _(M, df, go, mo, plotly_available):
    # Объем торгов (оборот, млн руб) со средним за 20 дней
    if not plotly_available or not M or 'value_rub' not in df.columns:
        block_volume = mo.md("")
    else:
        _val = df['value_rub'].astype(float).dropna() / 1e6
        _figvol = go.Figure()
        _figvol.add_bar(x=_val.index, y=_val.values, name='оборот/день',
                        marker_color='rgba(31,119,180,0.45)',
                        hovertemplate='%{y:,.0f} млн<extra></extra>')
        _ma = _val.rolling(20).mean()
        _figvol.add_scatter(x=_ma.index, y=_ma.values, name='среднее 20д',
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
