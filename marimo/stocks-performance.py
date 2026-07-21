"""
# Stocks Performance Analysis

Интерактивный анализ performance акций с учетом market cap.
"""

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium", app_title="Обзор фондового рынка РФ", css_file="styles.css")


@app.cell(hide_code=True)
def _():
    # moex_utils лежит в корне проекта (родительская папка от marimo/)
    import sys as _sys
    import os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    import moex_utils as moex
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime, timedelta
    import marimo as mo
    import io
    import base64
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        plotly_available = True
    except ImportError:
        px, go = None, None
        plotly_available = False

    return base64, go, io, mo, moex, np, pd, plotly_available, plt, px


@app.cell(hide_code=True)
def _(mo):
    # UI элементы для выбора периода
    period_dropdown = mo.ui.dropdown(
        options={
            "1 день": "1d",
            "1 неделя": "1w",
            "2 недели": "2w",
            "1 месяц": "1m",
            "3 месяца": "3m",
            "6 месяцев": "6m",
            "1 год": "1y",
            "С начала года (YTD)": "ytd",
        },
        value="1 неделя",
        label="Период:",
    )

    show_market_cap = mo.ui.checkbox(
        value=True,
        label="Показывать анализ по market cap"
    )

    sort_by = mo.ui.dropdown(
        options={
            "Изм. цены": "price_performance",
            "Изм. капитализации (%)": "market_cap_performance",
            "Изм. капитализации (млрд)": "market_cap_change",
            "Тикер": "ticker",
        },
        value="Изм. цены",
        label="Сортировка таблицы:"
    )

    min_market_cap = mo.ui.number(
        start=0,
        stop=1e15,
        step=1e9,
        value=0,
        label="Мин. market cap (млрд руб):"
    )
    return min_market_cap, period_dropdown, show_market_cap, sort_by


@app.cell
def _(moex):
    # Загружаем данные; цены приводим к пост-сплитовой базе (metadata/splits.csv),
    # иначе дробления акций (T 1:10 в 2026 и др.) выглядят как обвал цены
    combined_df = moex.adjust_for_splits(moex.combine_moex_stocks())
    return (combined_df,)


@app.cell(hide_code=True)
def _(moex, pd):
    # Справочник тикер→сектор (используется картой рынка, секторным разрезом и структурой)
    import os as _oss
    _sec_path = _oss.path.join(moex.BASE_DIR, 'metadata', 'sectors.csv')
    if _oss.path.exists(_sec_path):
        sectors_map = pd.read_csv(_sec_path)
    else:
        sectors_map = pd.DataFrame(columns=['ticker', 'sector'])
    return (sectors_map,)


@app.cell(hide_code=True)
def _(combined_df, pd, period_dropdown):
    # Метки периодов для заголовков
    PERIOD_LABELS = {
        "1d": "1 день",
        "1w": "1 неделя",
        "2w": "2 недели",
        "1m": "1 месяц",
        "3m": "3 месяца",
        "6m": "6 месяцев",
        "1y": "1 год",
        "ytd": "с начала года",
    }

    # Функция для расчета performance
    def calculate_performance(df, period_code):
        """Рассчитывает performance за период, отсчитанный от последней даты в данных"""
        _all_dates = df.index.unique().sort_values()
        end_date = _all_dates.max()

        if period_code == "1d":
            # последние два торговых дня
            start_date = _all_dates[-2] if len(_all_dates) >= 2 else end_date
        elif period_code == "ytd":
            start_date = pd.Timestamp(end_date.year, 1, 1)
        else:
            _offsets = {
                "1w": pd.DateOffset(weeks=1),
                "2w": pd.DateOffset(weeks=2),
                "1m": pd.DateOffset(months=1),
                "3m": pd.DateOffset(months=3),
                "6m": pd.DateOffset(months=6),
                "1y": pd.DateOffset(years=1),
            }
            start_date = end_date - _offsets.get(period_code, pd.DateOffset(weeks=1))

        performances = []

        for ticker in df['ticker'].unique():
            stock_data = df[df['ticker'] == ticker].copy()

            # Фильтруем по периоду
            mask = (stock_data.index >= start_date) & (stock_data.index <= end_date)
            period_data = stock_data[mask].sort_index()

            if len(period_data) < 2:
                continue

            # Получаем первую и последнюю даты с данными
            first_day = period_data.index.min()
            last_day = period_data.index.max()

            try:
                # Performance по цене
                if 'close' in period_data.columns:
                    first_price = period_data.loc[first_day, 'close']
                    last_price = period_data.loc[last_day, 'close']

                    if pd.isna(first_price) or pd.isna(last_price) or first_price <= 0:
                        continue

                    price_performance = ((last_price - first_price) / first_price) * 100
                else:
                    continue

                # Performance по market cap (если есть данные)
                market_cap_performance = None
                market_cap_change = None
                first_market_cap = None
                last_market_cap = None

                if 'market_cap' in period_data.columns:
                    first_market_cap = period_data.loc[first_day, 'market_cap']
                    last_market_cap = period_data.loc[last_day, 'market_cap']

                    if not pd.isna(first_market_cap) and not pd.isna(last_market_cap) and first_market_cap > 0:
                        market_cap_performance = ((last_market_cap - first_market_cap) / first_market_cap) * 100
                        market_cap_change = last_market_cap - first_market_cap

                performances.append({
                    'ticker': ticker,
                    'price_performance': price_performance,
                    'first_price': first_price,
                    'last_price': last_price,
                    'market_cap_performance': market_cap_performance,
                    'market_cap_change': market_cap_change,
                    'first_market_cap': first_market_cap,
                    'last_market_cap': last_market_cap,
                    'start_date': first_day,
                    'end_date': last_day,
                })
            except (KeyError, IndexError) as e:
                continue

        return pd.DataFrame(performances), start_date, end_date

    perf_df, period_start, period_end = calculate_performance(combined_df, period_dropdown.value)
    period_label = PERIOD_LABELS.get(period_dropdown.value, str(period_dropdown.value))
    return perf_df, period_end, period_label, period_start


@app.cell(hide_code=True)
def _(combined_df, moex, np, pd):
    # Годовые метрики риска по бумагам: волатильность (аннуализированная),
    # бета к IMOEX, max drawdown и расстояние от 52-недельного максимума
    _last_date_r = combined_df.index.max()
    _start_1y = _last_date_r - pd.DateOffset(years=1)
    _wide_r = combined_df.pivot_table(index=combined_df.index, columns='ticker',
                                      values='close', aggfunc='last').sort_index()
    _wide_1y = _wide_r[_wide_r.index >= _start_1y]
    _rets = _wide_1y.pct_change()
    _counts = _rets.count()

    _vol_1y = _rets.std() * np.sqrt(252) * 100
    _vol_1y[_counts < 60] = np.nan  # меньше ~3 месяцев наблюдений — оценка ненадежна

    _cummax = _wide_1y.cummax()
    _mdd_1y = ((_wide_1y / _cummax) - 1).min() * 100
    _off_high = (_wide_1y.ffill().iloc[-1] / _wide_1y.max() - 1) * 100

    _beta = pd.Series(np.nan, index=_rets.columns)
    try:
        _imx_r = moex.read_moex_index('IMOEX')
        _imx_r.index = pd.to_datetime(_imx_r.index)
        _imx_ret_s = _imx_r['close'].pct_change()
        _imx_ret_s = _imx_ret_s[_imx_ret_s.index >= _start_1y]
        _aligned = _rets.join(_imx_ret_s.rename('_IMOEX_'), how='inner')
        _ivar = float(_aligned['_IMOEX_'].var())
        if _ivar > 0:
            _beta = _aligned.drop(columns='_IMOEX_').apply(
                lambda _s: _s.cov(_aligned['_IMOEX_'])) / _ivar
            _beta[_counts < 60] = np.nan
    except Exception:
        pass

    risk_df = pd.DataFrame({
        'vol_1y': _vol_1y,
        'beta': _beta,
        'mdd_1y': _mdd_1y,
        'off_high': _off_high,
    })
    risk_df.index.name = 'ticker'
    risk_df = risk_df.reset_index()
    return (risk_df,)


@app.cell(hide_code=True)
def _(filtered_df, imoex_ret, pd, period_end, period_start, risk_df):
    # Обогащение риск-метриками: σ-движение (аномальность хода за период)
    # и альфа к IMOEX (изменение бумаги минус бета × изменение индекса)
    enriched_df = filtered_df.merge(risk_df, on='ticker', how='left')

    _years = max((period_end - period_start).days, 1) / 365.25
    _denom = (enriched_df['vol_1y'] * (_years ** 0.5)).replace(0, pd.NA)
    enriched_df['sigma_move'] = enriched_df['price_performance'] / _denom

    if imoex_ret is not None:
        enriched_df['alpha'] = enriched_df['price_performance'] - enriched_df['beta'] * imoex_ret
    else:
        enriched_df['alpha'] = pd.NA
    return (enriched_df,)


@app.cell(hide_code=True)
def _(min_market_cap, perf_df, sort_by):
    # Фильтруем и сортируем данные
    filtered_df = perf_df.copy()

    # Фильтр по минимальному market cap (конвертируем из миллиардов в рубли)
    if 'last_market_cap' in filtered_df.columns:
        min_cap_rub = min_market_cap.value * 1e9
        filtered_df = filtered_df[
            (filtered_df['last_market_cap'].isna()) | 
            (filtered_df['last_market_cap'] >= min_cap_rub)
        ]

    # Сортировка
    if sort_by.value in filtered_df.columns:
        ascending = sort_by.value != "ticker"
        filtered_df = filtered_df.sort_values(sort_by.value, ascending=ascending)
    elif sort_by.value == "ticker":
        filtered_df = filtered_df.sort_values('ticker', ascending=True)
    return (filtered_df,)


@app.cell(hide_code=True)
def _(enriched_df, mo, pd, show_market_cap):
    # Таблица с результатами
    display_cols = ['ticker', 'price_performance', 'sigma_move', 'alpha',
                    'vol_1y', 'beta', 'mdd_1y', 'off_high',
                    'first_price', 'last_price']
    display_cols = [c for c in display_cols if c in enriched_df.columns]

    # Добавляем даты в таблицу
    if 'start_date' in enriched_df.columns and 'end_date' in enriched_df.columns:
        display_cols.extend(['start_date', 'end_date'])

    if show_market_cap.value and 'market_cap_performance' in enriched_df.columns:
        display_cols.extend(['market_cap_performance', 'market_cap_change', 'last_market_cap'])

    display_df = enriched_df[display_cols].copy()

    # Округление риск-метрик
    for _rc, _nd in (('sigma_move', 1), ('alpha', 1), ('vol_1y', 0),
                     ('beta', 2), ('mdd_1y', 1), ('off_high', 1)):
        if _rc in display_df.columns:
            display_df[_rc] = pd.to_numeric(display_df[_rc], errors='coerce').round(_nd)

    # Форматирование
    if 'price_performance' in display_df.columns:
        display_df['price_performance'] = display_df['price_performance'].round(2)
    if 'market_cap_performance' in display_df.columns:
        display_df['market_cap_performance'] = display_df['market_cap_performance'].round(2)
    if 'market_cap_change' in display_df.columns:
        display_df['market_cap_change'] = (display_df['market_cap_change'] / 1e9).round(2)  # в миллиардах
    if 'last_market_cap' in display_df.columns:
        display_df['last_market_cap'] = (display_df['last_market_cap'] / 1e9).round(2)  # в миллиардах
    if 'first_price' in display_df.columns:
        display_df['first_price'] = display_df['first_price'].round(2)
    if 'last_price' in display_df.columns:
        display_df['last_price'] = display_df['last_price'].round(2)

    # Форматирование дат
    if 'start_date' in display_df.columns:
        display_df['start_date'] = pd.to_datetime(display_df['start_date']).dt.strftime('%d.%m.%Y')
    if 'end_date' in display_df.columns:
        display_df['end_date'] = pd.to_datetime(display_df['end_date']).dt.strftime('%d.%m.%Y')

    # Переименование для читаемости
    column_mapping = {
        'ticker': 'Тикер',
        'price_performance': 'Изм. цены (%)',
        'sigma_move': 'σ-движение',
        'alpha': 'Альфа (%)',
        'vol_1y': 'Волат. 1Y (%)',
        'beta': 'Бета',
        'mdd_1y': 'Max DD 1Y (%)',
        'off_high': 'От 52н max (%)',
        'first_price': 'Цена нач.',
        'last_price': 'Цена кон.',
        'start_date': 'Дата нач.',
        'end_date': 'Дата кон.',
        'market_cap_performance': 'Изм. mcap (%)',
        'market_cap_change': 'Δ mcap (млрд)',
        'last_market_cap': 'Mcap (млрд)',
    }
    display_df = display_df.rename(columns=column_mapping)

    table = mo.ui.table(display_df, pagination=True, page_size=20)
    return (table,)


@app.cell(hide_code=True)
def _(base64, filtered_df, io, mo, pd, period_label, plt):
    # Лидеры и аутсайдеры: топ-15 в обе стороны (полный список — в таблице)
    if len(filtered_df) > 0:
        _n_show = 15
        _mv = pd.concat([
            filtered_df.nlargest(_n_show, 'price_performance'),
            filtered_df.nsmallest(_n_show, 'price_performance'),
        ]).drop_duplicates(subset='ticker').sort_values('price_performance')

        _fig1, _ax1 = plt.subplots(figsize=(9.5, max(6.0, 0.32 * len(_mv))))
        _colors1 = ['green' if _x >= 0 else 'red' for _x in _mv['price_performance']]
        _bars1 = _ax1.barh(_mv['ticker'], _mv['price_performance'], color=_colors1, alpha=0.75)
        _ax1.set_xlabel('Изменение цены (%)')
        _ax1.set_title(f'Лидеры и аутсайдеры (топ-{_n_show} в обе стороны) — {period_label}')
        _ax1.axvline(x=0, color='black', linewidth=0.8)
        _ax1.grid(axis='x', linestyle='--', alpha=0.5)
        _ax1.margins(x=0.12)

        for _bar, _val in zip(_bars1, _mv['price_performance']):
            _w = _bar.get_width()
            _ax1.text(_w, _bar.get_y() + _bar.get_height() / 2, f' {_val:+.1f}% ',
                      ha='left' if _w >= 0 else 'right', va='center', fontsize=8)

        plt.tight_layout()
        _buf = io.BytesIO()
        _fig1.savefig(_buf, format='png', bbox_inches='tight', dpi=100)
        _buf.seek(0)
        _img_base64 = base64.b64encode(_buf.read()).decode()
        plt.close(_fig1)
        chart = mo.Html(f'<img src="data:image/png;base64,{_img_base64}" style="max-width: 100%; height: auto;" />')
    else:
        chart = mo.md("Нет данных для отображения")
    return (chart,)


@app.cell(hide_code=True)
def _(base64, filtered_df, io, mo, period_label, plt, show_market_cap):
    # Крупнейшие изменения капитализации: топ-10 по модулю
    if show_market_cap.value and 'market_cap_change' in filtered_df.columns and len(filtered_df) > 0:
        mc_change_data = filtered_df[filtered_df['market_cap_change'].notna()].copy()
        if len(mc_change_data) > 0:
            _top_idx = mc_change_data['market_cap_change'].abs().nlargest(10).index
            mc_change_data = mc_change_data.loc[_top_idx].sort_values('market_cap_change')
            _fig2, _ax2 = plt.subplots(figsize=(9.5, 4.5))

            _colors3 = ['green' if x >= 0 else 'red' for x in mc_change_data['market_cap_change']]
            _bars3 = _ax2.barh(mc_change_data['ticker'], mc_change_data['market_cap_change'] / 1e9, color=_colors3, alpha=0.7)
            _ax2.set_xlabel('Изменение market cap (млрд руб)')
            _ax2.set_title(f'Крупнейшие изменения market cap (топ-10) — {period_label}')
            _ax2.margins(x=0.12)
            _ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            _ax2.grid(axis='x', linestyle='--', alpha=0.7)

            # Добавляем значения
            for _bar3, _val3 in zip(_bars3, mc_change_data['market_cap_change'] / 1e9):
                _width3 = _bar3.get_width()
                _ax2.text(_width3, _bar3.get_y() + _bar3.get_height()/2,
                       f'{_val3:.1f}',
                       ha='left' if _width3 >= 0 else 'right',
                       va='center', fontsize=9)

            plt.tight_layout()
            # Конвертируем фигуру в base64 для отображения
            _buf2 = io.BytesIO()
            _fig2.savefig(_buf2, format='png', bbox_inches='tight', dpi=100)
            _buf2.seek(0)
            _img_base64_2 = base64.b64encode(_buf2.read()).decode()
            plt.close(_fig2)
            market_cap_chart = mo.Html(f'<img src="data:image/png;base64,{_img_base64_2}" style="max-width: 100%; height: auto;" />')
        else:
            market_cap_chart = mo.md("Нет данных по market cap")
    else:
        market_cap_chart = mo.md("")
    return (market_cap_chart,)


@app.cell(hide_code=True)
def _(
    anchor_date,
    breadth_block,
    chart,
    heatmap_block,
    index_block,
    marimekko_block,
    market_cap_chart,
    market_summary,
    min_market_cap,
    mo,
    period_dropdown,
    period_end,
    period_label,
    period_start,
    sector_block,
    show_market_cap,
    sort_by,
    structure_block,
    table,
    volume_block,
):
    # Основной layout: сводка и ключевые картинки сверху, детали — в аккордеоне
    period_start_str = period_start.strftime('%d.%m.%Y')
    period_end_str = period_end.strftime('%d.%m.%Y')

    mo.vstack([
        mo.hstack([period_dropdown, show_market_cap, sort_by, min_market_cap], justify="start"),
        mo.md(f"## Что произошло на рынке — {period_label}\n**Период:** {period_start_str} - {period_end_str}"),
        market_summary,
        breadth_block,
        index_block,
        sector_block,
        heatmap_block,
        volume_block,
        mo.md("---\n## 🧭 Структура рынка: сравнение с опорной датой"),
        anchor_date,
        structure_block,
        mo.accordion({
            "📊 Marimekko: динамика с учетом веса в рынке": marimekko_block,
            "🏆 Лидеры и аутсайдеры (топ-15)": chart,
            "💰 Крупнейшие изменения капитализации (топ-10)": market_cap_chart,
            "📋 Таблица по всем бумагам": table,
        }),
    ])
    return


@app.cell(hide_code=True)
def _(filtered_df, imoex_ret, mo, period_end, period_label):
    # Сводка: рынок в целом за период

    def _sgn(_v, _suffix='%', _nd=2):
        """Число со знаком, раскрашенное классами pos/neg из styles.css"""
        _cls = 'pos' if _v >= 0 else 'neg'
        _txt = format(_v, f'+,.{_nd}f').replace(',', ' ')
        return f'<span class="{_cls}">{_txt}{_suffix}</span>'

    _n = len(filtered_df)
    _up = int((filtered_df['price_performance'] > 0).sum()) if _n else 0
    _down = int((filtered_df['price_performance'] < 0).sum()) if _n else 0

    # Взвешенная по капитализации динамика: суммарная капитализация конец/начало
    _mc = filtered_df.dropna(subset=['first_market_cap', 'last_market_cap']) if _n else filtered_df
    if _n and len(_mc) > 0 and _mc['first_market_cap'].sum() > 0:
        _market_ret = (_mc['last_market_cap'].sum() / _mc['first_market_cap'].sum() - 1) * 100
        _mkt_str = _sgn(_market_ret)
        _mc_total = _mc['last_market_cap'].sum() / 1e12
        _mc_delta = (_mc['last_market_cap'].sum() - _mc['first_market_cap'].sum()) / 1e9
        _mkt_extra = (f" — капитализация на {period_end.strftime('%d.%m.%Y')}: "
                      f"{_mc_total:.1f} трлн руб ({_sgn(_mc_delta, ' млрд', 0)} за период)")
    else:
        _mkt_str = "н/д"
        _mkt_extra = ""

    if _n:
        _med = filtered_df['price_performance'].median()
        _top = filtered_df.nlargest(5, 'price_performance')
        _bot = filtered_df.nsmallest(5, 'price_performance')
        _top_str = ", ".join(f"**{_r.ticker}** {_sgn(_r.price_performance, '%', 1)}"
                             for _r in _top.itertuples())
        _bot_str = ", ".join(f"**{_r.ticker}** {_sgn(_r.price_performance, '%', 1)}"
                             for _r in _bot.itertuples())
        _imx_line = f"; IMOEX: {_sgn(imoex_ret)}" if imoex_ret is not None else ""
        market_summary = mo.md(f"""
    ### Итоги — {period_label}

    - **Рынок (взвешенно по капитализации):** {_mkt_str}{_mkt_extra}{_imx_line}; медианная бумага: {_sgn(_med)}
    - Выросло: **{_up}** | Упало: **{_down}** | Всего: {_n}
    - 📈 Лидеры: {_top_str}
    - 📉 Аутсайдеры: {_bot_str}
    """)
    else:
        market_summary = mo.md("Нет данных за выбранный период")
    return (market_summary,)


@app.cell(hide_code=True)
def _(filtered_df, go, mo, period_label, plotly_available):
    # Вертикальный Marimekko (интерактивный): толщина бара = доля в капитализации,
    # длина = performance, лучшие сверху. Тонкие бары читаются через hover.
    _mk = filtered_df.dropna(subset=['last_market_cap', 'price_performance']).copy()
    if len(_mk) == 0:
        marimekko_block = mo.md("")
    elif not plotly_available:
        marimekko_block = mo.md("*Для Marimekko нужен plotly: `pip install plotly`*")
    else:
        _mk = _mk.sort_values('price_performance', ascending=False)
        _mk['share'] = _mk['last_market_cap'] / _mk['last_market_cap'].sum() * 100
        _cums = _mk['share'].cumsum()
        _mk['y_center'] = -( _cums - _mk['share'] / 2)

        _figm = go.Figure(go.Bar(
            x=_mk['price_performance'],
            y=_mk['y_center'],
            width=(_mk['share'] * 0.94).clip(lower=0.12),
            orientation='h',
            marker_color=['green' if _v >= 0 else 'red' for _v in _mk['price_performance']],
            marker_line=dict(color='white', width=0.5),
            text=[f'{_t} {_v:+.1f}%' if _s >= 0.8 else ''
                  for _t, _v, _s in zip(_mk['ticker'], _mk['price_performance'], _mk['share'])],
            textposition='outside',
            textfont_size=11,
            customdata=[
                (_t, f'{_s:.2f}%', f'{_v:+.2f}%')
                for _t, _s, _v in zip(_mk['ticker'], _mk['share'], _mk['price_performance'])
            ],
            hovertemplate='<b>%{customdata[0]}</b><br>Изменение: %{customdata[2]}'
                          '<br>Доля в капитализации: %{customdata[1]}<extra></extra>',
        ))
        _figm.update_layout(
            height=900,
            title=dict(text=f'Marimekko: толщина = доля в капитализации — {period_label}', font_size=15),
            xaxis=dict(title='Изменение цены (%)', zeroline=True, zerolinecolor='black', zerolinewidth=1),
            yaxis=dict(
                title='Накопленная доля капитализации (лучшие — сверху)',
                tickvals=[0, -20, -40, -60, -80, -100],
                ticktext=['0%', '20%', '40%', '60%', '80%', '100%'],
                range=[-101, 1],
            ),
            margin=dict(t=40, l=10, r=10, b=10),
            showlegend=False,
        )
        marimekko_block = _figm
    return (marimekko_block,)


@app.cell(hide_code=True)
def _(go, mo, moex, pd, period_end, period_label, period_start, plotly_available):
    # IMOEX (интерактивный) с линиями EWMAC — пара EWMA 16/64 дня (по Р. Карверу):
    # быстрая выше медленной = восходящий тренд. Кэш: indexes/IMOEX.parquet.
    imoex_ret = None
    try:
        _idx_df = moex.read_moex_index('IMOEX')
        _idx_df.index = pd.to_datetime(_idx_df.index)
        _close = _idx_df['close'].astype(float).sort_index()

        # Доходность за выбранный период — для сводки "Итоги"
        _win = _close[(_close.index >= period_start) & (_close.index <= period_end)]
        if len(_win) >= 2:
            imoex_ret = (float(_win.iloc[-1]) / float(_win.iloc[0]) - 1) * 100

        if not plotly_available:
            index_block = mo.md("*Для графика IMOEX нужен plotly: `pip install plotly`*")
        elif len(_close) < 70:
            index_block = mo.md("*IMOEX: недостаточно истории в кэше — обновите: `python update_data.py`*")
        else:
            # EWMA считаем по всей истории (без прогревочного смещения),
            # показываем динамику с 2022 года
            _ew16 = _close.ewm(span=16, adjust=False).mean()
            _ew64 = _close.ewm(span=64, adjust=False).mean()
            _show_from = min(pd.Timestamp('2022-01-01'), pd.Timestamp(period_start))
            _c = _close[_close.index >= _show_from]
            _e16 = _ew16[_ew16.index >= _show_from]
            _e64 = _ew64[_ew64.index >= _show_from]
            _last_close = float(_c.iloc[-1])

            _figi = go.Figure()
            _figi.add_scatter(x=_c.index, y=_c.values, name='IMOEX',
                              line=dict(color='#1f77b4', width=1.8),
                              hovertemplate='%{y:.0f}<extra>IMOEX</extra>')
            _figi.add_scatter(x=_e16.index, y=_e16.values, name='EWMA 16',
                              line=dict(color='#2ca02c', width=1.1),
                              hovertemplate='%{y:.0f}<extra>EWMA 16</extra>')
            _figi.add_scatter(x=_e64.index, y=_e64.values, name='EWMA 64',
                              line=dict(color='#d62728', width=1.1, dash='dot'),
                              hovertemplate='%{y:.0f}<extra>EWMA 64</extra>')
            # Подсветка выбранного периода анализа
            _figi.add_vrect(x0=period_start, x1=period_end,
                            fillcolor='gray', opacity=0.08, line_width=0)
            # Последнее значение
            _figi.add_scatter(x=[_c.index[-1]], y=[_last_close], mode='markers',
                              marker=dict(color='#1f77b4', size=7),
                              showlegend=False, hoverinfo='skip')
            _figi.add_annotation(x=_c.index[-1], y=_last_close,
                                 text=f'<b>{_last_close:,.0f}</b>'.replace(',', ' '),
                                 showarrow=False, xanchor='left', xshift=8,
                                 font=dict(color='#1f77b4', size=13))

            _ret_str = f'{imoex_ret:+.2f}%' if imoex_ret is not None else 'н/д'
            _trend = ('восходящий (EWMA16 > EWMA64)'
                      if float(_ew16.iloc[-1]) > float(_ew64.iloc[-1])
                      else 'нисходящий (EWMA16 < EWMA64)')
            _figi.update_layout(
                height=360,
                title=dict(text=(f'IMOEX {_last_close:,.0f} | {period_label}: {_ret_str} | '
                                 f'тренд: {_trend}').replace(',', ' '), font_size=14),
                hovermode='x unified',
                legend=dict(orientation='h', y=1.12, x=1, xanchor='right'),
                margin=dict(t=48, l=10, r=70, b=10),
            )
            index_block = _figi
    except FileNotFoundError:
        index_block = mo.md("*Локальный кэш IMOEX не найден — выполните `python update_data.py` (шаг 1b)*")
    except Exception as _e_idx:
        index_block = mo.md(f"*IMOEX: ошибка чтения кэша — {_e_idx}*")
    return imoex_ret, index_block


@app.cell(hide_code=True)
def _(combined_df, mo, pd):
    # Ширина рынка: 52-недельные экстремумы + доля бумаг выше MA50/MA200.
    # Классика: >50% бумаг выше MA200 — здоровый рынок, дивергенция с индексом — ранний сигнал.
    _last_date = combined_df.index.max()
    _ydf = combined_df[combined_df.index >= _last_date - pd.DateOffset(years=1)]
    _g = _ydf.groupby('ticker')['close']
    _hi = _g.max()
    _lo = _g.min()
    _lastp = _ydf.sort_index().groupby('ticker')['close'].last()

    _near_hi = sorted(_lastp[_lastp >= _hi * 0.98].index)
    _near_lo = sorted(_lastp[_lastp <= _lo * 1.02].index)

    def _fmt_tickers(_lst, _limit=12):
        if not _lst:
            return "—"
        _s = ", ".join(_lst[:_limit])
        return _s + (f" и еще {len(_lst) - _limit}" if len(_lst) > _limit else "")

    # Доля бумаг выше скользящих средних (по всей истории, показываем последний год)
    _wide = combined_df.pivot_table(index=combined_df.index, columns='ticker',
                                    values='close', aggfunc='last').sort_index()
    _ma50 = _wide.rolling(50, min_periods=50).mean()
    _ma200 = _wide.rolling(200, min_periods=200).mean()

    def _pct_above(_prices, _ma):
        _valid = _ma.notna() & _prices.notna()
        _cnt = _valid.sum(axis=1)
        return ((_prices > _ma) & _valid).sum(axis=1) / _cnt.replace(0, pd.NA) * 100

    _above50_series = _pct_above(_wide, _ma50).dropna()
    _above200_series = _pct_above(_wide, _ma200).dropna()
    _above50 = float(_above50_series.iloc[-1]) if len(_above50_series) else float('nan')
    _above200 = float(_above200_series.iloc[-1]) if len(_above200_series) else float('nan')

    breadth_block = mo.md(
        f"**Ширина рынка:** выше MA50: **{_above50:.0f}%** | выше MA200: **{_above200:.0f}%** | "
        f"у 52-нед. максимумов (≤2%): **{len(_near_hi)}** ({_fmt_tickers(_near_hi)}) | "
        f"у минимумов: **{len(_near_lo)}** ({_fmt_tickers(_near_lo)})"
    )
    return (breadth_block,)


@app.cell(hide_code=True)
def _(filtered_df, go, mo, pd, period_label, plotly_available, sectors_map):
    # Секторный разрез: динамика секторов, взвешенная по капитализации
    if len(filtered_df) == 0 or len(sectors_map) == 0:
        sector_block = mo.md("")
    elif not plotly_available:
        sector_block = mo.md("*Для секторного графика нужен plotly: `pip install plotly`*")
    else:
        _sdf = filtered_df.merge(sectors_map, on='ticker', how='left')
        _sdf['sector'] = _sdf['sector'].fillna('Прочее')

        _rows = []
        for _sec, _grp in _sdf.groupby('sector'):
            _gmc = _grp.dropna(subset=['first_market_cap', 'last_market_cap'])
            if len(_gmc) > 0 and _gmc['first_market_cap'].sum() > 0:
                _ret = (_gmc['last_market_cap'].sum() / _gmc['first_market_cap'].sum() - 1) * 100
            else:
                _ret = float(_grp['price_performance'].median())
            _rows.append({'sector': f"{_sec} ({len(_grp)})", 'ret': _ret,
                          'tickers': ", ".join(sorted(_grp['ticker'])[:15])})
        _sec_df = pd.DataFrame(_rows).sort_values('ret')

        _figs = go.Figure(go.Bar(
            x=_sec_df['ret'],
            y=_sec_df['sector'],
            orientation='h',
            marker_color=['green' if _x >= 0 else 'red' for _x in _sec_df['ret']],
            text=[f'{_v:+.1f}%' for _v in _sec_df['ret']],
            textposition='outside',
            customdata=[
                (_tk, f'{_v:+.2f}%') for _tk, _v in zip(_sec_df['tickers'], _sec_df['ret'])
            ],
            hovertemplate='<b>%{y}</b>: %{customdata[1]}<br>%{customdata[0]}<extra></extra>',
        ))
        _figs.update_layout(
            height=max(300, 34 * len(_sec_df) + 80),
            title=dict(text=f'Сектора (взвешенно по капитализации) — {period_label}', font_size=15),
            xaxis=dict(title='Изменение (%)', zeroline=True, zerolinecolor='black'),
            margin=dict(t=40, l=10, r=10, b=10),
        )
        sector_block = _figs
    return (sector_block,)


@app.cell(hide_code=True)
def _(filtered_df, mo, np, period_label, plotly_available, px, sectors_map):
    # Карта рынка (finviz-style treemap): сектора → бумаги,
    # площадь = капитализация, цвет = изменение цены. Hover — точные цифры.
    _hm = filtered_df.dropna(subset=['price_performance', 'last_market_cap']).copy()
    if len(_hm) == 0:
        heatmap_block = mo.md("")
    elif not plotly_available:
        heatmap_block = mo.md("*Для карты рынка нужен plotly: `pip install plotly`*")
    else:
        _hm = _hm.merge(sectors_map, on='ticker', how='left')
        _hm['sector'] = _hm['sector'].fillna('Прочее')
        # Форматируем подписи заранее: форматы в шаблонах plotly с флагом "+"
        # применяются ненадежно, и на плитки попадают числа с 13 знаками
        _hm['perf_str'] = _hm['price_performance'].map(lambda _v: f'{_v:+.1f}%')
        _hm['mc_str'] = (_hm['last_market_cap'] / 1e9).map(
            lambda _v: f'{_v:,.0f}'.replace(',', ' '))

        # Шкала цвета по 95-му перцентилю, чтобы один выброс не обесцвечивал карту
        _vmax = max(float(np.percentile(np.abs(_hm['price_performance']), 95)), 1e-9)

        _figt = px.treemap(
            _hm,
            path=[px.Constant(f'Рынок — {period_label}'), 'sector', 'ticker'],
            values='last_market_cap',
            color='price_performance',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            range_color=(-_vmax, _vmax),
            custom_data=['perf_str', 'mc_str'],
        )
        _figt.update_traces(
            texttemplate='%{label}<br>%{customdata[0]}',
            hovertemplate='<b>%{label}</b><br>Изменение: %{customdata[0]}'
                          '<br>Капитализация: %{customdata[1]} млрд руб<extra></extra>',
            textfont_size=13,
            marker_line_width=1,
        )
        _figt.update_layout(
            height=640,
            margin=dict(t=34, l=2, r=2, b=2),
            coloraxis_colorbar=dict(title='%'),
            title=dict(text=f'Карта рынка — {period_label}', font_size=15),
        )
        heatmap_block = _figt
    return (heatmap_block,)


@app.cell(hide_code=True)
def _(combined_df, enriched_df, filtered_df, mo, pd, period_end, period_start):
    # Необычная активность: среднедневной оборот за период против 90 дней до него
    _per = combined_df[(combined_df.index >= period_start) & (combined_df.index <= period_end)]
    _base = combined_df[
        (combined_df.index >= period_start - pd.DateOffset(days=90)) & (combined_df.index < period_start)
    ]

    _va = pd.DataFrame({
        'per': _per.groupby('ticker')['value_rub'].mean(),
        'base': _base.groupby('ticker')['value_rub'].mean(),
    }).dropna()
    _va = _va[_va['base'] > 1e7]  # отсекаем неликвид: база < 10 млн руб/день
    _va['ratio'] = _va['per'] / _va['base']
    _va = _va.sort_values('ratio', ascending=False).head(10)

    _parts = []
    if len(_va) > 0:
        _perf_map = (
            filtered_df.set_index('ticker')['price_performance'] if len(filtered_df) else pd.Series(dtype=float)
        )
        _lines = []
        for _tv, _rv in _va.iterrows():
            _pperf = _perf_map.get(_tv)
            _pstr = f"{_pperf:+.1f}%" if pd.notna(_pperf) else "—"
            _lines.append(
                f"| {_tv} | {_rv['per'] / 1e6:,.0f} | {_rv['base'] / 1e6:,.0f} | ×{_rv['ratio']:.1f} | {_pstr} |".replace(",", " ")
            )
        _parts.append(mo.md(
            "### Необычная активность\n\n"
            "Среднедневной оборот за период против среднего за предыдущие 90 дней:\n\n"
            "| Тикер | Оборот/день, млн руб | База, млн руб | Всплеск | Изм. цены |\n"
            "|---|---|---|---|---|\n" + "\n".join(_lines)
        ))

    # Необычные движения цены: ход за период в единицах годовой волатильности бумаги.
    # |σ| ≥ 2 — статистически редкое движение, даже если процент скромный
    _sm = enriched_df.dropna(subset=['sigma_move']) if len(enriched_df) else enriched_df
    if len(_sm) > 0:
        _sm = _sm[_sm['sigma_move'].abs() >= 2]
        _sm = _sm.sort_values('sigma_move', key=lambda s: s.abs(), ascending=False).head(10)
        if len(_sm) > 0:
            _lines2 = [
                f"| {_r.ticker} | {_r.price_performance:+.1f}% | {_r.sigma_move:+.1f}σ | {_r.vol_1y:.0f}% |"
                for _r in _sm.itertuples()
            ]
            _parts.append(mo.md(
                "### Необычные движения цены (|σ| ≥ 2)\n\n"
                "Изменение за период в единицах собственной годовой волатильности бумаги:\n\n"
                "| Тикер | Изм. цены | Движение | Волат. 1Y |\n"
                "|---|---|---|---|\n" + "\n".join(_lines2)
            ))

    volume_block = mo.vstack(_parts) if _parts else mo.md("")
    return (volume_block,)


@app.cell(hide_code=True)
def _(mo):
    # Опорная дата для анализа структуры рынка
    anchor_date = mo.ui.date(value="2022-02-21", label="Опорная дата:")
    return (anchor_date,)


@app.cell(hide_code=True)
def _(anchor_date, combined_df, go, mo, moex, pd, plotly_available, sectors_map):
    # Структура рынка: что изменилось с опорной даты.
    # Отвечает на вопросы: индекс на том же уровне — а рынок тот же?
    # Кто вытащил/утопил капитализацию, как перекроились веса секторов,
    # выросла ли концентрация.
    _anchor = pd.Timestamp(anchor_date.value)
    _last_date = combined_df.index.max()

    # Срез "тогда": последняя котировка каждой бумаги в окне 45 дней до опорной даты
    _win_then = combined_df[(combined_df.index <= _anchor) &
                            (combined_df.index >= _anchor - pd.DateOffset(days=45))]
    _then = _win_then.sort_index().groupby('ticker').tail(1)
    _then = _then.reset_index()[['ticker', 'close', 'market_cap']].rename(
        columns={'close': 'close_then', 'market_cap': 'mc_then'})

    # Срез "сейчас": только бумаги, торговавшиеся в последние 30 дней (без делистингов)
    _now = combined_df.sort_index().groupby('ticker').tail(1)
    _now = _now[_now.index >= _last_date - pd.DateOffset(days=30)]
    _now = _now.reset_index()[['ticker', 'close', 'market_cap']].rename(
        columns={'close': 'close_now', 'market_cap': 'mc_now'})

    _st = _then.merge(_now, on='ticker', how='inner').dropna(subset=['close_then', 'close_now'])

    if len(_st) < 5:
        structure_block = mo.md("*Недостаточно данных на выбранную опорную дату*")
    else:
        _st['px_chg'] = (_st['close_now'] / _st['close_then'] - 1) * 100

        def _sgn(_v, _suffix='%', _nd=1):
            """Число со знаком, раскрашенное классами pos/neg из styles.css"""
            _cls = 'pos' if _v >= 0 else 'neg'
            _txt = format(_v, f'+,.{_nd}f').replace(',', ' ')
            return f'<span class="{_cls}">{_txt}{_suffix}</span>'

        # --- IMOEX тогда и сейчас
        _imx_line2 = ""
        try:
            _idx = moex.read_moex_index('IMOEX')
            _idx.index = pd.to_datetime(_idx.index)
            _idx_then = _idx[_idx.index <= _anchor]
            if len(_idx_then):
                _iv_then = float(_idx_then['close'].iloc[-1])
                _iv_now = float(_idx['close'].iloc[-1])
                _imx_line2 = (f"- **IMOEX:** {_iv_then:,.0f} → {_iv_now:,.0f} ".replace(",", " ")
                              + f"({_sgn((_iv_now / _iv_then - 1) * 100)})\n")
        except Exception:
            pass

        # --- счет выше/ниже уровня опорной даты
        _n_up = int((_st['px_chg'] > 0).sum())
        _n_down = int((_st['px_chg'] < 0).sum())
        _med_chg = float(_st['px_chg'].median())

        # --- капитализация и концентрация (по бумагам с mc в обеих точках)
        _mc = _st.dropna(subset=['mc_then', 'mc_now'])
        _conc_line = ""
        _total_line = ""
        if len(_mc) >= 5 and _mc['mc_then'].sum() > 0:
            _tot_then = _mc['mc_then'].sum()
            _tot_now = _mc['mc_now'].sum()
            _top5_then = _mc.nlargest(5, 'mc_then')['mc_then'].sum() / _tot_then * 100
            _top5_now = _mc.nlargest(5, 'mc_now')['mc_now'].sum() / _tot_now * 100
            _t5_then_names = ", ".join(_mc.nlargest(5, 'mc_then')['ticker'])
            _t5_now_names = ", ".join(_mc.nlargest(5, 'mc_now')['ticker'])
            _total_line = (f"- **Капитализация (сопоставимые бумаги):** "
                           f"{_tot_then / 1e12:.1f} → {_tot_now / 1e12:.1f} трлн руб "
                           f"({_sgn((_tot_now / _tot_then - 1) * 100)})\n")
            _conc_line = (f"- **Концентрация (доля топ-5):** {_top5_then:.0f}% → {_top5_now:.0f}%\n"
                          f"  - тогда: {_t5_then_names}\n  - сейчас: {_t5_now_names}\n")

        _tops = _st.nlargest(5, 'px_chg')
        _bots = _st.nsmallest(5, 'px_chg')
        _tops_str = ", ".join(f"**{_r.ticker}** {_sgn(_r.px_chg, '%', 0)}" for _r in _tops.itertuples())
        _bots_str = ", ".join(f"**{_r.ticker}** {_sgn(_r.px_chg, '%', 0)}" for _r in _bots.itertuples())

        _md_struct = mo.md(
            f"### С {_anchor.strftime('%d.%m.%Y')} (сопоставимых бумаг: {len(_st)})\n\n"
            + _imx_line2
            + f"- **Выше уровня той даты: {_n_up}**, ниже: **{_n_down}**; медианная бумага: {_sgn(_med_chg)}\n"
            + _total_line + _conc_line
            + f"- 📈 Сильнее всех: {_tops_str}\n- 📉 Слабее всех: {_bots_str}"
        )

        _blocks = [_md_struct]

        if plotly_available and len(_mc) >= 5 and _mc['mc_then'].sum() > 0:
            # --- вклад бумаг в изменение суммарной капитализации (п.п.)
            _mc = _mc.copy()
            _mc['contrib'] = (_mc['mc_now'] - _mc['mc_then']) / _mc['mc_then'].sum() * 100
            _cb = _mc.reindex(_mc['contrib'].abs().nlargest(12).index).sort_values('contrib')
            _figc = go.Figure(go.Bar(
                x=_cb['contrib'], y=_cb['ticker'], orientation='h',
                marker_color=['green' if _v >= 0 else 'red' for _v in _cb['contrib']],
                text=[f'{_v:+.1f} п.п.' for _v in _cb['contrib']],
                textposition='outside',
                customdata=[
                    (f'{_c:+.2f}', f'{_p:+.1f}%')
                    for _c, _p in zip(_cb['contrib'], _cb['px_chg'])
                ],
                hovertemplate='<b>%{y}</b>: %{customdata[0]} п.п. к капитализации рынка'
                              '<br>Цена: %{customdata[1]}<extra></extra>',
            ))
            _figc.update_layout(
                height=max(320, 30 * len(_cb) + 90),
                title=dict(text='Кто изменил капитализацию рынка (вклад, п.п.)', font_size=14),
                xaxis=dict(zeroline=True, zerolinecolor='black'),
                margin=dict(t=40, l=10, r=10, b=10),
            )
            _blocks.append(_figc)

            # --- веса секторов: тогда vs сейчас
            if len(sectors_map):
                _ms = _mc.merge(sectors_map, on='ticker', how='left')
                _ms['sector'] = _ms['sector'].fillna('Прочее')
                _w = _ms.groupby('sector').agg(mc_then=('mc_then', 'sum'), mc_now=('mc_now', 'sum'))
                _w = (_w / _w.sum() * 100).sort_values('mc_now')
                _figw = go.Figure()
                _figw.add_bar(x=_w['mc_then'], y=_w.index, orientation='h', name='Тогда',
                              marker_color='#9ecae1',
                              hovertemplate='%{y}: %{x:.1f}%<extra>тогда</extra>')
                _figw.add_bar(x=_w['mc_now'], y=_w.index, orientation='h', name='Сейчас',
                              marker_color='#1f77b4',
                              hovertemplate='%{y}: %{x:.1f}%<extra>сейчас</extra>')
                _figw.update_layout(
                    barmode='group',
                    height=max(360, 34 * len(_w) + 90),
                    title=dict(text='Веса секторов в капитализации: тогда vs сейчас', font_size=14),
                    xaxis=dict(ticksuffix='%'),
                    legend=dict(orientation='h', y=1.08, x=1, xanchor='right'),
                    margin=dict(t=46, l=10, r=10, b=10),
                )
                _blocks.append(_figw)

        structure_block = mo.vstack(_blocks)
    return (structure_block,)


if __name__ == "__main__":
    app.run()
