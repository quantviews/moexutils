"""
# Stocks Performance Analysis

Интерактивный анализ performance акций с учетом market cap.
"""

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
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

    return base64, io, mo, moex, np, pd, plt


@app.cell
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
        options=["price_performance", "market_cap_performance", "market_cap_change", "ticker"],
        value="price_performance",
        label="Сортировка:"
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
    # Загружаем данные
    combined_df = moex.combine_moex_stocks()
    return (combined_df,)


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
def _(filtered_df, mo, pd, show_market_cap):
    # Таблица с результатами
    display_cols = ['ticker', 'price_performance', 'first_price', 'last_price']

    # Добавляем даты в таблицу
    if 'start_date' in filtered_df.columns and 'end_date' in filtered_df.columns:
        display_cols.extend(['start_date', 'end_date'])

    if show_market_cap.value and 'market_cap_performance' in filtered_df.columns:
        display_cols.extend(['market_cap_performance', 'market_cap_change', 'last_market_cap'])

    display_df = filtered_df[display_cols].copy()

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
        'Ticker': 'Тикер',
        'Price Performance': 'Performance по цене (%)',
        'First Price': 'Цена нач. (руб)',
        'Last Price': 'Цена кон. (руб)',
        'Start Date': 'Дата нач.',
        'End Date': 'Дата кон.',
        'Market Cap Performance': 'Performance по market cap (%)',
        'Market Cap Change': 'Изменение market cap (млрд руб)',
        'Last Market Cap': 'Market cap кон. (млрд руб)',
    }
    display_df = display_df.rename(columns=column_mapping)

    table = mo.ui.table(display_df, pagination=True, page_size=20)
    return (table,)


@app.cell(hide_code=True)
def _(base64, filtered_df, io, mo, np, period_label, plt, show_market_cap):
    # График performance по цене
    if len(filtered_df) > 0:
        _fig1, axes = plt.subplots(1, 2 if show_market_cap.value and 'market_cap_performance' in filtered_df.columns else 1, 
                                figsize=(16, max(8, len(filtered_df) * 0.25)))

        if not isinstance(axes, np.ndarray):
            axes = [axes]

        # График 1: Performance по цене
        ax1 = axes[0]
        _colors1 = ['green' if x >= 0 else 'red' for x in filtered_df['price_performance']]
        bars1 = ax1.barh(filtered_df['ticker'], filtered_df['price_performance'], color=_colors1, alpha=0.7)
        ax1.set_xlabel('Performance (%)')
        ax1.set_title(f'Performance по цене — {period_label}')
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax1.grid(axis='x', linestyle='--', alpha=0.7)

        # Добавляем значения на столбцы
        for _i, (_bar, _val) in enumerate(zip(bars1, filtered_df['price_performance'])):
            _width = _bar.get_width()
            ax1.text(_width, _bar.get_y() + _bar.get_height()/2,
                    f'{_val:.1f}%',
                    ha='left' if _width >= 0 else 'right',
                    va='center', fontsize=9)

        # График 2: Performance по market cap (если включено)
        if show_market_cap.value and 'market_cap_performance' in filtered_df.columns and len(axes) > 1:
            ax2 = axes[1]
            # Фильтруем только те, у которых есть данные по market cap
            mc_data = filtered_df[filtered_df['market_cap_performance'].notna()].copy()
            if len(mc_data) > 0:
                colors2 = ['green' if x >= 0 else 'red' for x in mc_data['market_cap_performance']]
                bars2 = ax2.barh(mc_data['ticker'], mc_data['market_cap_performance'], color=colors2, alpha=0.7)
                ax2.set_xlabel('Performance (%)')
                ax2.set_title(f'Performance по market cap — {period_label}')
                ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                ax2.grid(axis='x', linestyle='--', alpha=0.7)

                # Добавляем значения на столбцы
                for _i2, (_bar2, _val2) in enumerate(zip(bars2, mc_data['market_cap_performance'])):
                    _width2 = _bar2.get_width()
                    ax2.text(_width2, _bar2.get_y() + _bar2.get_height()/2,
                            f'{_val2:.1f}%',
                            ha='left' if _width2 >= 0 else 'right',
                            va='center', fontsize=9)

        plt.tight_layout()
        # Конвертируем фигуру в base64 для отображения
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
    # Дополнительный график: изменение market cap в абсолютных значениях
    if show_market_cap.value and 'market_cap_change' in filtered_df.columns and len(filtered_df) > 0:
        mc_change_data = filtered_df[filtered_df['market_cap_change'].notna()].copy()
        if len(mc_change_data) > 0:
            _fig2, _ax2 = plt.subplots(figsize=(14, max(6, len(mc_change_data) * 0.2)))

            # Сортируем по изменению
            mc_change_data = mc_change_data.sort_values('market_cap_change', ascending=True)

            _colors3 = ['green' if x >= 0 else 'red' for x in mc_change_data['market_cap_change']]
            _bars3 = _ax2.barh(mc_change_data['ticker'], mc_change_data['market_cap_change'] / 1e9, color=_colors3, alpha=0.7)
            _ax2.set_xlabel('Изменение market cap (млрд руб)')
            _ax2.set_title(f'Абсолютное изменение market cap — {period_label}')
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
    breadth_block,
    charts_display,
    heatmap_block,
    index_block,
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
    table,
    volume_block,
):
    # Основной layout
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
        table,
        charts_display,
    ])
    return


@app.cell(hide_code=True)
def _(filtered_df, imoex_ret, mo, period_label):
    # Сводка: рынок в целом за период
    _n = len(filtered_df)
    _up = int((filtered_df['price_performance'] > 0).sum()) if _n else 0
    _down = int((filtered_df['price_performance'] < 0).sum()) if _n else 0

    # Взвешенная по капитализации динамика: суммарная капитализация конец/начало
    _mc = filtered_df.dropna(subset=['first_market_cap', 'last_market_cap']) if _n else filtered_df
    if _n and len(_mc) > 0 and _mc['first_market_cap'].sum() > 0:
        _market_ret = (_mc['last_market_cap'].sum() / _mc['first_market_cap'].sum() - 1) * 100
        _mkt_str = f"{_market_ret:+.2f}%"
        _mc_total = _mc['last_market_cap'].sum() / 1e12
        _mc_delta = (_mc['last_market_cap'].sum() - _mc['first_market_cap'].sum()) / 1e9
        _mkt_extra = f" (капитализация {_mc_total:.1f} трлн руб, {_mc_delta:+,.0f} млрд руб)".replace(",", " ")
    else:
        _mkt_str = "н/д"
        _mkt_extra = ""

    if _n:
        _med = filtered_df['price_performance'].median()
        _top = filtered_df.nlargest(5, 'price_performance')
        _bot = filtered_df.nsmallest(5, 'price_performance')
        _top_str = ", ".join(f"**{_r.ticker}** {_r.price_performance:+.1f}%" for _r in _top.itertuples())
        _bot_str = ", ".join(f"**{_r.ticker}** {_r.price_performance:+.1f}%" for _r in _bot.itertuples())
        _imx_line = f"; IMOEX: **{imoex_ret:+.2f}%**" if imoex_ret is not None else ""
        market_summary = mo.md(f"""
    ### Итоги — {period_label}

    - **Рынок (взвешенно по капитализации): {_mkt_str}**{_mkt_extra}{_imx_line}; медианная бумага: {_med:+.2f}%
    - Выросло: **{_up}** | Упало: **{_down}** | Всего: {_n}
    - 📈 Лидеры: {_top_str}
    - 📉 Аутсайдеры: {_bot_str}
    """)
    else:
        market_summary = mo.md("Нет данных за выбранный период")
    return (market_summary,)


@app.cell(hide_code=True)
def _(base64, chart, filtered_df, io, market_cap_chart, mo, period_label, plt):
    # Графики включая Marimekko chart
    _all_charts = [
        mo.md("## Визуализация"),
        chart,
        market_cap_chart,
    ]

    # Добавляем Marimekko chart
    try:
        if len(filtered_df) > 0 and 'last_market_cap' in filtered_df.columns and 'price_performance' in filtered_df.columns:
            _marimekko_data = filtered_df[
                (filtered_df['last_market_cap'].notna()) & 
                (filtered_df['price_performance'].notna())
            ].copy()

            if len(_marimekko_data) > 0:
                _marimekko_data = _marimekko_data.sort_values('price_performance', ascending=False)
                _total_mc = _marimekko_data['last_market_cap'].sum()
                _marimekko_data['width'] = (_marimekko_data['last_market_cap'] / _total_mc) * 100

                # Рассчитываем позиции так, чтобы бары шли без пропусков
                _cumulative_width = 0
                _positions = []
                for _w in _marimekko_data['width']:
                    _positions.append(_cumulative_width)
                    _cumulative_width += _w
                _marimekko_data['position'] = _positions

                # Убеждаемся, что последний бар доходит до 100%
                if len(_marimekko_data) > 0:
                    _last_idx = _marimekko_data.index[-1]
                    _last_pos = _marimekko_data.loc[_last_idx, 'position']
                    _last_width = _marimekko_data.loc[_last_idx, 'width']
                    # Если последний бар не доходит до 100%, корректируем его ширину
                    if _last_pos + _last_width < 100:
                        _marimekko_data.loc[_last_idx, 'width'] = 100 - _last_pos

                _fig_m, _ax_m = plt.subplots(figsize=(16, 8))
                _colors_m = ['green' if x >= 0 else 'red' for x in _marimekko_data['price_performance']]

                # Масштаб подписей завязан на размах performance, а не на фиксированные
                # проценты — иначе на коротких периодах подписи "висят в воздухе"
                _perf_span = max(
                    abs(float(_marimekko_data['price_performance'].max())),
                    abs(float(_marimekko_data['price_performance'].min())),
                    1e-9,
                )
                _lbl_off = 0.03 * _perf_span

                for _idx, _row in _marimekko_data.iterrows():
                    _h = _row['price_performance']
                    _w = _row['width']
                    _l = _row['position']
                    _c = _colors_m[_marimekko_data.index.get_loc(_idx)]

                    if _h < 0:
                        _b = _h
                        _bh = abs(_h)
                    else:
                        _b = 0
                        _bh = _h

                    _ax_m.bar(_l, _bh, width=_w, bottom=_b, color=_c, alpha=0.7, edgecolor='black', linewidth=0.5)

                    if _w > 0.8:
                        _ax_m.text(_l + _w / 2, _h / 2, _row['ticker'], ha='center', va='center',
                                  fontsize=8 if _w > 2 else 6, rotation=0 if _w > 2 else 90,
                                  fontweight='bold', color='white' if abs(_h) > 0.4 * _perf_span else 'black')
                        if _w > 2.5:
                            _ax_m.text(_l + _w / 2, _h + (_lbl_off if _h >= 0 else -_lbl_off), f'{_h:.1f}%',
                                      ha='center', va='bottom' if _h >= 0 else 'top', fontsize=8, fontweight='bold')

                _ax_m.set_xlabel('Доля market cap (%)', fontsize=12)
                _ax_m.set_ylabel('Performance по цене (%)', fontsize=12)
                _ax_m.set_xlim(0, 100)
                _ax_m.set_title(f'Marimekko Chart: Performance по цене (ширина = market cap) — {period_label}',
                              fontsize=14, fontweight='bold')
                _ax_m.axhline(y=0, color='black', linestyle='-', linewidth=1)
                _ax_m.grid(axis='y', linestyle='--', alpha=0.5)

                from matplotlib.patches import Patch as _Patch2
                _ax_m.legend(handles=[
                    _Patch2(facecolor='green', alpha=0.7, label='Положительный performance'),
                    _Patch2(facecolor='red', alpha=0.7, label='Отрицательный performance')
                ], loc='upper right')

                plt.tight_layout()
                _buf_m = io.BytesIO()
                _fig_m.savefig(_buf_m, format='png', bbox_inches='tight', dpi=100)
                _buf_m.seek(0)
                _img_m = base64.b64encode(_buf_m.read()).decode()
                plt.close(_fig_m)
                _marimekko_html = mo.Html(f'<img src="data:image/png;base64,{_img_m}" style="max-width: 100%; height: auto;" />')

                _all_charts.extend([
                    mo.md("## Marimekko Chart (ширина = market cap, высота = performance)"),
                    _marimekko_html,
                ])
    except Exception:
        pass

    charts_display = mo.vstack(_all_charts)
    return (charts_display,)


@app.cell(hide_code=True)
def _(filtered_df, mo, pd):
    # Статистика
    stats_items = [
        f"- **Всего акций:** {len(filtered_df)}",
        f"- **С данными по market cap:** {len(filtered_df[filtered_df['market_cap_performance'].notna()]) if 'market_cap_performance' in filtered_df.columns else 0}",
        f"- **Средний performance по цене:** {filtered_df['price_performance'].mean():.2f}%",
        f"- **Медианный performance по цене:** {filtered_df['price_performance'].median():.2f}%",
    ]

    if 'market_cap_performance' in filtered_df.columns:
        mc_perf = filtered_df['market_cap_performance'].dropna()
        if len(mc_perf) > 0:
            stats_items.extend([
                f"- **Средний performance по market cap:** {mc_perf.mean():.2f}%",
                f"- **Медианный performance по market cap:** {mc_perf.median():.2f}%",
            ])

            # Топ-5 по росту и падению market cap
            top_gainers = filtered_df.nlargest(5, 'market_cap_performance')
            top_losers = filtered_df.nsmallest(5, 'market_cap_performance')

            stats_items.append("\n### Топ-5 по росту market cap:")
            for _, _row_g in top_gainers.iterrows():
                if pd.notna(_row_g['market_cap_performance']):
                    stats_items.append(f"- **{_row_g['ticker']}**: {_row_g['market_cap_performance']:.2f}%")

            stats_items.append("\n### Топ-5 по падению market cap:")
            for _, _row_l in top_losers.iterrows():
                if pd.notna(_row_l['market_cap_performance']):
                    stats_items.append(f"- **{_row_l['ticker']}**: {_row_l['market_cap_performance']:.2f}%")

    stats_text = "## Статистика\n\n" + "\n".join(stats_items)

    mo.md(stats_text)
    return


@app.cell(hide_code=True)
def _(base64, io, mo, moex, pd, period_end, period_label, period_start, plt):
    # IMOEX за период — из локального кэша indexes/IMOEX.parquet
    # (обновляется офлайн: python update_data.py, шаг 1b)
    imoex_ret = None
    try:
        _idx_df = moex.read_moex_index('IMOEX')
        _idx_df.index = pd.to_datetime(_idx_df.index)
        _win = _idx_df[(_idx_df.index >= period_start) & (_idx_df.index <= period_end)].sort_index()
        if len(_win) >= 2:
            imoex_ret = (float(_win['close'].iloc[-1]) / float(_win['close'].iloc[0]) - 1) * 100
            _figi, _axi = plt.subplots(figsize=(10, 2.6))
            _axi.plot(_win.index, _win['close'], color='#1f77b4', linewidth=1.8)
            _axi.fill_between(_win.index, _win['close'], float(_win['close'].min()), alpha=0.12, color='#1f77b4')
            _axi.set_title(f'IMOEX — {period_label}: {imoex_ret:+.2f}%', fontsize=12, fontweight='bold')
            _axi.grid(alpha=0.3)
            _axi.spines['top'].set_visible(False)
            _axi.spines['right'].set_visible(False)
            plt.tight_layout()
            _bufi = io.BytesIO()
            _figi.savefig(_bufi, format='png', bbox_inches='tight', dpi=100)
            _bufi.seek(0)
            _imgi = base64.b64encode(_bufi.read()).decode()
            plt.close(_figi)
            index_block = mo.Html(f'<img src="data:image/png;base64,{_imgi}" style="max-width: 100%; height: auto;" />')
        else:
            index_block = mo.md("*IMOEX: недостаточно данных за период — обновите кэш: `python update_data.py`*")
    except FileNotFoundError:
        index_block = mo.md("*Локальный кэш IMOEX не найден — выполните `python update_data.py` (шаг 1b)*")
    except Exception as _e_idx:
        index_block = mo.md(f"*IMOEX: ошибка чтения кэша — {_e_idx}*")
    return imoex_ret, index_block


@app.cell(hide_code=True)
def _(combined_df, mo, pd):
    # Ширина рынка: близость к 52-недельным экстремумам (по close за последний год)
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

    breadth_block = mo.md(
        f"**Ширина рынка (52 нед.):** у максимумов (в пределах 2%): **{len(_near_hi)}** ({_fmt_tickers(_near_hi)}) | "
        f"у минимумов: **{len(_near_lo)}** ({_fmt_tickers(_near_lo)})"
    )
    return (breadth_block,)


@app.cell(hide_code=True)
def _(base64, filtered_df, io, mo, moex, pd, period_label, plt):
    # Секторный разрез: динамика секторов, взвешенная по капитализации
    # Справочник тикер→сектор: metadata/sectors.csv
    import os as _osx
    _sec_path = _osx.path.join(moex.BASE_DIR, 'metadata', 'sectors.csv')
    if not _osx.path.exists(_sec_path):
        sector_block = mo.md("*Справочник секторов `metadata/sectors.csv` не найден — секторный разрез пропущен*")
    elif len(filtered_df) == 0:
        sector_block = mo.md("")
    else:
        _sec_map = pd.read_csv(_sec_path)
        _sdf = filtered_df.merge(_sec_map, on='ticker', how='left')
        _sdf['sector'] = _sdf['sector'].fillna('Прочее')

        _rows = []
        for _sec, _grp in _sdf.groupby('sector'):
            _gmc = _grp.dropna(subset=['first_market_cap', 'last_market_cap'])
            if len(_gmc) > 0 and _gmc['first_market_cap'].sum() > 0:
                _ret = (_gmc['last_market_cap'].sum() / _gmc['first_market_cap'].sum() - 1) * 100
            else:
                _ret = float(_grp['price_performance'].median())
            _rows.append({'sector': f"{_sec} ({len(_grp)})", 'ret': _ret})
        _sec_df = pd.DataFrame(_rows).sort_values('ret')

        _figs, _axs = plt.subplots(figsize=(10, max(3.0, 0.45 * len(_sec_df))))
        _colss = ['green' if _x >= 0 else 'red' for _x in _sec_df['ret']]
        _axs.barh(_sec_df['sector'], _sec_df['ret'], color=_colss, alpha=0.75)
        for _yi, _v in enumerate(_sec_df['ret']):
            _axs.text(_v, _yi, f' {_v:+.1f}% ', va='center',
                      ha='left' if _v >= 0 else 'right', fontsize=9, fontweight='bold')
        _axs.set_title(f'Сектора (взвешенно по капитализации) — {period_label}', fontsize=12, fontweight='bold')
        _axs.axvline(0, color='black', linewidth=0.8)
        _axs.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        _bufs = io.BytesIO()
        _figs.savefig(_bufs, format='png', bbox_inches='tight', dpi=100)
        _bufs.seek(0)
        _imgs = base64.b64encode(_bufs.read()).decode()
        plt.close(_figs)
        sector_block = mo.Html(f'<img src="data:image/png;base64,{_imgs}" style="max-width: 100%; height: auto;" />')
    return (sector_block,)


@app.cell(hide_code=True)
def _(base64, filtered_df, io, mo, np, period_label, plt):
    # Карта рынка: плитки по убыванию капитализации, цвет = performance
    _hm = filtered_df.dropna(subset=['price_performance']).copy()
    if len(_hm) == 0:
        heatmap_block = mo.md("")
    else:
        _hm['_mc'] = _hm['last_market_cap'].fillna(0)
        _hm = _hm.sort_values('_mc', ascending=False)

        _ncols = 10
        _nrows = int(np.ceil(len(_hm) / _ncols))
        _vmax = max(float(np.percentile(np.abs(_hm['price_performance']), 95)), 1e-9)
        _cmap = plt.get_cmap('RdYlGn')

        _figh, _axh = plt.subplots(figsize=(16, 0.9 * _nrows))
        for _ih, _rowh in enumerate(_hm.itertuples()):
            _rr, _cc = divmod(_ih, _ncols)
            _pv = float(_rowh.price_performance)
            _normv = 0.5 + max(-1.0, min(1.0, _pv / _vmax)) / 2
            _axh.add_patch(plt.Rectangle((_cc + 0.02, -_rr - 0.98), 0.96, 0.94, color=_cmap(_normv)))
            _axh.text(_cc + 0.5, -_rr - 0.40, str(_rowh.ticker),
                      ha='center', va='center', fontsize=8, fontweight='bold')
            _axh.text(_cc + 0.5, -_rr - 0.74, f'{_pv:+.1f}%', ha='center', va='center', fontsize=7)
        _axh.set_xlim(0, _ncols)
        _axh.set_ylim(-_nrows, 0)
        _axh.axis('off')
        _axh.set_title(f'Карта рынка (порядок = размер компании) — {period_label}',
                       fontsize=13, fontweight='bold')
        plt.tight_layout()
        _bufh = io.BytesIO()
        _figh.savefig(_bufh, format='png', bbox_inches='tight', dpi=110)
        _bufh.seek(0)
        _imgh = base64.b64encode(_bufh.read()).decode()
        plt.close(_figh)
        heatmap_block = mo.Html(f'<img src="data:image/png;base64,{_imgh}" style="max-width: 100%; height: auto;" />')
    return (heatmap_block,)


@app.cell(hide_code=True)
def _(combined_df, filtered_df, mo, pd, period_end, period_start):
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

    if len(_va) == 0:
        volume_block = mo.md("")
    else:
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
        volume_block = mo.md(
            "### Необычная активность\n\n"
            "Среднедневной оборот за период против среднего за предыдущие 90 дней:\n\n"
            "| Тикер | Оборот/день, млн руб | База, млн руб | Всплеск | Изм. цены |\n"
            "|---|---|---|---|---|\n" + "\n".join(_lines)
        )
    return (volume_block,)


if __name__ == "__main__":
    app.run()
