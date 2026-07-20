"""
# Stocks Performance Analysis

Интерактивный анализ performance акций с учетом market cap.
"""

import marimo

__generated_with = "0.8.0"
app = marimo.App(width="full")


@app.cell
def __():
    import moex_utils as moex
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime, timedelta
    import marimo as mo
    import io
    import base64
    return base64, datetime, io, mdates, mo, moex, np, pd, plt, timedelta


@app.cell
def __(mo):
    # UI элементы для выбора периода
    period_months = mo.ui.slider(
        start=1, 
        stop=24, 
        step=1, 
        value=1, 
        label="Период (месяцев):"
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
    
    return (
        min_market_cap,
        period_months,
        show_market_cap,
        sort_by,
    )


@app.cell
def __(moex):
    # Загружаем данные
    combined_df = moex.combine_moex_stocks()
    return combined_df,


@app.cell
def __(combined_df, datetime, period_months, pd, timedelta):
    # Функция для расчета performance
    def calculate_performance(df, period_months):
        """Рассчитывает performance за указанный период"""
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.DateOffset(months=period_months)
        
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
    
    perf_df, period_start, period_end = calculate_performance(combined_df, period_months.value)
    return calculate_performance, perf_df, period_end, period_start


@app.cell
def __(min_market_cap, perf_df, sort_by):
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
    
    return filtered_df,


@app.cell
def __(filtered_df, mo, pd, show_market_cap):
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
    return display_df, table


@app.cell
def __(base64, filtered_df, io, mo, np, period_months, plt, show_market_cap):
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
        ax1.set_title(f'Performance по цене за {period_months.value} месяц(ев)')
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
                ax2.set_title(f'Performance по market cap за {period_months.value} месяц(ев)')
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
    
    return chart,


@app.cell
def __(base64, filtered_df, io, mo, period_months, plt, show_market_cap):
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
            _ax2.set_title(f'Абсолютное изменение market cap за {period_months.value} месяц(ев)')
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
    
    return market_cap_chart,


@app.cell
def __(charts_display, min_market_cap, mo, period_end, period_months, period_start, show_market_cap, sort_by, table):
    # Основной layout
    period_start_str = period_start.strftime('%d.%m.%Y')
    period_end_str = period_end.strftime('%d.%m.%Y')
    
    mo.vstack([
        mo.hstack([period_months, show_market_cap, sort_by, min_market_cap], justify="start"),
        mo.md(f"## Результаты анализа performance за {period_months.value} месяц(ев)\n**Период:** {period_start_str} - {period_end_str}"),
        table,
        charts_display,
    ])
    return period_end_str, period_start_str


@app.cell
def __(base64, filtered_df, io, mo, period_months, plt):
    # Marimekko chart: Performance по цене, ширина пропорциональна market cap
    # Инициализируем переменную в начале
    marimekko_chart = mo.md("Загрузка...")
    
    try:
        if len(filtered_df) > 0 and 'last_market_cap' in filtered_df.columns and 'price_performance' in filtered_df.columns:
            marimekko_data = filtered_df[
                (filtered_df['last_market_cap'].notna()) & 
                (filtered_df['price_performance'].notna())
            ].copy()
            
            if len(marimekko_data) > 0:
                # Сортируем по performance от большего к меньшему
                marimekko_data = marimekko_data.sort_values('price_performance', ascending=False)
                
                # Рассчитываем ширину каждого бара на основе market cap
                total_market_cap = marimekko_data['last_market_cap'].sum()
                marimekko_data['width'] = (marimekko_data['last_market_cap'] / total_market_cap) * 100
                
                # Рассчитываем позиции так, чтобы бары шли без пропусков
                cumulative_width = 0
                positions = []
                for w in marimekko_data['width']:
                    positions.append(cumulative_width)
                    cumulative_width += w
                marimekko_data['position'] = positions
                
                # Убеждаемся, что последний бар доходит до 100%
                if len(marimekko_data) > 0:
                    last_idx = marimekko_data.index[-1]
                    last_pos = marimekko_data.loc[last_idx, 'position']
                    last_width = marimekko_data.loc[last_idx, 'width']
                    # Если последний бар не доходит до 100%, корректируем его ширину
                    if last_pos + last_width < 100:
                        marimekko_data.loc[last_idx, 'width'] = 100 - last_pos
                
                # Создаем график
                _fig4, _ax4 = plt.subplots(figsize=(16, 8))
                
                # Цвета: зеленый для положительных, красный для отрицательных
                _colors4 = ['green' if x >= 0 else 'red' for x in marimekko_data['price_performance']]
                
                # Рисуем бары (вертикальные)
                _bottom = 0
                for _idx_m, _row_m in marimekko_data.iterrows():
                    _bar_color = _colors4[marimekko_data.index.get_loc(_idx_m)]
                    _height = _row_m['price_performance']
                    _width = _row_m['width']
                    _left = _row_m['position']
                    
                    # Определяем bottom для отрицательных значений
                    if _height < 0:
                        _bar_bottom = _height
                        _bar_height = abs(_height)
                    else:
                        _bar_bottom = 0
                        _bar_height = _height
                    
                    _ax4.bar(
                        _left,
                        _bar_height,
                        width=_width,
                        bottom=_bar_bottom,
                        color=_bar_color,
                        alpha=0.7,
                        edgecolor='black',
                        linewidth=0.5
                    )
                    
                    # Добавляем подпись тикера и значение performance
                    if _width > 0.5:  # Показываем только если ширина достаточна
                        _text_y = _height / 2 if _height >= 0 else _height / 2
                        # Тикер
                        _ax4.text(
                            _left + _width / 2,
                            _text_y,
                            _row_m['ticker'],
                            ha='center',
                            va='center',
                            fontsize=7 if _width > 1 else 6,
                            rotation=90 if _width < 2 else 0,
                            fontweight='bold',
                            color='white' if abs(_height) > 10 else 'black'
                        )
                        # Значение performance (если место позволяет)
                        if _width > 2 and abs(_height) > 5:
                            _ax4.text(
                                _left + _width / 2,
                                _height + (5 if _height >= 0 else -5),
                                f'{_height:.1f}%',
                                ha='center',
                                va='bottom' if _height >= 0 else 'top',
                                fontsize=8,
                                fontweight='bold'
                            )
                
                _ax4.set_xlabel('Доля market cap (%)', fontsize=12)
                _ax4.set_ylabel('Performance по цене (%)', fontsize=12)
                _ax4.set_xlim(0, 100)
                _ax4.set_title(f'Marimekko Chart: Performance по цене (ширина = market cap) за {period_months.value} месяц(ев)', 
                              fontsize=14, fontweight='bold')
                _ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
                _ax4.grid(axis='y', linestyle='--', alpha=0.5)
                
                # Добавляем легенду
                from matplotlib.patches import Patch as _Patch
                _legend_elements = [
                    _Patch(facecolor='green', alpha=0.7, label='Положительный performance'),
                    _Patch(facecolor='red', alpha=0.7, label='Отрицательный performance')
                ]
                _ax4.legend(handles=_legend_elements, loc='upper right')
                
                plt.tight_layout()
                # Конвертируем фигуру в base64 для отображения
                _buf4 = io.BytesIO()
                _fig4.savefig(_buf4, format='png', bbox_inches='tight', dpi=100)
                _buf4.seek(0)
                _img_base64_4 = base64.b64encode(_buf4.read()).decode()
                plt.close(_fig4)
                marimekko_chart = mo.Html(f'<img src="data:image/png;base64,{_img_base64_4}" style="max-width: 100%; height: auto;" />')
            else:
                marimekko_chart = mo.md("Нет данных для построения Marimekko chart (нужны данные по market cap)")
        else:
            marimekko_chart = mo.md("Нет данных для построения Marimekko chart")
    except Exception as e:
        marimekko_chart = mo.md(f"Ошибка при построении Marimekko chart: {str(e)}")
    
    return marimekko_chart,


@app.cell
def __(base64, chart, filtered_df, io, market_cap_chart, mo, period_months, plt):
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
                    
                    if _w > 0.5:
                        _ax_m.text(_l + _w / 2, _h / 2, _row['ticker'], ha='center', va='center',
                                  fontsize=7 if _w > 1 else 6, rotation=90 if _w < 2 else 0,
                                  fontweight='bold', color='white' if abs(_h) > 10 else 'black')
                        if _w > 2 and abs(_h) > 5:
                            _ax_m.text(_l + _w / 2, _h + (5 if _h >= 0 else -5), f'{_h:.1f}%',
                                      ha='center', va='bottom' if _h >= 0 else 'top', fontsize=8, fontweight='bold')
                
                _ax_m.set_xlabel('Доля market cap (%)', fontsize=12)
                _ax_m.set_ylabel('Performance по цене (%)', fontsize=12)
                _ax_m.set_xlim(0, 100)
                _ax_m.set_title(f'Marimekko Chart: Performance по цене (ширина = market cap) за {period_months.value} месяц(ев)', 
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
    return charts_display,


@app.cell
def __(filtered_df, mo, pd):
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
    return mc_perf, stats_items, stats_text, top_gainers, top_losers


if __name__ == "__main__":
    app.run()
