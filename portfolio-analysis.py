"""
marimo notebook: Анализ портфеля - доходность и волатильность

Использование:
1. Установите marimo: pip install marimo
2. Запустите ноутбук: marimo edit portfolio-analysis.py
3. Выберите период анализа через слайдер
4. Ноутбук автоматически отфильтрует тикеры и построит график

Функционал:
- Выбор периода анализа через слайдер (годы)
- Автоматический отбор тикеров с полной историей за период
- Расчет CAGR по лог-доходностям (с учетом дивидендов через adj_close)
- Расчет годовой волатильности
- График доходность-волатильность (scatter plot)
"""

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import moex_utils as moex
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime, timedelta
    import marimo as mo
    return mo, moex, np, pd, plt


@app.cell
def _(moex):
    # Загрузка всех данных
    combined_stocks = moex.combine_moex_stocks()
    latest_date = combined_stocks.index.max()
    earliest_date = combined_stocks.index.min()

    # Вычисляем максимальный доступный период в годах
    max_years = (latest_date - earliest_date).days / 365.25

    print(f"✅ Данные загружены")
    print(f"📅 Период данных: {earliest_date.strftime('%Y-%m-%d')} - {latest_date.strftime('%Y-%m-%d')}")
    print(f"📊 Максимальный период анализа: {max_years:.1f} лет")
    print(f"📈 Всего тикеров: {combined_stocks['ticker'].nunique()}")

    return combined_stocks, latest_date, max_years


@app.cell
def _(max_years, mo):
    # Слайдер для выбора периода анализа (в годах)
    years_slider = mo.ui.slider(
        start=1,
        stop=int(max_years),
        step=1,
        value=5,
        label="Период анализа (лет):"
    )
    
    # Безрисковая ставка для расчета Sharpe Ratio (в процентах)
    risk_free_rate_slider = mo.ui.slider(
        start=0,
        stop=20,
        step=0.5,
        value=8.0,
        label="Безрисковая ставка (%):"
    )
    
    [years_slider, risk_free_rate_slider]
    return (risk_free_rate_slider, years_slider)


@app.cell
def _(combined_stocks, latest_date, pd, years_slider):
    # Фильтрация данных по выбранному периоду
    years = years_slider.value
    start_date = latest_date - pd.DateOffset(years=years)

    # Фильтруем данные за выбранный период
    df_filtered = combined_stocks[combined_stocks.index >= start_date].copy()
    df_filtered = df_filtered.sort_values(by='ticker').sort_index()

    print(f"📅 Выбранный период: {start_date.strftime('%Y-%m-%d')} - {latest_date.strftime('%Y-%m-%d')} ({years} лет)")
    print(f"📊 Записей в отфильтрованных данных: {len(df_filtered)}")

    return df_filtered, start_date, years


@app.cell
def _(df_filtered, latest_date, start_date):
    # Фильтрация тикеров с полной историей за период
    # Считаем историю полной, если первая дата - не позже 30 дней от начала периода,
    # а последняя - не раньше 30 дней до конца периода
    valid_tickers = []

    for ticker, group in df_filtered.groupby('ticker'):
        min_date = group.index.min()
        max_date = group.index.max()

        # Проверка полноты истории
        days_from_start = (min_date - start_date).days
        days_to_end = (latest_date - max_date).days

        # История считается полной, если отклонение не более 30 дней
        if days_from_start <= 30 and days_to_end <= 30:
            valid_tickers.append(ticker)

    print(f"✅ Тикеров с полной историей: {len(valid_tickers)} из {df_filtered['ticker'].nunique()}")

    return (valid_tickers,)


@app.cell
def _(df_filtered, np, pd, risk_free_rate_slider, valid_tickers):
    # Расчет CAGR, волатильности и Sharpe Ratio для каждого тикера
    risk_free_rate = risk_free_rate_slider.value / 100.0  # Конвертируем проценты в десятичную дробь
    results = []

    for ticker_name in valid_tickers:
        ticker_data = df_filtered[df_filtered['ticker'] == ticker_name].sort_index()

        # Используем adj_close если есть, иначе close
        if 'adj_close' in ticker_data.columns:
            price_series = ticker_data['adj_close'].dropna()
        else:
            price_series = ticker_data['close'].dropna()

        if len(price_series) < 2:
            continue

        # Проверка на валидность цен (должны быть положительными)
        price_series = price_series[price_series > 0]
        if len(price_series) < 2:
            continue
        
        # Проверка на выбросы/аномалии в данных
        # Убираем значения, которые отличаются от медианы более чем в 100 раз
        # Это помогает отфильтровать ошибки в данных (например, неправильно рассчитанный adj_close)
        median_price = price_series.median()
        if median_price > 0:
            # Фильтруем значения в разумном диапазоне (от 0.01 до 1000 раз от медианы)
            price_series = price_series[
                (price_series >= median_price * 0.01) & 
                (price_series <= median_price * 1000)
            ]
        
        if len(price_series) < 2:
            continue

        # Расчет лог-доходностей с проверкой на валидность
        price_ratio = price_series / price_series.shift(1)
        price_ratio = price_ratio[price_ratio > 0]  # Убираем нулевые и отрицательные значения
        log_returns = np.log(price_ratio).dropna()

        if len(log_returns) == 0:
            continue

        # Расчет дат для информации
        start_date_ticker = price_series.index.min()
        end_date_ticker = price_series.index.max()
        
        # Расчет периода в годах через торговые дни (более точный для финансовых расчетов)
        # 252 - стандартное количество торговых дней в году
        num_trading_days = len(log_returns)
        years_ticker = num_trading_days / 252.0

        if years_ticker <= 0:
            continue

        # CAGR через лог-доходности (годовое выражение)
        # total_log_return - сумма всех дневных лог-доходностей
        # Деление на years_ticker (количество лет в торговых днях) дает годовую доходность
        total_log_return = log_returns.sum()
        cagr = np.exp(total_log_return / years_ticker) - 1

        # Годовая волатильность
        daily_volatility = log_returns.std()
        annual_volatility = daily_volatility * np.sqrt(252)

        # Коэффициент Шарпа (Sharpe Ratio)
        # Sharpe = (CAGR - risk_free_rate) / volatility
        if annual_volatility > 0:
            sharpe_ratio = (cagr - risk_free_rate) / annual_volatility
        else:
            sharpe_ratio = None  # Нельзя рассчитать при нулевой волатильности

        results.append({
            'ticker': ticker_name,
            'cagr': cagr,
            'volatility': annual_volatility,
            'sharpe': sharpe_ratio,
            'start_date': start_date_ticker,
            'end_date': end_date_ticker,
            'years': years_ticker
        })

    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        print(f"✅ Рассчитано метрик для {len(results_df)} тикеров")
        print(f"📈 Средняя CAGR: {results_df['cagr'].mean():.2%}")
        print(f"📊 Средняя волатильность: {results_df['volatility'].mean():.2%}")
        sharpe_mean = results_df['sharpe'].dropna().mean()
        if not pd.isna(sharpe_mean):
            print(f"📉 Средний Sharpe Ratio: {sharpe_mean:.2f}")
    else:
        print("⚠️ Нет данных для расчета")

    return (results_df,)


@app.cell
def _(mo, results_df, risk_free_rate_slider):
    # Вывод статистики
    if len(results_df) > 0:
        stats_text = mo.md(f"""
        ## 📊 Статистика по отобранным тикерам

        - **Количество тикеров**: {len(results_df)}
        - **Средняя CAGR**: {results_df['cagr'].mean():.2%}
        - **Медианная CAGR**: {results_df['cagr'].median():.2%}
        - **Средняя волатильность**: {results_df['volatility'].mean():.2%}
        - **Медианная волатильность**: {results_df['volatility'].median():.2%}
        - **Средний Sharpe Ratio**: {results_df['sharpe'].dropna().mean():.2f} (безрисковая ставка: {risk_free_rate_slider.value}%)
        - **Медианный Sharpe Ratio**: {results_df['sharpe'].dropna().median():.2f}
        - **Максимальная CAGR**: {results_df['cagr'].max():.2%} ({results_df.loc[results_df['cagr'].idxmax(), 'ticker']})
        - **Минимальная волатильность**: {results_df['volatility'].min():.2%} ({results_df.loc[results_df['volatility'].idxmin(), 'ticker']})
        - **Максимальный Sharpe Ratio**: {results_df['sharpe'].dropna().max():.2f} ({results_df.loc[results_df['sharpe'].dropna().idxmax(), 'ticker']})
        """)
    else:
        stats_text = mo.md("⚠️ Нет данных для отображения")
    
    stats_text
    return


@app.cell(hide_code=True)
def _(pd, plt, results_df, years):
    # График доходность-волатильность
    if len(results_df) > 0:
        try:
            plt.close(1)
        except:
            pass

        fig1, ax1 = plt.subplots(num=1, figsize=(14, 8))

        # Scatter plot с цветом по Sharpe Ratio
        # Используем Sharpe Ratio для цвета, если доступен, иначе используем CAGR/Volatility
        color_data = results_df['sharpe'].fillna(results_df['cagr'] / results_df['volatility'])
        
        scatter = ax1.scatter(
            results_df['volatility'] * 100,  # Волатильность в процентах
            results_df['cagr'] * 100,  # CAGR в процентах
            s=100,
            alpha=0.6,
            c=color_data,  # Цвет по Sharpe Ratio
            cmap='RdYlGn',
            edgecolors='black',
            linewidths=0.5
        )

        # Подписи тикеров (топ по Sharpe Ratio и топ/низ по доходности)
        top_n = 5
        top_sharpe = results_df.nlargest(3, 'sharpe')
        top_tickers = results_df.nlargest(top_n, 'cagr')
        bottom_tickers = results_df.nsmallest(3, 'cagr')
        
        # Объединяем для подписей (убираем дубликаты)
        labels_df = pd.concat([top_sharpe, top_tickers, bottom_tickers]).drop_duplicates(subset=['ticker'])

        for _, ticker_row in labels_df.iterrows():
            ax1.annotate(
                ticker_row['ticker'],
                xy=(ticker_row['volatility'] * 100, ticker_row['cagr'] * 100),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', alpha=0.5)
            )

        # Линия нулевой доходности
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

        # Настройка осей
        ax1.set_xlabel('Годовая волатильность (%)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('CAGR (%)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Доходность vs Волатильность (период: {years} лет)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Цветовая шкала
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Sharpe Ratio', fontsize=10)

        plt.tight_layout()
        plt.show()
        del fig1, ax1
    else:
        print("Нет данных для построения графика")
    return


@app.cell
def _(mo, pd, results_df):
    # Таблица с результатами (топ тикеров)
    if len(results_df) > 0:
        # Сортируем по Sharpe Ratio (топ по качеству доходности)
        top_results = results_df.nlargest(20, 'sharpe')[['ticker', 'cagr', 'volatility', 'sharpe']].copy()

        # Форматируем таблицу вручную
        table_rows = []
        table_rows.append("| Тикер | CAGR | Волатильность | Sharpe Ratio |")
        table_rows.append("|-------|------|----------------|--------------|")
        for _, result_row in top_results.iterrows():
            sharpe_str = f"{result_row['sharpe']:.2f}" if pd.notna(result_row['sharpe']) else "N/A"
            table_rows.append(f"| {result_row['ticker']} | {result_row['cagr']:.2%} | {result_row['volatility']:.2%} | {sharpe_str} |")

        table_text = mo.md(f"""
        ## 📋 Топ-20 тикеров по Sharpe Ratio

        {chr(10).join(table_rows)}
        """)
    else:
        table_text = mo.md("⚠️ Нет данных для отображения")
    
    table_text
    return


if __name__ == "__main__":
    app.run()
