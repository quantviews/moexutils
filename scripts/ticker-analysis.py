"""
marimo notebook: Анализ доходности и волатильности тикера с учетом дивидендов

Использование:
1. Установите marimo: pip install marimo
2. Запустите ноутбук: marimo edit ticker-analysis.py
3. Выберите тикер из выпадающего списка
4. Ноутбук автоматически загрузит данные и рассчитает метрики

Функционал:
- Выбор тикера через интерактивный виджет
- Расчет метрик доходности (CAGR, общая доходность, дивидендная доходность)
- Расчет волатильности (годовая, downside)
- Визуализация: цены, распределение доходностей, просадки, волатильность, объемы

Все расчеты учитывают дивиденды через adj_close.
"""

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import moex_utils as moex
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime, timedelta
    import os
    import marimo as mo
    from scipy import stats
    return mo, moex, np, os, pd, plt, stats


@app.cell
def _(os):
    # Получаем список доступных тикеров
    DATA_FOLDER = "data"
    available_tickers = sorted([
        d for d in os.listdir(DATA_FOLDER) 
        if os.path.isdir(os.path.join(DATA_FOLDER, d)) and 
           os.path.exists(os.path.join(DATA_FOLDER, d, f"{d}.parquet"))
    ]) if os.path.exists(DATA_FOLDER) else ["SBER", "LKOH", "GAZP"]
    return (available_tickers,)


@app.cell
def _(available_tickers, mo):
    # Виджет выбора тикера
    ticker = mo.ui.dropdown(
        options=available_tickers,
        value=available_tickers[0] if available_tickers else "SBER",
        label="Выберите тикер:"
    )
    ticker
    return (ticker,)


@app.cell
def _(moex, np, pd, ticker):
    # Загрузка данных
    try:
        df = moex.read_moex_stock(ticker.value)

        # Проверяем наличие adj_close
        if 'adj_close' not in df.columns:
            print(f"⚠️  Внимание: adj_close не найден для {ticker.value}. Используется close.")
            df['adj_close'] = df['close']

        # Используем adj_close для анализа (с учетом дивидендов)
        price_series = df['adj_close'].dropna().sort_index()

        # Расчет лог-доходностей
        log_returns = np.log(price_series / price_series.shift(1)).dropna()
        simple_returns = price_series.pct_change().dropna()

        print(f"✅ Данные загружены для {ticker.value}")
        print(f"📅 Период: {df.index.min().strftime('%Y-%m-%d')} - {df.index.max().strftime('%Y-%m-%d')}")
        print(f"📊 Количество торговых дней: {len(df)}")

    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        df = pd.DataFrame()
        price_series = pd.Series()
        log_returns = pd.Series()
        simple_returns = pd.Series()

    # df, price_series, log_returns, simple_returns
    return df, log_returns, price_series, simple_returns


@app.cell
def _(df, log_returns, np, price_series):
    # Расчет метрик доходности
    if len(price_series) > 0:
        start_date = price_series.index.min()
        end_date = price_series.index.max()
        years = (end_date - start_date).days / 365.25

        # CAGR (Compound Annual Growth Rate)
        start_price = price_series.iloc[0]
        end_price = price_series.iloc[-1]
        cagr = ((end_price / start_price) ** (1 / years)) - 1 if years > 0 and start_price > 0 else 0

        # CAGR через лог-доходности (более точный)
        if len(log_returns) > 0:
            total_log_return = log_returns.sum()
            cagr_log = np.exp(total_log_return / years) - 1 if years > 0 else 0
        else:
            cagr_log = 0

        # Средняя годовая доходность
        mean_daily_return = log_returns.mean() if len(log_returns) > 0 else 0
        mean_annual_return = mean_daily_return * 252  # 252 торговых дня в году

        # Общая доходность за период
        total_return = (end_price / start_price) - 1 if start_price > 0 else 0

        # Дивидендная доходность (разница между adj_close и close)
        if 'close' in df.columns and 'adj_close' in df.columns:
            close_cagr = ((df['close'].iloc[-1] / df['close'].iloc[0]) ** (1 / years)) - 1 if years > 0 and df['close'].iloc[0] > 0 else 0
            dividend_yield = cagr - close_cagr
        else:
            dividend_yield = 0
            close_cagr = 0

        metrics_return = {
            'CAGR (цена + дивиденды)': f"{cagr:.2%}",
            'CAGR (через лог-доходности)': f"{cagr_log:.2%}",
            'Средняя годовая доходность': f"{mean_annual_return:.2%}",
            'Общая доходность за период': f"{total_return:.2%}",
            'CAGR (только цена)': f"{close_cagr:.2%}",
            'Дивидендная доходность (годовая)': f"{dividend_yield:.2%}",
            'Период анализа (лет)': f"{years:.2f}"
        }
    else:
        metrics_return = {}
        cagr = 0
        cagr_log = 0
        mean_annual_return = 0
        total_return = 0
        dividend_yield = 0
        years = 0

    metrics_return, cagr, cagr_log, mean_annual_return, total_return, dividend_yield, years
    return (metrics_return,)


@app.cell
def _(log_returns, np):
    # Расчет волатильности
    if len(log_returns) > 0:
        # Волатильность (стандартное отклонение)
        daily_volatility = log_returns.std()
        annual_volatility = daily_volatility * np.sqrt(252)

        # Downside volatility (только отрицательные доходности)
        downside_returns = log_returns[log_returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0

        # Дневная волатильность
        daily_volatility_pct = daily_volatility * 100

        metrics_volatility = {
            'Дневная волатильность': f"{daily_volatility_pct:.2f}%",
            'Годовая волатильность': f"{annual_volatility:.2%}",
            'Downside волатильность (годовая)': f"{downside_volatility:.2%}"
        }
    else:
        metrics_volatility = {}
        annual_volatility = 0
        daily_volatility = 0
        downside_volatility = 0

    metrics_volatility, annual_volatility, daily_volatility, downside_volatility
    return (metrics_volatility,)


@app.cell
def _(metrics_return, metrics_volatility, mo):
    # Вывод всех метрик
    return_metrics = mo.md(f"""
    ## 📈 Метрики доходности

    {chr(10).join([f"- **{k}**: {v}" for k, v in metrics_return.items()])}
    """)

    volatility_metrics = mo.md(f"""
    ## 📊 Метрики волатильности

    {chr(10).join([f"- **{k}**: {v}" for k, v in metrics_volatility.items()])}
    """)

    [return_metrics, volatility_metrics]
    return


@app.cell
def _(df, plt, ticker):
    # График 1: Цены Close vs Adj Close
    if len(df) > 0 and 'close' in df.columns and 'adj_close' in df.columns:
        try:
            plt.close(1)  # Закрываем предыдущую фигуру, если она существует
        except:
            pass
        fig1, ax1_plot = plt.subplots(num=1, figsize=(14, 6))

        ax1_plot.plot(df.index, df['close'], label='Close (цена)', color='blue', alpha=0.7, linewidth=1.5)
        ax1_plot.plot(df.index, df['adj_close'], label='Adj Close (цена + дивиденды)', color='orange', alpha=0.7, linewidth=1.5)

        ax1_plot.set_title(f'{ticker.value} - Сравнение цен Close и Adj Close', fontsize=14, fontweight='bold')
        ax1_plot.set_xlabel('Дата', fontsize=12)
        ax1_plot.set_ylabel('Цена (RUB)', fontsize=12)
        ax1_plot.legend(fontsize=11)
        ax1_plot.grid(True, alpha=0.3)
        ax1_plot.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()
        del fig1, ax1_plot
    else:
        print("Нет данных для построения графика")
    return


@app.cell
def _(log_returns, plt, stats):
    # График 2: Распределение доходностей
    # Добавлен ticker в зависимости для принудительного обновления при смене тикера
    if len(log_returns) > 0:
        try:
            plt.close(2)  # Закрываем предыдущую фигуру, если она существует
        except:
            pass
        fig2, (ax2_hist, ax2_qq) = plt.subplots(1, 2, num=2, figsize=(14, 5))

        # Гистограмма доходностей
        ax2_hist.hist(log_returns * 100, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax2_hist.axvline(log_returns.mean() * 100, color='red', linestyle='--', linewidth=2, label=f'Среднее: {log_returns.mean()*100:.2f}%')
        ax2_hist.set_title('Распределение дневных доходностей', fontsize=12, fontweight='bold')
        ax2_hist.set_xlabel('Доходность (%)', fontsize=11)
        ax2_hist.set_ylabel('Частота', fontsize=11)
        ax2_hist.legend()
        ax2_hist.grid(True, alpha=0.3)

        # Q-Q plot для проверки нормальности
        stats.probplot(log_returns, dist="norm", plot=ax2_qq)
        ax2_qq.set_title('Q-Q Plot (проверка нормальности)', fontsize=12, fontweight='bold')
        ax2_qq.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        del fig2, ax2_hist, ax2_qq
    else:
        print("Нет данных для построения графика")
    return


@app.cell
def _(plt, price_series, simple_returns, ticker):
    # График 3: Просадки (Drawdown)
    if len(price_series) > 0 and len(simple_returns) > 0:
        cumulative = (1 + simple_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        fig3, (ax3_cum, ax3_dd) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # График цены и максимума
        ax3_cum.plot(cumulative.index, cumulative.values, label='Накопленная доходность', color='blue', linewidth=2)
        ax3_cum.plot(running_max.index, running_max.values, label='Растущий максимум', color='green', linestyle='--', linewidth=1.5)
        ax3_cum.fill_between(cumulative.index, cumulative.values, running_max.values, 
                         where=(cumulative < running_max), alpha=0.3, color='red', label='Просадка')
        ax3_cum.set_title(f'{ticker.value} - Накопленная доходность и просадки', fontsize=14, fontweight='bold')
        ax3_cum.set_ylabel('Накопленная доходность', fontsize=12)
        ax3_cum.legend(fontsize=11)
        ax3_cum.grid(True, alpha=0.3)

        # График просадок
        ax3_dd.fill_between(drawdown.index, 0, drawdown.values * 100, alpha=0.5, color='red')
        ax3_dd.plot(drawdown.index, drawdown.values * 100, color='darkred', linewidth=1.5)
        ax3_dd.set_title('Просадка (Drawdown) в процентах', fontsize=12, fontweight='bold')
        ax3_dd.set_xlabel('Дата', fontsize=12)
        ax3_dd.set_ylabel('Просадка (%)', fontsize=12)
        ax3_dd.grid(True, alpha=0.3)
        ax3_dd.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()
        del fig3, ax3_cum, ax3_dd
    else:
        print("Нет данных для построения графика")
    return


@app.cell
def _(log_returns, np, plt, ticker):
    # График 4: Скользящая волатильность
    if len(log_returns) > 0:
        # Скользящая волатильность (30, 60, 90 дней)
        windows = [30, 60, 90, 252]
        fig4, ax4 = plt.subplots(figsize=(14, 6))

        for window in windows:
            rolling_vol = log_returns.rolling(window=window).std() * np.sqrt(252) * 100
            ax4.plot(rolling_vol.index, rolling_vol.values, 
                   label=f'{window} дней', linewidth=1.5, alpha=0.8)

        ax4.set_title(f'{ticker.value} - Скользящая годовая волатильность', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Дата', fontsize=12)
        ax4.set_ylabel('Волатильность (%)', fontsize=12)
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()
        del fig4, ax4
    else:
        print("Нет данных для построения графика")
    return


@app.cell
def _(df, plt, ticker):
    # График 5: Объемы торгов
    if len(df) > 0 and 'volume' in df.columns:
        fig5, (ax5_price, ax5_vol) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # Цена и объем
        ax5_price.plot(df.index, df['adj_close'], color='blue', linewidth=1.5)
        ax5_price.set_ylabel('Цена (RUB)', fontsize=12, color='blue')
        ax5_price.tick_params(axis='y', labelcolor='blue')
        ax5_price.set_title(f'{ticker.value} - Цена и объем торгов', fontsize=14, fontweight='bold')
        ax5_price.grid(True, alpha=0.3)

        ax5_vol.bar(df.index, df['volume'], color='gray', alpha=0.6, width=0.8)
        ax5_vol.set_ylabel('Объем', fontsize=12)
        ax5_vol.set_xlabel('Дата', fontsize=12)
        ax5_vol.grid(True, alpha=0.3, axis='y')
        ax5_vol.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()
        del fig5, ax5_price, ax5_vol
    else:
        print("Нет данных для построения графика")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
