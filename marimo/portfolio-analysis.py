"""
marimo notebook: Анализ портфеля - доходность, волатильность, эффективная граница

Использование:
1. Установите marimo и PyPortfolioOpt: pip install marimo PyPortfolioOpt
2. Запустите ноутбук: marimo edit portfolio-analysis.py
3. Выберите период анализа и при необходимости исключите тикеры
4. Ноутбук построит scatter доходность-волатильность, эффективную границу и веса

Функционал:
- Выбор периода анализа, безрисковая ставка, исключение тикеров
- CAGR, волатильность, Sharpe по тикерам; график активов
- Эффективная граница и портфель max Sharpe (PyPortfolioOpt)
- График весов эффективного портфеля, корреляция доходностей
- Сравнение max Sharpe vs min Volatility, текстовые пояснения теории
"""

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path

    # moex_utils лежит в корне проекта (родительская папка от marimo/)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import moex_utils as moex
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime, timedelta
    import marimo as mo
    try:
        from pypfopt import expected_returns, risk_models, EfficientFrontier
        pypfopt_available = True
    except ImportError:
        expected_returns, risk_models, EfficientFrontier = None, None, None
        pypfopt_available = False
    return (
        EfficientFrontier,
        expected_returns,
        mo,
        moex,
        np,
        pd,
        plt,
        pypfopt_available,
        risk_models,
    )


@app.cell(hide_code=True)
def _(mo, moex):
    # Папка с данными (поддиректории по тикерам, в каждой <ticker>.parquet)
    data_folder_input = mo.ui.text(
        value=moex.DATA_FOLDER,
        label="Папка с данными:",
    )
    data_folder_input
    return (data_folder_input,)


@app.cell(hide_code=True)
def _(data_folder_input, moex):
    # Загрузка всех данных
    data_folder = data_folder_input.value or None
    combined_stocks = moex.combine_moex_stocks(data_folder=data_folder)
    latest_date = combined_stocks.index.max()
    earliest_date = combined_stocks.index.min()

    # Вычисляем максимальный доступный период в годах
    max_years = (latest_date - earliest_date).days / 365.25

    print(f"✅ Данные загружены")
    print(f"📅 Период данных: {earliest_date.strftime('%Y-%m-%d')} - {latest_date.strftime('%Y-%m-%d')}")
    print(f"📊 Максимальный период анализа: {max_years:.1f} лет")
    print(f"📈 Всего тикеров: {combined_stocks['ticker'].nunique()}")
    return combined_stocks, latest_date, max_years


@app.cell(hide_code=True)
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

    # Тикеры для исключения из анализа (через запятую)
    exclude_tickers_input = mo.ui.text(
        value="",
        label="Исключить тикеры (через запятую):",
    )

    # Период для построения портфеля: по умолчанию весь датасет
    portfolio_period_dropdown = mo.ui.dropdown(
        {"Весь датасет": "full", "Как период анализа": "analysis"},
        value="Весь датасет",
        label="Период для портфеля:",
    )

    # Макс. доля одной бумаги в портфеле (%); 100 = без ограничения
    max_weight_slider = mo.ui.slider(
        start=5,
        stop=100,
        step=5,
        value=100,
        label="Макс. доля одной бумаги (%); 100 = без ограничения:",
    )

    # Оценка ковариационной матрицы: выборочная или сжатие Ledoit-Wolf.
    # Ledoit-Wolf дает лучше обусловленную матрицу (меньше предупреждений
    # решателя и стабильнее граница при большом числе бумаг)
    cov_method_dropdown = mo.ui.dropdown(
        {"Выборочная (sample_cov)": "sample", "Ledoit-Wolf (сжатие)": "lw"},
        value="Выборочная (sample_cov)",
        label="Оценка ковариации:",
    )

    [years_slider, risk_free_rate_slider, exclude_tickers_input, portfolio_period_dropdown, max_weight_slider, cov_method_dropdown]
    return (
        cov_method_dropdown,
        exclude_tickers_input,
        max_weight_slider,
        portfolio_period_dropdown,
        risk_free_rate_slider,
        years_slider,
    )


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(df_filtered, exclude_tickers_input, latest_date, start_date):
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

    # Исключаем тикеры по вводу пользователя (через запятую, без учёта регистра)
    excluded = {t.strip().upper() for t in exclude_tickers_input.value.split(",") if t.strip()}
    valid_tickers = [t for t in valid_tickers if t.upper() not in excluded]
    if excluded:
        print(f"🚫 Исключено тикеров: {len(excluded)} ({', '.join(sorted(excluded))})")
    print(f"✅ Тикеров с полной историей: {len(valid_tickers)} из {df_filtered['ticker'].nunique()}")
    return (valid_tickers,)


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(combined_stocks, df_filtered, portfolio_period_dropdown, valid_tickers):
    # Матрица цен для PyPortfolioOpt: строки — даты, столбцы — тикеры
    # Период: весь датасет (по умолчанию) или как период анализа
    price_col = 'adj_close' if 'adj_close' in df_filtered.columns else 'close'
    use_full = portfolio_period_dropdown.value == "full"
    if use_full:
        _df = combined_stocks[combined_stocks['ticker'].isin(valid_tickers)].copy()
    else:
        _df = df_filtered[df_filtered['ticker'].isin(valid_tickers)].copy()
    prices_wide = _df.pivot_table(index=_df.index, columns='ticker', values=price_col)
    prices_wide = prices_wide[[t for t in valid_tickers if t in prices_wide.columns]].dropna(how='any')
    return (prices_wide,)


@app.cell(hide_code=True)
def _(mo):
    # --- Теория: портфельный анализ и эффективная граница ---
    theory_intro = mo.md(r"""
    ## Портфельный анализ (Markowitz)

    **Диверсификация**: объединение активов в портфель может снизить риск без пропорционального снижения доходности — за счёт того, что доходности активов не движутся идеально одинаково (корреляция < 1).

    **Эффективная граница** — множество портфелей с максимальной доходностью при заданном уровне риска (или минимальным риском при заданной доходности). Оптимальный портфель по Шарпу — точка касания прямой от безрискового актива к этой границе (максимум отношения (доходность − безрисковая) / волатильность).

    **Риск** здесь измеряется волатильностью (с.к.о. доходности). 

    **Доходность** — средняя историческая (или ожидаемая). PyPortfolioOpt строит границу по модели Марковица (mean-variance).
    """)
    theory_intro
    return


@app.cell(hide_code=True)
def _(
    EfficientFrontier,
    cov_method_dropdown,
    expected_returns,
    max_weight_slider,
    np,
    prices_wide,
    pypfopt_available,
    risk_models,
):
    # Расчёт эффективной границы и оптимального (max Sharpe) портфеля
    ef_curve_vol, ef_curve_ret, weights_max_sharpe, perf_max_sharpe, mu_series, S_df = [], [], {}, None, None, None
    max_weight = max_weight_slider.value / 100.0  # 1.0 = без ограничения
    weight_bounds = (0, max_weight)
    if pypfopt_available and len(prices_wide.columns) >= 2 and len(prices_wide) >= 2:
        mu_series = expected_returns.mean_historical_return(prices_wide)
        if cov_method_dropdown.value == "lw":
            S_df = risk_models.CovarianceShrinkage(prices_wide).ledoit_wolf()
        else:
            S_df = risk_models.sample_cov(prices_wide)
        # Точки границы: для каждой целевой доходности — минимальный риск.
        # Крайние точки диапазона отбрасываем: задачи на mu.min()/mu.max()
        # вырожденные, решатель на них дает "Solution may be inaccurate"
        target_returns = np.linspace(float(mu_series.min()), float(mu_series.max()), 52)[1:-1]
        for r in target_returns:
            ef = EfficientFrontier(mu_series, S_df, weight_bounds=weight_bounds)
            try:
                ef.efficient_return(target_return=r)
                r_eff, v_eff, _ = ef.portfolio_performance()
                ef_curve_ret.append(r_eff)
                ef_curve_vol.append(v_eff)
            except Exception:
                pass
        ef_max_sharpe = EfficientFrontier(mu_series, S_df, weight_bounds=weight_bounds)
        weights_max_sharpe = ef_max_sharpe.max_sharpe()
        weights_max_sharpe = {k: v for k, v in weights_max_sharpe.items() if v > 1e-6}
        perf_max_sharpe = ef_max_sharpe.portfolio_performance()
    else:
        if not pypfopt_available:
            print("⚠️ PyPortfolioOpt не установлен: pip install PyPortfolioOpt")
        elif len(prices_wide.columns) < 2 or len(prices_wide) < 2:
            print("⚠️ Недостаточно данных для оптимизации (нужно ≥2 тикеров и наблюдений)")
    return (
        S_df,
        ef_curve_ret,
        ef_curve_vol,
        mu_series,
        perf_max_sharpe,
        weights_max_sharpe,
    )


@app.cell(hide_code=True)
def _(mo, perf_max_sharpe, weights_max_sharpe):
    # Вывод характеристик оптимального портфеля (max Sharpe)
    if perf_max_sharpe is not None and weights_max_sharpe:
        ret_opt, vol_opt, sr_opt = perf_max_sharpe
        opt_text = mo.md(f"""
        ## Оптимальный портфель (максимум Sharpe)

        - **Ожидаемая годовая доходность**: {ret_opt:.2%}
        - **Годовая волатильность**: {vol_opt:.2%}
        - **Sharpe Ratio**: {sr_opt:.2f}
        """)
    else:
        opt_text = mo.md("Оптимальный портфель не рассчитан (недостаточно данных или PyPortfolioOpt не установлен).")
    opt_text
    return


@app.cell(hide_code=True)
def _(
    S_df,
    ef_curve_ret,
    ef_curve_vol,
    mu_series,
    np,
    perf_max_sharpe,
    plt,
    weights_max_sharpe,
):
    # График: эффективная граница, отдельные активы, точка оптимального портфеля
    if ef_curve_vol and ef_curve_ret and mu_series is not None and S_df is not None:
        try:
            plt.close(2)
        except Exception:
            pass
        fig2, ax2 = plt.subplots(num=2, figsize=(12, 7))
        ax2.plot([v * 100 for v in ef_curve_vol], [r * 100 for r in ef_curve_ret], 'b-', lw=2, label='Эффективная граница')
        # Активы: волатильность = sqrt(diag(S)), доходность = mu
        vol_assets = np.sqrt(np.diag(S_df)) * 100
        ret_assets = mu_series.values * 100
        ax2.scatter(vol_assets, ret_assets, s=60, alpha=0.7, c='gray', edgecolors='black', label='Отдельные активы')
        for idx, t in enumerate(mu_series.index):
            ax2.annotate(t, (vol_assets[idx], ret_assets[idx]), xytext=(4, 4), textcoords='offset points', fontsize=8)
        if perf_max_sharpe is not None and weights_max_sharpe:
            r_star, v_star, _ = perf_max_sharpe
            ax2.scatter([v_star * 100], [r_star * 100], s=200, marker='*', c='gold', edgecolors='black', label='Портфель max Sharpe', zorder=5)
        ax2.set_xlabel('Волатильность (%)')
        ax2.set_ylabel('Доходность (%)')
        ax2.set_title('Эффективная граница и отдельные активы')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        del fig2, ax2
    return


@app.cell(hide_code=True)
def _(np, plt, prices_wide, weights_max_sharpe):
    # График весов в эффективном портфеле (max Sharpe): сверху вниз от макс к мин, все тикеры датасета
    if weights_max_sharpe is not None and prices_wide is not None and len(prices_wide.columns) > 0:
        try:
            plt.close(3)
        except Exception:
            pass
        # Все тикеры датасета, веса (0 если не вошли в портфель)
        all_tickers = list(prices_wide.columns)
        all_weights = [weights_max_sharpe.get(t, 0.0) for t in all_tickers]
        # Сортировка от макс к мин
        pairs = sorted(zip(all_tickers, all_weights), key=lambda x: x[1], reverse=True)
        tickers_sorted = [p[0] for p in pairs]
        w_sorted = [p[1] * 100 for p in pairs]
        n_tickers = len(tickers_sorted)
        fig3, ax3 = plt.subplots(num=3, figsize=(10, max(5, n_tickers * 0.35)))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_tickers))
        ax3.barh(tickers_sorted, w_sorted, color=colors)
        ax3.set_xlabel('Доля (%)')
        ax3.set_ylabel('Тикер')
        ax3.set_title('Веса в портфеле с максимальным Sharpe Ratio (сверху вниз: от макс к мин)')
        ax3.invert_yaxis()
        plt.tight_layout()
        plt.show()
        del fig3, ax3
    return


@app.cell(hide_code=True)
def _(plt, prices_wide):
    # Корреляционная матрица доходностей (для понимания диверсификации)
    if len(prices_wide.columns) >= 2 and len(prices_wide) >= 2:
        rets = prices_wide.pct_change().dropna()
        corr = rets.corr()
        try:
            plt.close(4)
        except Exception:
            pass
        fig4, ax4 = plt.subplots(num=4, figsize=(10, 8))
        im = ax4.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax4.set_xticks(range(len(corr.columns)))
        ax4.set_yticks(range(len(corr.columns)))
        ax4.set_xticklabels(corr.columns, rotation=45, ha='right')
        ax4.set_yticklabels(corr.columns)
        for ri in range(len(corr)):
            for cj in range(len(corr)):
                ax4.text(cj, ri, f'{corr.iloc[ri, cj]:.2f}', ha='center', va='center', fontsize=7)
        plt.colorbar(im, ax=ax4, label='Корреляция')
        ax4.set_title('Корреляция дневных доходностей активов')
        plt.tight_layout()
        plt.show()
        del fig4, ax4
    return


@app.cell(hide_code=True)
def _(mo):
    theory_diversification = mo.md(r"""
    ## Диверсификация и корреляция

    Чем ниже корреляция между активами, тем сильнее снижение риска портфеля при объединении. 

    Корреляция +1 — активы движутся одинаково, диверсификация не даёт выигрыша. Корреляция −1 — идеальная хеджирующая пара, не встречается в реальности.  

    На практике корреляции между акциями обычно положительные, но меньше 1, поэтому портфель всё равно менее волатилен, чем отдельные активы.
    """)
    theory_diversification
    return


@app.cell(hide_code=True)
def _(mo):
    theory_weights = mo.md(r"""
    ## Интерпретация весов

    **Веса** — доли капитала в каждом активе (сумма = 100%). Портфель **max Sharpe** максимизирует отношение (доходность − безрисковая ставка) / волатильность: он лежит на эффективной границе в точке касания прямой от безрискового актива. 

    Портфель **min Volatility** минимизирует волатильность; его доходность обычно ниже, но и риск меньше. Оба решения — long-only (короткие продажи отключены).
    """)
    theory_weights
    return


@app.cell
def _(
    EfficientFrontier,
    cov_method_dropdown,
    expected_returns,
    max_weight_slider,
    prices_wide,
    pypfopt_available,
    risk_models,
):
    # Портфель минимальной волатильности (для сравнения с max Sharpe)
    weights_min_vol, perf_min_vol = {}, None
    max_weight_mv = max_weight_slider.value / 100.0
    weight_bounds_mv = (0, max_weight_mv)
    if pypfopt_available and len(prices_wide.columns) >= 2 and len(prices_wide) >= 2:
        mu_mv = expected_returns.mean_historical_return(prices_wide)
        # Та же оценка ковариации, что и для max Sharpe, — иначе сравнение некорректно
        if cov_method_dropdown.value == "lw":
            S_mv = risk_models.CovarianceShrinkage(prices_wide).ledoit_wolf()
        else:
            S_mv = risk_models.sample_cov(prices_wide)
        ef_minv = EfficientFrontier(mu_mv, S_mv, weight_bounds=weight_bounds_mv)
        weights_min_vol = ef_minv.min_volatility()
        weights_min_vol = {k: v for k, v in weights_min_vol.items() if v > 1e-6}
        perf_min_vol = ef_minv.portfolio_performance()
    return perf_min_vol, weights_min_vol


@app.cell(hide_code=True)
def _(
    mo,
    perf_max_sharpe,
    perf_min_vol,
    plt,
    weights_max_sharpe,
    weights_min_vol,
):
    # Сравнение: портфель max Sharpe vs min Volatility (таблица + столбчатые веса)
    if not weights_max_sharpe and not weights_min_vol:
        pass
    else:
        if perf_max_sharpe and perf_min_vol:
            r1, v1, s1 = perf_max_sharpe
            r2, v2, s2 = perf_min_vol
            comp_md = mo.md(f"""
        ## Сравнение стратегий оптимизации

        | Критерий | Max Sharpe | Min Volatility |
        |----------|------------|----------------|
        | Доходность | {r1:.2%} | {r2:.2%} |
        | Волатильность | {v1:.2%} | {v2:.2%} |
        | Sharpe | {s1:.2f} | {s2:.2f} |
        """)
        elif perf_max_sharpe:
            r1, v1, s1 = perf_max_sharpe
            comp_md = mo.md(f"""
        ## Сравнение стратегий оптимизации

        | Критерий | Max Sharpe |
        |----------|------------|
        | Доходность | {r1:.2%} |
        | Волатильность | {v1:.2%} |
        | Sharpe | {s1:.2f} |
        """)
        else:
            comp_md = mo.md("## Сравнение стратегий оптимизации\n\nДанные не рассчитаны.")
        comp_md
        if weights_max_sharpe or weights_min_vol:
            try:
                plt.close(5)
            except Exception:
                pass
            fig5, (ax5a, ax5b) = plt.subplots(1, 2, num=5, figsize=(14, 5))
            if weights_max_sharpe:
                t1, w1 = list(weights_max_sharpe.keys()), [x * 100 for x in weights_max_sharpe.values()]
                ax5a.bar(t1, w1, color='steelblue', alpha=0.8)
                ax5a.set_title('Max Sharpe')
                ax5a.set_ylabel('Доля (%)')
                ax5a.tick_params(axis='x', rotation=45)
            if weights_min_vol:
                t2, w2 = list(weights_min_vol.keys()), [x * 100 for x in weights_min_vol.values()]
                ax5b.bar(t2, w2, color='darkgreen', alpha=0.6)
                ax5b.set_title('Min Volatility')
                ax5b.set_ylabel('Доля (%)')
                ax5b.tick_params(axis='x', rotation=45)
            plt.suptitle('Веса: Max Sharpe vs Min Volatility')
            plt.tight_layout()
            plt.show()
            del fig5, ax5a, ax5b
    return


@app.cell(hide_code=True)
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

        # Markdown-таблица без ведущих пробелов (иначе не рендерится)
        table_rows = [
            "| Тикер | CAGR | Волатильность | Sharpe Ratio |",
            "|:------|-----:|-------------:|-------------:|",
        ]
        for _, result_row in top_results.iterrows():
            sharpe_str = f"{result_row['sharpe']:.2f}" if pd.notna(result_row['sharpe']) else "N/A"
            table_rows.append(f"| {result_row['ticker']} | {result_row['cagr']:.2%} | {result_row['volatility']:.2%} | {sharpe_str} |")

        table_text = mo.md("## 📋 Топ-20 тикеров по Sharpe Ratio\n\n" + "\n".join(table_rows))
    else:
        top_results = pd.DataFrame()
        table_text = mo.md("⚠️ Нет данных для отображения")

    table_text
    return (top_results,)


@app.cell
def _(mo, top_results):
    mo.ui.table(top_results)
    return


if __name__ == "__main__":
    app.run()
