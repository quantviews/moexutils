"""
marimo notebook: ARIMA-анализ тикера MOEX

Использование:
1. Установите зависимости: pip install marimo statsmodels pmdarima scikit-learn
2. Запустите: marimo edit arima-analysis.py
3. Выберите тикер, задайте параметры модели
4. Ноутбук: загрузка данных → выбор порядка ARIMA → диагностика → walk-forward прогноз

Функционал:
- Загрузка данных по тикеру из хранилища (stocks, polars)
- ACF/PACF коррелограммы для идентификации порядка
- Grid-search по (p,d,q) с AIC/BIC
- Диагностика остатков (Ljung-Box, ARCH-тест, ACF/PACF residuals)
- Walk-forward out-of-sample прогноз с rolling RMSE/MAE
"""

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium", css_file="styles.css")


@app.cell(hide_code=True)
def _():
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    import sys
    import math
    import warnings
    import itertools
    from pathlib import Path
    from datetime import datetime, timedelta

    import marimo as mo

    import statsmodels.api as sm
    import statsmodels.tsa.api as smt
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    from statsmodels.tsa.stattools import adfuller, kpss

    from sklearn.metrics import mean_squared_error, mean_absolute_error

    warnings.filterwarnings("ignore")
    return (
        ARIMA,
        Path,
        SARIMAX,
        acorr_ljungbox,
        adfuller,
        het_arch,
        itertools,
        kpss,
        math,
        mean_absolute_error,
        mean_squared_error,
        mo,
        np,
        pl,
        plot_acf,
        plot_pacf,
        plt,
        sys,
        timedelta,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # ARIMA-анализ тикера Московской биржи

    Пайплайн анализа:

    1. **Загрузка данных** — parquet-файлы с дневными ценами
    2. **Стационарность** — ADF/KPSS тесты на лог-доходностях
    3. **Идентификация порядка** — ACF/PACF коррелограммы + grid search по AIC/BIC
    4. **Диагностика модели** — остатки должны быть белым шумом (Ljung-Box), проверка ARCH-эффекта
    5. **Walk-forward прогноз** — out-of-sample оценка: модель обучается на прошлом, прогнозирует 1 шаг, сдвигается
    6. **Прогноз вперёд** — точечный прогноз за пределы имеющихся данных
    """)
    return


@app.cell
def _(Path, sys):
    # stocks.py лежит в корне проекта (родительская папка от marimo/)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import stocks  # pyright: ignore[reportMissingImports]
    return (stocks,)


@app.cell(hide_code=True)
def _(mo, stocks):
    # Тикеры из хранилища, без старых имен переименованных бумаг (TCSG, YNDX...)
    _olds = set(stocks.load_renames()['old'].to_list())
    _available = sorted(_d for _d in stocks.list_tickers(include_delisted=True) if _d not in _olds)
    if not _available:
        _available = ["SBER"]

    ticker_dropdown = mo.ui.dropdown(
        options=_available,
        value="SBER" if "SBER" in _available else _available[0],
        label="Тикер:",
        searchable=True,
    )
    # ARIMA чувствительна к смене режимов рынка: короткое окно обычно осмысленнее
    sample_choice = mo.ui.dropdown(
        options={"1 год": 1, "3 года": 3, "5 лет": 5, "10 лет": 10, "Вся история": 0},
        value="3 года",
        label="Период выборки:",
    )
    mo.hstack([ticker_dropdown, sample_choice], justify="start")
    return sample_choice, ticker_dropdown


@app.cell(hide_code=True)
def _(mo, pl, sample_choice, stocks, ticker_dropdown):
    # Загрузка: склейка переименований + сплит-коррекция;
    # анализируем ряд полной доходности (adj_close = дивиденды + сплиты)
    _t = ticker_dropdown.value
    try:
        # read_stocks подтягивает историю старых тикеров (TCSG → T), склеивает
        # ее и приводит цены к пост-сплитовой базе
        _raw = stocks.read_stocks(_t, split_adjusted=True)
        _raw = _raw.filter(pl.col("ticker") == _t).sort("date", maintain_order=True)
        _raw = _raw.unique(subset="date", keep="last", maintain_order=True)
        if _raw.is_empty():
            raise ValueError(f"нет данных по тикеру {_t}")

        _price_col = "adj_close" if "adj_close" in _raw.columns else "close"
        # close_df: date, close — цена полной доходности без пропусков
        close_df = (
            _raw.select("date", pl.col(_price_col).cast(pl.Float64).alias("close"))
            .filter(pl.col("close").is_not_null() & pl.col("close").is_not_nan())
        )

        if sample_choice.value:
            _cutoff = close_df.select(
                pl.col("date").max().dt.offset_by(f"-{sample_choice.value}y")).item()
            close_df = close_df.filter(pl.col("date") >= _cutoff)

        # returns_df: date, log_return
        returns_df = (
            close_df.select(
                "date", (pl.col("close") / pl.col("close").shift(1)).log().alias("log_return"))
            .filter(pl.col("log_return").is_not_null() & pl.col("log_return").is_not_nan())
        )

        _stitch = ""
        if "source_ticker" in _raw.columns and _raw["source_ticker"].n_unique() > 1:
            _stitch = ("- История склеена из: "
                       f"**{' → '.join(_raw['source_ticker'].unique(maintain_order=True).to_list())}**\n")
        status_block = mo.md(
            f"### Данные загружены: **{_t}**\n"
            f"- Период: **{close_df['date'].min().strftime('%Y-%m-%d')}** — "
            f"**{close_df['date'].max().strftime('%Y-%m-%d')}**, "
            f"наблюдений: **{close_df.height}**\n"
            f"- Столбец цены: `{_price_col}` (дивиденды + сплиты учтены)\n"
            + _stitch
        )
    except Exception as e:
        close_df = pl.DataFrame(schema={"date": pl.Date, "close": pl.Float64})
        returns_df = pl.DataFrame(schema={"date": pl.Date, "log_return": pl.Float64})
        status_block = mo.md(f"**Ошибка загрузки:** {e}")
    status_block
    return close_df, returns_df


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 1. Данные и лог-доходности

    Верхний график — цена актива. Нижний — лог-доходности: $r_t = \ln(P_t / P_{t-1})$.

    Лог-доходности используются потому что:
    - они аддитивны во времени ($r_{1..T} = \sum r_t$)
    - приближённо стационарны (цена — нет)
    - ARIMA работает со стационарными рядами
    """)
    return


@app.cell
def _(close_df, plt, returns_df, ticker_dropdown):
    if close_df.height > 1:
        fig_price, (ax_p, ax_r) = plt.subplots(2, 1, figsize=(14, 7), sharex=False)

        ax_p.plot(close_df["date"].to_list(), close_df["close"].to_numpy(), linewidth=1.2, color="steelblue")
        ax_p.set_title(f"{ticker_dropdown.value} — Цена", fontsize=13, fontweight="bold")
        ax_p.set_ylabel("Цена")
        ax_p.grid(True, alpha=0.3)

        ax_r.plot(returns_df["date"].to_list(), returns_df["log_return"].to_numpy(), linewidth=0.8, color="coral")
        ax_r.set_title("Лог-доходности", fontsize=13, fontweight="bold")
        ax_r.set_ylabel("log return")
        ax_r.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        del fig_price, ax_p, ax_r
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## 2. Тесты стационарности

    ARIMA требует стационарный ряд (или ряд, который становится стационарным после дифференцирования).

    - **ADF** (Augmented Dickey-Fuller): H0 = ряд нестационарен. Если p < 0.05 — отвергаем H0, ряд стационарен.
    - **KPSS**: H0 = ряд стационарен. Если p > 0.05 — не отвергаем H0, ряд стационарен.

    Идеальный случай: ADF говорит «стационарен» **и** KPSS говорит «стационарен». Если оба теста согласны — можно использовать d=0.
    """)
    return


@app.cell(hide_code=True)
def _(adfuller, kpss, mo, pl, returns_df):
    _out = mo.md("")
    if returns_df.height > 30:
        _y = returns_df["log_return"].to_numpy()
        _adf = adfuller(_y, autolag="AIC")
        _kpss_res = kpss(_y, regression="c", nlags="auto")

        _stab = pl.DataFrame({
            "Тест": ["ADF (H0: ряд нестационарен)", "KPSS (H0: ряд стационарен)"],
            "Статистика": [f"{_adf[0]:.4f}", f"{_kpss_res[0]:.4f}"],
            "p-value": [f"{_adf[1]:.6f}", f"{_kpss_res[1]:.6f}"],
            "Вывод": [
                "Стационарен" if _adf[1] < 0.05 else "Нестационарен",
                "Стационарен" if _kpss_res[1] > 0.05 else "Нестационарен",
            ],
        })
        _out = mo.vstack([mo.md("**Тесты стационарности лог-доходностей**"), mo.ui.table(_stab)])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 3. ACF и PACF — идентификация порядка

    - **ACF** (автокорреляционная функция) — корреляция ряда с самим собой на лаге $k$. Медленно затухающая ACF -> нужен AR-компонент.
    - **PACF** (частичная ACF) — корреляция на лаге $k$ за вычетом влияния промежуточных лагов. Обрыв PACF на лаге $p$ -> AR($p$).

    Правила подбора (Box-Jenkins):

    | Паттерн | ACF | PACF | Модель |
    |---------|-----|------|--------|
    | Обрыв на лаге $q$ | Обрыв | Затухание | MA($q$) |
    | Затухание | Затухание | Обрыв на лаге $p$ | AR($p$) |
    | Затухание | Затухание | Затухание | ARMA($p$,$q$) |

    Столбики за пределами голубой зоны (95% CI) — значимые автокорреляции.
    """)
    return


@app.cell
def _(plot_acf, plot_pacf, plt, returns_df):
    if returns_df.height > 30:
        fig_acf, axes_acf = plt.subplots(1, 2, figsize=(14, 4))

        plot_acf(returns_df["log_return"].to_numpy(), lags=30, ax=axes_acf[0], alpha=0.05)
        axes_acf[0].set_title("ACF лог-доходностей")

        plot_pacf(returns_df["log_return"].to_numpy(), lags=30, ax=axes_acf[1], alpha=0.05)
        axes_acf[1].set_title("PACF лог-доходностей")

        plt.tight_layout()
        plt.show()
        del fig_acf, axes_acf
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 4. Подбор порядка ARIMA(p,d,q) по информационному критерию

    Grid search перебирает все комбинации (p,d,q) и выбирает модель с минимальным AIC или BIC.

    - **AIC** = $-2 \ln L + 2k$ — штрафует за количество параметров $k$, но мягко. Склонен к переобучению.
    - **BIC** = $-2 \ln L + k \ln n$ — штраф растёт с размером выборки $n$. Выбирает более экономные модели.

    На практике: AIC чаще выбирает более сложную модель, BIC — более простую. Для прогнозирования на 1 шаг обычно хватает BIC.
    """)
    return


@app.cell
def _(mo):
    max_p_slider = mo.ui.slider(
        start=1, stop=5, step=1, value=3, label="max p (AR):"
    )
    max_d_slider = mo.ui.slider(
        start=0, stop=2, step=1, value=0, label="max d (дифференцирование):"
    )
    max_q_slider = mo.ui.slider(
        start=1, stop=5, step=1, value=3, label="max q (MA):"
    )
    criterion_select = mo.ui.dropdown(
        options=["AIC", "BIC"], value="AIC", label="Критерий:"
    )
    mo.hstack([max_p_slider, max_d_slider, max_q_slider, criterion_select])
    return criterion_select, max_d_slider, max_p_slider, max_q_slider


@app.cell
def _(
    ARIMA,
    criterion_select,
    itertools,
    max_d_slider,
    max_p_slider,
    max_q_slider,
    mo,
    np,
    pl,
    returns_df,
):
    _results = []
    _best_ic = np.inf
    _best_order = None

    _criterion = criterion_select.value.lower()  # "aic" or "bic"

    _y = returns_df["log_return"].to_numpy()
    if len(_y) > 60:
        for _p, _d, _q in itertools.product(
            range(max_p_slider.value + 1),
            range(max_d_slider.value + 1),
            range(max_q_slider.value + 1),
        ):
            if _p == 0 and _q == 0:
                continue
            try:
                _model = ARIMA(_y, order=(_p, _d, _q))
                _fit = _model.fit()
                _ic = getattr(_fit, _criterion)
                _results.append({
                    "p": _p, "d": _d, "q": _q,
                    "order_str": str((_p, _d, _q)),
                    "AIC": round(_fit.aic, 2),
                    "BIC": round(_fit.bic, 2),
                    "Log-Lik": round(_fit.llf, 2),
                })
                if _ic < _best_ic:
                    _best_ic = _ic
                    _best_order = (_p, _d, _q)
            except Exception:
                pass

    # grid_df: p, d, q, order_str, AIC, BIC, Log-Lik — по возрастанию критерия
    grid_df = pl.DataFrame(_results, schema={
        "p": pl.Int64, "d": pl.Int64, "q": pl.Int64, "order_str": pl.Utf8,
        "AIC": pl.Float64, "BIC": pl.Float64, "Log-Lik": pl.Float64,
    })
    if grid_df.height > 0:
        grid_df = grid_df.sort(_criterion.upper(), maintain_order=True)

    best_order = _best_order if _best_order is not None else (1, 0, 1)

    if _best_order is None:
        _msg = ("### Grid search не выполнен: недостаточно данных (> 60 наблюдений). "
                f"Используется порядок по умолчанию ARIMA{best_order}")
    else:
        _msg = (f"### Лучший порядок по {criterion_select.value}: **ARIMA{best_order}** "
                f"({criterion_select.value} = {_best_ic:.2f})")
    mo.md(_msg)
    return best_order, grid_df


@app.cell
def _(grid_df, mo):
    if grid_df.height > 0:
        table_view = mo.ui.table(
            grid_df.select("order_str", "AIC", "BIC", "Log-Lik").head(20),
            label="Top-20 моделей",
        )
    else:
        table_view = mo.md("")
    table_view
    return


@app.cell
def _(criterion_select, grid_df, np, pl, plt):
    if grid_df.height > 0:
        _crit = criterion_select.value.upper()
        # Берём только d=0 (или d с лучшим порядком) для 2D визуализации
        _best_d = grid_df["d"][0]
        _sub = grid_df.filter(pl.col("d") == _best_d)

        if _sub.height > 1:
            _ps = sorted(_sub["p"].unique().to_list())
            _qs = sorted(_sub["q"].unique().to_list())
            _heat = np.full((len(_ps), len(_qs)), np.nan)

            for row in _sub.iter_rows(named=True):
                _pi = _ps.index(row["p"])
                _qi = _qs.index(row["q"])
                _heat[_pi, _qi] = row[_crit]

            fig_heat, ax_heat = plt.subplots(figsize=(8, 6))
            _im = ax_heat.imshow(_heat, aspect="auto", origin="lower", cmap="RdYlGn_r")
            ax_heat.set_xticks(range(len(_qs)))
            ax_heat.set_xticklabels(_qs)
            ax_heat.set_yticks(range(len(_ps)))
            ax_heat.set_yticklabels(_ps)
            ax_heat.set_xlabel("q (MA)")
            ax_heat.set_ylabel("p (AR)")
            ax_heat.set_title(f"{_crit} heatmap (d={_best_d})")

            for i in range(len(_ps)):
                for j in range(len(_qs)):
                    if not np.isnan(_heat[i, j]):
                        ax_heat.text(j, i, f"{_heat[i,j]:.0f}", ha="center", va="center", fontsize=8)

            plt.colorbar(_im, ax=ax_heat)
            plt.tight_layout()
            plt.show()
            del fig_heat, ax_heat
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 5. Оценка лучшей модели

    В Таблице summary обращаем внимание на:

    - **Коэффициенты** (ar.L1, ma.L1 и т.д.) — значимы ли они (p-value < 0.05)?
    - **Log Likelihood** — чем больше, тем лучше подгонка
    - **Ljung-Box (L1)** — p > 0.05 означает, что остатки не автокоррелированы (хорошо)
    - **Jarque-Bera** — тест на нормальность остатков (для финансовых рядов обычно отвергается — это нормально, тяжёлые хвосты)
    - **Heteroskedasticity** — p < 0.05 указывает на ARCH-эффект (волатильность кластеризуется)
    """)
    return


@app.cell
def _(ARIMA, best_order, mo, returns_df):
    best_model_fit = ARIMA(returns_df["log_return"].to_numpy(), order=best_order).fit()
    mo.md(f"""
    ### Результаты ARIMA{best_order}
    ```
    {best_model_fit.summary().as_text()}
    ```
    """)
    return (best_model_fit,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 6. Диагностика остатков

    Если модель хорошая, остатки должны быть **белым шумом** — без автокорреляции и без паттернов.

    - **Ljung-Box / Box-Pierce** — тест на наличие автокорреляции в остатках. p > 0.05 → OK, автокорреляции нет.
    - **ARCH LM тест** — проверяет, есть ли кластеры волатильности в остатках. p < 0.05 → ARCH-эффект присутствует, стоит рассмотреть GARCH-модель для волатильности.

    На графиках:
    - Остатки должны выглядеть как случайный шум вокруг нуля
    - ACF/PACF остатков — все столбики внутри голубой зоны (незначимы)
    - Гистограмма — колоколообразная, но возможны тяжёлые хвосты
    """)
    return


@app.cell
def _(acorr_ljungbox, best_model_fit, het_arch, mo, np, pl, returns_df):
    # resid_df: date, resid — остатки модели без NaN, с датами наблюдений
    _res = np.asarray(best_model_fit.resid, dtype=float)
    _ok = ~np.isnan(_res)
    resid_df = pl.DataFrame({
        "date": returns_df["date"].filter(pl.Series(_ok)),
        "resid": _res[_ok],
    })
    _resid = resid_df["resid"].to_numpy()

    # Ljung-Box тест (statsmodels отдает таблицу — берем колонки как numpy)
    _lb_t = acorr_ljungbox(_resid, lags=[10, 20], boxpierce=True, return_df=True)
    _lb = {_c: _lb_t[_c].to_numpy() for _c in ("lb_stat", "lb_pvalue", "bp_stat", "bp_pvalue")}

    # ARCH-тест (гетероскедастичность)
    _arch = het_arch(_resid, nlags=10)
    _arch_stat, _arch_p = _arch[0], _arch[1]

    diagnostics_table = pl.DataFrame({
        "Тест": [
            "Ljung-Box (lag=10)",
            "Ljung-Box (lag=20)",
            "Box-Pierce (lag=10)",
            "Box-Pierce (lag=20)",
            "ARCH LM (10 lags)",
        ],
        "Статистика": [
            f"{_lb['lb_stat'][0]:.4f}",
            f"{_lb['lb_stat'][1]:.4f}",
            f"{_lb['bp_stat'][0]:.4f}",
            f"{_lb['bp_stat'][1]:.4f}",
            f"{_arch_stat:.4f}",
        ],
        "p-value": [
            f"{_lb['lb_pvalue'][0]:.4f}",
            f"{_lb['lb_pvalue'][1]:.4f}",
            f"{_lb['bp_pvalue'][0]:.4f}",
            f"{_lb['bp_pvalue'][1]:.4f}",
            f"{_arch_p:.4f}",
        ],
        "Вывод": [
            "OK (белый шум)" if _lb['lb_pvalue'][0] > 0.05 else "Автокорреляция!",
            "OK (белый шум)" if _lb['lb_pvalue'][1] > 0.05 else "Автокорреляция!",
            "OK" if _lb['bp_pvalue'][0] > 0.05 else "Автокорреляция!",
            "OK" if _lb['bp_pvalue'][1] > 0.05 else "Автокорреляция!",
            "OK (гомоскедаст.)" if _arch_p > 0.05 else "ARCH-эффект!",
        ],
    })

    mo.vstack([mo.md("**Диагностика остатков**"), mo.ui.table(diagnostics_table)])
    return (resid_df,)


@app.cell
def _(np, plot_acf, plot_pacf, plt, resid_df):
    if resid_df.height > 30:
        _resid = resid_df["resid"].to_numpy()
        fig_diag, axes_diag = plt.subplots(2, 2, figsize=(14, 8))

        # Остатки
        axes_diag[0, 0].plot(resid_df["date"].to_list(), _resid, linewidth=0.7, color="gray")
        axes_diag[0, 0].axhline(0, color="red", linewidth=0.8, linestyle="--")
        axes_diag[0, 0].set_title("Остатки модели")
        axes_diag[0, 0].grid(True, alpha=0.3)

        # Гистограмма
        axes_diag[0, 1].hist(_resid, bins=50, alpha=0.7, color="steelblue", edgecolor="black", density=True)
        _x = np.linspace(_resid.min(), _resid.max(), 200)
        _mu, _sigma = _resid.mean(), _resid.std(ddof=1)
        axes_diag[0, 1].plot(_x, (1 / (_sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((_x - _mu) / _sigma) ** 2),
                             color="red", linewidth=1.5, label="N(μ,σ²)")
        axes_diag[0, 1].set_title("Распределение остатков")
        axes_diag[0, 1].legend()

        # ACF остатков
        plot_acf(_resid, lags=25, ax=axes_diag[1, 0], alpha=0.05)
        axes_diag[1, 0].set_title("ACF остатков")

        # PACF остатков
        plot_pacf(_resid, lags=25, ax=axes_diag[1, 1], alpha=0.05)
        axes_diag[1, 1].set_title("PACF остатков")

        plt.tight_layout()
        plt.show()
        del fig_diag, axes_diag
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Квадраты остатков — визуализация ARCH-эффекта

    Если ACF квадратов остатков ($r_t^2$) показывает значимые лаги — волатильность персистентна (кластеры высокой/низкой волатильности). Это типично для финансовых рядов и означает, что ARIMA моделирует только среднее, но не дисперсию. Для моделирования волатильности нужен GARCH.
    """)
    return


@app.cell
def _(plot_acf, plt, resid_df):
    if resid_df.height > 30:
        _sq = resid_df["resid"].to_numpy() ** 2
        fig_sq, ax_sq = plt.subplots(1, 2, figsize=(14, 4))
        ax_sq[0].plot(resid_df["date"].to_list(), _sq, linewidth=0.6, color="purple")
        ax_sq[0].set_title("Квадраты остатков (r²)")
        ax_sq[0].grid(True, alpha=0.3)
        plot_acf(_sq, lags=25, ax=ax_sq[1], alpha=0.05)
        ax_sq[1].set_title("ACF квадратов остатков")
        plt.tight_layout()
        plt.show()
        del fig_sq, ax_sq
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 7. GARCH — моделирование волатильности

    ARIMA моделирует **условное среднее** (направление), но не волатильность. Если ARCH-тест выше показал значимый эффект — волатильность кластеризуется, и её можно моделировать отдельно.

    GARCH(p,q): $\sigma_t^2 = \omega + \sum_{i=1}^{p} \alpha_i \cdot r_{t-i}^2 + \sum_{j=1}^{q} \beta_j \cdot \sigma_{t-j}^2$

    - $\alpha$ — реакция на новые шоки (чем больше, тем сильнее реагирует на вчерашний выброс)
    - $\beta$ — персистентность волатильности (чем ближе к 1, тем дольше сохраняется высокая волатильность)
    - $\alpha + \beta < 1$ — модель стационарна

    Подход: берём остатки ARIMA и оцениваем GARCH(1,1) на них. Если остатки GARCH — белый шум, модель адекватна.
    """)
    return


@app.cell
def _(mo):
    garch_p_slider = mo.ui.slider(start=1, stop=3, step=1, value=1, label="GARCH p:")
    garch_q_slider = mo.ui.slider(start=0, stop=3, step=1, value=1, label="GARCH q:")
    mo.hstack([garch_p_slider, garch_q_slider])
    return garch_p_slider, garch_q_slider


@app.cell
def _(garch_p_slider, garch_q_slider, mo, resid_df):
    try:
        from arch import arch_model as _arch_model
    except ImportError:
        _arch_model = None

    if _arch_model is None:
        garch_fit = None
        garch_block = mo.md("**Пакет `arch` не установлен** — GARCH-блок пропущен: `pip install arch`")
    elif resid_df.height > 60:
        _gp = garch_p_slider.value
        _gq = garch_q_slider.value

        # Оцениваем GARCH на остатках ARIMA (масштабируем ×100 для численной стабильности)
        _am = _arch_model(resid_df["resid"].to_numpy() * 100, mean="Zero", vol="Garch", p=_gp, q=_gq)
        garch_fit = _am.fit(disp="off", show_warning=False)

        garch_block = mo.md(f"""
    ### GARCH({_gp},{_gq}) на остатках ARIMA

    ```
    {garch_fit.summary().as_text()}
    ```
    """)
    else:
        garch_fit = None
        garch_block = mo.md("**Недостаточно данных для оценки GARCH.**")
    garch_block
    return (garch_fit,)


@app.cell
def _(acorr_ljungbox, garch_fit, het_arch, mo, np, pl):
    if garch_fit is not None:
        _garch_resid = np.asarray(garch_fit.resid, dtype=float)
        _std_resid = _garch_resid / np.asarray(garch_fit.conditional_volatility, dtype=float)
        _std_resid = _std_resid[~np.isnan(_std_resid)]

        # Ljung-Box на стандартизованных остатках
        _t1 = acorr_ljungbox(_std_resid, lags=[10, 20], boxpierce=True, return_df=True)
        _lb_g = {_c: _t1[_c].to_numpy() for _c in ("lb_stat", "lb_pvalue")}

        # Ljung-Box на квадратах стандартизованных остатков (проверяем снял ли GARCH ARCH-эффект)
        _t2 = acorr_ljungbox(_std_resid ** 2, lags=[10, 20], boxpierce=False, return_df=True)
        _lb_g2 = {_c: _t2[_c].to_numpy() for _c in ("lb_stat", "lb_pvalue")}

        # ARCH-тест на стандартизованных остатках
        _arch_g = het_arch(_std_resid, nlags=10)

        garch_diag_table = pl.DataFrame({
            "Тест": [
                "Ljung-Box std resid (lag=10)",
                "Ljung-Box std resid (lag=20)",
                "Ljung-Box std_resid² (lag=10)",
                "Ljung-Box std_resid² (lag=20)",
                "ARCH LM на std resid (10 lags)",
            ],
            "Статистика": [
                f"{_lb_g['lb_stat'][0]:.4f}",
                f"{_lb_g['lb_stat'][1]:.4f}",
                f"{_lb_g2['lb_stat'][0]:.4f}",
                f"{_lb_g2['lb_stat'][1]:.4f}",
                f"{_arch_g[0]:.4f}",
            ],
            "p-value": [
                f"{_lb_g['lb_pvalue'][0]:.4f}",
                f"{_lb_g['lb_pvalue'][1]:.4f}",
                f"{_lb_g2['lb_pvalue'][0]:.4f}",
                f"{_lb_g2['lb_pvalue'][1]:.4f}",
                f"{_arch_g[1]:.4f}",
            ],
            "Вывод": [
                "OK" if _lb_g['lb_pvalue'][0] > 0.05 else "Автокорреляция!",
                "OK" if _lb_g['lb_pvalue'][1] > 0.05 else "Автокорреляция!",
                "OK (ARCH снят)" if _lb_g2['lb_pvalue'][0] > 0.05 else "ARCH остался!",
                "OK (ARCH снят)" if _lb_g2['lb_pvalue'][1] > 0.05 else "ARCH остался!",
                "OK (ARCH снят)" if _arch_g[1] > 0.05 else "ARCH остался!",
            ],
        })

        _garch_out = mo.vstack([mo.md("**Диагностика GARCH**"), mo.ui.table(garch_diag_table)])
    else:
        _garch_out = mo.md("")
    _garch_out
    return


@app.cell
def _(garch_fit, np, pl, plot_acf, plt, resid_df):
    if garch_fit is not None:
        # GARCH оценивался на остатках resid_df — даты наблюдений оттуда
        _dates = resid_df["date"]
        _cond_vol = np.asarray(garch_fit.conditional_volatility, dtype=float) # обратно в исходный масштаб
        _std_all = np.asarray(garch_fit.resid, dtype=float) / _cond_vol
        _ok = ~np.isnan(_std_all)
        _std_resid = _std_all[_ok]
        _std_dates = _dates.filter(pl.Series(_ok)).to_list()

        fig_garch, axes_garch = plt.subplots(2, 2, figsize=(14, 8))

        # Условная волатильность
        axes_garch[0, 0].plot(_dates.to_list(), _cond_vol, linewidth=0.8, color="darkred")
        axes_garch[0, 0].set_title("Условная волатильность σ_t (GARCH)")
        axes_garch[0, 0].grid(True, alpha=0.3)

        # Стандартизованные остатки
        axes_garch[0, 1].plot(_std_dates, _std_resid, linewidth=0.5, color="gray")
        axes_garch[0, 1].axhline(0, color="red", linewidth=0.7, linestyle="--")
        axes_garch[0, 1].set_title("Стандартизованные остатки (resid / σ_t)")
        axes_garch[0, 1].grid(True, alpha=0.3)

        # ACF стандартизованных остатков²
        plot_acf(_std_resid ** 2, lags=25, ax=axes_garch[1, 0], alpha=0.05)
        axes_garch[1, 0].set_title("ACF стандартизованных остатков²")

        # Гистограмма стандартизованных остатков
        axes_garch[1, 1].hist(_std_resid, bins=50, alpha=0.7, color="steelblue",
                              edgecolor="black", density=True)
        _x = np.linspace(_std_resid.min(), _std_resid.max(), 200)
        _mu, _sigma = _std_resid.mean(), _std_resid.std(ddof=1)
        axes_garch[1, 1].plot(
            _x, (1 / (_sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((_x - _mu) / _sigma) ** 2),
            color="red", linewidth=1.5, label="N(μ,σ²)")
        axes_garch[1, 1].set_title("Распределение стандартизованных остатков")
        axes_garch[1, 1].legend()

        plt.tight_layout()
        plt.show()
        del fig_garch, axes_garch
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 8. Walk-Forward прогноз (out-of-sample)

    Единственный честный способ оценить прогнозную силу модели — **out-of-sample** тестирование.

    Алгоритм walk-forward:
    1. Разбиваем данные на train (первые N%) и test (оставшиеся)
    2. Обучаем модель на train, делаем прогноз на 1 шаг вперёд
    3. Сдвигаемся на 1 наблюдение, добавляем реальное значение в train
    4. Повторяем; полный refit (переоценка AR/MA коэффициентов через MLE) — каждые K шагов. Между refit-ами коэффициенты фиксированы, модель просто «видит» новые наблюдения для корректного прогноза — это быстро

    Метрики:
    - **RMSE** — корень из среднеквадратичной ошибки. Чувствителен к выбросам.
    - **MAE** — средняя абсолютная ошибка. Более робастная.

    Для дневных лог-доходностей RMSE ≈ дневная волатильность ряда — это нормально: модель не может предсказать направление лучше, чем случайное блуждание.

    Расчёт занимает время — нажмите кнопку ниже для запуска.
    """)
    return


@app.cell
def _(mo):
    train_pct_slider = mo.ui.slider(
        start=50, stop=90, step=5, value=80, label="% данных на train:"
    )
    refit_slider = mo.ui.slider(
        start=5, stop=60, step=5, value=20, label="Refit каждые N шагов:"
    )
    use_auto_order = mo.ui.checkbox(value=False, label="Авто-подбор порядка при каждом refit")
    run_wf_button = mo.ui.run_button(label="Запустить Walk-Forward")
    mo.hstack([train_pct_slider, refit_slider, use_auto_order, run_wf_button])
    return refit_slider, run_wf_button, train_pct_slider, use_auto_order


@app.cell
def _(
    SARIMAX,
    best_order,
    math,
    mean_absolute_error,
    mean_squared_error,
    mo,
    np,
    pl,
    refit_slider,
    returns_df,
    run_wf_button,
    train_pct_slider,
    use_auto_order,
):
    if not run_wf_button.value:
        wf_status = mo.md("Нажмите **«Запустить Walk-Forward»** для начала расчёта.")
        wf_forecast_df = pl.DataFrame()
    elif returns_df.height < 100:
        wf_status = mo.md("**Недостаточно данных для walk-forward прогноза (нужно >100 наблюдений).**")
        wf_forecast_df = pl.DataFrame()
    else:
        _train_size = int(returns_df.height * train_pct_slider.value / 100)
        _refit_every = refit_slider.value
        _auto = use_auto_order.value

        _r = returns_df["log_return"].to_numpy()
        _r_dates = returns_df["date"].to_list()

        # Для скорости, используем SARIMAX с enforce_* = False
        _p_grid = list(range(0, 4))
        _q_grid = list(range(0, 3))

        def _fit_best(y, order_hint):
            """Grid search по AIC; возвращает (fitted_model, best_order)."""
            _best_aic, _best_res, _best_ord = np.inf, None, order_hint
            for p in _p_grid:
                for q in _q_grid:
                    if p == 0 and q == 0:
                        continue
                    try:
                        res = SARIMAX(
                            y, order=(p, 0, q),
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        ).fit(disp=False)
                        if res.aic < _best_aic:
                            _best_aic, _best_res, _best_ord = res.aic, res, (p, 0, q)
                    except Exception:
                        pass
            return _best_res, _best_ord

        _fcast_values = []
        _fcast_dates = []
        _actual_values = []

        _model = None
        _cur_order = best_order
        _last_refit = -10**9

        for _i in range(_train_size, len(_r) - 1):
            # Refit?
            if (_i - _last_refit) >= _refit_every or _model is None:
                _y_train = _r[:_i]
                if _auto:
                    _model, _cur_order = _fit_best(_y_train, _cur_order)
                else:
                    try:
                        _model = SARIMAX(
                            _y_train, order=_cur_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        ).fit(disp=False)
                    except Exception:
                        _model, _cur_order = _fit_best(_y_train, _cur_order)
                _last_refit = _i
            else:
                # Incremental update (filter)
                try:
                    _model = SARIMAX(
                        _r[:_i],
                        order=_cur_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).filter(_model.params)
                except Exception:
                    _model = None
                    _last_refit = -10**9
                    continue

            # One-step-ahead forecast
            try:
                _fc = float(_model.get_forecast(steps=1).predicted_mean[0])
            except Exception:
                _fc = np.nan

            _fcast_dates.append(_r_dates[_i + 1])
            _fcast_values.append(_fc)
            _actual_values.append(float(_r[_i + 1]))

        # wf_forecast_df: date, actual, forecast — без несостоявшихся прогнозов (NaN)
        wf_forecast_df = pl.DataFrame(
            {"date": _fcast_dates, "actual": _actual_values, "forecast": _fcast_values},
            schema={"date": pl.Date, "actual": pl.Float64, "forecast": pl.Float64},
        ).filter(~pl.col("actual").is_nan() & ~pl.col("forecast").is_nan())

        if wf_forecast_df.height > 0:
            _act = wf_forecast_df["actual"].to_numpy()
            _fct = wf_forecast_df["forecast"].to_numpy()
            wf_rmse = math.sqrt(mean_squared_error(_act, _fct))
            wf_mae = mean_absolute_error(_act, _fct)
        else:
            wf_rmse = np.nan
            wf_mae = np.nan

        wf_status = mo.md(f"""
    ### Walk-forward результаты
    - Обучающая выборка: **{_train_size}** наблюдений ({train_pct_slider.value}%)
    - Тестовая выборка: **{wf_forecast_df.height}** шагов
    - Порядок модели: **ARIMA{_cur_order}**
    - Refit каждые **{_refit_every}** шагов
    - **RMSE**: {wf_rmse:.6f}
    - **MAE**: {wf_mae:.6f}
    """)
    wf_status
    return (wf_forecast_df,)


@app.cell
def _(plt, ticker_dropdown, wf_forecast_df):
    if wf_forecast_df.height > 0:
        fig_wf, axes_wf = plt.subplots(3, 1, figsize=(14, 12))
        _dates = wf_forecast_df["date"].to_list()

        # Actual vs Forecast
        axes_wf[0].plot(_dates, wf_forecast_df["actual"].to_numpy(),
                        linewidth=0.8, label="Actual", color="steelblue")
        axes_wf[0].plot(_dates, wf_forecast_df["forecast"].to_numpy(),
                        linewidth=0.8, label="Forecast", color="coral", alpha=0.8)
        axes_wf[0].set_title(f"{ticker_dropdown.value} — Walk-Forward: Actual vs Forecast (лог-доходности)", fontweight="bold")
        axes_wf[0].legend()
        axes_wf[0].grid(True, alpha=0.3)

        # Forecast errors
        _errors = wf_forecast_df["actual"] - wf_forecast_df["forecast"]
        axes_wf[1].plot(_dates, _errors.to_numpy(), linewidth=0.7, color="gray")
        axes_wf[1].axhline(0, color="red", linewidth=0.8, linestyle="--") 
        axes_wf[1].set_title("Ошибки прогноза (actual − forecast)")
        axes_wf[1].grid(True, alpha=0.3)

        # Rolling RMSE (window=20)
        _window = min(20, len(_errors) // 3)
        if _window > 1:
            _rolling_rmse = (_errors ** 2).rolling_mean(_window).sqrt()
            axes_wf[2].plot(_dates, _rolling_rmse.to_numpy(), linewidth=1.2, color="darkred")
            axes_wf[2].set_title(f"Rolling RMSE (окно={_window})")
            axes_wf[2].grid(True, alpha=0.3)
        else:
            axes_wf[2].set_visible(False)

        plt.tight_layout()
        plt.show()
        del fig_wf, axes_wf
    return


@app.cell(hide_code=True)
def _(mo):
    _s = "### Простая торговая стратегия на основе прогноза\n\n"
    _s += r"Правило: если ARIMA прогнозирует положительную доходность завтра ($\hat{r}_{t+1} > 0$) — держим long, иначе — cash (вне рынка)." + "\n\n"
    _s += r"Сравниваем с Buy & Hold (всегда в рынке). Sharpe ratio = $\frac{\bar{r}}{\sigma_r} \cdot \sqrt{252}$ — аннуализированный." + "\n\n"
    _s += "**Ожидание:** на дневных данных ARIMA редко обыгрывает Buy & Hold. Доходности близки к белому шуму, и прогноз ≈ 0. Но на отдельных тикерах и периодах бывают исключения."
    mo.md(_s)
    return


@app.cell
def _(mo, np, pl, plt, ticker_dropdown, wf_forecast_df):
    if wf_forecast_df.height > 0:
        _df = wf_forecast_df.with_columns(
            # Стратегия: long, если прогноз > 0 (ожидаем рост)
            (pl.col("forecast") > 0).cast(pl.Int64).alias("signal"),
        ).with_columns(
            # Доходность стратегии: signal_{t} * actual_{t}
            (pl.col("signal") * pl.col("actual")).alias("strategy_return"),
        ).with_columns(
            # Кумулятивные доходности
            pl.col("actual").cum_sum().alias("cum_actual"),
            pl.col("strategy_return").cum_sum().alias("cum_strategy"),
        )
        _dates = _df["date"].to_list()

        fig_cum, ax_cum = plt.subplots(figsize=(14, 6))
        ax_cum.plot(_dates, (np.exp(_df["cum_actual"].to_numpy()) - 1) * 100,
                    label="Buy & Hold", linewidth=1.5, color="steelblue")
        ax_cum.plot(_dates, (np.exp(_df["cum_strategy"].to_numpy()) - 1) * 100,
                    label="ARIMA Long/Cash", linewidth=1.5, color="coral")
        ax_cum.axhline(0, color="black", linewidth=0.5)
        ax_cum.set_title(f"{ticker_dropdown.value} — Buy&Hold vs ARIMA (out-of-sample)", fontweight="bold")
        ax_cum.set_ylabel("Доходность (%)")
        ax_cum.legend(fontsize=11)
        ax_cum.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        del fig_cum, ax_cum

        # Статистика стратегии
        _total_bh = (np.exp(_df["cum_actual"][-1]) - 1) * 100
        _total_strat = (np.exp(_df["cum_strategy"][-1]) - 1) * 100
        _sharpe_bh = _df["actual"].mean() / _df["actual"].std() * np.sqrt(252) if _df["actual"].std() > 0 else 0
        _sharpe_st = _df["strategy_return"].mean() / _df["strategy_return"].std() * np.sqrt(252) if _df["strategy_return"].std() > 0 else 0
        _pct_long = _df["signal"].mean() * 100

        _t = "### Сравнение стратегий (out-of-sample)\n\n"
        _t += "| Метрика | Buy & Hold | ARIMA Long/Cash |\n"
        _t += "|---------|-----------|------------------|\n"
        _t += f"| Доходность | {_total_bh:.2f}% | {_total_strat:.2f}% |\n"
        _t += f"| Sharpe (ann.) | {_sharpe_bh:.3f} | {_sharpe_st:.3f} |\n"
        _t += f"| % дней в long | 100% | {_pct_long:.1f}% |\n"
        strategy_report = mo.md(_t)
    else:
        strategy_report = mo.md("")
    strategy_report
    return


@app.cell(hide_code=True)
def _(mo):
    _f = "---\n## 9. Прогноз за пределы данных\n\n"
    _f += r"Прогноз лог-доходностей конвертируется в уровни цен: $\hat{P}_{T+h} = P_T \cdot \exp(\sum_{i=1}^{h} \hat{r}_{T+i})$." + "\n\n"
    _f += "95% доверительный интервал показывает диапазон, в котором цена окажется с вероятностью 95% (при условии, что модель верна).\n\n"
    _f += "**Важно:** с ростом горизонта *h* CI быстро расширяется — неопределённость накапливается. На практике прогноз ARIMA на 1 день — это максимум полезного горизонта для дневных данных."
    mo.md(_f)
    return


@app.cell
def _(mo):
    forecast_horizon_slider = mo.ui.slider(
        start=1, stop=30, step=1, value=5, label="Горизонт прогноза (дней):"
    )
    return (forecast_horizon_slider,)


@app.cell
def _(
    best_model_fit,
    best_order,
    close_df,
    forecast_horizon_slider,
    np,
    pl,
    plt,
    ticker_dropdown,
    timedelta,
):
    _h = forecast_horizon_slider.value

    if close_df.height > 30:
        # Прогноз лог-доходностей
        _forecast_obj = best_model_fit.get_forecast(steps=_h)
        _fc_mean = np.asarray(_forecast_obj.predicted_mean, dtype=float)
        _fc_ci = np.asarray(_forecast_obj.conf_int(alpha=0.05), dtype=float)

        # Конвертация лог-доходностей → уровни цен
        _last_price = close_df["close"][-1]
        _cum_returns = np.cumsum(_fc_mean)
        _fc_prices = _last_price * np.exp(_cum_returns)

        # CI для цен
        _cum_lower = np.cumsum(_fc_ci[:, 0])
        _cum_upper = np.cumsum(_fc_ci[:, 1])
        _fc_lower = _last_price * np.exp(_cum_lower)
        _fc_upper = _last_price * np.exp(_cum_upper)

        # Даты прогноза (рабочие дни)
        _last_date = close_df["date"][-1]
        _forecast_dates = []
        _d = _last_date
        while len(_forecast_dates) < _h:
            _d = _d + timedelta(days=1)
            if _d.weekday() < 5:
                _forecast_dates.append(_d)

        # График
        _n_history = min(60, close_df.height)
        fig_fwd, ax_fwd = plt.subplots(figsize=(14, 6))
        ax_fwd.plot(close_df["date"].to_list()[-_n_history:], close_df["close"].to_numpy()[-_n_history:],
                    linewidth=1.5, color="steelblue", label="Исторические данные")
        ax_fwd.plot(_forecast_dates, _fc_prices, linewidth=2, color="coral", label="Прогноз")
        ax_fwd.fill_between(_forecast_dates, _fc_lower, _fc_upper, alpha=0.2, color="coral", label="95% CI")
        ax_fwd.axvline(_last_date, color="gray", linestyle="--", linewidth=0.8)
        ax_fwd.set_title(f"{ticker_dropdown.value} — Прогноз на {_h} дней ARIMA{best_order}", fontweight="bold")
        ax_fwd.set_ylabel("Цена")
        ax_fwd.legend(fontsize=11)
        ax_fwd.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        del fig_fwd, ax_fwd

        # Таблица прогноза
        forecast_table_df = pl.DataFrame({
            "Дата": [_fd.strftime("%Y-%m-%d") for _fd in _forecast_dates],
            "Прогноз цены": [f"{p:.2f}" for p in _fc_prices],
            "Нижняя граница (95%)": [f"{p:.2f}" for p in _fc_lower],
            "Верхняя граница (95%)": [f"{p:.2f}" for p in _fc_upper],
            "Прогноз лог-доходности": [f"{r:.6f}" for r in _fc_mean],
        })
    else:
        forecast_table_df = pl.DataFrame()
    return (forecast_table_df,)


@app.cell
def _(forecast_table_df, mo):
    # Таблица прогноза по дням (раньше собиралась, но не отображалась)
    if forecast_table_df.height > 0:
        forecast_table_view = mo.ui.table(forecast_table_df, label="Прогноз по дням")
    else:
        forecast_table_view = mo.md("")
    forecast_table_view
    return


if __name__ == "__main__":
    app.run()
