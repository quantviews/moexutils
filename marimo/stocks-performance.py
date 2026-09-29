"""
# Stocks Performance Analysis

Интерактивный анализ performance акций с учетом market cap.
"""

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium", app_title="Обзор фондового рынка РФ", css_file="styles.css")


@app.cell(hide_code=True)
def _():
    # stocks лежит в корне проекта (родительская папка от marimo/)
    import sys as _sys
    import os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    import stocks
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import date, datetime, timedelta
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

    return base64, date, go, io, mo, np, pl, plotly_available, plt, px, stocks, timedelta


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
def _(stocks):
    # Загружаем данные; цены приводим к пост-сплитовой базе (metadata/splits.csv),
    # иначе дробления акций (T 1:10 в 2026 и др.) выглядят как обвал цены
    combined_df = stocks.read_stocks(split_adjusted=True)
    return (combined_df,)


@app.cell(hide_code=True)
def _(pl, stocks):
    # Справочник тикер→сектор (используется картой рынка, секторным разрезом и структурой)
    import os as _oss
    _sec_path = _oss.path.join(stocks.BASE_DIR, 'metadata', 'sectors.csv')
    if _oss.path.exists(_sec_path):
        sectors_map = pl.read_csv(_sec_path)
    else:
        sectors_map = pl.DataFrame(schema={'ticker': pl.String, 'sector': pl.String})
    return (sectors_map,)


@app.cell(hide_code=True)
def _(combined_df, date, period_dropdown, pl):
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
        _all_dates = df['date'].unique().sort()
        end_date = _all_dates.max()

        if period_code == "1d":
            # последние два торговых дня
            start_date = _all_dates[-2] if len(_all_dates) >= 2 else end_date
        elif period_code == "ytd":
            start_date = date(end_date.year, 1, 1)
        else:
            # Календарные сдвиги (конец месяца обрезается, как у DateOffset)
            _offsets = {
                "1w": "-1w",
                "2w": "-2w",
                "1m": "-1mo",
                "3m": "-3mo",
                "6m": "-6mo",
                "1y": "-1y",
            }
            start_date = pl.select(
                pl.lit(end_date).dt.offset_by(_offsets.get(period_code, "-1w"))).item()

        # Первая и последняя котировка каждой бумаги в периоде
        _close = pl.col('close').fill_nan(None)
        _mcap = pl.col('market_cap').fill_nan(None)
        performances = (
            df.filter(pl.col('date').is_between(start_date, end_date))
            .sort('ticker', 'date')
            .group_by('ticker', maintain_order=True)
            .agg(
                _n=pl.len(),
                first_price=_close.first(),
                last_price=_close.last(),
                first_market_cap=_mcap.first(),
                last_market_cap=_mcap.last(),
                start_date=pl.col('date').first(),
                end_date=pl.col('date').last(),
            )
            # Нужно минимум 2 дня и валидные цены
            .filter(
                (pl.col('_n') >= 2)
                & pl.col('first_price').is_not_null()
                & pl.col('last_price').is_not_null()
                & (pl.col('first_price') > 0)
            )
        )

        # Performance по цене и по market cap (если есть данные)
        _mc_ok = (pl.col('first_market_cap').is_not_null()
                  & pl.col('last_market_cap').is_not_null()
                  & (pl.col('first_market_cap') > 0))
        performances = performances.select(
            'ticker',
            price_performance=(pl.col('last_price') - pl.col('first_price')) / pl.col('first_price') * 100,
            first_price='first_price',
            last_price='last_price',
            market_cap_performance=pl.when(_mc_ok).then(
                (pl.col('last_market_cap') - pl.col('first_market_cap')) / pl.col('first_market_cap') * 100),
            market_cap_change=pl.when(_mc_ok).then(pl.col('last_market_cap') - pl.col('first_market_cap')),
            first_market_cap='first_market_cap',
            last_market_cap='last_market_cap',
            start_date='start_date',
            end_date='end_date',
        )

        return performances, start_date, end_date

    perf_df, period_start, period_end = calculate_performance(combined_df, period_dropdown.value)
    period_label = PERIOD_LABELS.get(period_dropdown.value, str(period_dropdown.value))
    return perf_df, period_end, period_label, period_start


@app.cell(hide_code=True)
def _(combined_df, np, pl, stocks):
    # Годовые метрики риска по бумагам: волатильность (аннуализированная),
    # бета к IMOEX, max drawdown и расстояние от 52-недельного максимума
    _last_date_r = combined_df['date'].max()
    _start_1y = pl.select(pl.lit(_last_date_r).dt.offset_by('-1y')).item()
    _wide_r = combined_df.pivot(on='ticker', index='date', values='close',
                                aggregate_function='last', sort_columns=True).sort('date')
    _tk = [_c for _c in _wide_r.columns if _c != 'date']
    _wide_1y = _wide_r.filter(pl.col('date') >= _start_1y)
    # Дневные доходности по сетке торговых дат рынка — только когда бумага торговалась
    # в оба соседних дня: дни без сделок и период после снятия с торгов не дают
    # нулевых доходностей, занижающих волатильность и бету
    _rets = _wide_1y.select('date', pl.col(_tk) / pl.col(_tk).shift(1) - 1)

    _rets_long = _rets.unpivot(index='date', variable_name='ticker', value_name='ret')
    _px_long = _wide_1y.unpivot(index='date', variable_name='ticker', value_name='close')

    _stats = _rets_long.group_by('ticker', maintain_order=True).agg(
        _count=pl.col('ret').count(),
        vol_1y=pl.col('ret').std() * np.sqrt(252) * 100,
    ).with_columns(
        # меньше ~3 месяцев наблюдений — оценка ненадежна
        vol_1y=pl.when(pl.col('_count') >= 60).then(pl.col('vol_1y'))
    )

    _dd = _px_long.group_by('ticker', maintain_order=True).agg(
        mdd_1y=((pl.col('close') / pl.col('close').cum_max()) - 1).min() * 100,
        off_high=(pl.col('close').drop_nulls().last() / pl.col('close').max() - 1) * 100,
    )

    _beta = pl.DataFrame({'ticker': _tk, 'beta': [None] * len(_tk)},
                         schema={'ticker': pl.String, 'beta': pl.Float64})
    try:
        _imx_r = stocks.read_index('IMOEX').sort('date')
        _imx_ret_s = _imx_r.select('date', _IMOEX_=pl.col('close') / pl.col('close').shift(1) - 1)
        _imx_ret_s = _imx_ret_s.filter(pl.col('date') >= _start_1y)
        _aligned = _rets.join(_imx_ret_s, on='date', how='inner')
        _ivar = float(_aligned['_IMOEX_'].var())
        if _ivar > 0:
            # Ковариация по парам, где есть обе доходности (как Series.cov)
            _al = _aligned.select('date', '_IMOEX_').join(
                _rets_long, on='date', how='inner')
            _m = pl.col('ret').is_not_null() & pl.col('_IMOEX_').is_not_null()
            _a = pl.col('ret').filter(_m)
            _b = pl.col('_IMOEX_').filter(_m)
            _beta = _al.group_by('ticker').agg(
                beta=((_a - _a.mean()) * (_b - _b.mean())).sum() / (_m.sum() - 1) / _ivar
            ).join(_stats.select('ticker', '_count'), on='ticker', how='left').select(
                'ticker', beta=pl.when(pl.col('_count') >= 60).then(pl.col('beta'))
            )
    except Exception:
        pass

    risk_df = (
        pl.DataFrame({'ticker': _tk})
        .join(_stats.select('ticker', 'vol_1y'), on='ticker', how='left', maintain_order='left')
        .join(_beta, on='ticker', how='left', maintain_order='left')
        .join(_dd, on='ticker', how='left', maintain_order='left')
        .select('ticker', 'vol_1y', 'beta', 'mdd_1y', 'off_high')
    )
    return (risk_df,)


@app.cell(hide_code=True)
def _(filtered_df, imoex_ret, pl, period_end, period_start, risk_df):
    # Обогащение риск-метриками: σ-движение (аномальность хода за период)
    # и альфа к IMOEX (изменение бумаги минус бета × изменение индекса)
    enriched_df = filtered_df.join(risk_df, on='ticker', how='left', maintain_order='left')

    _years = max((period_end - period_start).days, 1) / 365.25
    _denom = pl.col('vol_1y') * (_years ** 0.5)
    enriched_df = enriched_df.with_columns(
        sigma_move=pl.when(_denom != 0).then(pl.col('price_performance') / _denom)
    )

    if imoex_ret is not None:
        enriched_df = enriched_df.with_columns(
            alpha=pl.col('price_performance') - pl.col('beta') * imoex_ret)
    else:
        enriched_df = enriched_df.with_columns(alpha=pl.lit(None, dtype=pl.Float64))
    return (enriched_df,)


@app.cell(hide_code=True)
def _(min_market_cap, perf_df, pl, sort_by):
    # Фильтруем и сортируем данные
    filtered_df = perf_df.clone()

    # Фильтр по минимальному market cap (конвертируем из миллиардов в рубли)
    if 'last_market_cap' in filtered_df.columns:
        min_cap_rub = min_market_cap.value * 1e9
        filtered_df = filtered_df.filter(
            pl.col('last_market_cap').is_null() |
            (pl.col('last_market_cap') >= min_cap_rub)
        )

    # Сортировка
    if sort_by.value in filtered_df.columns:
        # числа — от больших к меньшим, тикеры — по алфавиту
        filtered_df = filtered_df.sort(sort_by.value, descending=sort_by.value != "ticker",
                                       nulls_last=True, maintain_order=True)
    elif sort_by.value == "ticker":
        filtered_df = filtered_df.sort('ticker')
    return (filtered_df,)


@app.cell(hide_code=True)
def _(enriched_df, mo, pl, show_market_cap):
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

    display_df = enriched_df.select(display_cols)

    # Округление риск-метрик
    for _rc, _nd in (('sigma_move', 1), ('alpha', 1), ('vol_1y', 0),
                     ('beta', 2), ('mdd_1y', 1), ('off_high', 1)):
        if _rc in display_df.columns:
            display_df = display_df.with_columns(pl.col(_rc).cast(pl.Float64).round(_nd))

    # Форматирование
    _fmt = []
    if 'price_performance' in display_df.columns:
        _fmt.append(pl.col('price_performance').round(2))
    if 'market_cap_performance' in display_df.columns:
        _fmt.append(pl.col('market_cap_performance').round(2))
    if 'market_cap_change' in display_df.columns:
        _fmt.append((pl.col('market_cap_change') / 1e9).round(2))  # в миллиардах
    if 'last_market_cap' in display_df.columns:
        _fmt.append((pl.col('last_market_cap') / 1e9).round(2))  # в миллиардах
    if 'first_price' in display_df.columns:
        _fmt.append(pl.col('first_price').round(2))
    if 'last_price' in display_df.columns:
        _fmt.append(pl.col('last_price').round(2))

    # Форматирование дат
    if 'start_date' in display_df.columns:
        _fmt.append(pl.col('start_date').dt.strftime('%d.%m.%Y'))
    if 'end_date' in display_df.columns:
        _fmt.append(pl.col('end_date').dt.strftime('%d.%m.%Y'))
    if _fmt:
        display_df = display_df.with_columns(_fmt)

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
    display_df = display_df.rename({_k: _v for _k, _v in column_mapping.items()
                                    if _k in display_df.columns})

    table = mo.ui.table(display_df, pagination=True, page_size=20)
    return (table,)


@app.cell(hide_code=True)
def _(base64, filtered_df, io, mo, pl, period_label, plt):
    # Лидеры и аутсайдеры: топ-15 в обе стороны (полный список — в таблице)
    if len(filtered_df) > 0:
        _n_show = 15
        _mv = pl.concat([
            filtered_df.sort('price_performance', descending=True, maintain_order=True).head(_n_show),
            filtered_df.sort('price_performance', maintain_order=True).head(_n_show),
        ]).unique(subset='ticker', keep='first', maintain_order=True).sort(
            'price_performance', maintain_order=True)

        _perf1 = _mv['price_performance'].to_list()
        _fig1, _ax1 = plt.subplots(figsize=(9.5, max(6.0, 0.32 * len(_mv))))
        _colors1 = ['green' if _x >= 0 else 'red' for _x in _perf1]
        _bars1 = _ax1.barh(_mv['ticker'].to_list(), _perf1, color=_colors1, alpha=0.75)
        _ax1.set_xlabel('Изменение цены (%)')
        _ax1.set_title(f'Лидеры и аутсайдеры (топ-{_n_show} в обе стороны) — {period_label}')
        _ax1.axvline(x=0, color='black', linewidth=0.8)
        _ax1.grid(axis='x', linestyle='--', alpha=0.5)
        _ax1.margins(x=0.12)

        for _bar, _val in zip(_bars1, _perf1):
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
def _(base64, filtered_df, io, mo, period_label, pl, plt, show_market_cap):
    # Крупнейшие изменения капитализации: топ-10 по модулю
    if show_market_cap.value and 'market_cap_change' in filtered_df.columns and len(filtered_df) > 0:
        mc_change_data = filtered_df.filter(pl.col('market_cap_change').is_not_null())
        if len(mc_change_data) > 0:
            mc_change_data = (
                mc_change_data
                .sort(pl.col('market_cap_change').abs(), descending=True, maintain_order=True)
                .head(10)
                .sort('market_cap_change', maintain_order=True)
            )
            _fig2, _ax2 = plt.subplots(figsize=(9.5, 4.5))

            _mcb = (mc_change_data['market_cap_change'] / 1e9).to_list()
            _colors3 = ['green' if x >= 0 else 'red' for x in mc_change_data['market_cap_change'].to_list()]
            _bars3 = _ax2.barh(mc_change_data['ticker'].to_list(), _mcb, color=_colors3, alpha=0.7)
            _ax2.set_xlabel('Изменение market cap (млрд руб)')
            _ax2.set_title(f'Крупнейшие изменения market cap (топ-10) — {period_label}')
            _ax2.margins(x=0.12)
            _ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            _ax2.grid(axis='x', linestyle='--', alpha=0.7)

            # Добавляем значения
            for _bar3, _val3 in zip(_bars3, _mcb):
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
    date,
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
    return_mode,
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

    # Для "1 дня" пишем явно, изменение какой сессии показано
    if period_dropdown.value == '1d':
        _period_line = f"**За {period_end_str}** (к закрытию {period_start_str})"
    else:
        _period_line = f"**Период:** {period_start_str} - {period_end_str}"

    # Если последняя дата в данных — сегодня, цены могут быть внутридневными
    if period_end == date.today():
        _period_line += " · ⏳ *последняя дата — сегодня: цены на момент обновления данных, сессия может быть не закрыта*"

    mo.vstack([
        mo.hstack([period_dropdown, show_market_cap, sort_by, min_market_cap], justify="start"),
        mo.md(f"## Что произошло на рынке — {period_label}\n{_period_line}"),
        market_summary,
        breadth_block,
        index_block,
        sector_block,
        heatmap_block,
        volume_block,
        mo.md("---\n## 🧭 Структура рынка: сравнение с опорной датой"),
        mo.hstack([anchor_date, return_mode], justify="start"),
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
    _mc = filtered_df.drop_nulls(subset=['first_market_cap', 'last_market_cap']) if _n else filtered_df
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
        _top = filtered_df.sort('price_performance', descending=True, maintain_order=True).head(5)
        _bot = filtered_df.sort('price_performance', maintain_order=True).head(5)
        _top_str = ", ".join(f"**{_r['ticker']}** {_sgn(_r['price_performance'], '%', 1)}"
                             for _r in _top.iter_rows(named=True))
        _bot_str = ", ".join(f"**{_r['ticker']}** {_sgn(_r['price_performance'], '%', 1)}"
                             for _r in _bot.iter_rows(named=True))
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
def _(filtered_df, go, mo, period_label, pl, plotly_available):
    # Вертикальный Marimekko (интерактивный): толщина бара = доля в капитализации,
    # длина = performance, лучшие сверху. Тонкие бары читаются через hover.
    _mk = filtered_df.drop_nulls(subset=['last_market_cap', 'price_performance'])
    if len(_mk) == 0:
        marimekko_block = mo.md("")
    elif not plotly_available:
        marimekko_block = mo.md("*Для Marimekko нужен plotly: `pip install plotly`*")
    else:
        _mk = _mk.sort('price_performance', descending=True, maintain_order=True)
        _mk = _mk.with_columns(share=pl.col('last_market_cap') / pl.col('last_market_cap').sum() * 100)
        _mk = _mk.with_columns(y_center=-(pl.col('share').cum_sum() - pl.col('share') / 2))

        _tickers_m = _mk['ticker'].to_list()
        _perf_m = _mk['price_performance'].to_list()
        _share_m = _mk['share'].to_list()
        _figm = go.Figure(go.Bar(
            x=_perf_m,
            y=_mk['y_center'].to_list(),
            width=(_mk['share'] * 0.94).clip(lower_bound=0.12).to_list(),
            orientation='h',
            marker_color=['green' if _v >= 0 else 'red' for _v in _perf_m],
            marker_line=dict(color='white', width=0.5),
            text=[f'{_t} {_v:+.1f}%' if _s >= 0.8 else ''
                  for _t, _v, _s in zip(_tickers_m, _perf_m, _share_m)],
            textposition='outside',
            textfont_size=11,
            customdata=[
                (_t, f'{_s:.2f}%', f'{_v:+.2f}%')
                for _t, _s, _v in zip(_tickers_m, _share_m, _perf_m)
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
def _(date, go, mo, pl, period_end, period_label, period_start, plotly_available, stocks):
    # IMOEX (интерактивный) с линиями EWMAC — пара EWMA 16/64 дня (по Р. Карверу):
    # быстрая выше медленной = восходящий тренд. Данные: хранилище, таблица indexes.
    imoex_ret = None
    try:
        _idx_df = stocks.read_index('IMOEX')
        if _idx_df.is_empty():
            raise FileNotFoundError("IMOEX нет в хранилище")
        _idx_df = _idx_df.select('date', pl.col('close').cast(pl.Float64)).sort('date')

        # Доходность за выбранный период — для сводки "Итоги"
        _win = _idx_df.filter(pl.col('date').is_between(period_start, period_end))
        if len(_win) >= 2:
            imoex_ret = (float(_win['close'][-1]) / float(_win['close'][0]) - 1) * 100

        if not plotly_available:
            index_block = mo.md("*Для графика IMOEX нужен plotly: `pip install plotly`*")
        elif len(_idx_df) < 70:
            index_block = mo.md("*IMOEX: недостаточно истории в хранилище — обновите: `python update_data.py`*")
        else:
            # EWMA считаем по всей истории (без прогревочного смещения),
            # показываем динамику с 2022 года
            _idx_df = _idx_df.with_columns(
                ew16=pl.col('close').ewm_mean(span=16, adjust=False),
                ew64=pl.col('close').ewm_mean(span=64, adjust=False),
            )
            _show_from = min(date(2022, 1, 1), period_start)
            _c = _idx_df.filter(pl.col('date') >= _show_from)
            _dates_c = _c['date'].to_list()
            _last_close = float(_c['close'][-1])

            _figi = go.Figure()
            _figi.add_scatter(x=_dates_c, y=_c['close'].to_list(), name='IMOEX',
                              line=dict(color='#1f77b4', width=1.8),
                              hovertemplate='%{y:.0f}<extra>IMOEX</extra>')
            _figi.add_scatter(x=_dates_c, y=_c['ew16'].to_list(), name='EWMA 16',
                              line=dict(color='#2ca02c', width=1.1),
                              hovertemplate='%{y:.0f}<extra>EWMA 16</extra>')
            _figi.add_scatter(x=_dates_c, y=_c['ew64'].to_list(), name='EWMA 64',
                              line=dict(color='#d62728', width=1.1, dash='dot'),
                              hovertemplate='%{y:.0f}<extra>EWMA 64</extra>')
            # Подсветка выбранного периода анализа
            _figi.add_vrect(x0=period_start, x1=period_end,
                            fillcolor='gray', opacity=0.08, line_width=0)
            # Последнее значение
            _figi.add_scatter(x=[_dates_c[-1]], y=[_last_close], mode='markers',
                              marker=dict(color='#1f77b4', size=7),
                              showlegend=False, hoverinfo='skip')
            _figi.add_annotation(x=_dates_c[-1], y=_last_close,
                                 text=f'<b>{_last_close:,.0f}</b>'.replace(',', ' '),
                                 showarrow=False, xanchor='left', xshift=8,
                                 font=dict(color='#1f77b4', size=13))

            if imoex_ret is not None:
                _ret_str = (f'{imoex_ret:+.2f}% (от закрытия {_win["date"][0].strftime("%d.%m")}: '
                            f'{float(_win["close"][0]):,.0f})')
            else:
                _ret_str = 'н/д'
            _trend = ('восходящий (EWMA16 > EWMA64)'
                      if float(_idx_df['ew16'][-1]) > float(_idx_df['ew64'][-1])
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
        index_block = mo.md("*IMOEX нет в хранилище — выполните `python update_data.py` (шаг 1b)*")
    except Exception as _e_idx:
        index_block = mo.md(f"*IMOEX: ошибка чтения хранилища — {_e_idx}*")
    return imoex_ret, index_block


@app.cell(hide_code=True)
def _(combined_df, mo, np, pl):
    # Ширина рынка: 52-недельные экстремумы + доля бумаг выше MA50/MA200.
    # Классика: >50% бумаг выше MA200 — здоровый рынок, дивергенция с индексом — ранний сигнал.
    _last_date = combined_df['date'].max()
    _ydf = combined_df.filter(
        pl.col('date') >= pl.select(pl.lit(_last_date).dt.offset_by('-1y')).item())
    _ext = _ydf.sort('ticker', 'date').group_by('ticker', maintain_order=True).agg(
        _hi=pl.col('close').max(),
        _lo=pl.col('close').min(),
        _lastp=pl.col('close').drop_nulls().last(),
    )

    _near_hi = sorted(_ext.filter(pl.col('_lastp') >= pl.col('_hi') * 0.98)['ticker'].to_list())
    _near_lo = sorted(_ext.filter(pl.col('_lastp') <= pl.col('_lo') * 1.02)['ticker'].to_list())

    def _fmt_tickers(_lst, _limit=12):
        if not _lst:
            return "—"
        _s = ", ".join(_lst[:_limit])
        return _s + (f" и еще {len(_lst) - _limit}" if len(_lst) > _limit else "")

    # Доля бумаг выше скользящих средних (по всей истории, показываем последний год)
    _wide = combined_df.pivot(on='ticker', index='date', values='close',
                              aggregate_function='last', sort_columns=True).sort('date')
    _tk = [_c for _c in _wide.columns if _c != 'date']
    _prices = _wide.select(_tk).to_numpy().astype(float)
    _ma50 = _wide.select(pl.col(_tk).rolling_mean(50, min_samples=50)).to_numpy().astype(float)
    _ma200 = _wide.select(pl.col(_tk).rolling_mean(200, min_samples=200)).to_numpy().astype(float)

    def _pct_above(_p, _ma):
        _valid = ~np.isnan(_ma) & ~np.isnan(_p)
        _cnt = _valid.sum(axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            _above = ((_p > _ma) & _valid).sum(axis=1) / np.where(_cnt > 0, _cnt, np.nan) * 100
        return _above[~np.isnan(_above)]

    _above50_series = _pct_above(_prices, _ma50)
    _above200_series = _pct_above(_prices, _ma200)
    _above50 = float(_above50_series[-1]) if len(_above50_series) else float('nan')
    _above200 = float(_above200_series[-1]) if len(_above200_series) else float('nan')

    breadth_block = mo.md(
        f"**Ширина рынка:** выше MA50: **{_above50:.0f}%** | выше MA200: **{_above200:.0f}%** | "
        f"у 52-нед. максимумов (≤2%): **{len(_near_hi)}** ({_fmt_tickers(_near_hi)}) | "
        f"у минимумов: **{len(_near_lo)}** ({_fmt_tickers(_near_lo)})"
    )
    return (breadth_block,)


@app.cell(hide_code=True)
def _(filtered_df, go, mo, period_label, pl, plotly_available, sectors_map):
    # Секторный разрез: динамика секторов, взвешенная по капитализации
    if len(filtered_df) == 0 or len(sectors_map) == 0:
        sector_block = mo.md("")
    elif not plotly_available:
        sector_block = mo.md("*Для секторного графика нужен plotly: `pip install plotly`*")
    else:
        _sdf = filtered_df.join(sectors_map, on='ticker', how='left', maintain_order='left')
        _sdf = _sdf.with_columns(pl.col('sector').fill_null('Прочее'))

        _rows = []
        for (_sec,), _grp in sorted(_sdf.partition_by('sector', as_dict=True, maintain_order=True).items()):
            _gmc = _grp.drop_nulls(subset=['first_market_cap', 'last_market_cap'])
            if len(_gmc) > 0 and _gmc['first_market_cap'].sum() > 0:
                _ret = (_gmc['last_market_cap'].sum() / _gmc['first_market_cap'].sum() - 1) * 100
            else:
                _ret = float(_grp['price_performance'].median())
            _rows.append({'sector': f"{_sec} ({len(_grp)})", 'ret': _ret,
                          'tickers': ", ".join(sorted(_grp['ticker'].to_list())[:15])})
        _sec_df = pl.DataFrame(_rows).sort('ret', maintain_order=True)

        _ret_s = _sec_df['ret'].to_list()
        _figs = go.Figure(go.Bar(
            x=_ret_s,
            y=_sec_df['sector'].to_list(),
            orientation='h',
            marker_color=['green' if _x >= 0 else 'red' for _x in _ret_s],
            text=[f'{_v:+.1f}%' for _v in _ret_s],
            textposition='outside',
            customdata=[
                (_tk, f'{_v:+.2f}%') for _tk, _v in zip(_sec_df['tickers'].to_list(), _ret_s)
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
def _(filtered_df, mo, np, period_label, pl, plotly_available, px, sectors_map):
    # Карта рынка (finviz-style treemap): сектора → бумаги,
    # площадь = капитализация, цвет = изменение цены. Hover — точные цифры.
    _hm = filtered_df.drop_nulls(subset=['price_performance', 'last_market_cap'])
    if len(_hm) == 0:
        heatmap_block = mo.md("")
    elif not plotly_available:
        heatmap_block = mo.md("*Для карты рынка нужен plotly: `pip install plotly`*")
    else:
        _hm = _hm.join(sectors_map, on='ticker', how='left', maintain_order='left')
        _hm = _hm.with_columns(pl.col('sector').fill_null('Прочее'))
        # Форматируем подписи заранее: форматы в шаблонах plotly с флагом "+"
        # применяются ненадежно, и на плитки попадают числа с 13 знаками
        _hm = _hm.with_columns(
            perf_str=pl.Series([f'{_v:+.1f}%' for _v in _hm['price_performance'].to_list()],
                               dtype=pl.String),
            mc_str=pl.Series([f'{_v:,.0f}'.replace(',', ' ')
                              for _v in (_hm['last_market_cap'] / 1e9).to_list()], dtype=pl.String),
        )

        # Шкала цвета по 95-му перцентилю, чтобы один выброс не обесцвечивал карту
        _vmax = max(float(np.percentile(np.abs(_hm['price_performance'].to_numpy()), 95)), 1e-9)

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
def _(combined_df, enriched_df, filtered_df, mo, pl, period_end, period_start, timedelta):
    # Необычная активность: среднедневной оборот за период против 90 дней до него
    _per = combined_df.filter(pl.col('date').is_between(period_start, period_end))
    _base = combined_df.filter(
        (pl.col('date') >= period_start - timedelta(days=90)) & (pl.col('date') < period_start)
    )

    _va = (
        _per.group_by('ticker').agg(per=pl.col('value_rub').mean())
        .join(_base.group_by('ticker').agg(base=pl.col('value_rub').mean()), on='ticker', how='inner')
        .sort('ticker')
        .drop_nulls()
    )
    _va = _va.filter(pl.col('base') > 1e7)  # отсекаем неликвид: база < 10 млн руб/день
    _va = _va.with_columns(ratio=pl.col('per') / pl.col('base'))
    _va = _va.sort('ratio', descending=True, maintain_order=True).head(10)

    _parts = []
    if len(_va) > 0:
        _perf_map = (
            dict(zip(filtered_df['ticker'].to_list(), filtered_df['price_performance'].to_list()))
            if len(filtered_df) else {}
        )
        _lines = []
        for _rv in _va.iter_rows(named=True):
            _tv = _rv['ticker']
            _pperf = _perf_map.get(_tv)
            _pstr = f"{_pperf:+.1f}%" if _pperf is not None else "—"
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
    _sm = enriched_df.drop_nulls(subset=['sigma_move']) if len(enriched_df) else enriched_df
    if len(_sm) > 0:
        _sm = _sm.filter(pl.col('sigma_move').abs() >= 2)
        _sm = _sm.sort(pl.col('sigma_move').abs(), descending=True, maintain_order=True).head(10)
        if len(_sm) > 0:
            _lines2 = [
                f"| {_r['ticker']} | {_r['price_performance']:+.1f}% | {_r['sigma_move']:+.1f}σ | {_r['vol_1y']:.0f}% |"
                for _r in _sm.iter_rows(named=True)
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
    # Контролы анализа структуры рынка
    anchor_date = mo.ui.date(value="2022-02-21", label="Опорная дата:")
    return_mode = mo.ui.radio(
        options={"Цена": "close", "Полная доходность (дивиденды + сплиты)": "adj_close"},
        value="Цена",
        label="Метрика:",
        inline=True,
    )
    return anchor_date, return_mode


@app.cell(hide_code=True)
def _(anchor_date, combined_df, go, mo, pl, plotly_available, return_mode, sectors_map, stocks, timedelta):
    # Структура рынка: что изменилось с опорной даты.
    # Отвечает на вопросы: индекс на том же уровне — а рынок тот же?
    # Кто вытащил/утопил капитализацию, как перекроились веса секторов,
    # выросла ли концентрация.
    _anchor = anchor_date.value
    _last_date = combined_df['date'].max()

    # Метрика: цена (сплит-скорр. close) или полная доходность (adj_close)
    _pc = return_mode.value if return_mode.value in combined_df.columns else 'close'
    _mode_label = 'полная доходность (дивиденды + сплиты)' if _pc == 'adj_close' else 'цена'

    # Срез "тогда": последняя котировка каждой бумаги в окне 45 дней до опорной даты
    _win_then = combined_df.filter((pl.col('date') <= _anchor) &
                                   (pl.col('date') >= _anchor - timedelta(days=45)))
    _then = (_win_then.sort('date', 'ticker').group_by('ticker', maintain_order=True).last()
             .select('ticker', close_then=pl.col(_pc), mc_then=pl.col('market_cap')))

    # Срез "сейчас": только бумаги, торговавшиеся в последние 30 дней (без делистингов)
    _now = combined_df.sort('date', 'ticker').group_by('ticker', maintain_order=True).last()
    _now = _now.filter(pl.col('date') >= _last_date - timedelta(days=30))
    _now = _now.select('ticker', close_now=pl.col(_pc), mc_now=pl.col('market_cap'))

    _st = (_then.join(_now, on='ticker', how='inner', maintain_order='left')
           .drop_nulls(subset=['close_then', 'close_now']))

    if len(_st) < 5:
        structure_block = mo.md("*Недостаточно данных на выбранную опорную дату*")
    else:
        _st = _st.with_columns(px_chg=(pl.col('close_now') / pl.col('close_then') - 1) * 100)

        def _sgn(_v, _suffix='%', _nd=1):
            """Число со знаком, раскрашенное классами pos/neg из styles.css"""
            _cls = 'pos' if _v >= 0 else 'neg'
            _txt = format(_v, f'+,.{_nd}f').replace(',', ' ')
            return f'<span class="{_cls}">{_txt}{_suffix}</span>'

        # --- IMOEX тогда и сейчас
        _imx_line2 = ""
        try:
            _idx = stocks.read_index('IMOEX').sort('date')
            _idx_then = _idx.filter(pl.col('date') <= _anchor)
            if len(_idx_then):
                _iv_then = float(_idx_then['close'][-1])
                _iv_now = float(_idx['close'][-1])
                _imx_note = " *(ценовой индекс, без дивидендов)*" if _pc == 'adj_close' else ""
                _imx_line2 = (f"- **IMOEX:** {_iv_then:,.0f} → {_iv_now:,.0f} ".replace(",", " ")
                              + f"({_sgn((_iv_now / _iv_then - 1) * 100)}){_imx_note}\n")
        except Exception:
            pass

        # --- счет выше/ниже уровня опорной даты
        _n_up = int((_st['px_chg'] > 0).sum())
        _n_down = int((_st['px_chg'] < 0).sum())
        _med_chg = float(_st['px_chg'].median())

        # --- капитализация и концентрация (по бумагам с mc в обеих точках)
        _mc = _st.drop_nulls(subset=['mc_then', 'mc_now'])
        _conc_line = ""
        _total_line = ""
        if len(_mc) >= 5 and _mc['mc_then'].sum() > 0:
            _tot_then = _mc['mc_then'].sum()
            _tot_now = _mc['mc_now'].sum()
            _big_then = _mc.sort('mc_then', descending=True, maintain_order=True).head(5)
            _big_now = _mc.sort('mc_now', descending=True, maintain_order=True).head(5)
            _top5_then = _big_then['mc_then'].sum() / _tot_then * 100
            _top5_now = _big_now['mc_now'].sum() / _tot_now * 100
            _t5_then_names = ", ".join(_big_then['ticker'].to_list())
            _t5_now_names = ", ".join(_big_now['ticker'].to_list())
            _total_line = (f"- **Капитализация (сопоставимые бумаги):** "
                           f"{_tot_then / 1e12:.1f} → {_tot_now / 1e12:.1f} трлн руб "
                           f"({_sgn((_tot_now / _tot_then - 1) * 100)})\n")
            _conc_line = (f"- **Концентрация (доля топ-5):** {_top5_then:.0f}% → {_top5_now:.0f}%\n"
                          f"  - тогда: {_t5_then_names}\n  - сейчас: {_t5_now_names}\n")

        _tops = _st.sort('px_chg', descending=True, maintain_order=True).head(5)
        _bots = _st.sort('px_chg', maintain_order=True).head(5)
        _tops_str = ", ".join(f"**{_r['ticker']}** {_sgn(_r['px_chg'], '%', 0)}"
                              for _r in _tops.iter_rows(named=True))
        _bots_str = ", ".join(f"**{_r['ticker']}** {_sgn(_r['px_chg'], '%', 0)}"
                              for _r in _bots.iter_rows(named=True))

        _md_struct = mo.md(
            f"### С {_anchor.strftime('%d.%m.%Y')} — {_mode_label} (сопоставимых бумаг: {len(_st)})\n\n"
            + _imx_line2
            + f"- **Выше уровня той даты: {_n_up}**, ниже: **{_n_down}**; медианная бумага: {_sgn(_med_chg)}\n"
            + _total_line + _conc_line
            + f"- 📈 Сильнее всех: {_tops_str}\n- 📉 Слабее всех: {_bots_str}"
        )

        _blocks = [_md_struct]

        if plotly_available and len(_mc) >= 5 and _mc['mc_then'].sum() > 0:
            # --- вклад бумаг в изменение суммарной капитализации (п.п.)
            _mc = _mc.with_columns(
                contrib=(pl.col('mc_now') - pl.col('mc_then')) / pl.col('mc_then').sum() * 100)
            _cb = (_mc.sort(pl.col('contrib').abs(), descending=True, maintain_order=True).head(12)
                   .sort('contrib', maintain_order=True))
            _contrib = _cb['contrib'].to_list()
            _figc = go.Figure(go.Bar(
                x=_contrib, y=_cb['ticker'].to_list(), orientation='h',
                marker_color=['green' if _v >= 0 else 'red' for _v in _contrib],
                text=[f'{_v:+.1f} п.п.' for _v in _contrib],
                textposition='outside',
                customdata=[
                    (f'{_c:+.2f}', f'{_p:+.1f}%')
                    for _c, _p in zip(_contrib, _cb['px_chg'].to_list())
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
                _ms = _mc.join(sectors_map, on='ticker', how='left', maintain_order='left')
                _ms = _ms.with_columns(pl.col('sector').fill_null('Прочее'))
                _w = (_ms.group_by('sector').agg(pl.col('mc_then').sum(), pl.col('mc_now').sum())
                      .sort('sector'))
                _w = (_w.with_columns(pl.col('mc_then', 'mc_now') / pl.col('mc_then', 'mc_now').sum() * 100)
                      .sort('mc_now', maintain_order=True))
                _sectors_w = _w['sector'].to_list()
                _figw = go.Figure()
                _figw.add_bar(x=_w['mc_then'].to_list(), y=_sectors_w, orientation='h', name='Тогда',
                              marker_color='#9ecae1',
                              hovertemplate='%{y}: %{x:.1f}%<extra>тогда</extra>')
                _figw.add_bar(x=_w['mc_now'].to_list(), y=_sectors_w, orientation='h', name='Сейчас',
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
