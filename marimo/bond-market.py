"""
marimo notebook: Долговой рынок — кривая ОФЗ, доходности выпусков, RGBITR

Использование:
1. Данные — полная история рынка облигаций в хранилище (lake.bonds); первичная
   выгрузка: python update_data.py --history-init bonds
   (дальше обновляется ночным шагом 1c update_data.py)
2. marimo edit bond-market.py

Функционал:
- Кривая доходности ОФЗ (YTM × срок до погашения): сегодня против выбранной
  прошлой даты; только выпуски с фиксированным купоном (ОФЗ-ПД)
- Сводка: короткий/длинный конец, наклон кривой, сдвиг за период
- История кривой: ОФЗ 2 года / 10 лет, ключевая ставка, терм-спред 10л−2г
- Корпоративные выпуски (TQCB): G-спреды к кривой ОФЗ (ликвидные, оборот
  ≥ 1 млн руб) + история медианного G-спреда
- RGBITR (гособлигации, полная доходность) против IMOEX
- Таблица всех выпусков: цена, YTM, дюрация, выпуклость, оборот

Методика: YTM/дюрация — биржевые (YIELDCLOSE/DURATION из истории ISS),
упрощенная модель — фоллбэк; выпуклость всегда модельная.
"""

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium", app_title="Долговой рынок", css_file="styles.css")


@app.cell(hide_code=True)
def _():
    # moex_utils лежит в корне проекта (родительская папка от marimo/)
    import sys as _sys
    import os as _os
    import datetime as dt
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    import moex_utils as moex
    import polars as pl
    import numpy as np
    import marimo as mo
    try:
        import plotly.graph_objects as go
        plotly_available = True
    except ImportError:
        go = None
        plotly_available = False
    return dt, go, mo, moex, np, pl, plotly_available


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Долговой рынок: кривая ОФЗ

    Кривая доходности гособлигаций — главный макрограф долгового рынка:
    короткий конец повторяет ключевую ставку, длинный — ожидания по ставке
    и инфляции на годы вперед. **Инверсия** (короткие доходности выше длинных)
    исторически означает ожидания снижения ставки.

    Для кривой берутся только **ОФЗ-ПД** (фиксированный купон, серии 25xxx/26xxx):
    у флоатеров (ПК) и линкеров (ИН) доходность из цены так считать нельзя.

    **G-спред** корпоративной облигации = ее YTM минус доходность кривой ОФЗ,
    интерполированная на тот же срок: премия за кредитный риск эмитента.
    Широкий спред — рынок сомневается в эмитенте (или выпуск неликвиден),
    сжатие спредов сегмента — аппетит к риску растет.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    compare_dropdown = mo.ui.dropdown(
        options={"1 месяц назад": 1, "3 месяца назад": 3,
                 "6 месяцев назад": 6, "1 год назад": 12},
        value="3 месяца назад",
        label="Сравнить кривую с:",
    )
    compare_dropdown
    return (compare_dropdown,)


@app.cell(hide_code=True)
def _(mo, moex, pl):
    # Все выпуски досок TQOB (гособлигации) и TQCB (корпоративные) из хранилища
    _COLS = ['date', 'SECID', 'BOARDID', 'SHORTNAME', 'CLOSE', 'LEGALCLOSEPRICE',
             'YIELDCLOSE', 'DURATION', 'VALUE', 'MATDATE', 'FACEVALUE', 'FACEUNIT',
             'COUPONPERCENT']
    _OFZ_SERIES = {'25': 'ПД', '26': 'ПД', '24': 'ПК', '29': 'ПК',
                   '52': 'ИН', '46': 'АД', '48': 'АД'}

    try:
        bonds_long = (
            moex.read_bonds_market(boards=['TQOB', 'TQCB'], columns=_COLS)
            .rename({'LEGALCLOSEPRICE': 'PRICE_ALT', 'BOARDID': 'segment'})
            .with_columns(
                pl.col('MATDATE').cast(pl.Utf8).str.to_date('%Y-%m-%d', strict=False),
                # тип ОФЗ по серии в SECID: SU26238RMFS4 -> 26 -> ПД
                pl.when(pl.col('SECID').str.starts_with('SU'))
                  .then(pl.col('SECID').str.slice(2, 2).replace_strict(_OFZ_SERIES, default='прочее'))
                  .otherwise(pl.lit('прочее')).alias('bond_type'),
            )
        )
        _error = None
    except Exception as _e:
        bonds_long = pl.DataFrame()
        _error = str(_e)

    bonds_ready = bonds_long.height > 0
    if bonds_ready:
        _status = mo.md(
            f"**Данные:** {bonds_long['SECID'].n_unique()} выпусков досок TQOB и TQCB "
            f"(хранилище, с {bonds_long['date'].min():%d.%m.%Y}) · "
            f"последняя дата: {bonds_long['date'].max():%d.%m.%Y}")
    else:
        _status = mo.md(
            "⚠️ **Данные облигаций не загружены** из хранилища"
            + (f" ({_error})" if _error else "") + ". Первичная выгрузка:\n\n"
            "```\npython update_data.py --history-init bonds\n```"
        )
    _status
    return bonds_long, bonds_ready


@app.cell(hide_code=True)
def _(bonds_long, bonds_ready, dt, moex, pl):
    # Метрики всех выпусков на дату (последняя котировка не старше 14 дней до нее)
    def bonds_snapshot(asof=None):
        _w = bonds_long if asof is None else bonds_long.filter(pl.col('date') <= asof)
        if _w.height == 0:
            return pl.DataFrame()
        _last = _w.sort('date').group_by('SECID').last()
        _ref = _last['date'].max() if asof is None else asof
        _last = _last.filter(pl.col('date') >= _ref - dt.timedelta(days=14))

        rows = []
        for _r in _last.iter_rows(named=True):
            _price = _r['CLOSE'] if _r['CLOSE'] is not None else _r['PRICE_ALT']
            _mat = _r['MATDATE']
            if _price is None or _mat is None:
                continue
            _years = (_mat - _r['date']).days / 365.25
            if _years <= 0.05:
                continue
            _face = _r['FACEVALUE'] if _r['FACEVALUE'] is not None else 1000.0
            _coupon = _r['COUPONPERCENT'] if _r['COUPONPERCENT'] is not None else 0.0

            # Биржевые YTM/дюрация из истории ISS (точные: с НКД и фактическим
            # графиком купонов); упрощенная модель — фоллбэк для строк без них
            _y = _r['YIELDCLOSE']
            if _y is not None and 0 < _y < 100:
                _ytm, _src = _y, 'ISS'
            else:
                _ytm, _src = moex.calculate_ytm(_price, _face, _coupon, _years), 'модель'
            _d = _r['DURATION']  # ISS отдает дюрацию в днях
            _dur = (_d / 365.25 if _d is not None and _d > 0
                    else moex.calculate_duration(_price, _face, _coupon, _years, _ytm))

            rows.append({
                'SECID': _r['SECID'],
                'name': _r['SHORTNAME'] or _r['SECID'],
                'type': _r['bond_type'],
                'segment': _r['segment'] or '',
                'faceunit': _r['FACEUNIT'] or 'SUR',
                'maturity': _mat,
                'years': _years,
                'coupon': _coupon,
                'price': _price,
                'ytm': _ytm,
                'duration': _dur,
                'convexity': moex.calculate_convexity(_price, _face, _coupon, _years, _ytm),
                'value': _r['VALUE'] if _r['VALUE'] is not None else 0.0,
                'src': _src,
                'date': _r['date'],
            })
        return pl.DataFrame(rows)

    snap_now = bonds_snapshot() if bonds_ready else pl.DataFrame()
    return bonds_snapshot, snap_now


@app.cell(hide_code=True)
def _(bonds_ready, bonds_snapshot, compare_dropdown, dt, pl, snap_now):
    # Срез кривой на прошлую дату
    def _minus_months(_d, _m):
        _y, _mo = divmod(_d.month - 1 - _m, 12)
        return dt.date(_d.year + _y, _mo + 1, min(_d.day, 28))

    if bonds_ready and snap_now.height:
        compare_date = _minus_months(snap_now['date'].max(), compare_dropdown.value)
        snap_past = bonds_snapshot(asof=compare_date)
    else:
        compare_date = None
        snap_past = pl.DataFrame()
    return compare_date, snap_past


@app.cell(hide_code=True)
def _(compare_dropdown, go, mo, pl, plotly_available, snap_now, snap_past):
    # Кривая ОФЗ-ПД: сейчас против прошлой даты
    if not plotly_available or snap_now.height == 0:
        curve_block = mo.md("")
    else:
        _now = snap_now.filter(pl.col('type') == 'ПД').sort('years')
        _past = (snap_past.filter(pl.col('type') == 'ПД').sort('years')
                 if snap_past.height else snap_past)

        _figc = go.Figure()
        _figc.add_scatter(
            x=_now['years'].to_list(), y=_now['ytm'].to_list(), mode='lines+markers',
            name=f"сейчас ({_now['date'].max():%d.%m.%Y})",
            line=dict(color='#102D69', width=2.2), marker=dict(size=7),
            customdata=_now.select('name', 'SECID', 'duration').rows(),
            hovertemplate='<b>%{customdata[0]}</b> (%{customdata[1]})'
                          '<br>срок: %{x:.1f} лет · YTM: %{y:.2f}%'
                          '<br>дюрация: %{customdata[2]:.1f}<extra></extra>')
        if _past.height:
            _figc.add_scatter(
                x=_past['years'].to_list(), y=_past['ytm'].to_list(), mode='lines+markers',
                name=f"{compare_dropdown.selected_key or 'ранее'} "
                     f"({_past['date'].max():%d.%m.%Y})",
                line=dict(color='#9AAFD4', width=1.6, dash='dash'),
                marker=dict(size=5),
                customdata=_past.select('name').rows(),
                hovertemplate='<b>%{customdata[0]}</b>'
                              '<br>срок: %{x:.1f} лет · YTM: %{y:.2f}%<extra></extra>')
        _figc.update_layout(
            height=460, hovermode='closest',
            title=dict(text='Кривая доходности ОФЗ-ПД', font_size=15),
            xaxis=dict(title='Срок до погашения, лет'),
            yaxis=dict(title='YTM, % годовых', ticksuffix='%'),
            legend=dict(orientation='h', y=1.1, x=1, xanchor='right'),
            margin=dict(t=48, l=10, r=10, b=10),
        )
        curve_block = _figc
    curve_block
    return


@app.cell(hide_code=True)
def _(mo, pl, snap_now, snap_past):
    # Сводка по кривой
    def _sgn(_v, _suffix=' п.п.', _nd=2):
        if _v != _v:
            return 'н/д'
        _cls = 'pos' if _v >= 0 else 'neg'
        return f'<span class="{_cls}">{format(_v, f"+.{_nd}f")}{_suffix}</span>'

    def _bucket(_snap, _lo, _hi):
        _b = _snap.filter((pl.col('type') == 'ПД') & pl.col('years').is_between(_lo, _hi))['ytm']
        return float(_b.mean()) if _b.len() else float('nan')

    if snap_now.height == 0:
        curve_summary = mo.md("")
    else:
        _s2 = _bucket(snap_now, 1, 3)
        _s10 = _bucket(snap_now, 8, 12)
        _slope = _s10 - _s2
        _lines = [
            f"- **Короткий конец (1-3 года): {_s2:.2f}%** | "
            f"длинный (8-12 лет): **{_s10:.2f}%** | "
            f"наклон: {_sgn(_slope)}"
            + (" — **кривая инвертирована** (рынок ждет снижения ставки)"
               if _slope < -0.3 else "")
        ]
        if snap_past.height:
            _p2 = _bucket(snap_past, 1, 3)
            _p10 = _bucket(snap_past, 8, 12)
            _lines.append(
                f"- Сдвиг с прошлой даты: короткий {_sgn(_s2 - _p2)}, "
                f"длинный {_sgn(_s10 - _p10)}")
        curve_summary = mo.md("### Итоги\n\n" + "\n".join(_lines))
    curve_summary
    return


@app.cell(hide_code=True)
def _(bonds_long, bonds_ready, np, pl):
    # История кривой и спредов: на каждую дату — доходности ОФЗ в точках 2 и 10 лет
    # (интерполяция) и медианный G-спред ликвидных корпоратов
    curve_hist = pl.DataFrame()
    gspread_hist = pl.DataFrame()
    if bonds_ready:
        _h = (bonds_long
              .with_columns(yrs_h=(pl.col('MATDATE') - pl.col('date')).dt.total_days() / 365.25)
              .filter(pl.col('YIELDCLOSE').is_between(0.1, 60) & (pl.col('yrs_h') > 0.1)))

        _ofz_by_date = {
            _k[0]: _g.sort('yrs_h')
            for _k, _g in _h.filter(pl.col('bond_type') == 'ПД').partition_by('date', as_dict=True).items()
        }
        _rows_c = []
        for _d, _g in _ofz_by_date.items():
            _yrs = _g['yrs_h'].to_numpy()
            if _g.height >= 3 and _yrs[0] <= 2 and _yrs[-1] >= 10:
                _y2, _y10 = np.interp([2.0, 10.0], _yrs, _g['YIELDCLOSE'].to_numpy())
                _rows_c.append({'date': _d, 'y2': float(_y2), 'y10': float(_y10)})
        if _rows_c:
            curve_hist = pl.DataFrame(_rows_c).sort('date')

        _crp = _h.filter((pl.col('segment') == 'TQCB')
                         & pl.col('FACEUNIT').is_in(['SUR', 'RUB'])
                         & (pl.col('VALUE').fill_null(0) >= 1e6))
        _rows_g = []
        for _k, _g in _crp.partition_by('date', as_dict=True).items():
            _o = _ofz_by_date.get(_k[0])
            if _o is None or _o.height < 3 or _g.height < 20:
                continue
            _sp = (_g['YIELDCLOSE'].to_numpy()
                   - np.interp(_g['yrs_h'].to_numpy(), _o['yrs_h'].to_numpy(),
                               _o['YIELDCLOSE'].to_numpy())) * 100
            _rows_g.append({'date': _k[0], 'med_bp': float(np.median(_sp)), 'n': _g.height})
        if _rows_g:
            gspread_hist = pl.DataFrame(_rows_g).sort('date')
    return curve_hist, gspread_hist


@app.cell(hide_code=True)
def _(curve_hist, go, mo, moex, pl, plotly_available):
    # Динамика кривой: 2 года, 10 лет, ключевая ставка и терм-спред
    if not plotly_available or curve_hist.height < 20:
        term_block = mo.md("")
    else:
        from plotly.subplots import make_subplots
        _dates = curve_hist['date'].to_list()
        _figt = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              row_heights=[0.68, 0.32], vertical_spacing=0.06)
        _figt.add_scatter(x=_dates, y=curve_hist['y10'].to_list(),
                          name='ОФЗ 10 лет', line=dict(color='#102D69', width=1.8),
                          hovertemplate='%{y:.2f}%<extra>10 лет</extra>', row=1, col=1)
        _figt.add_scatter(x=_dates, y=curve_hist['y2'].to_list(),
                          name='ОФЗ 2 года', line=dict(color='#0050CF', width=1.6),
                          hovertemplate='%{y:.2f}%<extra>2 года</extra>', row=1, col=1)
        _kr = pl.read_csv(moex.KEY_RATE_FILE, try_parse_dates=True).sort('date')
        if _kr.height:
            _kr_s = curve_hist.select('date').join_asof(_kr, on='date', strategy='backward')
            _figt.add_scatter(x=_dates, y=_kr_s['rate'].to_list(), name='ключевая ставка',
                              line=dict(color='#7f7f7f', width=1.2, dash='dot'),
                              hovertemplate='%{y:.2f}%<extra>ключевая</extra>', row=1, col=1)
        _figt.add_scatter(x=_dates, y=(curve_hist['y10'] - curve_hist['y2']).to_list(),
                          name='спред 10л − 2г', fill='tozeroy',
                          line=dict(color='#d73027', width=1.2),
                          fillcolor='rgba(215,48,39,0.15)',
                          hovertemplate='%{y:+.2f} п.п.<extra>терм-спред</extra>', row=2, col=1)
        _figt.add_hline(y=0, line_width=1, line_color='#999', row=2, col=1)
        _figt.update_layout(
            height=520, hovermode='x unified',
            title=dict(text='Кривая ОФЗ во времени: 2 года, 10 лет и терм-спред', font_size=15),
            legend=dict(orientation='h', y=1.08, x=1, xanchor='right'),
            margin=dict(t=52, l=10, r=10, b=10),
        )
        _figt.update_yaxes(ticksuffix='%', row=1, col=1)
        _figt.update_yaxes(title_text='п.п.', row=2, col=1)
        term_block = mo.vstack([
            mo.md(
                "### Кривая во времени\n\n"
                "Терм-спред (10 лет − 2 года) — компактная мера формы кривой: "
                "отрицательный (инверсия) — рынок ждет снижения ставки, "
                "рост из отрицательной зоны к положительной обычно сопровождает "
                "цикл смягчения."),
            _figt,
        ])
    term_block
    return


@app.cell(hide_code=True)
def _(go, gspread_hist, mo, np, pl, plotly_available, snap_now):
    # Корпоративные облигации: G-спреды к интерполированной кривой ОФЗ
    _ofz = (snap_now.filter(pl.col('type') == 'ПД').sort('years')
            if snap_now.height else pl.DataFrame())
    _corp_all = (snap_now.filter((pl.col('segment') == 'TQCB')
                                 & pl.col('faceunit').is_in(['SUR', 'RUB'])
                                 & pl.col('ytm').is_between(0.1, 60))
                 if snap_now.height else pl.DataFrame())
    # Неликвид дает фиктивные спреды: оставляем выпуски с оборотом ≥ 1 млн руб
    _corp = _corp_all.filter(pl.col('value') >= 1e6) if _corp_all.height else _corp_all
    if _corp_all.height and _corp.height == 0:
        _corp = _corp_all

    if not plotly_available or snap_now.height == 0:
        corp_block = mo.md("")
    elif _corp_all.height == 0:
        corp_block = mo.md("*Нет котировок корпоративных облигаций (TQCB) на последнюю дату*")
    elif _ofz.height < 3:
        corp_block = mo.md("*Недостаточно точек кривой ОФЗ для расчета спредов*")
    else:
        # G-спред = YTM корпората − YTM ОФЗ, интерполированная на его срок
        _corp = _corp.with_columns(gspread_bp=pl.Series(
            (_corp['ytm'].to_numpy()
             - np.interp(_corp['years'].to_numpy(), _ofz['years'].to_numpy(),
                         _ofz['ytm'].to_numpy())) * 100))

        _figg = go.Figure()
        _figg.add_scatter(
            x=_ofz['years'].to_list(), y=_ofz['ytm'].to_list(), mode='lines',
            name='кривая ОФЗ', line=dict(color='#102D69', width=2),
            hovertemplate='ОФЗ %{x:.1f} лет: %{y:.2f}%<extra></extra>')
        _figg.add_scatter(
            x=_corp['years'].to_list(), y=_corp['ytm'].to_list(), mode='markers',
            name='корпораты (TQCB)',
            marker=dict(size=8, color=_corp['gspread_bp'].to_list(),
                        colorscale='RdYlGn', reversescale=True, cmin=0,
                        cmax=float(_corp['gspread_bp'].quantile(0.95)),
                        colorbar=dict(title='спред,<br>б.п.')),
            customdata=_corp.select('name', 'gspread_bp').rows(),
            hovertemplate='<b>%{customdata[0]}</b><br>срок: %{x:.1f} лет · '
                          'YTM: %{y:.2f}%<br>G-спред: %{customdata[1]:+.0f} б.п.'
                          '<extra></extra>')
        _figg.update_layout(
            height=440,
            title=dict(text='Корпоративные облигации против кривой ОФЗ', font_size=15),
            xaxis=dict(title='Срок до погашения, лет'),
            yaxis=dict(title='YTM, % годовых', ticksuffix='%'),
            legend=dict(orientation='h', y=1.1, x=1, xanchor='right'),
            margin=dict(t=48, l=10, r=10, b=10),
        )

        _med = float(_corp['gspread_bp'].median())
        _wide = _corp.sort('gspread_bp', descending=True).head(5)
        _tight = _corp.sort('gspread_bp').head(5)
        _md_corp = mo.md(
            f"**Корпоративный сегмент:** {_corp.height} ликвидных выпусков "
            f"(оборот ≥ 1 млн ₽, всего на доске {_corp_all.height}) · "
            f"медианный G-спред **{_med:.0f} б.п.**\n\n"
            f"- Самые широкие: " + ", ".join(
                f"**{_n}** {_g:+.0f}" for _n, _g in _wide.select('name', 'gspread_bp').rows()) + "\n"
            f"- Самые узкие: " + ", ".join(
                f"**{_n}** {_g:+.0f}" for _n, _g in _tight.select('name', 'gspread_bp').rows())
        )

        _tbl = (_corp.sort('gspread_bp', descending=True)
                .select(
                    'SECID',
                    pl.col('name').alias('Выпуск'),
                    pl.col('maturity').dt.strftime('%d.%m.%Y').alias('Погашение'),
                    pl.col('years').round(1).alias('Лет'),
                    pl.col('price').round(2).alias('Цена %'),
                    pl.col('ytm').round(2).alias('YTM %'),
                    pl.col('duration').round(1).alias('Дюрация'),
                    pl.col('gspread_bp').round(0).alias('G-спред, б.п.'),
                    (pl.col('value') / 1e6).round(1).alias('Оборот, млн ₽')))

        _parts = [_md_corp, _figg,
                  mo.ui.table(_tbl, pagination=True, page_size=15,
                              label='Корпоративные выпуски по G-спреду')]

        # История медианного G-спреда — барометр кредитных условий
        if gspread_hist.height >= 20:
            _gh = gspread_hist.with_columns(
                sm=pl.col('med_bp').rolling_median(window_size=21, min_samples=10))
            _figh = go.Figure()
            _figh.add_scatter(
                x=_gh['date'].to_list(), y=_gh['med_bp'].to_list(), name='медианный G-спред',
                line=dict(color='#0050CF', width=1.2),
                customdata=_gh.select('n').rows(),
                hovertemplate='%{y:.0f} б.п. · выпусков: %{customdata[0]}<extra></extra>')
            _figh.add_scatter(x=_gh['date'].to_list(), y=_gh['sm'].to_list(),
                              name='медиана за месяц', line=dict(color='#102D69', width=2),
                              hovertemplate='%{y:.0f} б.п.<extra>сглаженный</extra>')
            _figh.update_layout(
                height=340, hovermode='x unified',
                title=dict(text='Медианный G-спред ликвидных корпоратов во времени', font_size=14),
                yaxis=dict(title='б.п.'),
                legend=dict(orientation='h', y=1.12, x=1, xanchor='right'),
                margin=dict(t=48, l=10, r=10, b=10),
            )
            _parts.append(mo.md(
                "Рост медианного спреда — ужесточение кредитных условий "
                "(рынок требует большую премию за риск), сжатие — аппетит "
                "к риску возвращается."))
            _parts.append(_figh)

        corp_block = mo.vstack(_parts)
    corp_block
    return


@app.cell(hide_code=True)
def _(dt, go, mo, moex, pl, plotly_available):
    # RGBITR (гособлигации, полная доходность) против IMOEX за год
    def _index(_ticker):
        return moex.read_index(_ticker).select('date', 'close').sort('date')

    try:
        _rgb = _index('RGBITR')
        _rgb_ok = _rgb.height > 0
    except Exception:
        _rgb_ok = False

    if not plotly_available or not _rgb_ok:
        rgbitr_block = mo.md(
            "*RGBITR нет в хранилище — обновите индексы: `python update_data.py`*"
        ) if plotly_available else mo.md("")
    else:
        _last = _rgb['date'].max()
        _from = dt.date(_last.year - 1, _last.month, min(_last.day, 28))
        _r = _rgb.filter(pl.col('date') >= _from)
        _figr = go.Figure()
        _figr.add_scatter(x=_r['date'].to_list(), y=(_r['close'] / _r['close'][0] * 100).to_list(),
                          name='RGBITR (ОФЗ, полная дох.)',
                          line=dict(color='#102D69', width=1.8),
                          hovertemplate='%{y:.1f}<extra>RGBITR</extra>')
        try:
            _i = _index('IMOEX').filter(pl.col('date') >= _from)
            _figr.add_scatter(x=_i['date'].to_list(), y=(_i['close'] / _i['close'][0] * 100).to_list(),
                              name='IMOEX (акции)',
                              line=dict(color='#7f7f7f', width=1.4, dash='dot'),
                              hovertemplate='%{y:.1f}<extra>IMOEX</extra>')
        except Exception:
            pass
        _figr.update_layout(
            height=320, hovermode='x unified',
            title=dict(text='Облигации против акций, год (старт = 100)', font_size=14),
            legend=dict(orientation='h', y=1.12, x=1, xanchor='right'),
            margin=dict(t=44, l=10, r=10, b=10),
        )
        rgbitr_block = _figr
    rgbitr_block
    return


@app.cell(hide_code=True)
def _(mo, pl, snap_now):
    # Таблица всех выпусков
    if snap_now.height == 0:
        bonds_table = mo.md("")
    else:
        _t = snap_now.sort('years').select(
            'SECID',
            pl.col('name').alias('Выпуск'),
            pl.col('type').alias('Тип'),
            pl.col('segment').alias('Доска'),
            pl.col('maturity').dt.strftime('%d.%m.%Y').alias('Погашение'),
            pl.col('years').round(1).alias('Лет'),
            pl.col('coupon').round(2).alias('Купон %'),
            pl.col('price').round(2).alias('Цена %'),
            pl.col('ytm').round(2).alias('YTM %'),
            pl.col('duration').round(1).alias('Дюрация'),
            pl.col('convexity').round(1).alias('Выпуклость'),
            (pl.col('value') / 1e6).round(1).alias('Оборот, млн ₽'),
            pl.col('src').alias('Источник'))
        bonds_table = mo.ui.table(_t, pagination=True, page_size=20,
                                  label='Все выпуски (YTM у ПК/ИН — некорректен, это флоатеры/линкеры)')
    bonds_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    **Методика**: YTM и дюрация берутся **биржевые** (колонки `YIELDCLOSE` /
    `DURATION` из истории ISS — рассчитаны с НКД и фактическим графиком
    купонов); упрощенная модель (полугодовой купон, без НКД) — только фоллбэк
    для строк без биржевых значений, источник указан в таблице. Выпуклость
    всегда модельная — биржа ее не публикует. Для флоатеров (ПК) и линкеров
    (ИН) YTM из цены не имеет смысла — они показаны только в таблице.

    Точки «2 года» и «10 лет» в истории кривой — линейная интерполяция между
    соседними выпусками ОФЗ-ПД на каждую дату. В G-спредах участвуют только
    рублевые корпоративные выпуски с дневным оборотом от 1 млн руб: у неликвида
    цена последней сделки может быть недельной давности, и спред к сегодняшней
    кривой получается фиктивным. История медианного G-спреда считается по датам,
    где ликвидных выпусков не меньше 20.
    """)
    return


if __name__ == "__main__":
    app.run()
