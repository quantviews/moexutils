"""
marimo notebook: Долговой рынок — кривая ОФЗ, доходности выпусков, RGBITR

Использование:
1. Первичная выгрузка данных (мониторинг всех выпусков досок):
   python update_data.py --no-update --no-adj --no-cap --bonds-market-init TQOB,TQCB
   (дальше доски обновляются штатным шагом 1c update_data.py)
2. marimo edit bond-market.py

Функционал:
- Кривая доходности ОФЗ (YTM × срок до погашения): сегодня против выбранной
  прошлой даты; только выпуски с фиксированным купоном (ОФЗ-ПД)
- Сводка: короткий/длинный конец, наклон кривой, сдвиг за период
- RGBITR (гособлигации, полная доходность) против IMOEX
- Таблица всех выпусков: цена, YTM, дюрация, выпуклость

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
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    import moex_utils as moex
    import pandas as pd
    import numpy as np
    import marimo as mo
    try:
        import plotly.graph_objects as go
        plotly_available = True
    except ImportError:
        go = None
        plotly_available = False
    return go, mo, moex, np, pd, plotly_available


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
def _(mo, moex, pd):
    # Загрузка вселенной единым длинным DataFrame.
    # Приоритет — консолидированный мониторинг досок (bonds/market_*.parquet,
    # все выпуски, обновляется по датам); фоллбэк — пофайловые истории + params.
    import os as _osb

    def _bond_type(_secid):
        """Тип ОФЗ по серии в SECID: SU26238RMFS4 → 26 → ПД"""
        if not str(_secid).startswith('SU') or len(str(_secid)) < 4:
            return 'прочее'
        _series = str(_secid)[2:4]
        return {'25': 'ПД', '26': 'ПД', '24': 'ПК', '29': 'ПК',
                '52': 'ИН', '46': 'АД', '48': 'АД'}.get(_series, 'прочее')

    _LONG_COLS = ['date', 'SECID', 'SHORTNAME', 'CLOSE', 'PRICE_ALT', 'YIELDCLOSE',
                  'DURATION', 'MATDATE', 'FACEVALUE', 'FACEUNIT', 'COUPONPERCENT',
                  'segment']

    bonds_long = pd.DataFrame(columns=_LONG_COLS)
    _src_desc = ''
    try:
        _mkt = moex.read_bonds_market()
        _mkt = _mkt.rename(columns={'LEGALCLOSEPRICE': 'PRICE_ALT'})
        for _c in _LONG_COLS:
            if _c not in _mkt.columns:
                _mkt[_c] = None
        bonds_long = _mkt[_LONG_COLS]
        _segs = sorted(bonds_long['segment'].dropna().unique())
        _src_desc = 'мониторинг досок ' + ', '.join(_segs)
    except FileNotFoundError:
        try:
            _params = moex.read_bonds_params().dropna(subset=['SECID'])
            _params = _params.drop_duplicates(subset=['SECID'], keep='last')
            _frames = []
            for _r in _params.itertuples():
                _fp = _osb.path.join(moex.BONDS_FOLDER, f"{_r.SECID}.parquet")
                if not _osb.path.exists(_fp):
                    continue
                _px = pd.read_parquet(_fp).sort_index()
                _f = pd.DataFrame({
                    'date': _px.index,
                    'SECID': _r.SECID,
                    'CLOSE': _px.get('CLOSE'),
                    'PRICE_ALT': _px.get('WAPRICE'),
                    'YIELDCLOSE': _px.get('YIELDCLOSE'),
                    'DURATION': _px.get('DURATION'),
                })
                _f['SHORTNAME'] = getattr(_r, 'SHORTNAME', _r.SECID)
                _f['MATDATE'] = getattr(_r, 'MATDATE', None)
                _f['FACEVALUE'] = getattr(_r, 'FACEVALUE', 1000)
                _f['FACEUNIT'] = getattr(_r, 'FACEUNIT', 'SUR')
                _f['COUPONPERCENT'] = getattr(_r, 'COUPONPERCENT', 0)
                _f['segment'] = getattr(_r, 'segment', '')
                _frames.append(_f)
            if _frames:
                bonds_long = pd.concat(_frames, ignore_index=True)[_LONG_COLS]
                _src_desc = 'пофайловые истории выпусков'
        except FileNotFoundError:
            pass

    bonds_ready = len(bonds_long) > 0
    if bonds_ready:
        bonds_long = bonds_long.copy()
        bonds_long['date'] = pd.to_datetime(bonds_long['date'])
        bonds_long['bond_type'] = bonds_long['SECID'].map(_bond_type)
        _status = mo.md(
            f"**Данные:** {bonds_long['SECID'].nunique()} выпусков "
            f"({_src_desc}) · последняя дата: {bonds_long['date'].max():%d.%m.%Y}")
    else:
        _status = mo.md(
            "⚠️ **Данные облигаций не выгружены.** Для мониторинга всех выпусков "
            "выполните разово:\n\n"
            "```\npython update_data.py --no-update --no-adj --no-cap "
            "--bonds-market-init TQOB,TQCB\n```\n\n"
            "Дальше доски будут обновляться обычным `update_data.py` (шаг 1c)."
        )
    _status
    return bonds_long, bonds_ready


@app.cell(hide_code=True)
def _(bonds_long, bonds_ready, moex, pd):
    # Метрики всех выпусков на дату (последняя котировка не старше 14 дней до нее)
    def bonds_snapshot(asof=None):
        _w = bonds_long if asof is None else bonds_long[bonds_long['date'] <= asof]
        if len(_w) == 0:
            return pd.DataFrame()
        _last_rows = _w.sort_values('date').groupby('SECID', as_index=False).tail(1)
        _ref = _last_rows['date'].max() if asof is None else asof
        _last_rows = _last_rows[_last_rows['date'] >= _ref - pd.Timedelta(days=14)]

        rows = []
        for _r in _last_rows.itertuples():
            _date = _r.date
            _price = _r.CLOSE
            if pd.isna(_price):
                _price = _r.PRICE_ALT
            _mat = pd.to_datetime(_r.MATDATE, errors='coerce')
            _face = float(_r.FACEVALUE) if pd.notna(_r.FACEVALUE) else 1000.0
            _coupon = float(_r.COUPONPERCENT) if pd.notna(_r.COUPONPERCENT) else 0.0
            if pd.isna(_price) or pd.isna(_mat):
                continue
            _years = (_mat - _date).days / 365.25
            if _years <= 0.05:
                continue
            _price = float(_price)

            # Биржевые YTM/дюрация из истории ISS (точные: с НКД и фактическим
            # графиком купонов); упрощенная модель — фоллбэк для строк без них
            _y_exch = _r.YIELDCLOSE
            if _y_exch is not None and pd.notna(_y_exch) and 0 < float(_y_exch) < 100:
                _ytm = float(_y_exch)
                _src = 'ISS'
            else:
                _ytm = moex.calculate_ytm(_price, _face, _coupon, _years)
                _src = 'модель'

            _d_exch = _r.DURATION  # ISS отдает дюрацию в днях
            if _d_exch is not None and pd.notna(_d_exch) and float(_d_exch) > 0:
                _dur = float(_d_exch) / 365.25
            else:
                _dur = moex.calculate_duration(_price, _face, _coupon, _years, _ytm)

            rows.append({
                'SECID': _r.SECID,
                'name': _r.SHORTNAME if pd.notna(_r.SHORTNAME) else _r.SECID,
                'type': _r.bond_type,
                'segment': _r.segment if pd.notna(_r.segment) else '',
                'faceunit': _r.FACEUNIT if pd.notna(_r.FACEUNIT) else 'SUR',
                'maturity': _mat,
                'years': _years,
                'coupon': _coupon,
                'price': _price,
                'ytm': _ytm,
                'duration': _dur,
                'convexity': moex.calculate_convexity(_price, _face, _coupon, _years, _ytm),
                'src': _src,
                'date': _date,
            })
        return pd.DataFrame(rows)

    snap_now = bonds_snapshot() if bonds_ready else pd.DataFrame()
    return bonds_snapshot, snap_now


@app.cell(hide_code=True)
def _(bonds_ready, bonds_snapshot, compare_dropdown, pd, snap_now):
    # Срез кривой на прошлую дату
    if bonds_ready and len(snap_now):
        _last_date = snap_now['date'].max()
        compare_date = _last_date - pd.DateOffset(months=compare_dropdown.value)
        snap_past = bonds_snapshot(asof=compare_date)
    else:
        compare_date = None
        snap_past = pd.DataFrame()
    return compare_date, snap_past


@app.cell(hide_code=True)
def _(compare_dropdown, go, mo, plotly_available, snap_now, snap_past):
    # Кривая ОФЗ-ПД: сейчас против прошлой даты
    if not plotly_available or len(snap_now) == 0:
        curve_block = mo.md("")
    else:
        _now = snap_now[snap_now['type'] == 'ПД'].sort_values('years')
        _past = (snap_past[snap_past['type'] == 'ПД'].sort_values('years')
                 if len(snap_past) else snap_past)

        _figc = go.Figure()
        _figc.add_scatter(
            x=_now['years'], y=_now['ytm'], mode='lines+markers',
            name=f"сейчас ({_now['date'].max():%d.%m.%Y})",
            line=dict(color='#102D69', width=2.2), marker=dict(size=7),
            customdata=_now[['name', 'SECID', 'duration']].values,
            hovertemplate='<b>%{customdata[0]}</b> (%{customdata[1]})'
                          '<br>срок: %{x:.1f} лет · YTM: %{y:.2f}%'
                          '<br>дюрация: %{customdata[2]:.1f}<extra></extra>')
        if len(_past):
            _figc.add_scatter(
                x=_past['years'], y=_past['ytm'], mode='lines+markers',
                name=f"{compare_dropdown.selected_key or 'ранее'} "
                     f"({_past['date'].max():%d.%m.%Y})",
                line=dict(color='#9AAFD4', width=1.6, dash='dash'),
                marker=dict(size=5),
                customdata=_past[['name']].values,
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
def _(mo, snap_now, snap_past):
    # Сводка по кривой
    def _sgn(_v, _suffix=' п.п.', _nd=2):
        if _v != _v:
            return 'н/д'
        _cls = 'pos' if _v >= 0 else 'neg'
        return f'<span class="{_cls}">{format(_v, f"+.{_nd}f")}{_suffix}</span>'

    def _bucket(_snap, _lo, _hi):
        _pd_only = _snap[_snap['type'] == 'ПД']
        _b = _pd_only[(_pd_only['years'] >= _lo) & (_pd_only['years'] <= _hi)]['ytm']
        return float(_b.mean()) if len(_b) else float('nan')

    if len(snap_now) == 0:
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
        if len(snap_past):
            _p2 = _bucket(snap_past, 1, 3)
            _p10 = _bucket(snap_past, 8, 12)
            _lines.append(
                f"- Сдвиг с прошлой даты: короткий {_sgn(_s2 - _p2)}, "
                f"длинный {_sgn(_s10 - _p10)}")
        curve_summary = mo.md("### Итоги\n\n" + "\n".join(_lines))
    curve_summary
    return


@app.cell(hide_code=True)
def _(go, mo, np, pd, plotly_available, snap_now):
    # Корпоративные облигации: G-спреды к интерполированной кривой ОФЗ
    _ofz = (snap_now[snap_now['type'] == 'ПД'].sort_values('years')
            if len(snap_now) else pd.DataFrame())
    _corp = (snap_now[(snap_now['segment'] == 'TQCB')
                      & (snap_now['faceunit'].isin(['SUR', 'RUB']))
                      & snap_now['ytm'].between(0.1, 60)].copy()
             if len(snap_now) else pd.DataFrame())

    if not plotly_available or len(snap_now) == 0:
        corp_block = mo.md("")
    elif len(_corp) == 0:
        corp_block = mo.md(
            "*Корпоративные облигации не выгружены. Для G-спредов выполните разово:*\n\n"
            "```\npython update_data.py --no-update --no-adj --no-cap "
            "--bonds-market-init TQCB\n```"
        )
    elif len(_ofz) < 3:
        corp_block = mo.md("*Недостаточно точек кривой ОФЗ для расчета спредов*")
    else:
        # G-спред = YTM корпората − YTM ОФЗ, интерполированная на его срок
        _corp['gspread_bp'] = (
            _corp['ytm'] - np.interp(_corp['years'], _ofz['years'], _ofz['ytm'])
        ) * 100

        _figg = go.Figure()
        _figg.add_scatter(
            x=_ofz['years'], y=_ofz['ytm'], mode='lines',
            name='кривая ОФЗ', line=dict(color='#102D69', width=2),
            hovertemplate='ОФЗ %{x:.1f} лет: %{y:.2f}%<extra></extra>')
        _figg.add_scatter(
            x=_corp['years'], y=_corp['ytm'], mode='markers',
            name='корпораты (TQCB)',
            marker=dict(size=8, color=_corp['gspread_bp'],
                        colorscale='RdYlGn', reversescale=True, cmin=0,
                        cmax=float(_corp['gspread_bp'].quantile(0.95)),
                        colorbar=dict(title='спред,<br>б.п.')),
            customdata=_corp[['name', 'gspread_bp']].values,
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
        _wide = _corp.nlargest(5, 'gspread_bp')
        _tight = _corp.nsmallest(5, 'gspread_bp')
        _md_corp = mo.md(
            f"**Корпоративный сегмент:** {len(_corp)} выпусков · "
            f"медианный G-спред **{_med:.0f} б.п.**\n\n"
            f"- Самые широкие: " + ", ".join(
                f"**{_r.name}** {_r.gspread_bp:+.0f}" for _r in _wide.itertuples()) + "\n"
            f"- Самые узкие: " + ", ".join(
                f"**{_r.name}** {_r.gspread_bp:+.0f}" for _r in _tight.itertuples())
        )

        _tbl = _corp.sort_values('gspread_bp', ascending=False)[
            ['SECID', 'name', 'maturity', 'years', 'price', 'ytm', 'duration', 'gspread_bp']
        ].copy()
        _tbl['maturity'] = _tbl['maturity'].dt.strftime('%d.%m.%Y')
        _tbl.columns = ['SECID', 'Выпуск', 'Погашение', 'Лет', 'Цена %',
                        'YTM %', 'Дюрация', 'G-спред, б.п.']
        for _cc, _nd2 in (('Лет', 1), ('Цена %', 2), ('YTM %', 2),
                          ('Дюрация', 1), ('G-спред, б.п.', 0)):
            _tbl[_cc] = _tbl[_cc].round(_nd2)

        corp_block = mo.vstack([
            _md_corp, _figg,
            mo.ui.table(_tbl, pagination=True, page_size=15,
                        label='Корпоративные выпуски по G-спреду'),
        ])
    corp_block
    return


@app.cell(hide_code=True)
def _(go, mo, moex, pd, plotly_available):
    # RGBITR (гособлигации, полная доходность) против IMOEX за год
    try:
        _rgb = moex.read_moex_index('RGBITR')
        _rgb.index = pd.to_datetime(_rgb.index)
        _rgb_close = _rgb['close'].astype(float).sort_index()
        _rgb_ok = True
    except Exception:
        _rgb_ok = False

    if not plotly_available or not _rgb_ok:
        rgbitr_block = mo.md(
            "*Кэш RGBITR не найден — обновите индексы: `python update_data.py`*"
        ) if plotly_available else mo.md("")
    else:
        _from = _rgb_close.index.max() - pd.DateOffset(years=1)
        _r = _rgb_close[_rgb_close.index >= _from]
        _figr = go.Figure()
        _figr.add_scatter(x=_r.index, y=_r / _r.iloc[0] * 100, name='RGBITR (ОФЗ, полная дох.)',
                          line=dict(color='#102D69', width=1.8),
                          hovertemplate='%{y:.1f}<extra>RGBITR</extra>')
        try:
            _imx = moex.read_moex_index('IMOEX')
            _imx.index = pd.to_datetime(_imx.index)
            _i = _imx['close'].astype(float).sort_index()
            _i = _i[_i.index >= _from]
            _figr.add_scatter(x=_i.index, y=_i / _i.iloc[0] * 100, name='IMOEX (акции)',
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
def _(mo, snap_now):
    # Таблица всех выпусков
    if len(snap_now) == 0:
        bonds_table = mo.md("")
    else:
        _t = snap_now.sort_values('years').copy()
        _t['maturity'] = _t['maturity'].dt.strftime('%d.%m.%Y')
        _t = _t[['SECID', 'name', 'type', 'segment', 'maturity', 'years',
                 'coupon', 'price', 'ytm', 'duration', 'convexity', 'src']]
        _t.columns = ['SECID', 'Выпуск', 'Тип', 'Доска', 'Погашение', 'Лет',
                      'Купон %', 'Цена %', 'YTM %', 'Дюрация', 'Выпуклость', 'Источник']
        for _c, _nd in (('Лет', 1), ('Купон %', 2), ('Цена %', 2),
                        ('YTM %', 2), ('Дюрация', 1), ('Выпуклость', 1)):
            _t[_c] = _t[_c].round(_nd)
        bonds_table = mo.ui.table(_t, pagination=True, page_size=20,
                                  label='Все сохраненные выпуски (YTM у ПК/ИН — некорректен, это флоатеры/линкеры)')
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
    """)
    return


if __name__ == "__main__":
    app.run()
