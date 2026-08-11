"""
marimo notebook: Долговой рынок — кривая ОФЗ, доходности выпусков, RGBITR

Использование:
1. Первичная выгрузка данных: python update_data.py --bonds-init TQOB
   (дальше выпуски обновляются штатным шагом 1c update_data.py)
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
    # Загрузка вселенной: параметры + история цен каждого выпуска
    import os as _osb

    def _bond_type(_secid):
        """Тип ОФЗ по серии в SECID: SU26238RMFS4 → 26 → ПД"""
        if not str(_secid).startswith('SU') or len(str(_secid)) < 4:
            return 'прочее'
        _series = str(_secid)[2:4]
        return {'25': 'ПД', '26': 'ПД', '24': 'ПК', '29': 'ПК',
                '52': 'ИН', '46': 'АД', '48': 'АД'}.get(_series, 'прочее')

    try:
        bonds_params = moex.read_bonds_params()
        bonds_params = bonds_params.dropna(subset=['SECID']).copy()
        bonds_params['bond_type'] = bonds_params['SECID'].map(_bond_type)
        _files = {
            _f[:-len('.parquet')]
            for _f in _osb.listdir(moex.BONDS_FOLDER)
            if _f.endswith('.parquet') and _f != 'params.parquet'
        }
        bonds_params = bonds_params[bonds_params['SECID'].isin(_files)]
        bond_prices = {
            _s: pd.read_parquet(
                _osb.path.join(moex.BONDS_FOLDER, f"{_s}.parquet")).sort_index()
            for _s in bonds_params['SECID']
        }
        bonds_ready = len(bond_prices) > 0
        _status = mo.md(
            f"**Данные:** {len(bond_prices)} выпусков · последняя дата: "
            f"{max(_df.index.max() for _df in bond_prices.values()):%d.%m.%Y}"
        ) if bonds_ready else mo.md("")
    except FileNotFoundError:
        bonds_params = pd.DataFrame()
        bond_prices = {}
        bonds_ready = False
        _status = mo.md(
            "⚠️ **Данные облигаций не выгружены.** Выполните разово:\n\n"
            "```\npython update_data.py --bonds-init TQOB\n```\n\n"
            "Дальше выпуски будут обновляться обычным `update_data.py` (шаг 1c)."
        )
    _status
    return bond_prices, bonds_params, bonds_ready


@app.cell(hide_code=True)
def _(bond_prices, bonds_params, bonds_ready, moex, pd):
    # Метрики всех выпусков на дату (последняя котировка не старше 14 дней до нее)
    def bonds_snapshot(asof=None):
        rows = []
        for _r in bonds_params.itertuples():
            _px = bond_prices.get(_r.SECID)
            if _px is None or len(_px) == 0:
                continue
            _w = _px if asof is None else _px[_px.index <= asof]
            if len(_w) == 0:
                continue
            _last = _w.iloc[-1]
            _date = _w.index[-1]
            if asof is not None and _date < asof - pd.Timedelta(days=14):
                continue  # выпуск уже не торговался на эту дату

            _price = _last.get('CLOSE')
            if pd.isna(_price):
                _price = _last.get('WAPRICE')
            _mat = pd.to_datetime(getattr(_r, 'MATDATE', None), errors='coerce')
            _face = float(getattr(_r, 'FACEVALUE', 1000) or 1000)
            _coupon = float(getattr(_r, 'COUPONPERCENT', 0) or 0)
            if pd.isna(_price) or pd.isna(_mat):
                continue
            _years = (_mat - _date).days / 365.25
            if _years <= 0.05:
                continue
            _price = float(_price)

            # Биржевые YTM/дюрация из истории ISS (точные: с НКД и фактическим
            # графиком купонов); упрощенная модель — фоллбэк для строк без них
            _y_exch = _last.get('YIELDCLOSE')
            if _y_exch is not None and pd.notna(_y_exch) and 0 < float(_y_exch) < 100:
                _ytm = float(_y_exch)
                _src = 'ISS'
            else:
                _ytm = moex.calculate_ytm(_price, _face, _coupon, _years)
                _src = 'модель'

            _d_exch = _last.get('DURATION')  # ISS отдает дюрацию в днях
            if _d_exch is not None and pd.notna(_d_exch) and float(_d_exch) > 0:
                _dur = float(_d_exch) / 365.25
            else:
                _dur = moex.calculate_duration(_price, _face, _coupon, _years, _ytm)

            rows.append({
                'SECID': _r.SECID,
                'name': getattr(_r, 'SHORTNAME', _r.SECID),
                'type': _r.bond_type,
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
        _t = _t[['SECID', 'name', 'type', 'maturity', 'years',
                 'coupon', 'price', 'ytm', 'duration', 'convexity', 'src']]
        _t.columns = ['SECID', 'Выпуск', 'Тип', 'Погашение', 'Лет',
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
