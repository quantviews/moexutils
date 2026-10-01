"""
Генератор справочника колонок данных MOEX ISS: docs/iss-columns.md.

Берет из ISS метаданные полей (эндпоинты .../columns.json): код поля, короткое
и полное название, тип, признак скрытого поля — для истории торгов по рынкам
(акции, облигации, индексы, фьючерсы, опционы, валюта, валютные фиксинги), текущих данных торгов и
карточек бумаг (на примерах инструментов). Для каждого поля отмечает, хранится
ли оно в нашем хранилище и в какой таблице (по схеме lake).

Запуск: MOEX_DATA_ROOT=F:\\moex-data python scripts/gen_iss_columns.py
"""
import datetime as dt
import os

from moexutils import iss, lake

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs', 'iss-columns.md')

# Раздел -> (путь рынка, таблицы хранилища: первая хранит все поля ISS под теми же
# именами, следующие — рабочий набор полей под своими именами (RENAMED))
MARKETS = [
    ('Акции (рынок shares: акции, депозитарные расписки, паи)', 'stock/markets/shares', ('shares', 'stocks')),
    ('Облигации (рынок bonds, все доски)', 'stock/markets/bonds', ('bonds',)),
    ('Индексы (рынок index)', 'stock/markets/index', ('indexes_all', 'indexes')),
    ('Фьючерсы (FORTS)', 'futures/markets/forts', ('futures',)),
    ('Опционы (FORTS)', 'futures/markets/options', ('options',)),
    ('Валютный рынок (selt)', 'currency/markets/selt', ('currency',)),
    ('Валютные фиксинги (рынок index валютного рынка)', 'currency/markets/index', ('currency_fixings',)),
]
# Карточки бумаг: пример инструмента каждого вида
CARDS = [
    ('Акция', 'SBER'), ('ОФЗ', 'SU26238RMFS4'), ('Корпоративная облигация', 'RU000A10AT01'),
    ('Фьючерс', 'SiZ6'), ('Индекс', 'IMOEX'), ('Валютная пара', 'CNYRUB_TOM'),
]
# Карточки -> реестр хранилища
CARD_TABLES = {'ОФЗ': 'bonds_securities', 'Корпоративная облигация': 'bonds_securities',
               'Акция': 'shares_securities'}
# Поля, которые в рабочих таблицах акций и индексов переименованы из ISS
RENAMED = {
    'stocks': {'TRADEDATE': 'date', 'SECID': 'ticker', 'OPEN': 'open', 'LOW': 'low', 'HIGH': 'high',
               'CLOSE': 'close', 'WAPRICE': 'waprice', 'VOLUME': 'volume', 'VALUE': 'value_rub'},
    'indexes': {'TRADEDATE': 'date', 'SECID': 'ticker', 'BOARDID': 'BOARDID', 'CLOSE': 'close',
                'VALUE': 'value_rub', 'VOLUME': 'volume'},
}


def lake_columns() -> dict[str, set]:
    try:
        df = lake.query("SELECT table_name, column_name FROM information_schema.columns "
                        "WHERE table_catalog = 'lake'")
    except Exception as e:
        print(f"[WARN] хранилище недоступно ({e}) — колонка «Храним» будет пустой")
        return {}
    out = {}
    for t, c in df.iter_rows():
        out.setdefault(t, set()).add(c)
    return out


def md_escape(text) -> str:
    return '' if text is None else str(text).replace('|', '\\|').replace('\n', ' ').strip()


def stored(tables, field, cols):
    """Где поле хранится: `ПОЛЕ` в таблице всех полей, `таблица.поле` — в рабочих таблицах."""
    out = []
    for i, table in enumerate(tables):
        if table not in cols:
            continue
        ours = ('date' if field == 'TRADEDATE' else field) if i == 0 else RENAMED.get(table, {}).get(field)
        if ours and ours in cols[table]:
            out.append(f"`{ours}`" if i == 0 else f"`{table}.{ours}`")
    return ", ".join(out)


def columns_table(block, tables, cols, show_hidden=True):
    rows = ["| Поле | Название | Описание | Тип | Храним |", "|---|---|---|---|---|"]
    for r in block.iter_rows(named=True):
        if not show_hidden and r.get('is_hidden'):
            continue
        hidden = ' *(скрытое)*' if r.get('is_hidden') else ''
        rows.append(f"| `{r['name']}`{hidden} | {md_escape(r['short_title'])} | {md_escape(r['title'])} "
                    f"| {md_escape(r['type'])} | {stored(tables, r['name'], cols)} |")
    return "\n".join(rows)


def main():
    session = iss.make_session()
    cols = lake_columns()
    parts = [f"""# Колонки данных MOEX ISS

Справочник всех полей, которые отдает ISS Московской биржи, — по рынкам и видам
данных. Сгенерирован скриптом `scripts/gen_iss_columns.py` по метаданным ISS
(`.../columns.json`) {dt.date.today():%d.%m.%Y}; перегенерируйте после изменений у биржи
или в хранилище.

- **Поле** — код ISS; *(скрытое)* — поле есть в метаданных, но по умолчанию не отдается.
- **Храним** — имя колонки в нашем хранилище, если поле сохраняется. Наборы «все
  инструменты за дату» (`shares`, `bonds`, `indexes_all`, `futures`, `currency`,
  `currency_fixings`) хранят все отдаваемые поля под теми же именами (дата торгов —
  `date`); рабочие таблицы `stocks` и `indexes` — свой набор полей в нижнем регистре
  (в колонке — `таблица.поле`).

Модель наших таблиц — в [data-model.md](data-model.md).

## Содержание

""" + "\n".join(f"- [{title}](#{i})" for i, (title, _, _) in enumerate(MARKETS, 1))
        + "\n- [Карточки бумаг](#cards)\n"]

    for i, (title, path, tables) in enumerate(MARKETS, 1):
        if tables:
            where = "хранится в " + ", ".join(f"`lake.{t}`" for t in tables)
            missing = [t for t in tables if t not in cols]
            if missing:
                where += (" (в хранилище еще нет " + ", ".join(f"`{t}`" for t in missing)
                          + " — колонка «Храним» для них пуста)")
        else:
            where = "не хранится"
        parts.append(f'\n<a id="{i}"></a>\n\n## {title}\n\nПуть ISS: `{path}`; {where}.\n')
        hist = session.get(f"{iss.ISS_URL}/history/engines/{path}/securities/columns.json").json()
        parts.append("\n### История торгов (`/history/engines/" + path + "/securities`)\n\n"
                     + columns_table(iss.to_frame(hist.get('history')), tables, cols) + "\n")
        cur = session.get(f"{iss.ISS_URL}/engines/{path}/securities/columns.json").json()
        for block, label in (('securities', 'Справочник инструментов торгов'),
                             ('marketdata', 'Текущие данные торгов')):
            if cur.get(block):
                parts.append(f"\n### {label} (`/engines/{path}/securities`, блок `{block}`) — не храним\n\n"
                             + columns_table(iss.to_frame(cur[block]), (), cols, show_hidden=False) + "\n")
        print(f"[OK] {title}")

    parts.append('\n<a id="cards"></a>\n\n## Карточки бумаг (`/iss/securities/<SECID>`, блок `description`)\n\n'
                 'Набор полей зависит от вида бумаги. Карточки облигаций хранятся в '
                 '`lake.bonds_securities`, акций и фондов — в `lake.shares_securities` '
                 '(все поля, имена как в ISS).\n')
    for label, secid in CARDS:
        resp = session.get(f"{iss.ISS_URL}/securities/{secid}.json", params={'iss.only': 'description'}).json()
        desc = iss.to_frame(resp.get('description'))
        if desc.is_empty():
            continue
        table = CARD_TABLES.get(label)
        rows = [f"\n### {label} (пример: `{secid}`)\n", "| Поле | Описание | Тип | Храним |", "|---|---|---|---|"]
        for r in desc.iter_rows(named=True):
            keep = f"`{r['name']}`" if table and r['name'] in cols.get(table, set()) else ''
            rows.append(f"| `{r['name']}` | {md_escape(r['title'])} | {md_escape(r.get('type'))} | {keep} |")
        parts.append("\n".join(rows) + "\n")
        print(f"[OK] карточка {secid}")

    with open(OUT, 'w', encoding='utf-8') as f:
        f.write("\n".join(parts))
    print(f"→ {OUT}")


if __name__ == '__main__':
    main()
