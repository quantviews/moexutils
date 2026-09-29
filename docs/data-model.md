# Модель данных

Как устроены данные moexutils: где хранятся, какие таблицы и ключи, что означают колонки, как таблицы связаны между собой и с реестрами, какие шаги обновления их пишут. Полный список полей, которые отдает биржа по каждому рынку, — в [iss-columns.md](iss-columns.md).

## Слои хранения

| Слой | Где | Что | В git |
|------|-----|-----|-------|
| **Хранилище DuckLake** | каталог — PostgreSQL 17, база `moex_lake` (localhost); файлы — Parquet (zstd) в `<MOEX_DATA_ROOT>/lake` (`F:\moex-data\lake`) | рыночные данные: акции, индексы, облигации, фьючерсы, реестр карточек облигаций | нет |
| **Реестры** | `metadata/` в проекте | корпоративные события и справочники: сплиты, переименования, снятые с торгов, ключевая ставка, сектора, число акций | да |
| **Дивиденды** | соседний проект `../dividends/data/<TICKER>.csv` | история выплат с закрытияреестров.рф | в своем репозитории |
| **Прежние файлы** | `F:\moex-data\data`, `indexes` (Parquet по тикерам) | заморожены на момент перехода на хранилище; ноутбуки пока перебирают по ним тикеры | нет |

Источник рыночных данных — MOEX ISS (история торгов `/history`, карточки `/securities`), ключевой ставки — cbr.ru. Хранилище читается функциями модулей (`stocks`, `history`, `moex_utils` — результат polars) или SQL через `lake.query(...)`.

## Схема связей

```mermaid
erDiagram
    stocks }o--o| renames : "ticker = old / new"
    stocks }o--o{ splits : "ticker"
    stocks }o--o| delisted : "ticker"
    stocks }o--o{ dividends : "ticker"
    stocks }o--o{ shares_xlsx : "ticker = Code"
    stocks }o--|| indexes : "date (календарь IMOEX)"
    bonds }o--|| bonds_securities : "SECID"
    bonds }o--|| indexes : "date (календарь IMOEX)"
    futures }o--|| indexes : "date (календарь IMOEX)"
    empty_dates }o--|| bonds : "dataset = 'bonds'"
    empty_dates }o--|| futures : "dataset = 'futures'"

    stocks { date date PK; string ticker PK; double close; double adj_close; double market_cap }
    indexes { date date PK; string ticker PK; double close; double value_rub }
    bonds { date date PK; string SECID PK; string BOARDID PK; double CLOSE; double YIELDCLOSE; double DURATION }
    bonds_securities { string SECID PK; string ISIN; string MATDATE; double ISSUESIZE }
    futures { date date PK; string SECID PK; string BOARDID PK; string ASSETCODE; double SETTLEPRICE; double OPENPOSITION }
    empty_dates { string dataset PK; date date PK }
```

- **Календарь торгов** — будни, по которым есть `IMOEX` в `lake.indexes`. По нему проверяются пропуски и отставание всех наборов.
- **Акции ↔ реестры** — по тикеру: переименования склеивают истории, сплиты приводят цены к одной базе, снятые с торгов не обновляются, дивиденды и число акций дают `adj_close` и капитализацию.
- **Облигации ↔ карточки** — по `SECID`: история торгов (`bonds`) и параметры выпуска (`bonds_securities`), включая погашенные.

## Соглашения

- Даты торгов — колонка `date` типа DATE. Прочие даты ISS (`MATDATE`, `OFFERDATE`, ...) — строки `YYYY-MM-DD` (как их отдает биржа; перевод — `strptime`/`str.to_date`).
- Числа — DOUBLE, текст — VARCHAR. Типы колонок ISS берутся из метаданных биржи, поэтому колонка имеет один тип во все годы истории.
- Ключ каждой таблицы уникален; запись идет транзакциями по ключу (новые строки добавляются, существующие обновляются).
- Новые поля биржи добавляются в таблицы автоматически (эволюция схемы); в ранних годах часть колонок пустая.
- `bonds` и `futures` разбиты по годам даты торгов.
- Снимки хранилища хранятся 30 дней: `SELECT ... FROM lake.bonds AT (VERSION => n)` — таблица на момент снимка `n` (список — `lake.snapshots()`).

---

## Таблицы хранилища

### `lake.stocks` — акции

Дневные данные акций, включая снятые с торгов; ключ `date + ticker`. Источник — история торгов ISS по бумаге (рынок `shares`): на дату берется строка режима с максимальным оборотом (главная доска), `close` — закрытие основной сессии (та же методика, что у индексов). Пишет шаг 1 (`stocks.update_stocks`) и шаг 2 (`stocks.recompute_stocks`).

| Колонка | Тип | Смысл | Поле ISS / расчет |
|---------|-----|-------|-------------------|
| `date` | DATE | Дата торгов | `TRADEDATE` |
| `ticker` | VARCHAR | Тикер (код бумаги на бирже) | `SECID` |
| `open`, `low`, `high` | DOUBLE | Цены открытия, минимум, максимум, руб. | `OPEN`, `LOW`, `HIGH` |
| `close` | DOUBLE | Закрытие основной сессии, руб. — **как отдает ISS**, без поправки на сплиты | `CLOSE` |
| `waprice` | DOUBLE | Средневзвешенная цена дня | `WAPRICE` |
| `volume` | DOUBLE | Объем, шт. | `VOLUME` |
| `value_rub` | DOUBLE | Оборот, руб. (не цена) | `VALUE` |
| `adj_close` | DOUBLE | Цена с поправкой на дивиденды и сплиты, в текущей базе акции | расчет, см. ниже |
| `shares` | DOUBLE | Число акций на дату (в текущей базе) | срезы `metadata/stock-index-base.xlsx` |
| `market_cap` | DOUBLE | Капитализация, руб. = `close × shares` | расчет |

**`adj_close`.** База — `close`, приведенный к пост-сплитовой базе по реестру сплитов. Каждый дивиденд из CSV проекта dividends умножает все цены **строго до экс-даты** на `1 − D / P`, где `P` — последнее закрытие с дивидендом. Экс-дата выводится из даты закрытия реестра: с 31.07.2023 (расчеты T+1) — сама дата отсечки (или последний торговый день перед ней, если она выходная), раньше (T+2) — торговый день перед ней. Дивиденд, неправдоподобный ни к сырой, ни к сплит-скорректированной цене (доходность вне 0–50%), пропускается и попадает в отчет о качестве; отсечки после последней даты данных историю не меняют.

**`shares` и `market_cap`.** Число акций — срезы Excel по датам: между срезами действует последний, до первого — первый. Срезы до сплита приводятся к базе после события (реестр сплитов, виды `shares` и `auto`). Нет данных о числе акций — обе колонки пустые.

**Для доходностей** берите `adj_close` или цены через `stocks.read_stocks(..., split_adjusted=True)`: сырой `close` содержит разрывы на сплитах, если биржа не пересчитала историю.

### `lake.indexes` — индексы

Индексы MOEX (сейчас `IMOEX`, `MCFTR`, `RGBITR`); ключ `date + ticker`. Источник — история торгов ISS по индексу (рынок `index`). Пишет шаг 1b (`stocks.update_indexes`).

| Колонка | Тип | Смысл | Поле ISS |
|---------|-----|-------|----------|
| `date` | DATE | Дата расчета | `TRADEDATE` |
| `ticker` | VARCHAR | Код индекса | `SECID` |
| `BOARDID` | VARCHAR | Режим (для индексов — `SNDX` и т.п.) | `BOARDID` |
| `close` | DOUBLE | Значение индекса на закрытие | `CLOSE` |
| `value_rub` | DOUBLE | Оборот по бумагам индекса, руб. | `VALUE` |
| `volume` | DOUBLE | Объем | `VOLUME` |

### `lake.bonds` — облигации (весь рынок)

История торгов **всех** облигаций MOEX с 30.06.1997 по всем доскам: гособлигации, корпоративные, региональные, валютные, погашенные выпуски. Ключ `date + SECID + BOARDID` (выпуск может торговаться на нескольких досках в один день). Разбита по годам. Источник — история торгов ISS «все бумаги рынка за дату» (рынок `bonds`). Пишет шаг 1c (`history.update/repair('bonds')`).

Хранятся **все поля**, которые отдает ISS, под теми же именами (дата — `date`); полный список с описаниями — [iss-columns.md → Облигации](iss-columns.md#2). Основные группы:

| Группа | Колонки |
|--------|---------|
| Идентификация | `date`, `BOARDID`, `SECID`, `SHORTNAME` |
| Цены (% от номинала) | `OPEN`, `LOW`, `HIGH`, `CLOSE`, `WAPRICE`, `LEGALCLOSEPRICE`, `MARKETPRICE2`, `MARKETPRICE3`, `ADMITTEDQUOTE` |
| Доходность и риск | `YIELDCLOSE` (доходность к погашению, %), `YIELDATWAP`, `YIELDTOOFFER`, `YIELDLASTCOUPON`, `DURATION` (дни), `ZSPREAD`, `ZSPREADATWAPRICE`, `CALLOPTIONYIELD`, `CALLOPTIONDURATION` |
| ОФЗ-ИН | `BEICLOSE` (breakeven-инфляция), `IRICPICLOSE` (индекс потребительских цен) |
| Купон и номинал | `COUPONPERCENT`, `COUPONVALUE`, `ACCINT` (НКД), `FACEVALUE`, `FACEUNIT`, `FACEVALUE_TYPE`, `CURRENCYID`, `COUPON_DETAILS` |
| Даты (строки) | `MATDATE` (погашение), `OFFERDATE`, `BUYBACKDATE`, `CALLOPTIONDATE`, `PUTOPTIONDATE`, `LASTTRADEDATE` |
| Обороты | `VALUE`, `VOLUME`, `NUMTRADES`, `MP2VALTRD`, `MARKETPRICE3TRADESVALUE`, `ADMITTEDVALUE` |
| Тип | `BONDTYPE`, `BONDSUBTYPE` |

Доски: старые основные режимы (`EQOB`, `EQNB`, `EQOS`, `EQNO` и др.) до перехода на Т+, `TQOB` — гособлигации, `TQCB` — корпоративные, `TQOD`/`TQOE`/`TQOY`/`TQUD` — валютные, `TQRD`. Для анализа обычно фильтруют основные доски: `boards=['TQOB', 'TQCB']`.

### `lake.bonds_securities` — карточки облигаций

Строка на выпуск (`SECID`), включая погашенные: карточка ISS `/iss/securities/<SECID>` (блок `description`) — все поля под именами ISS: `ISIN`, `NAME`, `EMITTER_ID`, `REGNUMBER`, `ISSUEDATE`, `MATDATE`, `ISSUESIZE`, `FACEVALUE`, `INITIALFACEVALUE`, `FACEUNIT`, `COUPONFREQUENCY`, `TYPE`/`TYPENAME`, `BOND_TYPE`/`BOND_SUBTYPE`, `LISTLEVEL`, `HASDEFAULT`, `HASTECHNICALDEFAULT` и др. (набор зависит от выпуска), плюс `FETCHED` — дата запроса. Пишет шаг 1c (`history.update_securities('bonds')`, до 500 новых выпусков за ночь).

Флаги `HASDEFAULT`/`HASTECHNICALDEFAULT` — текущий статус, как правило на уровне эмитента (все его действующие выпуски), а не история: у погашенных выпусков они не выставлены. Для истории дефолтов не годятся.

### `lake.futures` — фьючерсы FORTS

История торгов **всех** фьючерсных контрактов с 03.01.2002, включая истекшие. Ключ `date + SECID + BOARDID`; разбита по годам. Источник — история торгов ISS «все контракты за дату» (`futures/markets/forts`). Пишет шаг 1e (`history.update/repair('futures')`). Все поля ISS под теми же именами — [iss-columns.md → Фьючерсы](iss-columns.md#4).

| Колонка | Смысл |
|---------|-------|
| `date`, `BOARDID`, `SECID`, `SHORTNAME` | Дата, режим, код контракта (`SiZ5`), имя с месяцем и годом экспирации (`Si-12.25`) |
| `ASSETCODE` | Базовый актив (`Si`, `RTS`, `BR`, `GD`, `MIX`, ...) |
| `OPEN`, `LOW`, `HIGH`, `CLOSE`, `WAPRICE` | Цены |
| `SETTLEPRICE` | Расчетная цена (для вариационной маржи) |
| `OPENPOSITION`, `OPENPOSITIONVALUE` | Открытый интерес: контракты и руб. |
| `VOLUME`, `VALUE`, `NUMTRADES`, `QTY` | Объемы и число сделок |
| `SWAPRATE`, `SWAPRATE_CURR`, `CHANGE` | Своп-ставка (для вечных фьючерсов), изменение |

Коды контрактов повторяются раз в 10 лет (`SiZ5` — декабрь 2015 и декабрь 2025): внутри даты код уникален, год контракта берите из `SHORTNAME`.

### `lake.empty_dates` — служебная

Торговые (по календарю IMOEX) даты, за которые ISS подтвержденно не вернул строк; ключ `dataset + date` (`dataset` — `bonds` или `futures`). При докачке пропусков такие даты больше не запрашиваются.

---

## Реестры `metadata/`

Корпоративные события и справочники ведутся только здесь — данные в хранилище руками не правятся.

| Файл | Колонки | Назначение | Кто пишет |
|------|---------|------------|-----------|
| `splits.csv` | `ticker, date, ratio, kind` | Сплиты и консолидации. `kind`: `price` — в истории цен разрыв, цены до даты делятся на `ratio`; `shares` — биржа пересчитала цены, но число акций в старых срезах в старой базе; `auto` — вид определяется по данным (есть ценовой разрыв — ценовая поправка, нет — поправка числа акций). `ratio` в ценовой семантике: дробление 1:10 → `10`, консолидация 100:1 → `0.01` | вручную; плюс внешний реестр `../dividends/metadata/splits.json` (записи получают `kind=auto`) |
| `renames.csv` | `old, new, date` | Переименования тикеров (`TCSG→T`, `YNDX→YDEX`, ...): `date` — первый день под новым тикером. При чтении истории склеиваются, исходный тикер строки — `source_ticker` | вручную |
| `delisted.csv` | `ticker, last_date, note` | Снятые с торгов: история хранится, но не обновляется. Кандидатов показывает проверка данных (`stock_stale`) | вручную |
| `key_rate.csv` | `date, rate` | Ключевая ставка ЦБ по датам изменения, % годовых (до 13.09.2013 — ставка рефинансирования) | шаг 1d, автоматически с cbr.ru |
| `sectors.csv` | `ticker, sector` | Отраслевой разрез в ноутбуках | вручную |
| `stock-index-base.xlsx` | листы-даты `DD.MM.YYYY`: `Code`, `Number of issued shares` (шапка на 4-й строке) | Срезы числа акций для капитализации | вручную |

## Дивиденды (`../dividends/data/<TICKER>.csv`)

Соседний проект собирает историю выплат с сайта закрытияреестров.рф и приводит суммы к текущей акции с учетом сплитов. Для `adj_close` используются `closing_date` (дата закрытия реестра) и `dividend_value` (руб. на акцию, > 0). Привилегированные акции — в файле `<TICKER>P.csv` по правилам того проекта. Если сайт еще не внес выплату, проверка данных покажет гэп открытия без дивиденда (`dividend_gap`).

---

## Обновление: какие шаги пишут какие таблицы

| Шаг `update_data.py` | Таблицы / файлы | Как |
|---|---|---|
| 1 — акции | `lake.stocks` | дозагрузка с последней даты тикера, сразу `adj_close` и капитализация по всей истории; пишутся только новые и изменившиеся строки |
| 1b — индексы | `lake.indexes` | дозагрузка с последней даты |
| 1c — облигации | `lake.bonds`, `lake.bonds_securities`, `lake.empty_dates` | хвост истории, докачка пропусков, до 500 новых карточек |
| 1d — ключевая ставка | `metadata/key_rate.csv` | новые решения ЦБ |
| 1e — фьючерсы | `lake.futures`, `lake.empty_dates` | хвост истории, докачка пропусков |
| 2 — пересчет | `lake.stocks` | `adj_close` и капитализация после изменений дивидендов, срезов или реестра сплитов — только изменившиеся строки |
| 3 — проверка | — | отчет о качестве в лог |
| 4 — обслуживание | файлы хранилища | слияние мелких файлов, удаление снимков старше 30 дней |

Первичные выгрузки полной истории облигаций и фьючерсов — `update_data.py --history-init bonds,futures` (многочасовые).

## Примеры доступа

```python
import polars as pl
import stocks, history, lake

sber = stocks.read_stocks('SBER', start='2024-01-01')                 # polars
all_adj = stocks.read_stocks(split_adjusted=True)                      # все тикеры, склейка + сплиты
imoex = stocks.read_index('IMOEX')
ofz = history.read('bonds', boards='TQOB', start='2026-01-01',
                   columns=['date', 'SECID', 'YIELDCLOSE', 'DURATION', 'MATDATE'])
si = history.read('futures', start='2025-01-01').filter(pl.col('ASSETCODE') == 'Si')
cards = history.read_securities('bonds')

# SQL: медианный Z-спред корпоративных облигаций по дням
lake.query("""
    SELECT date, median(ZSPREAD) AS zspread, count(*) AS n
    FROM lake.bonds WHERE BOARDID = 'TQCB' AND ZSPREAD IS NOT NULL
    GROUP BY date ORDER BY date""")

# SQL: история выпуска вместе с его карточкой
lake.query("""
    SELECT b.date, b.CLOSE, b.YIELDCLOSE, s.ISIN, s.ISSUESIZE
    FROM lake.bonds b JOIN lake.bonds_securities s USING (SECID)
    WHERE b.SECID = 'SU26238RMFS4' AND b.BOARDID = 'TQOB'""")
```
