# Модель данных

Как устроены данные moexutils: где хранятся, какие таблицы и ключи, что означают колонки, как таблицы связаны между собой и с реестрами, какие шаги обновления их пишут. Полный список полей, которые отдает биржа по каждому рынку, — в [iss-columns.md](iss-columns.md); что гарантируется потребителям — в [data-contract.md](data-contract.md).

## Слои хранения

| Слой | Где | Что | В git |
|------|-----|-----|-------|
| **Хранилище DuckLake** | каталог — PostgreSQL 17, база `moex_lake` (localhost); файлы — Parquet (zstd) в `<MOEX_DATA_ROOT>/lake` (`F:\moex-data\lake`) | рыночные данные MOEX, ставки, производные таблицы, копии реестров `metadata/`, история прогонов обновления | нет |
| **Копии каталога** | `<MOEX_DATA_ROOT>/backups/catalog` | `pg_dump` базы `moex_lake`, 14 последних (шаг 5 обновления) | нет |
| **Реестры** | `metadata/` в проекте | корпоративные события и справочники: сплиты, переименования, снятые с торгов, ключевая ставка, сектора, число акций | да |
| **Дивиденды** | соседний проект `../dividends/data/<TICKER>.csv` | история выплат с закрытияреестров.рф | в своем репозитории |
| **Прежние файлы** | `F:\moex-data\data`, `indexes`, `bonds`, `futures` | не используются кодом пакета; проект vectorbt еще читает их — удалить после его перевода на пакет (см. [план](../development-plan.md)) | нет |

Источники: MOEX ISS (история торгов `/history`, карточки `/securities`, КБД `/engines/stock/zcyc`, денежные потоки облигаций `bondization`, реестр фьючерсов `series`), cbr.ru (ключевая ставка, RUONIA). Хранилище читается функциями пакета (`stocks`, `history`, `rates`, `cashflows`, `contracts` — результат polars) или SQL через `lake.query(...)`.

## Таблицы кратко

| Таблица | Ключ | Источник | Пишет шаг |
|---------|------|----------|-----------|
| `stocks` | `date, ticker` | ISS, история по бумаге (рынок shares) + расчет | 1, 2 |
| `indexes` | `date, ticker` | ISS, история по индексу | 1b |
| `bonds` | `date, SECID, BOARDID` | ISS, все бумаги рынка bonds за дату | 1c |
| `bonds_securities` | `SECID` | ISS, карточки бумаг | 1c |
| `futures` | `date, SECID, BOARDID` | ISS, все контракты FORTS за дату | 1e |
| `futures_contracts` | `secid` | ISS, реестр серий FORTS | 1e |
| `futures_continuous` | `date, asset` | расчет по `futures` и `futures_contracts` | 1e |
| `shares` | `date, SECID, BOARDID` | ISS, все бумаги рынка shares за дату | 1f |
| `shares_securities` | `SECID` | ISS, карточки бумаг | 1f |
| `indexes_all` | `date, SECID, BOARDID` | ISS, все индексы за дату | 1f |
| `currency` | `date, SECID, BOARDID` | ISS, валютный рынок selt за дату | 1f |
| `currency_fixings` | `date, SECID, BOARDID` | ISS, валютные фиксинги за дату | 1f |
| `ruonia` | `date` | cbr.ru | 1g |
| `zcyc_params` | `date` | ISS, КБД | 1g |
| `zcyc_yields` | `date, period` | ISS, КБД | 1g |
| `zcyc_bonds` | `date, secid` | ISS, КБД | 1g |
| `bond_coupons` | `secid, coupondate` | ISS, bondization | 1g |
| `bond_amortizations` | `secid, amortdate, data_source` | ISS, bondization | 1g |
| `bond_offers` | `secid, offer_date` | ISS, bondization | 1g |
| `stocks_adjusted` | `date, ticker` | расчет по `stocks` и реестрам | 2b |
| `ref_splits`, `ref_renames`, `ref_delisted`, `ref_key_rate`, `ref_sectors` | см. [ниже](#реестры-ref_--копии-metadata) | `metadata/` | 2b |
| `empty_dates` | `dataset, date` | служебная | 1c, 1e, 1f |
| `update_runs`, `quality_log` | `run_id`; все поля, кроме `mode` | служебные | 3 (в конце прогона) |

Представления: `bonds_ofz`, `bonds_corporate` (шаг 2b).

## Схема связей

```mermaid
erDiagram
    stocks }o--o| ref_renames : "ticker = old / new"
    stocks }o--o{ ref_splits : "ticker"
    stocks }o--o| ref_delisted : "ticker"
    stocks }o--o| ref_sectors : "ticker"
    stocks }o--o{ dividends : "ticker"
    stocks ||--|| stocks_adjusted : "date, ticker (склейка, сплиты)"
    stocks }o--|| indexes : "date (календарь IMOEX)"
    stocks }o--o{ shares : "ticker = SECID"
    shares }o--|| shares_securities : "SECID"
    bonds }o--|| bonds_securities : "SECID"
    bonds_securities ||--o{ bond_coupons : "SECID = secid"
    bonds_securities ||--o{ bond_amortizations : "SECID = secid"
    bonds_securities ||--o{ bond_offers : "SECID = secid"
    zcyc_params ||--o{ zcyc_yields : "date"
    zcyc_params ||--o{ zcyc_bonds : "date"
    zcyc_bonds }o--|| bonds_securities : "secid = SECID"
    futures }o--|| futures_contracts : "SECID = secid"
    futures_contracts ||--o{ futures_continuous : "secid = SECID"
    empty_dates }o--|| bonds : "dataset = имя набора"

    stocks { date date PK; string ticker PK; double close; double adj_close; double market_cap }
    stocks_adjusted { date date PK; string ticker PK; string source_ticker; double close }
    indexes { date date PK; string ticker PK; double close; double value_rub }
    shares { date date PK; string SECID PK; string BOARDID PK; double CLOSE }
    bonds { date date PK; string SECID PK; string BOARDID PK; double CLOSE; double YIELDCLOSE; double DURATION }
    bonds_securities { string SECID PK; string ISIN; string MATDATE; double ISSUESIZE }
    bond_coupons { string secid PK; date coupondate PK; double value }
    futures { date date PK; string SECID PK; string BOARDID PK; string ASSETCODE; double SETTLEPRICE; double OPENPOSITION }
    futures_contracts { string secid PK; string base_secid; string asset_code; date expiration_date }
    futures_continuous { date date PK; string asset PK; string SECID; double settle_adj }
    zcyc_params { date date PK; double B1; double T1 }
    empty_dates { string dataset PK; date date PK }
```

- **Календарь торгов** — будни, по которым есть `IMOEX` в `lake.indexes`. По нему проверяются пропуски и отставание всех наборов «все инструменты за дату».
- **Акции ↔ реестры** — по тикеру: переименования склеивают истории, сплиты приводят цены к одной базе, снятые с торгов не обновляются, дивиденды и число акций дают `adj_close` и капитализацию. `stocks.ticker` — это `SECID` бумаги на рынке shares.
- **Облигации ↔ карточки ↔ потоки** — по `SECID` (в потоках и КБД — `secid` в нижнем регистре): история торгов (`bonds`), параметры выпуска (`bonds_securities`), купоны, амортизации и оферты, участие в кривой ОФЗ.
- **Фьючерсы ↔ реестр** — `futures.SECID = futures_contracts.secid`: после перекодировки (шаг 1e) код в истории однозначно определяет контракт, дата экспирации и базовый актив — в реестре.
- **RUONIA** ни с чем по ключу не связана, кроме даты.

## Соглашения

- Даты торгов — колонка `date` типа DATE. Прочие даты в таблицах истории ISS и карточках (`MATDATE`, `OFFERDATE`, `ISSUEDATE`, ...) — строки `YYYY-MM-DD`, как их отдает биржа (перевод — `str.to_date`). В таблицах денежных потоков и реестре фьючерсов даты приведены к DATE.
- Числа — DOUBLE, текст — VARCHAR. Типы колонок ISS берутся из метаданных биржи, поэтому колонка имеет один тип во все годы истории.
- Имена колонок: наборы «все инструменты за дату» и карточки — как в ISS (верхний регистр); `bond_*`, `zcyc_*`, `futures_contracts` — как в соответствующих выдачах ISS (нижний регистр); рабочие таблицы (`stocks`, `indexes`, `futures_continuous`, `ruonia`) — свои имена в нижнем регистре.
- Ключ каждой таблицы уникален; запись идет транзакциями по ключу (новые строки добавляются, существующие обновляются, в большинстве таблиц пишутся только изменившиеся строки).
- Новые поля биржи добавляются в таблицы автоматически (эволюция схемы); в ранних годах часть колонок пустая.
- По годам даты торгов разбиты `bonds`, `futures`, `shares`, `indexes_all`, `currency`.
- Снимки хранилища хранятся 30 дней: `SELECT ... FROM lake.bonds AT (VERSION => n)` — таблица на момент снимка `n` (список — `lake.snapshots()`, ссылка — `lake.ref(table, as_of)`, в функциях чтения — параметр `as_of`).

---

## Таблицы хранилища

### `lake.stocks` — акции

Дневные данные акций рабочего набора (~100 тикеров), включая снятые с торгов; ключ `date + ticker`. Источник — история торгов ISS по бумаге (рынок `shares`): на дату берется строка режима с максимальным оборотом (главная доска), `close` — закрытие основной сессии (та же методика, что у индексов). Пишет шаг 1 (`stocks.update_stocks`) и шаг 2 (`stocks.recompute_stocks`).

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

**Для доходностей** берите `adj_close`, `lake.stocks_adjusted` или `stocks.read_stocks(..., split_adjusted=True)`: сырой `close` содержит разрывы на сплитах, если биржа не пересчитала историю.

### `lake.stocks_adjusted` — акции со склейкой и поправкой на сплиты

То же, что `stocks.read_stocks(split_adjusted=True)`, сохраненное для SQL-потребителей; ключ `date + ticker`. Колонки `lake.stocks` плюс `source_ticker` — исходный тикер строки: строки переименованных тикеров (`TCSG`, `YNDX`, ...) получают итоговый тикер (`T`, `YDEX`), цены (`open`, `low`, `high`, `close`, `waprice`) до сплита приведены к текущей базе, `volume` — обратно; `adj_close`, `value_rub`, `market_cap` не меняются. Пишет шаг 2b (`stocks.update_adjusted` через `lake.sync`: изменившиеся строки пишутся, исчезнувшие ключи удаляются). Пересчитывается при любом изменении `stocks` или реестров.

### `lake.indexes` — индексы (рабочий набор)

Индексы `IMOEX`, `MCFTR`, `RGBITR`; ключ `date + ticker`. Источник — история торгов ISS по индексу (рынок `index`). Пишет шаг 1b (`stocks.update_indexes`). Даты `IMOEX` — торговый календарь проекта.

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

### Представления `lake.bonds_ofz`, `lake.bonds_corporate`

Строки `lake.bonds` по типу выпуска из карточки: `bonds_ofz` — `bonds_securities.TYPE = 'ofz_bond'`, `bonds_corporate` — `TYPE IN ('corporate_bond', 'exchange_bond')`. Все доски; колонки — как у `bonds`. Создаются шагом 2b (`lake.ensure_views`), если есть обе исходные таблицы; определения — `lake.VIEWS`.

### `lake.bonds_securities` — карточки облигаций

Строка на выпуск (`SECID`), включая погашенные: карточка ISS `/iss/securities/<SECID>` (блок `description`) — все поля под именами ISS: `ISIN`, `NAME`, `EMITTER_ID`, `REGNUMBER`, `ISSUEDATE`, `MATDATE`, `ISSUESIZE`, `FACEVALUE`, `INITIALFACEVALUE`, `FACEUNIT`, `COUPONFREQUENCY`, `TYPE`/`TYPENAME`, `BOND_TYPE`/`BOND_SUBTYPE`, `LISTLEVEL`, `HASDEFAULT`, `HASTECHNICALDEFAULT` и др. (набор зависит от выпуска), плюс `FETCHED` — дата запроса. Пишет шаг 1c (`history.update_securities('bonds')`, до 500 новых выпусков за ночь). Карточка запрашивается один раз: изменения параметров выпуска после первого запроса не отслеживаются.

Флаги `HASDEFAULT`/`HASTECHNICALDEFAULT` — текущий статус, как правило на уровне эмитента (все его действующие выпуски), а не история: у погашенных выпусков они не выставлены. Для истории дефолтов не годятся.

### `lake.bond_coupons`, `lake.bond_amortizations`, `lake.bond_offers` — денежные потоки облигаций

Купоны, амортизации и оферты **всех** облигаций, включая погашенные выпуски с 1997 года и будущие выплаты. Источник — сводная выдача ISS `/iss/statistics/engines/stock/markets/bonds/bondization`. Пишет шаг 1g (`cashflows.update_cashflows`): каждую ночь — потоки с датой от сегодня −10 до +60 дней, по субботам — все будущие потоки; полная история — `--history-init cashflows`. Колонки — как в выдаче ISS (нижний регистр), даты приведены к DATE (`0000-00-00` → пусто).

| Таблица | Ключ | Основные колонки |
|---------|------|------------------|
| `bond_coupons` | `secid, coupondate` | `isin`, `name`, `issuevalue`, `coupondate`, `recorddate`, `startdate`, `initialfacevalue`, `facevalue`, `faceunit`, `value` (купон на бумагу в валюте номинала), `valueprc` (% годовых), `primary_boardid` |
| `bond_amortizations` | `secid, amortdate, data_source` | `amortdate`, `facevalue`, `initialfacevalue`, `faceunit`, `valueprc`, `value`, `data_source` — `amortization` (частичное погашение) или `maturity` (погашение), `primary_boardid` |
| `bond_offers` | `secid, offer_date` | `offerdate`, `offerdatestart`, `offerdateend` (период предъявления), `offer_date`, `price`, `value`, `agent`, `offertype`, `facevalue`, `faceunit`, `primary_boardid` |

Особенности:

- **`facevalue` — текущий номинал**, а не номинал на дату выплаты: у амортизируемых выпусков `valueprc` прошлых купонов искажен — используйте `value`.
- **`value_rub` не хранится**: биржа пересчитывает его по сегодняшнему курсу.
- Будущие купоны флоатеров и ипотечных облигаций пусты (`value` = null), пока не зафиксированы.
- **`offer_date`** — `offerdate`, а если биржа его не указала, начало или конец периода предъявления. `offertype` меняется со временем («Оферта» → «Оферта (состоялось)»), поэтому в ключ не входит. Строки без всех трех дат не хранятся: это заглушки «Оферта/Погашение» без цены и суммы, по одной на выпуск (около 600 из 6,5 тыс. строк биржи).
- **Удаление отмененных.** Запрошенный диапазон дат приходит целиком: купоны и амортизации в нем, которых биржа больше не отдает (отмененные, перенесенные), удаляются. Оферты удаляются только при полной выгрузке — фильтр ISS по датам идет по `offerdate`, а у части оферт он пустой.

### `lake.futures` — фьючерсы FORTS

История торгов **всех** фьючерсных контрактов с 03.01.2002, включая истекшие. Ключ `date + SECID + BOARDID`; разбита по годам. Источник — история торгов ISS «все контракты за дату» (`futures/markets/forts`). Пишет шаг 1e (`history.update/repair('futures')`, затем `contracts.remap_futures_secids`). Все поля ISS под теми же именами — [iss-columns.md → Фьючерсы](iss-columns.md#4).

| Колонка | Смысл |
|---------|-------|
| `date`, `BOARDID`, `SECID`, `SHORTNAME` | Дата, режим, код контракта (`SiZ5`, `SiZ5_2015`), имя с месяцем и годом экспирации (`Si-12.25`) |
| `ASSETCODE` | Базовый актив (`Si`, `RTS`, `BR`, `GD`, `MIX`, ...) |
| `OPEN`, `LOW`, `HIGH`, `CLOSE`, `WAPRICE` | Цены |
| `SETTLEPRICE` | Расчетная цена (для вариационной маржи) |
| `OPENPOSITION`, `OPENPOSITIONVALUE` | Открытый интерес: контракты и руб. |
| `VOLUME`, `VALUE`, `NUMTRADES`, `QTY` | Объемы и число сделок |
| `SWAPRATE`, `SWAPRATE_CURR`, `CHANGE` | Своп-ставка (для вечных фьючерсов), изменение |

**Коды контрактов.** Коды повторяются раз в 10 лет, и при повторном листинге ISS **задним числом переименовывает старый контракт**: `SiZ5` декабря 2015 года теперь `SiZ5_2015` — и в реестре, и в истории торгов (так же `SiZ5_2005`). Строки, загруженные до переименования, `contracts.remap_futures_secids()` переводит на новый код по датам обращения контракта из реестра — история совпадает с биржей, повторная загрузка не создает дублей. Поэтому `SECID` однозначно определяет контракт на всей истории; дата экспирации и базовый актив — в `futures_contracts`.

### `lake.futures_contracts` — реестр фьючерсных контрактов

Все контракты FORTS с 2001 года, включая истекшие и календарные спреды (`SiZ5SiH6`); ключ `secid`. Источник — `/iss/statistics/engines/futures/markets/forts/series?show_expired=1` одним запросом. Пишет шаг 1e (`contracts.update_contracts` через `lake.sync`: исчезнувшие у биржи контракты удаляются).

| Колонка | Тип | Смысл |
|---------|-----|-------|
| `secid` | VARCHAR | Код контракта, как сейчас в ISS (`SiZ5`, `SiZ5_2015`) |
| `name` | VARCHAR | Наименование |
| `start_date`, `expiration_date` | DATE | Начало обращения, экспирация (вечные фьючерсы — 2100-01-01) |
| `asset_code` | VARCHAR | Базовый актив (= `futures.ASSETCODE`) |
| `underlying_asset` | VARCHAR | Код базового инструмента |
| `is_traded` | DOUBLE | 1 — торгуется |
| `base_secid` | VARCHAR | Код без суффикса года (`SiZ5_2015` → `SiZ5`) |

### `lake.futures_continuous` — непрерывные ряды фьючерсов

Непрерывные ряды по основным активам (`contracts.MAIN_ASSETS`: `Si`, `Eu`, `CNY`, `RTS`, `MIX`, `MXI`, `BR`, `NG`, `GOLD`, `SILV`, `SBRF`, `GAZR`); ключ `date + asset`. Расчет по `futures` и `futures_contracts`, пишет шаг 1e (`contracts.update_continuous` через `lake.sync` — пересчитывается вся история).

Правило выбора контракта на дату: среди контрактов актива (кроме вечных), до экспирации которых больше `ROLL_DAYS` = 7 календарных дней и у которых есть цена, берется контракт с наибольшим открытым интересом (при равенстве — объемом), то есть ликвидный, а не обязательно ближайший; ряд никогда не возвращается к контракту с более ранней экспирацией.

| Колонка | Смысл |
|---------|-------|
| `date`, `asset` | Дата, базовый актив (`ASSETCODE`) |
| `SECID`, `expiration_date` | Контракт ряда на дату и его экспирация |
| `open`, `high`, `low`, `close` | Цены контракта (`OPEN`, `HIGH`, `LOW`, `CLOSE`) без поправки |
| `settle` | Расчетная цена контракта (`SETTLEPRICE`, если нет — `CLOSE`) без поправки |
| `volume`, `open_interest` | `VOLUME`, `OPENPOSITION` |
| `roll` | True — в эту дату ряд перешел на новый контракт |
| `adj_factor` | Множитель склейки: произведение отношений цен всех переходов после этой даты |
| `settle_adj` | `settle × adj_factor` — ряд, приведенный к уровню текущего контракта: доходности по нему не содержат скачков на переходах |

Отношение на переходе — цена нового контракта к цене старого в последний день старого; если цены нового в тот день нет — по ценам обоих в день перехода. Цены `open`–`close` не склеиваются.

### Рынки «все инструменты за дату»: `shares`, `indexes_all`, `currency`, `currency_fixings`

Устроены как `bonds` и `futures` (`history.DATASETS`): один постраничный запрос ISS на торговую дату, **все поля** ISS под теми же именами (дата — `date`), ключ `date + SECID + BOARDID`, докачка пропусков по календарю IMOEX. Пишет шаг 1f; первичная выгрузка — `--history-init <набор>`. Полные списки полей — [iss-columns.md](iss-columns.md).

| Таблица | Рынок ISS | Начало в ISS | Что и основные колонки |
|---------|-----------|--------------|------------------------|
| `shares` | `stock/markets/shares` | 24.03.1997 | Все бумаги рынка акций: акции, депозитарные расписки, паи ПИФ и ETF по всем доскам (`TQBR` — акции, `TQTF` — фонды, `TQIF`, `SMAL`, `TQTD`, `TQPI`, `SPEQ`, ...). `SHORTNAME`, `OPEN`, `LOW`, `HIGH`, `CLOSE`, `LEGALCLOSEPRICE`, `WAPRICE`, `VOLUME`, `VALUE`, `NUMTRADES`, `MARKETPRICE2`, `MARKETPRICE3`, `ADMITTEDQUOTE`, `CURRENCYID`, `TRADINGSESSION`. Разбита по годам |
| `indexes_all` | `stock/markets/index` | 01.09.1995 | Все индексы MOEX: акций, облигаций (`RGBI`, ...), денежного рынка (`RUSFAR`), iNAV фондов (доска `INAV`) и др. `SHORTNAME`, `NAME`, `OPEN`, `HIGH`, `LOW`, `CLOSE`, `VALUE`, `VOLUME`, `CAPITALIZATION`, `DIVISOR`, `DURATION`, `YIELD` (для облигационных), `CURRENCYID`. Разбита по годам |
| `currency` | `currency/markets/selt` | 02.06.1997 | Валютный рынок: пары и инструменты (`CNYRUB_TOM`, `USD000UTSTOM`, свопы) по доскам (`CETS` — основная, `CNGD`, `LICU` и др.). `SHORTNAME`, `OPEN`, `LOW`, `HIGH`, `CLOSE`, `WAPRICE`, `NUMTRADES`. Хранятся **только строки со сделками** (`NUMTRADES > 0`): ISS отдает множество строк-заглушек с нулевыми ценами. Разбита по годам |
| `currency_fixings` | `currency/markets/index` | 01.08.2019 | Валютные фиксинги MOEX (доска `FIXI`): `USDFIXME`, `EURFIXME`, `CNYFIXME`, `CNYFIX`, `EURUSDFIXME`, `USDCNYFIXME`, фиксинги драгметаллов (`SILVFIXME`, `PLATFIXME`, `PALADFIXME`), других валют (`BYNFIXME`, `TRYFIXME`, ...). `OPEN`, `LOW`, `HIGH`, `CLOSE` |

**Осторожно с фиксингами.** С июня 2024 года значения `USDFIXME` и `EURFIXME` совпадают с официальными курсами Банка России — вероятно, после прекращения биржевых торгов долларом и евро (13.06.2024) смысл ряда изменился; для анализа биржевого курса до и после этой даты ряды несопоставимы.

**`shares` и `stocks`.** `lake.stocks` — рабочий набор с одной строкой на дату (главная доска) и расчетными колонками; `lake.shares` — сырые данные всех бумаг и досок. `stocks.ticker` = `shares.SECID`.

### `lake.shares_securities` — карточки бумаг рынка акций

Как `bonds_securities`, для бумаг из `lake.shares`: строка на `SECID`, все поля карточки ISS (`ISIN`, `NAME`, `SHORTNAME`, `REGNUMBER`, `ISSUESIZE`, `FACEVALUE`, `FACEUNIT`, `ISSUEDATE`, `LISTLEVEL`, `TYPE`/`TYPENAME` — `common_share`, `preferred_share`, `depositary_receipt`, `exchange_ppif`, `etf_ppif`, ...; `GROUP`, `EMITTER_ID`, `ISQUALIFIEDINVESTORS`, `HASDEFAULT` и др.) плюс `FETCHED`. Пишет шаг 1f (`history.update_securities('shares')`, до 500 новых бумаг за ночь). `ISSUESIZE` — текущий объем выпуска на дату запроса, а не история (исторический объем ISS отдает только с 01.04.2024 — см. [план](../development-plan.md)).

### `lake.ruonia` — ставка RUONIA

Ставка RUONIA Банка России с 11.01.2010; ключ `date`. Источник — cbr.ru (`/hd_base/ruonia/dynamics/`), в ISS RUONIA нет. Пишет шаг 1g (`rates.update_ruonia`: вся история одним запросом, в хранилище — новые и пересмотренные строки).

| Колонка | Тип | Смысл |
|---------|-----|-------|
| `date` | DATE | Дата ставки |
| `rate` | DOUBLE | RUONIA, % годовых |
| `volume_bn` | DOUBLE | Объем сделок, млрд руб. |
| `deals`, `participants` | DOUBLE | Число сделок и участников |
| `rate_min`, `rate_p25`, `rate_p75`, `rate_max` | DOUBLE | Минимум, 25-й и 75-й процентили, максимум ставок сделок |
| `status` | VARCHAR | Статус расчета с сайта (`Стандартный`, `Резервный`; `—` — нет пометки) |
| `published` | DATE | Дата публикации |

### `lake.zcyc_params`, `lake.zcyc_yields`, `lake.zcyc_bonds` — кривая бескупонной доходности

Кривая бескупонной доходности ОФЗ MOEX (КБД) с 06.01.2014. Источник — ISS `/iss/engines/stock/zcyc` на каждую дату. Грузится **по вчерашний день**: за сегодня ISS отдает промежуточную кривую. Пишет шаг 1g (`rates.update_zcyc`); первичная выгрузка — `--history-init zcyc`. Колонки — как в ISS, `tradedate` → `date`, `tradetime` — время расчета.

| Таблица | Ключ | Колонки |
|---------|------|---------|
| `zcyc_params` | `date` | Параметры Nelson–Siegel–Svensson: `B1`, `B2`, `B3`, `T1`, `G1`–`G9` |
| `zcyc_yields` | `date, period` | `period` — срок, лет (0.25, 0.5, 0.75, 1, 2, 3, 5, 7, 10, 15, 20), `value` — доходность, % |
| `zcyc_bonds` | `date, secid` | ОФЗ, по которым строилась кривая: `shortname`, `expdate` (строка), `benchmark`, цены и доходности заявок (`bidprice`, `bidyield`, `askprice`, `askyield`), сделок (`trdprice`, `trdyield`), расчетные (`clcprice`, `clcyield`, `crtprice`, `crtyield`, `correction`), дюрации (`crtduration`, `bidduration`, `askduration`) |

### Реестры `ref_*` — копии `metadata/`

Копии файлов `metadata/` для SQL-потребителей, колонки — как в файлах (см. [реестры](#реестры-metadata)). Пишет шаг 2b (`stocks.sync_registries` через `lake.sync`: таблица приводится к содержимому файла, удаленные записи удаляются).

| Таблица | Файл | Ключ | Колонки |
|---------|------|------|---------|
| `ref_splits` | `splits.csv` + `../dividends/metadata/splits.json` (объединенный реестр `stocks.load_splits`) | `ticker, date` | `ticker, date, ratio, kind` |
| `ref_renames` | `renames.csv` | `old` | `old, new, date` |
| `ref_delisted` | `delisted.csv` | `ticker` | `ticker, last_date, note` |
| `ref_key_rate` | `key_rate.csv` | `date` | `date, rate` |
| `ref_sectors` | `sectors.csv` | `ticker` | `ticker, sector` |

### `lake.empty_dates` — служебная

Торговые (по календарю IMOEX) даты, за которые ISS подтвержденно не вернул строк; ключ `dataset + date` (`dataset` — имя набора из `history.DATASETS`). При докачке пропусков такие даты больше не запрашиваются.

### `lake.update_runs`, `lake.quality_log` — история прогонов

Пишет `update_data.py` в конце каждого прогона с проверкой данных.

`update_runs` — ключ `run_id` (время начала прогона):

| Колонка | Смысл |
|---|---|
| `run_id`, `finished` | Начало и конец прогона |
| `mode` | `update` — обновление (ночное или ручное), `check` — `--check` |
| `status` | `ok` — без сбоев и замечаний, `issues` — есть замечания проверки, `error` — сбой хотя бы одного шага |
| `warnings`, `messages` | Число сбоев шагов и их тексты (через перевод строки) |
| `issues`, `new_issues` | Замечаний проверки всего и новых — которых не было в прошлом прогоне того же режима |

`quality_log` — замечания проверки по прогонам: `run_id`, `mode`, `check`, `object`, `detail` (как в `quality.data_quality_report`); ключ — все поля, кроме `mode`. Когда проблема появилась: `SELECT min(run_id) FROM lake.quality_log WHERE "check" = 'dividend_gap' AND object = 'MSNG'`.

---

## Реестры `metadata/`

Корпоративные события и справочники ведутся только здесь — данные в хранилище руками не правятся. Копии для SQL — таблицы `ref_*` (шаг 2b).

| Файл | Колонки | Назначение | Кто пишет |
|------|---------|------------|-----------|
| `splits.csv` | `ticker, date, ratio, kind` | Сплиты и консолидации. `kind`: `price` — в истории цен разрыв, цены до даты делятся на `ratio`; `shares` — биржа пересчитала цены, но число акций в старых срезах в старой базе; `auto` — вид определяется по данным (есть ценовой разрыв — ценовая поправка, нет — поправка числа акций). `ratio` в ценовой семантике: дробление 1:10 → `10`, консолидация 100:1 → `0.01` | вручную; плюс внешний реестр `../dividends/metadata/splits.json` (записи получают `kind=auto`) |
| `renames.csv` | `old, new, date` | Переименования тикеров (`TCSG→T`, `YNDX→YDEX`, ...): `date` — первый день под новым тикером. При чтении истории склеиваются, исходный тикер строки — `source_ticker` | вручную |
| `delisted.csv` | `ticker, last_date, note` | Снятые с торгов: история хранится, но не обновляется. Кандидатов показывает проверка данных (`stock_stale`) | вручную |
| `key_rate.csv` | `date, rate` | Ключевая ставка ЦБ по датам изменения, % годовых (до 13.09.2013 — ставка рефинансирования) | шаг 1d, автоматически с cbr.ru |
| `sectors.csv` | `ticker, sector` | Отраслевой справочник (`stocks.load_sectors`) | вручную |
| `stock-index-base.xlsx` | листы-даты `DD.MM.YYYY`: `Code`, `Number of issued shares` (шапка на 4-й строке) | Срезы числа акций для капитализации | вручную |

## Дивиденды (`../dividends/data/<TICKER>.csv`)

Соседний проект собирает историю выплат с сайта закрытияреестров.рф и приводит суммы к текущей акции с учетом сплитов. Для `adj_close` используются `closing_date` (дата закрытия реестра) и `dividend_value` (руб. на акцию, > 0). Привилегированные акции — в файле `<TICKER>P.csv` по правилам того проекта. Если сайт еще не внес выплату, проверка данных покажет гэп открытия без дивиденда (`dividend_gap`).

---

## Обновление: какие шаги пишут какие таблицы

| Шаг `update_data.py` | Таблицы / файлы | Как |
|---|---|---|
| 1 — акции | `stocks` | дозагрузка с последней даты тикера, сразу `adj_close` и капитализация по всей истории; пишутся только новые и изменившиеся строки |
| 1b — индексы | `indexes` | дозагрузка с последней даты |
| 1c — облигации | `bonds`, `bonds_securities`, `empty_dates` | хвост истории, докачка пропусков, до 500 новых карточек |
| 1d — ключевая ставка | `metadata/key_rate.csv` | новые решения ЦБ |
| 1e — фьючерсы | `futures`, `empty_dates`, `futures_contracts`, `futures_continuous` | хвост истории, докачка пропусков; реестр контрактов, перекодировка строк после повторного листинга, пересчет непрерывных рядов |
| 1f — прочие рынки | `shares`, `indexes_all`, `currency`, `currency_fixings`, `shares_securities`, `empty_dates` | хвост истории, докачка пропусков, до 500 новых карточек бумаг рынка акций |
| 1g — ставки и потоки | `ruonia`, `zcyc_params`, `zcyc_yields`, `zcyc_bonds`, `bond_coupons`, `bond_amortizations`, `bond_offers` | RUONIA — вся история, записываются изменения; КБД — по вчерашний день; потоки — окно −10…+60 дней, по субботам все будущие |
| 2 — пересчет | `stocks` | `adj_close` и капитализация после изменений дивидендов, срезов или реестра сплитов — только изменившиеся строки |
| 2b — копии для SQL | `ref_splits`, `ref_renames`, `ref_delisted`, `ref_key_rate`, `ref_sectors`, `stocks_adjusted`, представления `bonds_ofz`, `bonds_corporate` | синхронизация с файлами и `stocks`; недостающие представления |
| 3 — проверка | `update_runs`, `quality_log` | отчет о качестве в лог; итог прогона и замечания — в конце прогона |
| 4 — обслуживание | файлы хранилища | слияние мелких файлов, удаление снимков старше 30 дней |
| 5 — копия каталога | `<MOEX_DATA_ROOT>/backups/catalog` | `pg_dump` каталога, 14 последних копий |

Первичные выгрузки полной истории — `update_data.py --history-init bonds,futures,shares,indexes_all,currency,currency_fixings,zcyc,cashflows` (многочасовые; реестры `bonds_securities` и `shares_securities` наполняются следом).

## Примеры доступа

```python
import polars as pl
from moexutils import stocks, history, rates, cashflows, contracts, lake

sber = stocks.read_stocks('SBER', start='2024-01-01')                 # polars
all_adj = stocks.read_stocks(split_adjusted=True)                      # все тикеры, склейка + сплиты
imoex = stocks.read_index('IMOEX')
ofz = history.read('bonds', boards='TQOB', start='2026-01-01',
                   columns=['date', 'SECID', 'YIELDCLOSE', 'DURATION', 'MATDATE'])
si = history.read('futures', start='2025-01-01').filter(pl.col('ASSETCODE') == 'Si')
cards = history.read_securities('bonds')
etf = history.read('shares', boards='TQTF', start='2026-01-01')
cny = history.read('currency', secids='CNYRUB_TOM', start='2025-01-01')
si_cont = contracts.read_continuous('Si')
curve = rates.read_zcyc('yields', start='2026-01-01')
coupons = cashflows.read_cashflows('coupons', secids='SU26238RMFS4')

# SQL: медианный Z-спред корпоративных облигаций по дням
lake.query("""
    SELECT date, median(ZSPREAD) AS zspread, count(*) AS n
    FROM lake.bonds_corporate WHERE BOARDID = 'TQCB' AND ZSPREAD IS NOT NULL
    GROUP BY date ORDER BY date""")

# SQL: история выпуска вместе с его карточкой
lake.query("""
    SELECT b.date, b.CLOSE, b.YIELDCLOSE, s.ISIN, s.ISSUESIZE
    FROM lake.bonds b JOIN lake.bonds_securities s USING (SECID)
    WHERE b.SECID = 'SU26238RMFS4' AND b.BOARDID = 'TQOB'""")

# SQL: история фьючерса с датой экспирации из реестра
lake.query("""
    SELECT f.date, f.SECID, f.SETTLEPRICE, c.expiration_date
    FROM lake.futures f JOIN lake.futures_contracts c ON c.secid = f.SECID
    WHERE c.base_secid = 'SiZ5' ORDER BY f.date""")
```
