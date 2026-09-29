# Справочник API

Модули проекта:

| Модуль | Что в нем |
|--------|-----------|
| `stocks` | Акции и индексы в хранилище: загрузка из ISS, сплиты, переименования, снятые с торгов, `adj_close`, капитализация, ключевая ставка, безрисковая ставка |
| `history` | История рынков «все инструменты за дату» (облигации, фьючерсы) в хранилище, реестр карточек бумаг |
| `quality` | Проверка качества данных |
| `lake` | Хранилище DuckLake: подключение, SQL-запросы с результатом в polars, запись по ключу, обслуживание |
| `iss` | Доступ к ISS: HTTP-сессия, разбор ответов в polars |
| `moex_utils` | Фасад прежнего интерфейса: реэкспорт функций модулей, обертки облигаций и фьючерсов, математика облигаций |

Все функции модулей возвращают **polars**; pandas в проекте не используется. `moex_utils` реэкспортирует функции `stocks` под теми же именами (`read_stocks`, `read_index`, `adjust_for_splits`, `apply_renames`, `load_renames`, `load_key_rate`, `risk_free_monthly` и др.). Модель таблиц — [data-model.md](data-model.md), все поля биржи — [iss-columns.md](iss-columns.md). Скрипт обновления — `update_data.py` (раздел в конце).

## Константы и инфраструктура

Пути не зависят от текущего рабочего каталога. **Рыночные данные** — в хранилище DuckLake: файлы в `<MOEX_DATA_ROOT>/lake` (переменная окружения `MOEX_DATA_ROOT`, на рабочей машине `F:\moex-data`; без нее — папка проекта), каталог — PostgreSQL. **Реестры** `metadata/` — в папке проекта и в git. Переменная читается при импорте модулей: процессы, запущенные до ее установки, нужно перезапустить.

| Константа | Значение | Описание |
|-----------|----------|----------|
| `lake.LAKE_DATA_PATH` | `<MOEX_DATA_ROOT>/lake` | Файлы данных хранилища |
| `lake.PG_DATABASE`, `lake.PG_USER` | `moex_lake`, `moex` | Каталог хранилища (переопределяются `MOEX_PG_*`) |
| `stocks.METADATA_FILE` | `<проект>/metadata/stock-index-base.xlsx` | Срезы числа акций |
| `stocks.SPLITS_FILE` | `<проект>/metadata/splits.csv` | Реестр сплитов |
| `stocks.EXTERNAL_SPLITS_FILE` | `<проект>/../dividends/metadata/splits.json` | Внешний реестр сплитов проекта dividends |
| `stocks.RENAMES_FILE` | `<проект>/metadata/renames.csv` | Реестр переименований |
| `stocks.DELISTED_FILE` | `<проект>/metadata/delisted.csv` | Снятые с торгов |
| `stocks.KEY_RATE_FILE` | `<проект>/metadata/key_rate.csv` | Ключевая ставка ЦБ |
| `stocks.DIVIDENDS_FOLDER` | `<проект>/../dividends/data` | CSV дивидендов |
| `iss.ISS_TIMEOUT` | `(10, 60)` | Таймаут запроса к ISS: соединение, ответ (сек) |

**HTTP.** Все загрузчики ходят в ISS через `iss.make_session()` — `requests.Session` с таймаутом и повторами на сетевых сбоях, 429 и 5xx. Историю ISS отдает страницами по 100 строк, загрузчики листают все страницы. Типы колонок ответа берутся из метаданных ISS.

**Логи.** Сообщения идут через логгер `moex_utils` (по умолчанию — в stdout). Приглушить: `logging.getLogger("moex_utils").setLevel(logging.WARNING)`.

---

## Акции (`stocks`)

```python
stocks.read_stocks(tickers=None, start=None, end=None, merge_renames=True, split_adjusted=False,
                   columns=None) -> pl.DataFrame
stocks.list_tickers(include_delisted=True) -> list[str]
stocks.update_stocks(tickers=None, include_delisted=False, div_folder=None, rebuild=False,
                     session=None) -> int
stocks.recompute_stocks(tickers=None, div_folder=None, dry_run=False) -> pl.DataFrame
stocks.add_stock(ticker, start='2002-01-01', div_folder=None, session=None) -> int
stocks.fetch_stock(ticker, start, end=None, session=None) -> pl.DataFrame
```

- **`read_stocks`** — дневные данные из `lake.stocks`. `merge_renames` склеивает истории переименованных тикеров (исходный тикер строки — `source_ticker`); `split_adjusted` приводит цены к пост-сплитовой базе (после склейки: реестр сплитов записан на текущий тикер). Результат отсортирован по `ticker, date`.
- **`update_stocks`** — дозагрузка из ISS с последней даты тикера (она перекачивается) до сегодня; `rebuild` — вся история с 2002 года. Для каждого тикера сразу пересчитываются `adj_close` и капитализация по всей истории; в хранилище пишутся только новые и изменившиеся строки — одной транзакцией. Снятые с торгов (`metadata/delisted.csv`) пропускаются (`include_delisted=True` — опросить и их). Шаг 1 `update_data.py`.
- **`recompute_stocks`** — пересчет `adj_close` и капитализации после изменения дивидендов, срезов числа акций или реестра сплитов; пишет только изменившиеся строки, `dry_run` — только вернуть их. Шаг 2 `update_data.py`.
- **`add_stock`** — новый тикер: вся история с `start`, сразу с расчетными колонками.
- **`fetch_stock`** — дневные данные из истории торгов ISS: на дату — строка режима с максимальным оборотом (главная доска), `close` — закрытие основной сессии (та же методика, что у индексов).

Колонки и их смысл — [data-model.md → lake.stocks](data-model.md#lakestocks--акции). Внутридневные свечи не поддерживаются (не используются проектом).

## Индексы (`stocks`)

```python
stocks.read_index(ticker='IMOEX', start=None, end=None) -> pl.DataFrame
stocks.update_indexes(tickers=('IMOEX', 'MCFTR', 'RGBITR'), session=None) -> int
stocks.fetch_index(ticker, start, end=None, session=None) -> pl.DataFrame
```

История индексов в `lake.indexes`: `date, ticker, BOARDID, close, value_rub` (оборот), `volume`. `update_indexes` дозагружает с последней даты (без истории — с 2000 года) и пишет только новые и изменившиеся строки; шаг 1b. Даты IMOEX служат торговым календарем для проверок и докачки пропусков.

## Корпоративные события (`stocks`)

```python
stocks.load_splits(splits_file=None, external_file=None) -> pl.DataFrame
stocks.adjust_for_splits(df, splits=None) -> pl.DataFrame
stocks.price_jump_matches(dates, prices, date, divisor) -> bool
stocks.load_renames(renames_file=None) -> pl.DataFrame
stocks.apply_renames(df, renames=None) -> pl.DataFrame
stocks.load_delisted(delisted_file=None) -> pl.DataFrame
stocks.is_traded(ticker, session=None) -> Optional[bool]
```

**Сплиты.** Реестр `metadata/splits.csv` (`ticker, date, ratio, kind`) плюс внешний `../dividends/metadata/splits.json` (записи получают `kind=auto`; явная запись `splits.csv` в пределах 45 дней приоритетнее):

- **`price`** — в истории цен разрыв на дату: `adjust_for_splits` делит цены (`close`, `open`, `high`, `low`, `waprice`) до даты на `ratio`, объем умножает; `adj_close`, `value_rub`, `market_cap` не трогаются.
- **`shares`** — биржа пересчитала цены, но число акций в старых срезах метаданных в старой базе: число акций до даты делится на `ratio` (капитализация).
- **`auto`** — вид по данным (`price_jump_matches`): есть ценовой разрыв, соответствующий сплиту (допуск 2,5×) — ценовая поправка, нет — поправка числа акций.

`ratio` в ценовой семантике: дробление 1:10 → `10`, консолидация 100:1 → `0.01`.

**Переименования.** `metadata/renames.csv` (`old, new, date` — первый день нового тикера): `apply_renames` дает строкам старого тикера новый тикер, исходный — в `source_ticker`; строки старого тикера с даты переименования отбрасываются с предупреждением; цепочки A→B→C поддерживаются. `adj_close` старых имен при пересчете (`recompute_stocks`) считается по склеенной истории в базе итогового тикера: сплиты и дивиденды преемника распространяются на историю прежнего имени, стык непрерывен (дивиденды с одной датой реестра берутся из файла итогового тикера, несколько выплат на дату сохраняются). Конвертации с коэффициентом ≠ 1:1 в реестр не входят.

**Снятые с торгов.** `metadata/delisted.csv` (`ticker, last_date, note`): история хранится, но не обновляется. `is_traded` — торгуется ли акция хоть на одной доске рынка shares (флаг ISS `is_traded`; `None` — ISS бумагу не знает).

## Дивиденды, `adj_close`, капитализация (`stocks`)

```python
stocks.load_dividends(ticker, div_folder=None) -> pl.DataFrame
stocks.ex_dividend_pos(dates, record_date) -> int
stocks.adj_close(df, dividends=None, splits=None) -> tuple[pl.DataFrame, list]
stocks.load_shares(metadata_file=None) -> pl.DataFrame
stocks.market_cap(df, shares=None, splits=None) -> pl.DataFrame
stocks.enrich(df, div_folder=None, shares=None, splits=None) -> tuple[pl.DataFrame, list]
```

- **`adj_close`** (одна бумага) — цена с поправкой на дивиденды и сплиты в текущей базе. Методика и экс-дата (T+1 с 31.07.2023, раньше T+2) — [data-model.md → lake.stocks](data-model.md#lakestocks--акции). Возвращает также список отброшенных дивидендов `[(дата, сумма)]` — их показывает проверка данных.
- **`load_dividends`** — CSV проекта dividends: `closing_date`, `dividend_value` (> 0).
- **`load_shares`** — срезы числа акций из Excel (`ticker, date, shares`), кэш до изменения файла; читается через openpyxl.
- **`market_cap`** — `shares` (между срезами — последний, до первого — первый, с поправками сплитов) и `market_cap = close × shares`; нет данных — пустые колонки.
- **`enrich`** — `adj_close` и капитализация вместе.

Перенос логики с pandas на polars проверен сверкой: пересчет по всем 311 тыс. строкам дал нулевое расхождение с прежними значениями.

## Ставки (`stocks`)

```python
stocks.load_key_rate(key_rate_file=None) -> pl.DataFrame
stocks.update_key_rate(key_rate_file=None, session=None) -> int
stocks.risk_free_monthly(dates, key_rate_file=None) -> pl.DataFrame
```

Ключевая ставка ЦБ в `metadata/key_rate.csv` (`date` — дата изменения, `rate` — % годовых; до 13.09.2013 — ставка рефинансирования). `update_key_rate` дописывает решения ЦБ после последней записи (таблица ставки на cbr.ru, разбор HTML через lxml; в файл — только даты изменения); шаг 1d. `risk_free_monthly` — месячная ставка в долях на даты (`date, rf`; порядок входа сохраняется, до первой записи — первое значение).

---

## Хранилище DuckLake (`lake`)

```python
lake.query(sql, params=None) -> pl.DataFrame
lake.write(table, df: pl.DataFrame, key=None) -> int
lake.tables() -> list[str]
lake.session(read_only=False)      # контекстный менеджер: подключение DuckDB с хранилищем `lake`
lake.maintenance(retention_days=30) -> None
```

Каталог (метаданные, снимки, схема) — PostgreSQL на localhost: база `moex_lake`, роль `moex` (служба Windows `postgresql-x64-17`, автозапуск). Данные — Parquet со сжатием zstd в `<MOEX_DATA_ROOT>/lake`. Одновременная работа писателя и читателей проверена (20 транзакций записи при трех параллельных читателях — без ошибок).

- **Пароль** берется из файла паролей PostgreSQL (`%APPDATA%\postgresql\pgpass.conf`, либо `PGPASSFILE`) и передается во временный безымянный секрет DuckDB: встроенная в DuckDB libpq не читает pgpass и переменные окружения, а пароль в строке подключения попал бы в текст ошибок.
- **`query`** — SQL только на чтение, таблицы — `lake.<имя>`, результат — polars DataFrame.
- **`write`** — одна транзакция: `MERGE` по ключу таблицы (строки с тем же ключом обновляются, новые — добавляются), новые колонки добавляются автоматически (у ISS набор полей со временем расширяется), дубли ключа отклоняются. Таблица создается при первой записи; `bonds` и `futures` разбиты по годам.
- **`maintenance`** — слияние мелких файлов ежедневных дозаписей, удаление снимков старше 30 дней (окно для отката) и неиспользуемых файлов; шаг 5 `update_data.py`.
- **Тесты и работа без Postgres:** переменная `MOEX_LAKE_CATALOG` задает другой каталог, например файловый `ducklake:C:/tmp/catalog.ducklake`.

Таблицы и ключи: `bonds`, `futures` — `date, SECID, BOARDID`; `bonds_securities` — `SECID`; `empty_dates` — `dataset, date`; `stocks`, `indexes` — `date, ticker`. Подробно — [data-model.md](data-model.md).

Пример SQL поверх хранилища:

```python
import lake
lake.query('''
    SELECT date, median(ZSPREAD) AS zspread
    FROM lake.bonds
    WHERE BOARDID = 'TQCB' AND date >= DATE '2025-01-01'
    GROUP BY date ORDER BY date''')
```

---

## История рынков (`history`)

```python
history.update(dataset, start=None, max_days=3000, session=None, flush_every=50) -> int
history.repair(dataset, session=None, calendar=None) -> int
history.read(dataset, start=None, end=None, secids=None, boards=None, columns=None) -> pl.DataFrame
history.dataset_dates(dataset) -> list[date]
history.update_securities(dataset='bonds', max_new=500, session=None) -> int
history.read_securities(dataset='bonds') -> pl.DataFrame
```

Наборы (`dataset`): `bonds` — все облигации MOEX с 1997 года, `futures` — все контракты FORTS с 2002 года. Один постраничный запрос ISS на торговую дату отдает все инструменты рынка со **всеми колонками**; строки пишутся в таблицу хранилища. Типы колонок берутся из метаданных ответа ISS: числовые — Float64, остальные — строки (так одна колонка имеет один тип во все годы).

- **Хвост** — даты после последней сохраненной до сегодня (ночной прогон).
- **Бэкфилл** — только при явно заданном `start` раньше истории: даты от истории назад (при обрыве скачанное примыкает к истории).
- **Без дыр** — на сбое даты прогон останавливается, следующий запуск продолжит с нее.
- **Прогресс** — запись порциями по `flush_every` дат (многочасовая выгрузка не теряет результат).
- **`repair`** — докачка пропусков внутри истории по будням IMOEX; даты, за которые ISS подтвержденно пуст, запоминаются в `lake.empty_dates` и больше не запрашиваются.
- **Параллельная работа** — ночное обновление и идущая выгрузка могут писать одновременно: запись идемпотентна (MERGE по ключу), конфликты транзакций разрешает каталог.

`read` возвращает историю за период (`start`/`end` включительно) с фильтрами по бумагам и режимам торгов; `columns` ускоряет чтение полной истории.

---

## Облигации

```python
update_bonds_market(start=None, session=None, max_days=3000) -> int
repair_bonds_market(session=None, calendar=None) -> int
update_bonds_market_all(session=None) -> None
read_bonds_market(start=None, end=None, boards=None, secids=None, columns=None) -> pl.DataFrame
update_bonds_securities(max_new=500, session=None) -> int
read_bonds_securities() -> pl.DataFrame
get_security_description(secid, session=None) -> dict
```

Обертки над `history` для набора `bonds`. Полная история **всех** облигаций MOEX с 1997 года — все доски: старые EQOB/EQNB/EQOS (до 2016–2020), TQOB (гособлигации), TQCB (корпоративные), валютные TQOD/TQOE/TQOY/TQUD, TQRD. Колонки ISS: цены (`OPEN`, `LOW`, `HIGH`, `CLOSE`, `WAPRICE`, `LEGALCLOSEPRICE`), доходности (`YIELDCLOSE`, `YIELDATWAP`, `YIELDTOOFFER`), `DURATION` (дни), НКД `ACCINT`, `ZSPREAD`, `BEICLOSE`/`IRICPICLOSE` для ОФЗ-ИН, купон (`COUPONPERCENT`, `COUPONVALUE`), номинал и валюта, оферты и call/put-даты, `BONDTYPE`/`BONDSUBTYPE`, обороты. Погашенные выпуски остаются в истории. Выпуск может торговаться на нескольких досках в один день — для анализа фильтруйте `boards` (например `['TQOB', 'TQCB']`).

**Реестр карточек** `lake.bonds_securities` — строка на выпуск, включая погашенные: `ISIN`, `NAME`, `EMITTER_ID`, `REGNUMBER`, `ISSUEDATE`, `MATDATE`, `ISSUESIZE`, `FACEVALUE`/`INITIALFACEVALUE`, `FACEUNIT`, `COUPONFREQUENCY`, `TYPE`/`TYPENAME`, `BOND_TYPE`/`BOND_SUBTYPE`, `HASDEFAULT`, `HASTECHNICALDEFAULT`, `LISTLEVEL` и др., плюс `FETCHED` — дата запроса. Ночью добавляются до 500 новых выпусков. Флаги дефолта — текущий статус, как правило на уровне эмитента (все его действующие выпуски), а не история: у погашенных выпусков они не выставлены.

Первичная выгрузка — около 95 тыс. запросов, 6–7 часов: `python update_data.py --history-init bonds` (реестр карточек наполняется следом).

### Метрики облигаций

```python
calculate_ytm(price, face_value, coupon_rate, years_to_maturity, coupon_freq=2) -> float
calculate_duration(price, face_value, coupon_rate, years_to_maturity, ytm, coupon_freq=2) -> float
calculate_convexity(price, face_value, coupon_rate, years_to_maturity, ytm, coupon_freq=2) -> float
```

Учебные расчеты по упрощенной модели (равные купоны, без НКД и реального графика выплат): YTM бисекцией в диапазоне [−50%, 500%], модифицированные дюрация (годы) и выпуклость (годы²): dP/P ≈ −D·dy + 0.5·C·dy². В аналитике используйте биржевые `YIELDCLOSE` и `DURATION` из истории; модель — фоллбэк (так делает ноутбук `bond-market.py`).

---

## Фьючерсы FORTS

```python
update_futures_history(start=None, session=None, max_days=3000) -> int
repair_futures_history(session=None, calendar=None) -> int
read_futures_history(start=None, end=None, assets=None, columns=None) -> pl.DataFrame
```

Обертки над `history` для набора `futures`: история **всех** фьючерсных контрактов FORTS с 2002 года — `OPEN`, `LOW`, `HIGH`, `CLOSE`, расчетная цена `SETTLEPRICE`, открытый интерес `OPENPOSITION` и `OPENPOSITIONVALUE`, `VOLUME`, `VALUE`, `NUMTRADES`, `SWAPRATE`, базовый актив `ASSETCODE`, `SHORTNAME`. Истекшие контракты остаются в истории. Коды контрактов повторяются раз в 10 лет (`SiZ5` — декабрь 2015 и декабрь 2025), уникальна пара `date` + `SECID` (+ `BOARDID`); год контракта — в `SHORTNAME` (`Si-12.25`). `assets` фильтрует по базовому активу (строка или список). Счетчик `TOTAL` в ответе ISS бывает больше числа реально отдаваемых строк — особенность ISS. Первичная выгрузка — около 25 тыс. запросов: `python update_data.py --history-init futures`.

---

## Проверка качества данных (`quality`)

```python
quality.data_quality_report(days=30, div_folder=None, div_days=120, adj_jump=0.25,
                            check_iss=False, session=None) -> pl.DataFrame
quality.quality_summary(issues) -> str
quality.find_dividend_gap_candidates(df, div_folder=None, since=None, min_gap=0.04,
                                     market_returns=None, window_days=5, splits=None) -> pl.DataFrame
```

`data_quality_report` проверяет данные хранилища по торговому календарю IMOEX и возвращает замечания (`check`, `object`, `detail`); пустой результат — замечаний нет. Окно — последние `days` торговых дней (`None` — вся история), для дивидендов — `div_days`.

| check | Что значит |
|-------|------------|
| `index_stale` | индекс отстает от IMOEX |
| `stock_stale` | акция отстает от календаря; 20+ торговых дней без данных — кандидат в `metadata/delisted.csv` (с `check_iss=True` — со статусом ISS) |
| `stock_gaps` | пропущенные торговые даты в окне (бывают законные паузы торгов — сплит, редомициляция) |
| `adj_missing` | пустые `adj_close` / `market_cap` в окне |
| `adj_jump` | изменение `adj_close` за день больше `adj_jump` и расходится с изменением сплит-скорректированной цены больше чем на 5 п.п. — артефакт корректировки. Сильные движения самой цены ошибкой не считаются |
| `price_spike` | скачок цены больше `adj_jump` с разворотом на следующий день — возможна сбойная цена |
| `dividend_skipped` | дивиденд из CSV отброшен как неправдоподобный |
| `dividend_gap` | гэп открытия хуже −4%, не объясненный IMOEX, сплитом или дивидендом из CSV в пределах 5 дней, — кандидат в пропущенный дивиденд. Гэп в тот же день у 3+ бумаг помечается как возможное отраслевое движение |
| `bonds_stale` / `bonds_gaps` | история облигаций отстает от IMOEX или с пропусками в окне |
| `futures_stale` / `futures_gaps` | история фьючерсов отстает от IMOEX или с пропусками в окне |

При кандидатах `dividend_gap` обновите проект дивидендов (`python parse_all_dividends.py` в `../dividends`); `adj_close` пересчитается ночным шагом 2. Если выплаты нет и в источнике (закрытияреестров.рф), кандидат останется в отчете. `quality_summary` — одна строка итога для лога. Перенос на polars проверен сверкой: на реальных данных отчет совпал с прежним построчно.

---

## Скрипт update_data.py

Шаги по порядку:

| Шаг | Что делает | Отключить |
|-----|-----------|-----------|
| 1 | Акции: дозагрузка, сразу `adj_close` и капитализация, только изменившиеся строки (снятые с торгов пропускаются) | `--no-update` |
| 1b | Индексы IMOEX, MCFTR, RGBITR | `--no-index` |
| 1c | Облигации: хвост и пропуски истории, новые выпуски в реестр карточек (до 500) | `--no-bonds` |
| 1d | Ключевая ставка ЦБ | `--no-key-rate` |
| 1e | Фьючерсы: хвост и пропуски истории | `--no-futures` |
| 2 | Пересчет `adj_close` и капитализации по всей истории (после изменений дивидендов, срезов, реестра сплитов) — только изменившиеся строки | `--no-adj` / `--no-cap` |
| 3 | Проверка данных: замечания и строка итога | `--no-check` |
| 4 | Обслуживание хранилища: слияние файлов, снимки старше 30 дней | `--no-maintenance` |

Облигации и фьючерсы ночью только дообновляются: если набора в хранилище нет, шаг подсказывает команду первичной выгрузки и ничего не качает. Если хранилище недоступно (Postgres не запущен), шаги пишут предупреждение, а прогон продолжается.

Прочие опции:

| Опция | Описание |
|-------|----------|
| `--check` | Только проверка данных: без обновления, окно — год, статус ISS для отстающих бумаг |
| `--history-init` | Первичная выгрузка истории наборов через запятую: `bonds,futures` (многочасовая; для `bonds` следом наполняется реестр карточек) |
| `--history-start` | Начальная дата первичной выгрузки (по умолчанию — начало истории ISS: 1997 для облигаций, 2002 для фьючерсов) |
| `--rebuild` | Перескачать историю всех акций целиком |
| `--indexes` | Индексы через запятую (по умолчанию `IMOEX,MCFTR,RGBITR`) |
| `--div-folder` | Папка CSV дивидендов (по умолчанию `../dividends/data`) |
| `--metadata-file` | Excel с количеством акций |

Прежние флаги `--bonds-market-init` и `--futures-init` работают как синонимы `--history-init bonds` / `futures`.

Первичные выгрузки (многочасовые, запускать в фоне):

```bash
python update_data.py --history-init bonds,futures --no-update --no-index --no-key-rate --no-adj --no-check
```

Первой строкой скрипт пишет корень данных (`Данные: F:\moex-data`) — по ней в логе видно, подхватилась ли `MOEX_DATA_ROOT`. Вывод идет в UTF-8 и при перенаправлении в файл. Из кода: `from update_data import main; main(...)` — параметры повторяют опции (`do_update`, `do_indexes`, `do_bonds`, `do_key_rate`, `do_futures`, `do_adj_close`, `do_market_cap`, `do_check`, `do_maintenance`, `history_init`, `history_start`, `check_days`, `check_div_days`, `check_iss`, `rebuild`, `div_folder`, `metadata_file`, `index_tickers`).

### Ночной запуск

`update_data.bat` выбирает интерпретатор (`MOEX_PYTHON` → `H:\conda\envs\py312` → `python` из PATH) и проверяет, что в нем есть `polars` и `duckdb`; пауза в конце отключается переменной `MOEX_NO_PAUSE`. `scheduled_update.cmd` — обертка для планировщика: без паузы, вывод дописывается в `logs/update.log` (ротация после 5 МБ в `update.old.log`), код выхода передается планировщику. Задача Windows `MOEX data nightly` запускает ее вт–сб в 00:30: лимит 25 минут, пониженный приоритет, пропущенный запуск не догоняется (следующий докачает все сам).
