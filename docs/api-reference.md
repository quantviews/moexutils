# Справочник API

Пакет `moexutils` (`pip install -e .`), модули импортируются из него: `from moexutils import stocks, history, lake`.

| Модуль | Что в нем |
|--------|-----------|
| `stocks` | Акции и индексы (рабочий набор) в хранилище: загрузка из ISS, сплиты, переименования, снятые с торгов, сектора, `adj_close`, капитализация, ключевая и безрисковая ставки; копии реестров и `stocks_adjusted` для SQL |
| `history` | Рынки «все инструменты за дату» (облигации, фьючерсы, все акции, все индексы, валюта, фиксинги), реестры карточек бумаг |
| `rates` | RUONIA, кривая бескупонной доходности (КБД) |
| `cashflows` | Денежные потоки облигаций: купоны, амортизации, оферты |
| `contracts` | Реестр фьючерсных контрактов, перекодировка после повторного листинга, непрерывные ряды |
| `quality` | Проверка качества данных, история прогонов |
| `lake` | Хранилище DuckLake: подключение, SQL-запросы с результатом в polars, запись и синхронизация по ключу, снимки, представления, обслуживание |
| `iss` | Доступ к ISS: HTTP-сессия, разбор ответов в polars |
| `bondmath` | Доходность, дюрация, выпуклость облигации |
| `backup`, `notify` | Копия каталога хранилища, уведомления Windows |

Все функции чтения возвращают **polars** DataFrame; pandas в проекте не используется. Функции чтения принимают `as_of` — данные на момент снимка хранилища (см. [`lake.ref`](#хранилище-ducklake-lake)). Модель таблиц — [data-model.md](data-model.md), гарантии для потребителей — [data-contract.md](data-contract.md), все поля биржи — [iss-columns.md](iss-columns.md). Скрипт обновления — `update_data.py` (раздел в конце).

## Константы и инфраструктура

Пути не зависят от текущего рабочего каталога. **Рыночные данные** — в хранилище DuckLake: файлы в `<MOEX_DATA_ROOT>/lake` (переменная окружения `MOEX_DATA_ROOT`, на рабочей машине `F:\moex-data`; без нее — папка проекта), каталог — PostgreSQL. **Реестры** `metadata/` — в папке проекта и в git (пакет установлен в режиме `-e`, пути считаются от папки проекта). Переменные читаются при импорте модулей: процессы, запущенные до их установки, нужно перезапустить.

| Константа | Значение | Описание |
|-----------|----------|----------|
| `lake.DATA_ROOT`, `lake.LAKE_DATA_PATH` | `<MOEX_DATA_ROOT>`, `<MOEX_DATA_ROOT>/lake` | Корень данных, файлы хранилища |
| `lake.PG_HOST`, `lake.PG_PORT`, `lake.PG_DATABASE`, `lake.PG_USER` | `localhost`, `5432`, `moex_lake`, `moex` | Каталог хранилища (переопределяются `MOEX_PG_HOST`, `MOEX_PG_PORT`, `MOEX_PG_DATABASE`, `MOEX_PG_USER`) |
| `lake.SNAPSHOT_RETENTION_DAYS` | `30` | Сколько дней хранятся снимки |
| `lake.TABLE_KEYS`, `lake.VIEWS`, `lake.PARTITIONED_BY_YEAR` | — | Ключи таблиц, определения представлений, таблицы с разбиением по годам |
| `stocks.METADATA_FILE` | `<проект>/metadata/stock-index-base.xlsx` | Срезы числа акций |
| `stocks.SPLITS_FILE` | `<проект>/metadata/splits.csv` | Реестр сплитов |
| `stocks.EXTERNAL_SPLITS_FILE` | `<проект>/../dividends/metadata/splits.json` | Внешний реестр сплитов проекта dividends |
| `stocks.RENAMES_FILE` | `<проект>/metadata/renames.csv` | Реестр переименований |
| `stocks.DELISTED_FILE` | `<проект>/metadata/delisted.csv` | Снятые с торгов |
| `stocks.KEY_RATE_FILE` | `<проект>/metadata/key_rate.csv` | Ключевая ставка ЦБ |
| `stocks.SECTORS_FILE` | `<проект>/metadata/sectors.csv` | Отраслевой справочник |
| `stocks.DIVIDENDS_FOLDER` | `<проект>/../dividends/data` | CSV дивидендов |
| `stocks.DEFAULT_INDEXES` | `('IMOEX', 'MCFTR', 'RGBITR')` | Рабочие индексы |
| `stocks.T1_SETTLEMENT_DATE` | `2023-07-31` | Переход на расчеты T+1 (экс-дата дивидендов) |
| `history.DATASETS` | см. [История рынков](#история-рынков-history) | Наборы «все инструменты за дату» |
| `iss.ISS_URL`, `iss.ISS_TIMEOUT` | `https://iss.moex.com/iss`, `(10, 60)` | Адрес ISS; таймаут запроса: соединение, ответ (сек) |

**HTTP.** Все загрузчики ходят в ISS и на cbr.ru через `iss.make_session()` — `requests.Session` с таймаутом и повторами на сетевых сбоях, 429 и 5xx. Историю ISS отдает страницами по 100 строк, загрузчики листают все страницы. Типы колонок ответа берутся из метаданных ISS.

**Логи.** Сообщения идут через логгер `moexutils` (если логирование в приложении не настроено — в stdout). Приглушить: `logging.getLogger("moexutils").setLevel(logging.WARNING)`. Версия пакета — `moexutils.__version__`.

---

## Акции (`stocks`)

```python
stocks.read_stocks(tickers=None, start=None, end=None, merge_renames=True, split_adjusted=False,
                   columns=None, as_of=None) -> pl.DataFrame
stocks.list_tickers(include_delisted=True, as_of=None) -> list[str]
stocks.update_stocks(tickers=None, include_delisted=False, div_folder=None, rebuild=False,
                     session=None) -> int
stocks.recompute_stocks(tickers=None, div_folder=None, dry_run=False) -> pl.DataFrame
stocks.add_stock(ticker, start='2002-01-01', div_folder=None, session=None) -> int
stocks.fetch_stock(ticker, start, end=None, session=None) -> pl.DataFrame
```

- **`read_stocks`** — дневные данные из `lake.stocks`. `merge_renames` склеивает истории переименованных тикеров (исходный тикер строки — `source_ticker`; при фильтре по новому тикеру подтягиваются и старые); `split_adjusted` приводит цены к пост-сплитовой базе (после склейки: реестр сплитов записан на текущий тикер); `columns` — нужные колонки (`date`, `ticker` добавляются всегда). `as_of` — данные на момент снимка, реестры сплитов и переименований при этом текущие. Результат отсортирован по `ticker, date`.
- **`list_tickers`** — тикеры в хранилище; `include_delisted=False` — без снятых с торгов.
- **`update_stocks`** — дозагрузка из ISS с последней даты тикера (она перекачивается) до сегодня; `rebuild` — вся история с 2002 года. Для каждого тикера сразу пересчитываются `adj_close` и капитализация по всей истории; в хранилище пишутся только новые и изменившиеся строки — одной транзакцией. Снятые с торгов (`metadata/delisted.csv`) пропускаются (`include_delisted=True` — опросить и их). Шаг 1 `update_data.py`.
- **`recompute_stocks`** — пересчет `adj_close` и капитализации после изменения дивидендов, срезов числа акций или реестра сплитов; пишет только изменившиеся строки, `dry_run` — только вернуть их. Шаг 2.
- **`add_stock`** — новый тикер: вся история с `start`, сразу с расчетными колонками.
- **`fetch_stock`** — дневные данные из истории торгов ISS: на дату — строка режима с максимальным оборотом (главная доска), `close` — закрытие основной сессии (та же методика, что у индексов).

Колонки и их смысл — [data-model.md → lake.stocks](data-model.md#lakestocks--акции). Внутридневные свечи не поддерживаются. Все бумаги рынка акций со всеми полями ISS — набор `shares` в [`history`](#история-рынков-history).

## Индексы (`stocks`)

```python
stocks.read_index(ticker='IMOEX', start=None, end=None, as_of=None) -> pl.DataFrame
stocks.update_indexes(tickers=('IMOEX', 'MCFTR', 'RGBITR'), session=None) -> int
stocks.fetch_index(ticker, start, end=None, session=None) -> pl.DataFrame
```

Рабочие индексы в `lake.indexes`: `date, ticker, BOARDID, close, value_rub` (оборот), `volume`. `update_indexes` дозагружает с последней даты (без истории — с 2000 года) и пишет только новые и изменившиеся строки; шаг 1b. Даты IMOEX служат торговым календарем для проверок и докачки пропусков. Все индексы MOEX со всеми полями ISS — набор `indexes_all` в [`history`](#история-рынков-history).

## Корпоративные события и справочники (`stocks`)

```python
stocks.load_splits(splits_file=None, external_file=None) -> pl.DataFrame
stocks.adjust_for_splits(df, splits=None) -> pl.DataFrame
stocks.price_jump_matches(dates, prices, date, divisor) -> bool
stocks.load_renames(renames_file=None) -> pl.DataFrame
stocks.apply_renames(df, renames=None) -> pl.DataFrame
stocks.load_delisted(delisted_file=None) -> pl.DataFrame
stocks.load_sectors(sectors_file=None) -> pl.DataFrame
stocks.is_traded(ticker, session=None) -> Optional[bool]
```

**Сплиты.** Реестр `metadata/splits.csv` (`ticker, date, ratio, kind`) плюс внешний `../dividends/metadata/splits.json` (записи получают `kind=auto`; явная запись `splits.csv` в пределах 45 дней приоритетнее):

- **`price`** — в истории цен разрыв на дату: `adjust_for_splits` делит цены (`close`, `open`, `high`, `low`, `waprice`) до даты на `ratio`, объем умножает; `adj_close`, `value_rub`, `market_cap` не трогаются.
- **`shares`** — биржа пересчитала цены, но число акций в старых срезах метаданных в старой базе: число акций до даты делится на `ratio` (капитализация).
- **`auto`** — вид по данным (`price_jump_matches`): есть ценовой разрыв, соответствующий сплиту (допуск 2,5×) — ценовая поправка, нет — поправка числа акций.

`ratio` в ценовой семантике: дробление 1:10 → `10`, консолидация 100:1 → `0.01`.

**Переименования.** `metadata/renames.csv` (`old, new, date` — первый день нового тикера): `apply_renames` дает строкам старого тикера новый тикер, исходный — в `source_ticker`; строки старого тикера с даты переименования отбрасываются с предупреждением; цепочки A→B→C поддерживаются. `adj_close` старых имен при пересчете (`recompute_stocks`) считается по склеенной истории в базе итогового тикера: сплиты и дивиденды преемника распространяются на историю прежнего имени, стык непрерывен (дивиденды с одной датой реестра берутся из файла итогового тикера, несколько выплат на дату сохраняются). Конвертации с коэффициентом ≠ 1:1 в реестр не входят.

**Снятые с торгов.** `metadata/delisted.csv` (`ticker, last_date, note`): история хранится, но не обновляется. `is_traded` — торгуется ли акция хоть на одной доске рынка shares (флаг ISS `is_traded`; `None` — ISS бумагу не знает).

**Сектора.** `load_sectors` — `metadata/sectors.csv` (`ticker, sector`); нет файла — пустая таблица.

## Дивиденды, `adj_close`, капитализация (`stocks`)

```python
stocks.load_dividends(ticker, div_folder=None) -> pl.DataFrame
stocks.ex_dividend_pos(dates, record_date) -> int
stocks.adj_close(df, dividends=None, splits=None) -> tuple[pl.DataFrame, list]
stocks.load_shares(metadata_file=None) -> pl.DataFrame
stocks.market_cap(df, shares=None, splits=None) -> pl.DataFrame
stocks.enrich(df, div_folder=None, shares=None, splits=None) -> tuple[pl.DataFrame, list]
```

- **`adj_close`** (одна бумага) — цена с поправкой на дивиденды и сплиты в текущей базе. Методика и экс-дата (T+1 с 31.07.2023, раньше T+2; `ex_dividend_pos`) — [data-model.md → lake.stocks](data-model.md#lakestocks--акции). Возвращает также список отброшенных дивидендов `[(дата, сумма)]` — их показывает проверка данных.
- **`load_dividends`** — CSV проекта dividends: `closing_date`, `dividend_value` (> 0).
- **`load_shares`** — срезы числа акций из Excel (`ticker, date, shares`), кэш до изменения файла; читается через openpyxl.
- **`market_cap`** — `shares` (между срезами — последний, до первого — первый, с поправками сплитов) и `market_cap = close × shares`; нет данных — пустые колонки.
- **`enrich`** — `adj_close` и капитализация вместе.

## Ключевая ставка (`stocks`)

```python
stocks.load_key_rate(key_rate_file=None) -> pl.DataFrame
stocks.update_key_rate(key_rate_file=None, session=None) -> int
stocks.risk_free_monthly(dates, key_rate_file=None) -> pl.DataFrame
```

Ключевая ставка ЦБ в `metadata/key_rate.csv` (`date` — дата изменения, `rate` — % годовых; до 13.09.2013 — ставка рефинансирования). `update_key_rate` дописывает решения ЦБ после последней записи (таблица ставки на cbr.ru, разбор HTML через lxml; в файл — только даты изменения); шаг 1d. `risk_free_monthly` — месячная ставка в долях на даты (`date, rf`; порядок входа сохраняется, до первой записи — первое значение).

## Копии для SQL-потребителей (`stocks`)

```python
stocks.sync_registries() -> dict[str, tuple[int, int]]   # {таблица: (записано, удалено)}
stocks.update_adjusted() -> tuple[int, int]
stocks.REGISTRIES                                        # {'ref_splits': load_splits, ...}
```

- **`sync_registries`** — реестры `metadata/` → таблицы `ref_splits`, `ref_renames`, `ref_delisted`, `ref_key_rate`, `ref_sectors` через `lake.sync` (только изменения, удаленные из файла строки удаляются).
- **`update_adjusted`** — `lake.stocks_adjusted` = `read_stocks(split_adjusted=True)`: склейка переименований (`source_ticker` — исходный тикер) и цены в пост-сплитовой базе; через `lake.sync`.

Обе — шаг 2b `update_data.py`. Другие проекты читают их SQL-запросом без кода moexutils.

---

## Хранилище DuckLake (`lake`)

```python
lake.query(sql, params=None) -> pl.DataFrame
lake.write(table, df, key=None, retries=5, delete=None) -> int
lake.sync(table, df, key=None) -> tuple[int, int]          # (записано, удалено)
lake.changed_rows(old, new, key, rel_tol=1e-9) -> pl.DataFrame
lake.tables(con=None) -> list[str]
lake.views(con=None) -> list[str]
lake.ensure_views(replace=False) -> list[str]
lake.ref(table, as_of=None) -> str
lake.snapshots() -> pl.DataFrame
lake.connect(read_only=False, retries=5) -> duckdb.DuckDBPyConnection
lake.session(read_only=False)      # контекстный менеджер: подключение DuckDB с хранилищем `lake`
lake.maintenance(retention_days=30) -> None
lake.init() -> None
lake.pg_password(host=PG_HOST, port=PG_PORT, database=PG_DATABASE, user=PG_USER) -> str
```

Каталог (метаданные, снимки, схема) — PostgreSQL на localhost: база `moex_lake`, роль `moex` (служба Windows `postgresql-x64-17`, автозапуск). Данные — Parquet со сжатием zstd в `<MOEX_DATA_ROOT>/lake`. Одновременная работа писателя и читателей проверена (20 транзакций записи при трех параллельных читателях — без ошибок).

- **Пароль** берется из файла паролей PostgreSQL (`%APPDATA%\postgresql\pgpass.conf`, `~/.pgpass` в других ОС или `PGPASSFILE`) и передается во временный безымянный секрет DuckDB: встроенная в DuckDB libpq не читает pgpass и переменные окружения, а пароль в строке подключения попал бы в текст ошибок. Нет файла или строки — `lake.LakeConfigError`.
- **`query`** — SQL только на чтение, таблицы — `lake.<имя>`, результат — polars DataFrame.
- **`write`** — одна транзакция: `MERGE` по ключу таблицы (`TABLE_KEYS`; для `*_securities` — `SECID`; иначе параметр `key`): строки с тем же ключом обновляются, новые — добавляются; новые колонки добавляются автоматически (у ISS набор полей со временем расширяется), дубли ключа отклоняются. `delete` — ключи строк, которые удаляются в той же транзакции до записи (можно с пустым `df` — только удаление). Таблица создается при первой записи; таблицы из `PARTITIONED_BY_YEAR` (`bonds`, `futures`, `shares`, `indexes_all`, `currency`) разбиты по годам. На занятом каталоге — повтор с растущей паузой.
- **`changed_rows`** — строки `new`, которых нет в `old` или которые отличаются (числа — с относительным допуском `rel_tol`). Так все загрузчики пишут только изменения.
- **`sync`** — приводит таблицу к содержимому `df` одной транзакцией: пишет новые и изменившиеся строки, удаляет строки, ключей которых в `df` нет; ничего не изменилось — снимок не создается. Для производных таблиц и копий реестров.
- **`tables` / `views`** — имена таблиц и представлений хранилища.
- **`ensure_views`** — создает недостающие представления `VIEWS` (`bonds_ofz`, `bonds_corporate`); `replace=True` пересоздает все (после изменения их определений в коде); представление без исходных таблиц пропускается. Шаг 2b.
- **`ref`** — ссылка на таблицу для SQL: `lake."stocks"`, с `as_of` — на момент снимка: номер снимка (int) → `AT (VERSION => n)`, момент времени (datetime) или дата (`date` или `'YYYY-MM-DD'` — состояние на конец дня) → `AT (TIMESTAMP => ...)`. Доступны снимки не старше 30 дней.
- **`snapshots`** — список снимков: `snapshot_id`, `snapshot_time`, `changes`.
- **`connect` / `session`** — подключение DuckDB с присоединенным хранилищем под именем `lake`; `read_only=True` — для чтения.
- **`maintenance`** — слияние мелких файлов ежедневных дозаписей, удаление снимков старше `retention_days` (окно для отката) и неиспользуемых файлов; шаг 4.
- **`init`** — однократная настройка нового хранилища (сжатие zstd).
- **Тесты и работа без Postgres:** переменная `MOEX_LAKE_CATALOG` задает другой каталог, например файловый `ducklake:C:/tmp/catalog.ducklake` (один процесс-писатель).

Ключи всех таблиц — в [data-model.md](data-model.md). Пример SQL поверх хранилища:

```python
from moexutils import lake
lake.query('''
    SELECT date, median(ZSPREAD) AS zspread
    FROM lake.bonds_corporate
    WHERE BOARDID = 'TQCB' AND date >= DATE '2025-01-01'
    GROUP BY date ORDER BY date''')
# то же на момент снимка
lake.query(f"SELECT count(*) FROM {lake.ref('bonds', as_of='2026-09-25')}")
```

---

## История рынков (`history`)

```python
history.update(dataset, start=None, max_days=3000, session=None, flush_every=50) -> int
history.repair(dataset, session=None, calendar=None) -> int
history.read(dataset, start=None, end=None, secids=None, boards=None, columns=None,
             as_of=None) -> pl.DataFrame
history.dataset_dates(dataset) -> list[date]
history.empty_dates(dataset) -> set[date]
history.trading_calendar() -> list[date]
history.update_securities(dataset='bonds', max_new=500, session=None, flush_every=200) -> int
history.read_securities(dataset='bonds', as_of=None) -> pl.DataFrame
```

Наборы `history.DATASETS` — словарь `имя → Dataset(path, start, label, keep)`: путь рынка в ISS, начало истории, подпись для логов и выражение polars — какие строки ответа хранить (`None` — все). Имя набора = таблица хранилища, ключ `date + SECID + BOARDID`.

| Набор | Рынок ISS | Начало | Что |
|-------|-----------|--------|-----|
| `bonds` | `stock/markets/bonds` | 1997-01-01 | Все облигации, все доски, включая погашенные |
| `futures` | `futures/markets/forts` | 2002-01-01 | Все фьючерсы FORTS, включая истекшие |
| `shares` | `stock/markets/shares` | 1997-03-24 | Все акции, депозитарные расписки, паи и ETF, все доски |
| `indexes_all` | `stock/markets/index` | 1995-09-01 | Все индексы MOEX |
| `currency` | `currency/markets/selt` | 1997-06-02 | Валютный рынок; хранятся только строки со сделками (`NUMTRADES > 0`) |
| `currency_fixings` | `currency/markets/index` | 2019-08-01 | Валютные фиксинги (доска `FIXI`) |

Один постраничный запрос ISS на торговую дату отдает все инструменты рынка со **всеми колонками**; строки пишутся в таблицу хранилища. Типы колонок берутся из метаданных ответа ISS: числовые — Float64, остальные — строки (так одна колонка имеет один тип во все годы).

- **Хвост** — даты после последней сохраненной до сегодня (ночной прогон).
- **Бэкфилл** — только при явно заданном `start` раньше истории: даты от истории назад (при обрыве скачанное примыкает к истории); зазор в начале до 10 дней считается закрытым (праздники). Без истории — с `start` или начала набора.
- **Без дыр** — на сбое даты прогон останавливается, следующий запуск продолжит с нее.
- **Прогресс** — запись порциями по `flush_every` дат (многочасовая выгрузка не теряет результат); `max_days` ограничивает число дат за прогон.
- **`repair`** — докачка пропусков внутри истории по будням IMOEX (`trading_calendar`); даты, за которые ISS подтвержденно пуст, запоминаются в `lake.empty_dates` (`empty_dates`) и больше не запрашиваются.
- **Параллельная работа** — ночное обновление и идущая выгрузка могут писать одновременно: запись идемпотентна (MERGE по ключу), конфликты транзакций разрешает каталог.

`read` возвращает историю за период (`start`/`end` включительно); `secids` / `boards` — код или список кодов бумаг / режимов торгов; `columns` ускоряет чтение полной истории. Нет таблицы набора — `FileNotFoundError` с командой первичной выгрузки. `dataset_dates` — все сохраненные даты набора.

**Реестры карточек** `<набор>_securities` (сейчас `bonds_securities` и `shares_securities`) — строка на `SECID`, включая погашенные и снятые с торгов: все поля карточки ISS `/iss/securities/<SECID>` (блок `description`) плюс `FETCHED` — дата запроса. `update_securities` добавляет бумаги из истории набора, которых в реестре еще нет; `max_new` — лимит запросов за прогон (ночью 500, первичное наполнение — `None`); на сбое прогон останавливается. Флаги `HASDEFAULT`/`HASTECHNICALDEFAULT` — текущий статус, как правило на уровне эмитента (все его действующие выпуски), а не история: у погашенных выпусков они не выставлены.

**Облигации** (`bonds`). Доски: старые EQOB/EQNB/EQOS (до 2016–2020), TQOB (гособлигации), TQCB (корпоративные), валютные TQOD/TQOE/TQOY/TQUD, TQRD. Колонки ISS: цены (`OPEN`, `LOW`, `HIGH`, `CLOSE`, `WAPRICE`, `LEGALCLOSEPRICE`), доходности (`YIELDCLOSE`, `YIELDATWAP`, `YIELDTOOFFER`), `DURATION` (дни), НКД `ACCINT`, `ZSPREAD`, `BEICLOSE`/`IRICPICLOSE` для ОФЗ-ИН, купон (`COUPONPERCENT`, `COUPONVALUE`), номинал и валюта, оферты и call/put-даты, `BONDTYPE`/`BONDSUBTYPE`, обороты. Выпуск может торговаться на нескольких досках в один день — для анализа фильтруйте `boards` (например `['TQOB', 'TQCB']`) или читайте представления `bonds_ofz` / `bonds_corporate`. Первичная выгрузка — около 95 тыс. запросов, 6–7 часов.

**Фьючерсы** (`futures`). `OPEN`, `LOW`, `HIGH`, `CLOSE`, расчетная цена `SETTLEPRICE`, открытый интерес `OPENPOSITION` и `OPENPOSITIONVALUE`, `VOLUME`, `VALUE`, `NUMTRADES`, `SWAPRATE`, базовый актив `ASSETCODE`, `SHORTNAME`. Истекшие контракты остаются в истории. Коды контрактов повторяются раз в 10 лет, и при повторном листинге ISS **задним числом переименовывает старый контракт**: `SiZ5` декабря 2015 года теперь `SiZ5_2015` — и в реестре, и в истории торгов. Строки, загруженные до переименования, переводит на новый код `contracts.remap_futures_secids` (шаг 1e), поэтому `SECID` однозначно определяет контракт и совпадает с `futures_contracts.secid`. Счетчик `TOTAL` в ответе ISS бывает больше числа реально отдаваемых строк — особенность ISS. Первичная выгрузка — около 25 тыс. запросов.

**Прочие рынки** (`shares`, `indexes_all`, `currency`, `currency_fixings`) — шаг 1f; реестр `shares_securities` дополняется там же. Особенности данных — [data-model.md](data-model.md#рынки-все-инструменты-за-дату-shares-indexes_all-currency-currency_fixings).

### Метрики облигаций (`bondmath`)

```python
bondmath.calculate_ytm(price, face_value, coupon_rate, years_to_maturity, coupon_freq=2) -> float
bondmath.calculate_duration(price, face_value, coupon_rate, years_to_maturity, ytm, coupon_freq=2) -> float
bondmath.calculate_convexity(price, face_value, coupon_rate, years_to_maturity, ytm, coupon_freq=2) -> float
```

Учебные расчеты по упрощенной модели (равные купоны, без НКД и реального графика выплат): YTM бисекцией в диапазоне [−50%, 500%], модифицированные дюрация (годы) и выпуклость (годы²): dP/P ≈ −D·dy + 0.5·C·dy². В аналитике используйте биржевые `YIELDCLOSE` и `DURATION` из истории; модель — фоллбэк (так делает ноутбук `bond-market.py`).

---

## Ставки и кривая (`rates`)

```python
rates.update_ruonia(session=None) -> int
rates.read_ruonia(start=None, end=None, as_of=None) -> pl.DataFrame
rates.fetch_ruonia(start=None, end=None, session=None) -> pl.DataFrame
rates.update_zcyc(start=None, max_days=5000, session=None, flush_every=50) -> int
rates.read_zcyc(kind='params', start=None, end=None, as_of=None) -> pl.DataFrame
rates.fetch_zcyc(date, session=None) -> dict[str, pl.DataFrame]
```

- **RUONIA** — с cbr.ru (`RUONIA_URL`), с 11.01.2010 (`RUONIA_START`); в ISS RUONIA нет (RUSFAR — в `indexes_all`). `update_ruonia` берет всю историю одним запросом и пишет в `lake.ruonia` новые и пересмотренные строки (ЦБ может уточнить последние значения); пустой ответ — ошибка. `fetch_ruonia` — таблица с сайта без записи.
- **КБД** — кривая бескупонной доходности MOEX (`/iss/engines/stock/zcyc`) с 06.01.2014 (`ZCYC_START`). `update_zcyc` докачивает **по вчерашний день** (за сегодня ISS отдает промежуточную кривую): хвост после последней даты и, если `start` раньше истории, начало (назад); на сбое останавливается, скачанное сохраняется. Returns — число записанных дат. `fetch_zcyc` — кривая на дату: `{'params', 'yearyields', 'securities'}` (выходной — пустые таблицы). `read_zcyc(kind)`: `'params'` → `zcyc_params`, `'yields'` → `zcyc_yields`, `'bonds'` → `zcyc_bonds`.

Шаг 1g `update_data.py`; первичная выгрузка КБД — `--history-init zcyc`.

## Денежные потоки облигаций (`cashflows`)

```python
cashflows.update_cashflows(mode='window', session=None) -> dict[str, int]   # {таблица: записано}
cashflows.read_cashflows(kind='coupons', secids=None, start=None, end=None, as_of=None) -> pl.DataFrame
cashflows.fetch_block(block, start=None, till=None, session=None, max_pages=5000) -> pl.DataFrame
cashflows.prepare(block, df) -> pl.DataFrame
```

Источник — сводная выдача ISS по всем облигациям (`/iss/statistics/engines/stock/markets/bonds/bondization`), включая погашенные выпуски с 1997 года и будущие выплаты. Блоки (`BLOCKS`): `coupons` → `bond_coupons`, `amortizations` → `bond_amortizations`, `offers` → `bond_offers`.

- **`update_cashflows(mode)`**: `'window'` — потоки от сегодня −10 до +60 дней (`WINDOW_BACK`, `WINDOW_FORWARD`; ночью), `'future'` — все будущие потоки (по субботам), `'full'` — вся история (около 2,6 тыс. запросов; `--history-init cashflows`). Пишутся только новые и изменившиеся строки.
- **Удаление отмененных.** Запрошенный диапазон дат приходит целиком, поэтому купоны и амортизации в нем, которых биржа больше не отдает (отмененные, перенесенные), удаляются в той же транзакции; оферты — только при полной выгрузке (фильтр ISS по датам идет по `offerdate`, а у части оферт он пустой). Пустой ответ ISS ничего не удаляет.
- **`prepare`** — даты → Date (`0000-00-00` → null), без `value_rub`, у оферт — `offer_date`; строки без ключа и повторы ключа отбрасываются.
- **`read_cashflows`** — фильтр по бумагам (`secids`) и дате потока (`coupondate`, `amortdate`, `offer_date`), сортировка `secid`, дата.

Особенности данных (текущий номинал, пустые будущие купоны флоатеров) — [data-model.md](data-model.md#lakebond_coupons-lakebond_amortizations-lakebond_offers--денежные-потоки-облигаций).

## Фьючерсные контракты и непрерывные ряды (`contracts`)

```python
contracts.update_contracts(session=None) -> tuple[int, int]      # (записано, удалено)
contracts.read_contracts(assets=None, as_of=None) -> pl.DataFrame
contracts.fetch_contracts(session=None) -> pl.DataFrame
contracts.remap_futures_secids() -> int
contracts.update_continuous(assets=MAIN_ASSETS) -> tuple[int, int]
contracts.build_continuous(assets=MAIN_ASSETS, roll_days=ROLL_DAYS) -> pl.DataFrame
contracts.read_continuous(assets=None, start=None, end=None, as_of=None) -> pl.DataFrame
```

- **Реестр** `lake.futures_contracts` — все контракты с 2001 года одним запросом (`/iss/statistics/engines/futures/markets/forts/series?show_expired=1`), через `lake.sync`. `base_secid` — код без суффикса года (`SiZ5_2015` → `SiZ5`). `read_contracts(assets)` — фильтр по `asset_code` (строка или список), сортировка по активу и экспирации.
- **`remap_futures_secids`** — переводит строки `lake.futures` со старым кодом (`SiZ5`) на код, который биржа дала контракту после повторного листинга (`SiZ5_2015`), по датам обращения из реестра (`start_date` … `expiration_date`); старые ключи удаляются в той же транзакции. Returns — число перекодированных строк.
- **Непрерывные ряды** `lake.futures_continuous` по активам `MAIN_ASSETS` (`Si`, `Eu`, `CNY`, `RTS`, `MIX`, `MXI`, `BR`, `NG`, `GOLD`, `SILV`, `SBRF`, `GAZR`): на дату берется контракт с наибольшим открытым интересом (при равенстве — объемом) среди тех, до экспирации которых больше `ROLL_DAYS` = 7 календарных дней, и ряд никогда не возвращается к контракту с более ранней экспирацией. Вечные фьючерсы (экспирация `PERPETUAL` = 2100-01-01) исключены. `build_continuous` — расчет без записи, `update_continuous` — через `lake.sync`. Колонки и склейка — [data-model.md](data-model.md#lakefutures_continuous--непрерывные-ряды-фьючерсов).

Все три шага — в шаге 1e `update_data.py` после обновления `futures`.

---

## Проверка качества данных (`quality`)

```python
quality.data_quality_report(days=30, div_folder=None, div_days=120, adj_jump=0.25,
                            check_iss=False, session=None) -> pl.DataFrame
quality.quality_summary(issues) -> str
quality.find_dividend_gap_candidates(df, div_folder=None, since=None, min_gap=0.04,
                                     market_returns=None, window_days=5, splits=None) -> pl.DataFrame
quality.trading_calendar() -> list[date]
quality.previous_issues(mode, before) -> Optional[pl.DataFrame]
quality.new_issues(issues, previous) -> pl.DataFrame
quality.record_run(run_id, mode, warnings, issues, n_new) -> None
```

`data_quality_report` проверяет данные хранилища по торговому календарю IMOEX и возвращает замечания (`check`, `object`, `detail`); пустой результат — замечаний нет. Окно — последние `days` торговых дней (`None` — вся история), для дивидендов — `div_days`.

| check | Что значит |
|-------|------------|
| `index_stale` | рабочий индекс отстает от IMOEX |
| `stock_stale` | акция отстает от календаря; 20+ торговых дней без данных — кандидат в `metadata/delisted.csv` (с `check_iss=True` — со статусом ISS) |
| `stock_gaps` | пропущенные торговые даты в окне (бывают законные паузы торгов — сплит, редомициляция) |
| `adj_missing` | пустые `adj_close` / `market_cap` в окне |
| `adj_jump` | изменение `adj_close` за день больше `adj_jump` и расходится с изменением сплит-скорректированной цены больше чем на 5 п.п. — артефакт корректировки. Сильные движения самой цены ошибкой не считаются |
| `price_spike` | скачок цены больше `adj_jump` с разворотом на следующий день — возможна сбойная цена |
| `dividend_skipped` | дивиденд из CSV отброшен как неправдоподобный |
| `dividend_gap` | гэп открытия хуже −4%, не объясненный IMOEX, сплитом или дивидендом из CSV в пределах 5 дней, — кандидат в пропущенный дивиденд. Гэп в тот же день у 3+ бумаг помечается как возможное отраслевое движение |
| `<набор>_stale` / `<набор>_gaps` | для каждого набора `history.DATASETS` (`bonds`, `futures`, `shares`, `indexes_all`, `currency`, `currency_fixings`): история отстает от IMOEX или с пропусками в окне (кроме дат из `empty_dates`); `object` — подпись набора. Набора нет в хранилище — пропуск |

При кандидатах `dividend_gap` обновите проект дивидендов (`python parse_all_dividends.py` в `../dividends`); `adj_close` пересчитается ночным шагом 2. Если выплаты нет и в источнике (закрытияреестров.рф), кандидат останется в отчете. `quality_summary` — одна строка итога для лога.

**История прогонов.** `record_run` пишет итог прогона в `lake.update_runs` и замечания в `lake.quality_log`; `previous_issues` — замечания (`check`, `object`) последнего прогона того же режима до `before` (`None` — прогонов не было); `new_issues` — замечания, которых не было в прошлом прогоне (сравнение по `check`, `object`: `detail` меняется день ото дня). Использует `update_data.py`.

---

## Скрипт update_data.py

Шаги по порядку:

| Шаг | Что делает | Отключить |
|-----|-----------|-----------|
| 1 | Акции: дозагрузка, сразу `adj_close` и капитализация, только изменившиеся строки (снятые с торгов пропускаются) | `--no-update` |
| 1b | Индексы IMOEX, MCFTR, RGBITR (`--indexes`) | `--no-index` |
| 1c | Облигации: хвост и пропуски истории, новые выпуски в `bonds_securities` (до 500) | `--no-bonds` |
| 1d | Ключевая ставка ЦБ | `--no-key-rate` |
| 1e | Фьючерсы: хвост и пропуски истории; затем реестр контрактов, перекодировка после повторного листинга, непрерывные ряды | `--no-futures` |
| 1f | Прочие рынки: `shares`, `indexes_all`, `currency`, `currency_fixings` (хвост и пропуски), новые бумаги в `shares_securities` (до 500) | `--no-markets` |
| 1g | RUONIA, КБД, денежные потоки облигаций (по субботам — все будущие потоки, в остальные ночи — окно −10…+60 дней) | `--no-rates` |
| 2 | Пересчет `adj_close` и капитализации по всей истории (после изменений дивидендов, срезов, реестра сплитов) — только изменившиеся строки | `--no-adj` / `--no-cap` |
| 2b | Копии для SQL: реестры `ref_*`, `stocks_adjusted`, недостающие представления | `--no-derived` |
| 3 | Проверка данных: замечания и строка итога | `--no-check` |
| 4 | Обслуживание хранилища: слияние файлов, снимки старше 30 дней | `--no-maintenance` |
| 5 | Копия каталога хранилища (`backup.backup_catalog`) | `--no-backup` |

**Итог прогона.** После шагов (если проверка не отключена) итог пишется в `lake.update_runs`, замечания проверки — в `lake.quality_log` (модель — [data-model.md](data-model.md#lakeupdate_runs-lakequality_log--история-прогонов)). Сбой любого шага (исключение, недоступное хранилище, нет папки дивидендов) — строка `[WARN]` в логе, в конце «Готово со сбоями: N», код выхода 1 и уведомление Windows. Новые замечания проверки — те, которых (по паре `check`, `object`) не было в прошлом прогоне того же режима, — тоже уведомление; повторяющиеся не напоминают о себе каждую ночь. Падение вне шагов — уведомление «обновление упало» и трассировка в логе. `MOEX_NO_NOTIFY=1` отключает уведомления; в режиме `--check` их нет.

Наборы «все инструменты за дату» ночью только дообновляются: если набора в хранилище нет, шаг подсказывает команду первичной выгрузки и ничего не качает (реестр и непрерывные ряды фьючерсов, реестр `shares_securities` — тоже только при наличии истории). Если хранилище недоступно (Postgres не запущен), шаги пишут предупреждение, а прогон продолжается (и завершается с кодом 1).

Прочие опции:

| Опция | Описание |
|-------|----------|
| `--check` | Только проверка данных: без обновления (все шаги 1–2b, 4, 5 отключены), окно — год, статус ISS для отстающих бумаг |
| `--history-init` | Первичная выгрузка через запятую: наборы `bonds`, `futures`, `shares`, `indexes_all`, `currency`, `currency_fixings` (`history.update` без лимита дат; для `bonds` и `shares` следом наполняется реестр карточек), `zcyc` (КБД), `cashflows` (полная выгрузка потоков). Выполняется до шага 1 |
| `--history-start` | Начальная дата первичной выгрузки (по умолчанию — начало истории набора, для `zcyc` — 06.01.2014; на `cashflows` не влияет) |
| `--rebuild` | Перескачать историю всех акций целиком |
| `--indexes` | Рабочие индексы через запятую (по умолчанию `IMOEX,MCFTR,RGBITR`) |
| `--div-folder` | Папка CSV дивидендов (по умолчанию `../dividends/data`) |
| `--metadata-file` | Excel с количеством акций |

Прежние флаги `--bonds-market-init`, `--bonds-market-start`, `--futures-init`, `--futures-start` работают как синонимы `--history-init bonds` / `futures` и `--history-start`.

Первичные выгрузки (многочасовые, запускать в фоне):

```bash
python update_data.py --history-init bonds,futures --no-update --no-index --no-key-rate --no-markets --no-rates --no-adj --no-check
python update_data.py --history-init shares,indexes_all,currency,currency_fixings,zcyc,cashflows --no-update --no-index --no-bonds --no-key-rate --no-futures --no-adj --no-check
```

Первой строкой скрипт пишет корень данных (`Данные: F:\moex-data`) — по ней в логе видно, подхватилась ли `MOEX_DATA_ROOT`. Вывод идет в UTF-8 и при перенаправлении в файл. Из кода: `from update_data import main; main(...)` (из папки проекта) — параметры повторяют опции (`do_update`, `do_indexes`, `do_bonds`, `do_key_rate`, `do_futures`, `do_markets`, `do_rates`, `do_adj_close`, `do_market_cap`, `do_derived`, `do_check`, `do_maintenance`, `do_backup`, `history_init`, `history_start`, `check_days`, `check_div_days`, `check_iss`, `rebuild`, `div_folder`, `metadata_file`, `index_tickers`, `mode`, `notify_on`); возвращает код выхода.

### Резервная копия каталога (`backup`)

```python
backup.backup_catalog(folder=None, keep=14) -> str   # путь к новой копии
backup.list_backups(folder=None) -> list[str]         # от старых к новым
backup.verify(path) -> int                            # число таблиц с данными в копии
backup.pg_tool(name) -> str                           # путь к pg_dump / pg_restore
```

Каталог (база PostgreSQL `moex_lake`) — схема таблиц, снимки и список файлов данных: без него Parquet-файлы хранилища не собрать в таблицы. Шаг 5 делает `pg_dump -Fc` в `<MOEX_DATA_ROOT>/backups/catalog/moex_lake-ГГГГММДД-ччммсс.dump` (папка — `MOEX_BACKUP_DIR`), проверяет копию `pg_restore --list` и хранит 14 последних (`KEEP`); сбой — `backup.BackupError`. Копия делается после обслуживания, поэтому ссылается на файлы, которые остаются на диске. `pg_dump` ищется в `MOEX_PG_BIN`, PATH, затем в `C:\Program Files\PostgreSQL\<версия>\bin`; пароль он берет из `pgpass.conf` (`-w` — без запроса).

Восстановление (Postgres запущен, файлы `<MOEX_DATA_ROOT>/lake` на месте; берите последнюю копию — более старые могут ссылаться на файлы, уже удаленные обслуживанием):

```bat
rem база цела, каталог поврежден — перезаписать объекты каталога
pg_restore -h localhost -U moex -d moex_lake --clean --if-exists --no-owner F:\moex-data\backups\catalog\moex_lake-<дата>.dump
rem база потеряна (новая установка Postgres) — сначала роль и база от суперпользователя
psql -U postgres -c "CREATE ROLE moex LOGIN PASSWORD '...'"
createdb -U postgres -O moex moex_lake
pg_restore -h localhost -U moex -d moex_lake --no-owner F:\moex-data\backups\catalog\moex_lake-<дата>.dump
```

После восстановления в новую базу заново выполните `scripts/reader_role.sql` (роль `moex_reader`). Проверка: `python update_data.py --check`.

### Уведомления (`notify`)

```python
notify.toast(title, text) -> bool   # показано ли
```

Уведомление Windows через PowerShell (центр уведомлений); не Windows, `MOEX_NO_NOTIFY=1` или сбой показа — `False`, обновление не падает.

### Ночной запуск

`update_data.bat` выбирает интерпретатор (`MOEX_PYTHON` → `H:\conda\envs\py312` → `python` из PATH) и проверяет, что в нем импортируются `moexutils`, `polars` и `duckdb`; пауза в конце отключается переменной `MOEX_NO_PAUSE`. `scheduled_update.cmd` — обертка для планировщика: без паузы, вывод дописывается в `logs/update.log` (ротация после 5 МБ в `update.old.log`), код выхода передается планировщику. Задача Windows `MOEX data nightly` запускает ее вт–сб в 00:30: лимит 25 минут, пониженный приоритет, пропущенный запуск не догоняется (следующий докачает все сам).
