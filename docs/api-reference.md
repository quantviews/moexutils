# Справочник API

Модули проекта:

| Модуль | Что в нем |
|--------|-----------|
| `moex_utils` | Прежний интерфейс: акции, индексы, корпоративные события, adj_close, капитализация, ставка ЦБ, проверка качества; облигации и фьючерсы — тонкие обертки над `history` |
| `lake` | Хранилище DuckLake: подключение, SQL-запросы с результатом в polars, запись по ключу, обслуживание |
| `history` | История рынков «все инструменты за дату» (облигации, фьючерсы) в хранилище, реестр карточек бумаг |
| `iss` | Доступ к ISS: HTTP-сессия, разбор ответов в polars |

**Миграция на polars и DuckLake идет по частям.** Облигации и фьючерсы уже в хранилище и отдаются как polars DataFrame; акции и индексы пока в Parquet-файлах и pandas (переводятся следующей частью). Скрипт обновления — `update_data.py` (раздел в конце).

## Константы и инфраструктура

Пути не зависят от текущего рабочего каталога — импорт из `nb/`, `scripts/`, `marimo/` работает одинаково. **Рыночные данные** лежат в корне `DATA_ROOT`: это переменная окружения `MOEX_DATA_ROOT` (на рабочей машине — `F:\moex-data`, вне папки Яндекс.Диска: синхронизация частых перезаписей портила файлы), а если она не задана — папка проекта. **Реестры** `metadata/` всегда в папке проекта (`BASE_DIR`) и в git. Переменная читается при импорте модуля: процессы, запущенные до ее установки (терминалы, VS Code, marimo), нужно перезапустить. Константы можно переопределить до вызова функций (`moex.DATA_FOLDER = ...`); дефолтные пути разрешаются в момент вызова.

| Константа | По умолчанию | Описание |
|-----------|--------------|----------|
| `DATA_ROOT` | `MOEX_DATA_ROOT` или `<проект>` | Корень рыночных данных |
| `DATA_FOLDER` | `<DATA_ROOT>/data` | Акции: `<TICKER>/<TICKER>.parquet` |
| `INDEXES_FOLDER` | `<DATA_ROOT>/indexes` | Кэш индексов: `<TICKER>.parquet` |
| `lake.LAKE_DATA_PATH` | `<DATA_ROOT>/lake` | Файлы данных хранилища DuckLake (облигации, фьючерсы) |
| `METADATA_FILE` | `<проект>/metadata/stock-index-base.xlsx` | Количество акций по датам (для капитализации) |
| `SPLITS_FILE` | `<проект>/metadata/splits.csv` | Реестр сплитов: `ticker, date, ratio, kind` |
| `EXTERNAL_SPLITS_FILE` | `<проект>/../dividends/metadata/splits.json` | Внешний реестр сплитов проекта dividends |
| `RENAMES_FILE` | `<проект>/metadata/renames.csv` | Реестр переименований тикеров |
| `DELISTED_FILE` | `<проект>/metadata/delisted.csv` | Снятые с торгов тикеры (не обновляются) |
| `KEY_RATE_FILE` | `<проект>/metadata/key_rate.csv` | История ключевой ставки ЦБ |
| `DIVIDENDS_FOLDER` | `<проект>/../dividends/data` | CSV дивидендов соседнего проекта dividends |
| `ISS_TIMEOUT` | `(10, 60)` | Таймаут запроса к ISS: соединение, ответ (сек) |

**HTTP.** Все загрузчики ходят в ISS через `make_session()` — `requests.Session` с таймаутом `ISS_TIMEOUT` и повторами на сетевых сбоях, 429 и 5xx (зависший запрос не останавливает прогон). Историю ISS отдает страницами по 100 строк (больше не разрешает), загрузчики листают все страницы.

**Запись файлов.** Parquet пишется атомарно (`.tmp` + `os.replace`); если целевой файл занят другим процессом (чтение из ноутбука, облачная синхронизация, антивирус), замена повторяется до 10 раз с паузой. Неизменившиеся файлы не перезаписываются.

**Логи.** Сообщения идут через логгер `moex_utils` (по умолчанию — в stdout). Приглушить: `logging.getLogger("moex_utils").setLevel(logging.WARNING)`.

---

## Акции

### get_moex_stock

```python
get_moex_stock(ticker, start='2023-01-01', end=None, session=None, frequency=24) -> pd.DataFrame
```

Котировки бумаги. `frequency`: 1 / 10 / 60 минут, **24** — день, 7 — неделя, 31 — месяц, 4 — квартал.

**Методика:** дневные данные берутся из официальной истории торгов (`/history`): `close` — закрытие **основной сессии**, только завершенные дни — та же методика, что у индексов. На каждую дату остается строка главной доски (с максимальным оборотом). Остальные частоты — свечи ISS (включают вечернюю сессию и текущий незавершенный период).

**Возвращает:** индекс `date`; `open`, `low`, `high`, `close`, `waprice` (средневзвешенная, только дневные), `volume`, `value_rub` (оборот, руб. — не цена), `ticker`. Ошибки API и разбора оборачиваются в `RuntimeError` (пустой ответ — `RuntimeError(... empty ...)`).

### save_moex_stock / read_moex_stock

```python
save_moex_stock(ticker, start='2023-01-01', end=None, session=None, frequency=24,
                out_dir=None, calculate_market_cap_flag=True, metadata_file=None) -> Optional[str]
read_moex_stock(ticker, start='2023-01-01', end=None, session=None) -> pd.DataFrame
```

`save_moex_stock` скачивает историю и сохраняет в `out_dir/<TICKER>/<TICKER>.parquet` (тикер приводится к верхнему регистру), при `calculate_market_cap_flag=True` добавляет `shares` и `market_cap`; при ошибке возвращает `None` и пишет в лог. `read_moex_stock` читает локальный файл, а если его нет — сначала скачивает.

### update_moex_stock

```python
update_moex_stock(ticker, session=None, calculate_market_cap_flag=True,
                  metadata_file=None, frequency=24, div_folder=None) -> None
```

Дозагрузка с последней даты файла. Последняя дата перекачивается: сырые колонки берутся из ответа ISS, а уже посчитанные (`adj_close`, `shares`, `market_cap`) сохраняются из файла. Если передан `div_folder`, `adj_close` пересчитывается сразу (одна запись файла за прогон вместо нескольких). Если данные не изменились, файл не перезаписывается. `frequency` должна совпадать с частотой, с которой файл сохранялся.

### update_all_stocks

```python
update_all_stocks(calculate_market_cap_flag=True, rebuild=False, div_folder=None,
                  include_delisted=False) -> None
```

Обновляет все тикеры, для которых есть файл в `DATA_FOLDER`, одной HTTP-сессией. Тикеры из `metadata/delisted.csv` пропускаются (`include_delisted=True` — опросить и их). `rebuild=True` перескачивает историю каждого тикера с 2002 года (после смены методики данных).

### combine_moex_stocks

```python
combine_moex_stocks(data_folder=None, merge_renames=True) -> pd.DataFrame
```

Все тикеры одним DataFrame. При `merge_renames=True` истории переименованных бумаг склеиваются (см. `apply_renames`), исходный тикер строки — в колонке `source_ticker`.

---

## Индексы

```python
get_moex_index(ticker, start='2023-01-01', end=None, session=None) -> pd.DataFrame
save_moex_index(ticker='IMOEX', start='2010-01-01', end=None, session=None) -> Optional[str]
read_moex_index(ticker='IMOEX') -> pd.DataFrame
update_moex_index(ticker='IMOEX', session=None) -> None
```

История индекса (IMOEX, MCFTR, RGBITR и др.): индекс `date`, колонки `volume`, `close`, `ticker`. Локальный кэш — `INDEXES_FOLDER/<TICKER>.parquet`; `update_moex_index` дозагружает с последней даты (без файла — история с 2010 года) и не переписывает файл, если новых данных нет. Шаг 1b `update_data.py` обновляет IMOEX, MCFTR, RGBITR. Даты IMOEX служат торговым календарем для проверок и докачки пропусков.

---

## Корпоративные события

### Сплиты: load_splits / adjust_for_splits

```python
load_splits(splits_file=None, external_file=None) -> pd.DataFrame
adjust_for_splits(df, splits_file=None) -> pd.DataFrame
```

Реестр `metadata/splits.csv` (`ticker, date, ratio, kind`):

- **`price`** — в скачанной истории цен есть разрыв на дату события. `adjust_for_splits` делит ценовые колонки (`close`, `open`, `high`, `low`, `waprice`) до даты на `ratio`, объем умножает. `adj_close`, `value_rub`, `market_cap` не трогаются.
- **`shares`** — ISS уже рестейтнул цены, но число акций в листах метаданных за старые даты в старой базе; `calculate_market_cap` делит число акций до даты на `ratio`.
- **`auto`** — тип определяется по данным: есть ценовой разрыв, соответствующий сплиту, — ценовая поправка, нет — поправка числа акций.

`ratio` для price/auto — в ценовой семантике: дробление 1:10 → `10`, консолидация 100:1 → `0.01`. Дополнительно подхватывается внешний реестр `../dividends/metadata/splits.json` (записи получают `kind=auto`); при дубликатах в пределах 45 дней приоритет у `splits.csv`. Перед расчетом доходностей: `moex.adjust_for_splits(moex.combine_moex_stocks())`.

### Переименования: load_renames / apply_renames

```python
load_renames(renames_file=None) -> pd.DataFrame
apply_renames(df, renames_file=None) -> pd.DataFrame
```

ISS `/history` отдает данные только по текущему коду, поэтому на дате переименования история рвется (TCSG→T, YNDX→YDEX, HHRU→HEAD, MAIL→VKCO, EONR→UPRO, MRKH→RSTI). Реестр `metadata/renames.csv` (`old, new, date` — первый день нового тикера) склеивает истории: строки старого тикера получают `ticker=new`, исходный тикер — в `source_ticker`; строки старого тикера с даты переименования отбрасываются с предупреждением. Цепочки A→B→C поддерживаются. Конвертации с коэффициентом ≠ 1:1 — не переименования и в реестр не входят.

### Снятые с торгов: load_delisted / iss_is_traded

```python
load_delisted(delisted_file=None) -> pd.DataFrame
iss_is_traded(ticker, session=None) -> Optional[bool]
```

Реестр `metadata/delisted.csv` (`ticker, last_date, note`): история этих бумаг остается в `data/` и участвует в анализе, но `update_all_stocks` их не опрашивает. `iss_is_traded` — торгуется ли акция хоть на одной доске рынка shares (флаг ISS `is_traded`; `None` — ISS бумагу не знает). Кандидатов в реестр показывает проверка данных (`stock_stale`).

---

## Дивиденды и скорректированная цена

### calculate_adj_close / add_adj_close_to_all_stocks

```python
calculate_adj_close(df, div_folder) -> pd.DataFrame
add_adj_close_to_all_stocks(div_folder) -> None
```

`adj_close` — цена, скорректированная **на дивиденды и сплиты**, в текущей (пост-сплитовой) базе. База — сплит-скорректированный `close`; дивидендный фактор — `1 − дивиденд / цена` на последнее закрытие с дивидендом. Дивиденд в CSV может быть в валюте своей даты или рестейтнут в текущую базу (ВТБ после консолидации) — берется база с правдоподобной доходностью (0–50%), иначе дивиденд пропускается с предупреждением, а список пропущенных попадает в `df.attrs['skipped_dividends']`.

**Экс-дата.** В CSV хранится дата закрытия реестра R. Экс-дата (первый день без дивиденда, день гэпа цены) выводится из режима расчетов: с 31.07.2023 (T+1) — сам R или последний торговый день перед ним, если R выходной; раньше (T+2) — торговый день перед R. Корректируются все цены строго до экс-даты. Объявленный дивиденд с отсечкой позже последней даты данных историю не корректирует.

Источник дивидендов — CSV соседнего проекта `../dividends` (`<TICKER>.csv`, колонки `closing_date`, `dividend_value`; сайт закрытияреестров.рф, значения приведены к текущей акции). Публичного эндпоинта дивидендов в ISS больше нет. `add_adj_close_to_all_stocks` пересчитывает все тикеры и пишет только изменившиеся файлы.

---

## Капитализация

```python
load_shares_data(metadata_file=None) -> pd.DataFrame
calculate_market_cap(df, ticker, metadata_file=None) -> pd.DataFrame
add_market_cap_to_all_stocks(metadata_file=None) -> None
```

`load_shares_data` читает количество акций из Excel (листы с датами `DD.MM.YYYY`, колонки `Code`, `Number of issued shares`) и кэширует результат до изменения файла. `calculate_market_cap` добавляет `shares` (между срезами — ffill, до первого — bfill, с поправками `kind=shares/auto` из реестра сплитов) и `market_cap = close × shares`. `add_market_cap_to_all_stocks` пишет только изменившиеся файлы.

---

## Безрисковая ставка

```python
load_key_rate(key_rate_file=None) -> pd.DataFrame
risk_free_monthly(dates, key_rate_file=None) -> pd.Series
update_key_rate(key_rate_file=None, session=None) -> int
```

История ключевой ставки ЦБ в `metadata/key_rate.csv` (`date` — дата изменения, `rate` — % годовых; до 13.09.2013 — ставка рефинансирования как прокси). `risk_free_monthly` — месячная ставка в долях на заданные даты (для Sharpe по избыточной доходности). `update_key_rate` дописывает решения ЦБ после последней записи (источник — таблица ставки на cbr.ru, в файл попадают только даты изменения); шаг 1d `update_data.py`.

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

Таблицы и ключи: `bonds`, `futures` — `date, SECID, BOARDID`; `bonds_securities` — `SECID`; `empty_dates` — `dataset, date`; `stocks`, `indexes` — `date, ticker` (копия на 28.09.2026, рабочий источник акций и индексов пока — Parquet-файлы).

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

## Проверка качества данных

```python
data_quality_report(days=30, div_folder=None, div_days=120, adj_jump=0.25,
                    check_iss=False, session=None) -> pd.DataFrame
quality_summary(issues) -> str
find_dividend_gap_candidates(df, div_folder=None, since=None, min_gap=0.04,
                             market_returns=None, window_days=5) -> pd.DataFrame
```

`data_quality_report` проверяет локальные данные по торговому календарю IMOEX и возвращает замечания (`check`, `object`, `detail`); пустой результат — замечаний нет. Окно — последние `days` торговых дней (`None` — вся история), для дивидендов — `div_days`.

| check | Что значит |
|-------|------------|
| `index_stale` | индекс (MCFTR, RGBITR) отстает от IMOEX |
| `stock_stale` | акция отстает от календаря; 20+ торговых дней без данных — кандидат в `metadata/delisted.csv` (с `check_iss=True` — со статусом ISS) |
| `stock_gaps` | пропущенные торговые даты в окне (бывают законные паузы торгов — сплит, редомициляция) |
| `adj_missing` | пустые `adj_close` / `market_cap` в окне |
| `adj_jump` | изменение `adj_close` за день больше `adj_jump` и расходится с изменением сплит-скорректированной цены больше чем на 5 п.п. — артефакт корректировки. Сильные движения самой цены ошибкой не считаются |
| `price_spike` | скачок цены больше `adj_jump` с разворотом на следующий день — возможна сбойная цена |
| `dividend_skipped` | дивиденд из CSV отброшен как неправдоподобный |
| `dividend_gap` | гэп открытия хуже −4%, не объясненный IMOEX, сплитом или дивидендом из CSV в пределах 5 дней, — кандидат в пропущенный дивиденд. Гэп в тот же день у 3+ бумаг помечается как возможное отраслевое движение |
| `bonds_stale` / `bonds_gaps` | история облигаций в хранилище отстает от IMOEX или с пропусками в окне |
| `futures_stale` / `futures_gaps` | история фьючерсов отстает от IMOEX или с пропусками в окне |

При кандидатах `dividend_gap` обновите проект дивидендов (`python parse_all_dividends.py` в `../dividends`) и пересчитайте `adj_close`. Если выплаты нет и в источнике (закрытияреестров.рф), кандидат останется в отчете. `quality_summary` — одна строка итога для лога.

---

## Скрипт update_data.py

Шаги по порядку:

| Шаг | Что делает | Отключить |
|-----|-----------|-----------|
| 1 | Акции: дозагрузка, сразу adj_close и market_cap (снятые с торгов пропускаются) | `--no-update` |
| 1b | Индексы IMOEX, MCFTR, RGBITR | `--no-index` |
| 1c | Облигации: хвост и пропуски истории в хранилище, новые выпуски в реестр (до 500) | `--no-bonds` |
| 1d | Ключевая ставка ЦБ | `--no-key-rate` |
| 1e | Фьючерсы: хвост и пропуски истории в хранилище | `--no-futures` |
| 2 | Сверка adj_close по дивидендам (пишет только изменившиеся файлы) | `--no-adj` |
| 3 | Сверка market_cap | `--no-cap` |
| 4 | Проверка данных: замечания и строка итога | `--no-check` |
| 5 | Обслуживание хранилища: слияние файлов, снимки старше 30 дней | `--no-maintenance` |

Облигации и фьючерсы ночью только дообновляются: если набора в хранилище нет, шаг подсказывает команду первичной выгрузки и ничего не качает. Если хранилище недоступно (Postgres не запущен), шаги 1c, 1e и 5 пишут предупреждение, остальное обновление идет как обычно.

Прочие опции:

| Опция | Описание |
|-------|----------|
| `--check` | Только проверка данных: без обновления, окно — год, статус ISS для отстающих бумаг |
| `--history-init` | Первичная выгрузка истории наборов через запятую: `bonds,futures` (многочасовая; для `bonds` следом наполняется реестр карточек) |
| `--history-start` | Начальная дата первичной выгрузки (по умолчанию — начало истории ISS: 1997 для облигаций, 2002 для фьючерсов) |
| `--rebuild` | Перескачать историю всех акций целиком |
| `--indexes` | Индексы через запятую (по умолчанию `IMOEX,MCFTR,RGBITR`) |
| `--div-folder` | Папка CSV дивидендов (по умолчанию `../dividends/data`) |
| `--data-folder` | Папка акций |
| `--metadata-file` | Excel с количеством акций |

Прежние флаги `--bonds-market-init` и `--futures-init` работают как синонимы `--history-init bonds` / `futures`.

Первичные выгрузки (многочасовые, запускать в фоне):

```bash
python update_data.py --history-init bonds,futures --no-update --no-index --no-key-rate --no-adj --no-cap --no-check
```

Первой строкой скрипт пишет корень данных (`Данные: F:\moex-data`) — по ней в логе видно, подхватилась ли `MOEX_DATA_ROOT`. Вывод идет в UTF-8 и при перенаправлении в файл. Из кода: `from update_data import main; main(...)` — параметры повторяют опции (`do_update`, `do_indexes`, `do_bonds`, `do_key_rate`, `do_futures`, `do_adj_close`, `do_market_cap`, `do_check`, `do_maintenance`, `history_init`, `history_start`, `check_days`, `check_div_days`, `check_iss`, `rebuild`, `div_folder`, `data_folder`, `metadata_file`, `index_tickers`).

### Ночной запуск

`update_data.bat` выбирает интерпретатор (`MOEX_PYTHON` → `H:\conda\envs\py312` → `python` из PATH) и проверяет наличие `apimoex`; пауза в конце отключается переменной `MOEX_NO_PAUSE`. `scheduled_update.cmd` — обертка для планировщика: без паузы, вывод дописывается в `logs/update.log` (ротация после 5 МБ в `update.old.log`), код выхода передается планировщику. Задача Windows `MOEX data nightly` запускает ее вт–сб в 00:30: лимит 25 минут, пониженный приоритет, пропущенный запуск не догоняется (следующий докачает все сам).
