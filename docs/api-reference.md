# Справочник API

## Константы (moex_utils)

Все пути привязаны к папке модуля `moex_utils.py` (константа `BASE_DIR`) и не зависят от текущего рабочего каталога — импорт из `nb/`, `scripts/`, `marimo/` работает одинаково.

| Константа | По умолчанию | Описание |
|-----------|--------------|----------|
| `DATA_FOLDER` | `<корень проекта>/data` | Каталог с подпапками по тикерам и Parquet-файлами |
| `METADATA_FILE` | `<корень проекта>/metadata/stock-index-base.xlsx` | Excel с количеством акций по датам |
| `BONDS_FOLDER` | `<корень проекта>/bonds` | Parquet-файлы облигаций (`<SECID>.parquet`) |
| `INDEXES_FOLDER` | `<корень проекта>/indexes` | Локальный кэш индексов (`<TICKER>.parquet`) |
| `SPLITS_FILE` | `<корень проекта>/metadata/splits.csv` | Реестр сплитов акций (ticker, date, ratio) |
| `KEY_RATE_FILE` | `<корень проекта>/metadata/key_rate.csv` | История ключевой ставки ЦБ (для безрисковой ставки) |
| `DELISTED_FILE` | `<корень проекта>/metadata/delisted.csv` | Реестр снятых с торгов тикеров (не обновляются) |
| `DIVIDENDS_FOLDER` | `<корень проекта>/../dividends/data` | CSV дивидендов соседнего проекта dividends |

Все загрузчики ходят в ISS через `make_session()` — `requests.Session` с таймаутом по умолчанию (`ISS_TIMEOUT` = 10 с на соединение, 60 с на ответ) и повторами на сетевых сбоях, 429 и 5xx. Parquet пишется атомарно (через `.tmp`), а шаги adj_close/market_cap не перезаписывают неизменившиеся файлы — папка проекта синхронизируется облаком, и массовая перезапись порождает конфликтные копии.

Сообщения о ходе работы идут через логгер `moex_utils` (по умолчанию — в stdout, как обычный print; приглушить: `logging.getLogger("moex_utils").setLevel(logging.WARNING)`).

---

## Загрузка с MOEX

### get_moex_stock

```python
get_moex_stock(ticker, start='2023-01-01', end=None, session=None, frequency=24) -> pd.DataFrame
```

Котировки бумаги. Параметры: `ticker`, `start`, `end` (YYYY-MM-DD), `session`, `frequency` (1/10/60/**24**/7/31/4 — минуты/час/день/неделя/месяц/квартал).

**Методика:** дневные данные (`frequency=24`) берутся из официальной истории торгов (`/history`): `close` — закрытие **основной сессии**, только завершённые дни — та же методика, что у индексов. Остальные частоты идут через свечи ISS, которые включают вечернюю сессию и текущий незавершённый период.

**Возвращает:** DataFrame с индексом `date`, колонками `value_rub` (оборот за период, руб. — не цена), `close` (цена закрытия), `open`, `low`, `high`, `waprice` (средневзвешенная цена — только для дневных данных из `/history`), `volume`, `ticker`.

---

### get_moex_index

```python
get_moex_index(ticker, start='2023-01-01', end=None, session=None) -> pd.DataFrame
```

История индекса (например IMOEX, RGBI). **Возвращает:** DataFrame с индексом `date`, колонками `volume`, `close`.

---

### save_moex_index / read_moex_index / update_moex_index

```python
save_moex_index(ticker='IMOEX', start='2010-01-01', end=None, session=None) -> Optional[str]
read_moex_index(ticker='IMOEX') -> pd.DataFrame
update_moex_index(ticker='IMOEX', session=None) -> None
```

Локальный кэш индексов в `INDEXES_FOLDER/<TICKER>.parquet` (DatetimeIndex, колонки `volume`, `close`, `ticker`; атомарная запись). `update_moex_index` дозагружает с последней сохранённой даты, при отсутствии файла скачивает историю с 2010 года. Обновляется шагом 1b в `update_data.py`.

---

## Сохранение и чтение

### save_moex_stock

```python
save_moex_stock(ticker, start='2023-01-01', end=None, session=None, frequency=24,
                out_dir=None, calculate_market_cap_flag=True,
                metadata_file=None) -> Optional[str]
```

Скачивает данные и сохраняет в `out_dir/<TICKER>/<TICKER>.parquet` (запись атомарная: tmp-файл + `os.replace`). Тикер нормализуется к верхнему регистру. `out_dir=None` / `metadata_file=None` означают «текущие `DATA_FOLDER` / `METADATA_FILE`» (разрешаются в момент вызова). При `calculate_market_cap_flag=True` добавляет `shares`, `market_cap`.

---

### read_moex_stock

```python
read_moex_stock(ticker, start='2023-01-01', end=None, session=None) -> pd.DataFrame
```

Читает локальный Parquet; при отсутствии файла вызывает `save_moex_stock` и затем читает.

---

### update_moex_stock

```python
update_moex_stock(ticker, session=None, calculate_market_cap_flag=True,
                  metadata_file=None, frequency=24) -> None
```

Дозагрузка с последней даты в файле до текущей даты (запись атомарная). `frequency` должна совпадать с частотой, с которой файл сохранялся изначально. Пересчёт market_cap при `calculate_market_cap_flag=True`.

---

### update_all_stocks

```python
update_all_stocks(calculate_market_cap_flag=True) -> None
```

Обновляет все тикеры, для которых есть `data/<TICKER>/<TICKER>.parquet`, кроме снятых с торгов из `metadata/delisted.csv` (`include_delisted=True` — опросить и их; см. `load_delisted`, `iss_is_traded`). Использует одну HTTP-сессию на весь прогон. При `calculate_market_cap_flag=False` пропускает пересчёт капитализации (так делает `update_data.py`, когда пересчёт всё равно выполняется отдельным шагом).

---

### combine_moex_stocks

```python
combine_moex_stocks(data_folder=None) -> pd.DataFrame
```

Объединяет все Parquet из `data_folder` (по умолчанию `DATA_FOLDER`) в один DataFrame.

---

## Сплиты

### load_splits / adjust_for_splits

```python
load_splits(splits_file=None) -> pd.DataFrame
adjust_for_splits(df, splits_file=None) -> pd.DataFrame
```

Реестр `metadata/splits.csv` (колонки: `ticker, date, ratio, kind`) описывает два вида поправок:

- **`kind=price`** — скачанная история цен содержит разрыв на дату события (пример: T, дробление 1:10 20.02.2026 — цена «упала» в 10 раз). `adjust_for_splits` приводит ценовые колонки (`close`, `adj_close`, `open/high/low`) до даты к пост-сплитовой базе: делит на `ratio`, объем умножает. `value_rub` и `market_cap` не трогаются.
- **`kind=shares`** — ISS уже рестейтнул историю цен в новую базу (разрыва нет), но число акций в листах метаданных за старые даты осталось в старой базе, и market_cap до события кратно врет (ВТБ ×5000 после консолидации 2024, ГМК и Транснефть ×100 после дроблений 2024). `calculate_market_cap` делит количество акций до даты события на `ratio`.

- **`kind=auto`** — тип поправки определяется по данным: если в ценовом ряду на дату события есть разрыв, соответствующий сплиту, — применяется ценовая поправка; если ряд гладкий (рестейтнут) — корректируется число акций. Ratio для auto — в ценовой семантике.

Семантика `ratio` для price/auto: дробление 1:10 → `10`, консолидация 100:1 → `0.01`; для shares — прямой делитель числа акций.

Дополнительно подхватывается **внешний реестр** соседнего проекта `../dividends/metadata/splits.json` (формат: `{"GMKN": [{"date": "...", "ratio": 100, "kind": "split"|"reverse"}]}`) — его записи получают `kind=auto`, поэтому работают корректно на любой копии данных независимо от того, рестейтнута ли история цен. При дубликатах (±45 дней) приоритет у явной записи из `splits.csv`.

Применяйте к результату `combine_moex_stocks()` перед расчетом доходностей:

```python
combined = moex.adjust_for_splits(moex.combine_moex_stocks())
```

При добавлении нового сплита допишите строку в `metadata/splits.csv`.

---

## Безрисковая ставка

### load_key_rate / risk_free_monthly

```python
load_key_rate(key_rate_file=None) -> pd.DataFrame
risk_free_monthly(dates, key_rate_file=None) -> pd.Series
update_key_rate(key_rate_file=None, session=None) -> int
```

История ключевой ставки ЦБ из `metadata/key_rate.csv` (`date` — дата изменения, `rate` — % годовых; до 13.09.2013 — ставка рефинансирования как прокси). `risk_free_monthly` возвращает месячную ставку в долях (ставка/12) на заданные даты с ffill между изменениями — используется для Sharpe по избыточной доходности. `update_key_rate` дописывает в файл решения ЦБ после последней записи (источник — таблица ставки на cbr.ru, в файл попадают только даты изменения); выполняется шагом 1d `update_data.py`, вручную файл править не нужно.

---

## Переименования тикеров

### load_renames / apply_renames

```python
load_renames(renames_file=None) -> pd.DataFrame
apply_renames(df, renames_file=None) -> pd.DataFrame
```

ISS `/history` отдает данные только по текущему secid — на дате переименования история бумаги рвется (TCSG→T, YNDX→YDEX, HHRU→HEAD, MAIL→VKCO, EONR→UPRO, MRKH→RSTI). Реестр `metadata/renames.csv` (`old,new,date` — первый торговый день нового тикера) склеивает истории: строки старого тикера получают `ticker=new`, а исходный тикер каждой строки сохраняется в колонке **`source_ticker`** — склейка остается прозрачной. Строки старого тикера с даты переименования отбрасываются с предупреждением. Цепочки (A→B→C) поддерживаются.

`combine_moex_stocks(merge_renames=True)` применяет склейку по умолчанию; `merge_renames=False` возвращает сырые тикеры. Конвертации с коэффициентом ≠1:1 (например, RSTI→FEES) в реестр не включаются — это не переименования.

---

## Метаданные и капитализация

### load_shares_data

```python
load_shares_data(metadata_file=None) -> pd.DataFrame
```

Читает из Excel количество акций. Листы с датами в формате DD.MM.YYYY, колонки: `Code`, `Number of issued shares`. Результат кэшируется в памяти до изменения файла (по mtime) — повторные вызовы Excel не перечитывают.

---

### calculate_market_cap

```python
calculate_market_cap(df, ticker, metadata_file=None) -> pd.DataFrame
```

Добавляет колонки `shares` и `market_cap`. В `df` нужны индекс-даты и колонка `close` или `value_rub`.

---

### add_market_cap_to_all_stocks

```python
add_market_cap_to_all_stocks(metadata_file=None) -> None
```

Пересчитывает и сохраняет `shares` и `market_cap` для всех Parquet в `DATA_FOLDER`.

---

## Скорректированная цена (adj close)

### calculate_adj_close

```python
calculate_adj_close(df, div_folder) -> pd.DataFrame
```

Считает `adj_close` — цену, скорректированную **на дивиденды и сплиты**, в текущей (пост-сплитовой) базе. Базой служит сплит-скорректированный ряд `close`; дивидендные факторы считаются через дивидендную доходность, причем база дивиденда в CSV определяется автоматически (валюта своей даты или рестейтнутая в текущую, как у ВТБ после консолидации) по правдоподобию доходности (0–50%); несогласующиеся дивиденды пропускаются с предупреждением. Повторно применять `adjust_for_splits` к `adj_close` не нужно — и он ее не трогает.

В `df` — `ticker`, `close`, индекс-даты. В `div_folder` — CSV `<TICKER>.csv` с колонками `closing_date`, `dividend_value`.

**Экс-дата.** В CSV хранится дата закрытия реестра R; экс-дата (первый день без дивиденда, день гэпа цены) выводится из режима расчетов: с 31.07.2023 (T+1) — сам R или последний торговый день перед ним, если R выходной; раньше (T+2) — торговый день перед R. Корректируются все цены строго до экс-даты, доходность дивиденда считается от последнего закрытия с дивидендом (до гэпа). До исправления 09.2026 корректировка захватывала и день гэпа — это давало ложный скачок доходности на следующий день и отбрасывало крупные спецдивиденды (SFIN 12.2025).

---

### add_adj_close_to_all_stocks

```python
add_adj_close_to_all_stocks(div_folder) -> None
```

Для всех тикеров в `DATA_FOLDER` вычисляет `adj_close` и перезаписывает Parquet.

---

## Облигации

### get_moex_bonds_list

```python
get_moex_bonds_list(segment='TQCB', session=None) -> pd.DataFrame
```

Список облигаций доски (TQCB — корпоративные, TQOB — государственные и др.). Фильтрация по доске идёт через путь `/boards/<board>/` ISS API.

---

### get_moex_bond_params

```python
get_moex_bond_params(secid, session=None) -> pd.DataFrame
```

Параметры облигации: `FACEVALUE` (номинал), `COUPONPERCENT` (купон, %), `MATDATE` (погашение), ISIN и др.

---

### get_moex_bond_prices

```python
get_moex_bond_prices(secid, start='2023-01-01', end=None, session=None) -> pd.DataFrame
```

Исторические цены. ISS отдаёт историю страницами (~100 строк) — функция листает все страницы по курсору и склеивает результат. **Возвращает:** DataFrame с индексом `TRADEDATE`, колонками истории торгов (в т.ч. `CLOSE`, `WAPRICE` — цены в % от номинала) и `secid`.

---

### save_moex_bond / read_moex_bond / update_moex_bond

```python
save_moex_bond(secid, start='2023-01-01', end=None, session=None) -> None
read_moex_bond(secid) -> pd.DataFrame
update_moex_bond(secid, session=None) -> None
```

Сохранение в `BONDS_FOLDER/<SECID>.parquet` (атомарная запись), чтение и инкрементальное обновление со следующего дня после последней сохранённой даты (дедупликация по дате, `keep='last'`).

---

### Вселенная облигаций

```python
save_bonds_params(segment='TQOB', session=None) -> pd.DataFrame
read_bonds_params() -> pd.DataFrame
download_bonds_universe(segment='TQOB', start='2014-01-01', session=None,
                        min_issue_size=None, max_issues=None) -> int
update_all_bonds(session=None, refresh_params=True) -> None
```

Фильтры `download_bonds_universe`: `min_issue_size` — минимальный объем выпуска в рублях (ISSUESIZE × FACEVALUE), обязателен на практике для TQCB; `max_issues` — максимум выпусков (крупнейшие по объему); погашенные не скачиваются. Реестр параметров при этом сохраняет полную доску.

`download_bonds_universe` — первичная выгрузка доски: снапшот параметров всех выпусков в `bonds/params.parquet` (колонка `segment`; при повторном снапшоте записи доски заменяются, чужие доски не трогаются) + история цен каждого выпуска. `update_all_bonds` — инкрементальное обновление всех сохранённых выпусков и параметров (шаг 1c в `update_data.py`). Запуск из CLI: `python update_data.py --bonds-init TQOB` (разово), дальше — штатный `update_data.py`.

---

### Мониторинг всех выпусков доски (по датам)

```python
update_bonds_market(segment='TQOB', start='2024-01-01', session=None, max_days=3000) -> int
read_bonds_market(segment=None) -> pd.DataFrame
update_bonds_market_all(session=None) -> None
repair_bonds_market(segment='TQOB', session=None, calendar=None) -> int
```

Основной механизм регулярного мониторинга **всех** выпусков доски. Вместо запроса истории по каждому SECID запрашивается история торгов **всей доски за дату** (`/history/.../boards/<segment>/securities?date=...`, с пагинацией) — один проход по недостающим торговым датам (выходные пропускаются). Данные дозаписываются в годовые файлы `bonds/market_<SEGMENT>/<YYYY>.parquet` — перезаписываются только годы, куда попали новые строки (колонки `date`, `SECID`, `SHORTNAME`, `CLOSE`, `LEGALCLOSEPRICE`, `YIELDCLOSE`, `DURATION`, `VALUE`, `VOLUME`, `MATDATE`, `FACEVALUE`, `FACEUNIT`, `COUPONPERCENT`, `segment`; дедупликация по `date`+`SECID`, атомарная запись).

Если `start` раньше уже сохраненной истории, недостающие даты в начале докачиваются (бэкфилл): повторный `--bonds-market-init TQOB,TQCB --bonds-market-start 2021-01-01` углубит историю, не перекачивая уже сохраненный диапазон. Новые размещения появляются в данных автоматически, погашенные выпуски перестают приходить сами — реестр следить не нужно. `read_bonds_market(segment=None, start=None, end=None)` без `segment` читает все доски одним DataFrame; `start`/`end` ограничивают период и читают только нужные годовые файлы. Старый единый файл `market_<SEGMENT>.parquet` при первом обращении автоматически разбивается по годам (со сверкой числа строк) и удаляется. `update_bonds_market_all` обновляет каждую доску, по которой уже есть мониторинг (вызывается в шаге 1c `update_data.py`). Инициализация: `python update_data.py --bonds-market-init TQOB,TQCB` (история с `--bonds-market-start`, по умолчанию 2024-01-01). Ноутбук `marimo/bond-market.py` использует мониторинг как основной источник (пофайловые истории — фоллбэк).

Надежность: при сбое загрузки даты прогон останавливается и сохраняет скачанное (дата не перескакивается, иначе осталась бы дыра); бэкфилл идет от сохраненной истории назад. `repair_bonds_market` докачивает пропущенные торговые даты внутри сохраненного диапазона (календарь — будни IMOEX из кэша индексов); `update_bonds_market_all` вызывает его после каждого обновления.

---

### calculate_ytm

```python
calculate_ytm(price, face_value, coupon_rate, years_to_maturity, coupon_freq=2) -> float
```

Доходность к погашению, %. `price` — в % от номинала. Решается бисекцией в диапазоне ставок [-50%, 500%] — сходится при любом номинале и сроке.

---

### calculate_duration / calculate_convexity

```python
calculate_duration(price, face_value, coupon_rate, years_to_maturity, ytm, coupon_freq=2) -> float
calculate_convexity(price, face_value, coupon_rate, years_to_maturity, ytm, coupon_freq=2) -> float
```

Модифицированная дюрация (в годах) и модифицированная выпуклость (в годах²) по заданной YTM: dP/P ≈ −D·dy + 0.5·C·dy².

---

### add_bond_metrics

```python
add_bond_metrics(df, params) -> pd.DataFrame
```

Добавляет `years_to_maturity`, `ytm`, `duration`, `convexity` к ряду цен. `params` — строка из `get_moex_bond_params` (нужны `FACEVALUE`, `COUPONPERCENT`, `MATDATE`). Цена берётся из `CLOSE`, при отсутствии — из `WAPRICE`. Если `MATDATE` отсутствует, метрики заполняются NaN.

---

## Проверка качества данных

```python
data_quality_report(days=30, div_folder=None, div_days=120, adj_jump=0.25, check_iss=False, session=None) -> pd.DataFrame
quality_summary(issues) -> str
find_dividend_gap_candidates(df, div_folder=None, since=None, min_gap=0.04, market_returns=None, window_days=5) -> pd.DataFrame
```

`data_quality_report` проверяет локальные данные по торговому календарю IMOEX и возвращает замечания (`check`, `object`, `detail`); пустой результат — замечаний нет. Окно — последние `days` торговых дней (`None` — вся история), для дивидендов — `div_days`. Проверки:

| check | Что значит |
|-------|------------|
| `index_stale` | индекс (MCFTR, RGBITR) отстает от IMOEX |
| `stock_stale` | акция отстает от календаря; 20+ торговых дней без данных — кандидат в `metadata/delisted.csv` (с `check_iss=True` — со статусом ISS) |
| `stock_gaps` | пропущенные торговые даты в окне (бывают и законные паузы торгов — сплит, редомициляция) |
| `adj_missing` | пустые `adj_close` / `market_cap` в окне |
| `adj_jump` | дневное изменение `adj_close` больше `adj_jump` и расходится с изменением сплит-скорректированной цены больше чем на 5 п.п. — артефакт корректировки. Сильные движения самой цены не считаются ошибкой |
| `price_spike` | скачок цены больше `adj_jump` с разворотом на следующий день — возможна сбойная цена |
| `dividend_skipped` | дивиденд из CSV отброшен `calculate_adj_close` как неправдоподобный (`df.attrs['skipped_dividends']`) |
| `dividend_gap` | гэп открытия хуже −4%, не объясненный IMOEX, сплитом или дивидендом из CSV в пределах 5 дней, — кандидат в пропущенный дивиденд. Если гэп в тот же день у 3+ бумаг, помечается как возможное отраслевое движение |
| `bonds_stale` / `bonds_gaps` | мониторинг облигаций отстает от IMOEX или с пропусками в окне |

Публичного эндпоинта дивидендов в ISS больше нет (`/iss/securities/<SECID>/dividends.json` отдает только описание бумаги), поэтому сверка дивидендов — по ценовым гэпам. Источник дивидендов — соседний проект `../dividends`; при кандидатах `dividend_gap` его нужно обновить (`python parse_all_dividends.py` в папке проекта) и пересчитать `adj_close`.

`quality_summary` — одна строка итога для лога. Проверка выполняется шагом 4 `update_data.py`; `python update_data.py --check` — только проверка, без обновления, окно — год, со статусом ISS для отстающих бумаг.

## Скрипт update_data.py

Выполняет по порядку: 1 котировки акций → 1b индексы → 1c облигации (выпуски + мониторинг досок) → 1d ключевая ставка ЦБ → 2 adj_close → 3 market_cap → 4 проверка данных (итог одной строкой в логе).

**Командная строка:**

```bash
python update_data.py [--no-update] [--no-index] [--no-bonds] [--no-key-rate] [--no-adj] [--no-cap] [--div-folder PATH] [--data-folder PATH] [--metadata-file PATH]
```

| Опция | Описание |
|-------|----------|
| `--no-update` | Не обновлять котировки с MOEX |
| `--no-adj` | Не пересчитывать adj_close |
| `--no-cap` | Не пересчитывать market_cap |
| `--no-index` | Не обновлять индексы |
| `--indexes` | Индексы через запятую (по умолчанию `IMOEX,MCFTR,RGBITR`) |
| `--rebuild` | Перескачать историю всех тикеров целиком (после смены методики данных) |
| `--no-bonds` | Не обновлять облигации |
| `--no-key-rate` | Не обновлять ключевую ставку ЦБ |
| `--no-check` | Не выполнять проверку данных в конце |
| `--check` | Только проверка данных: без обновления, окно — год, статус ISS |
| `--bonds-init` | Первичная выгрузка вселенной облигаций доски (например `TQOB`) |
| `--bonds-min-issue` | Мин. объем выпуска в млрд руб при `--bonds-init` (для TQCB рекомендуется 10) |
| `--bonds-market-init` | Инициализация мониторинга всех выпусков досок через запятую (например `TQOB,TQCB`) |
| `--bonds-market-start` | Начальная дата мониторинга при `--bonds-market-init` (по умолчанию 2024-01-01) |
| `--div-folder` | Папка с CSV дивидендов (по умолчанию `../dividends/data`) |
| `--data-folder` | Папка с Parquet |
| `--metadata-file` | Путь к Excel с метаданными |

**Вызов из кода:**

```python
from update_data import main

main(
    do_update=True,
    do_adj_close=True,
    do_market_cap=True,
    div_folder=None,      # иначе ../dividends/data
    data_folder=None,
    metadata_file=None,
)
```
