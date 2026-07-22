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

Сообщения о ходе работы идут через логгер `moex_utils` (по умолчанию — в stdout, как обычный print; приглушить: `logging.getLogger("moex_utils").setLevel(logging.WARNING)`).

---

## Загрузка с MOEX

### get_moex_stock

```python
get_moex_stock(ticker, start='2023-01-01', end=None, session=None, frequency=24) -> pd.DataFrame
```

Котировки бумаги. Параметры: `ticker`, `start`, `end` (YYYY-MM-DD), `session`, `frequency` (1/10/60/**24**/7/31/4 — минуты/час/день/неделя/месяц/квартал).

**Методика:** дневные данные (`frequency=24`) берутся из официальной истории торгов (`/history`): `close` — закрытие **основной сессии**, только завершённые дни — та же методика, что у индексов. Остальные частоты идут через свечи ISS, которые включают вечернюю сессию и текущий незавершённый период.

**Возвращает:** DataFrame с индексом `date`, колонками `value_rub` (оборот за период, руб. — не цена), `close` (цена закрытия), `volume`, `ticker`.

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

Обновляет все тикеры, для которых есть `data/<TICKER>/<TICKER>.parquet`. Использует одну HTTP-сессию на весь прогон. При `calculate_market_cap_flag=False` пропускает пересчёт капитализации (так делает `update_data.py`, когда пересчёт всё равно выполняется отдельным шагом).

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

### calculate_ytm

```python
calculate_ytm(price, face_value, coupon_rate, years_to_maturity, coupon_freq=2) -> float
```

Доходность к погашению, %. `price` — в % от номинала. Решается бисекцией в диапазоне ставок [-50%, 500%] — сходится при любом номинале и сроке.

---

### calculate_duration

```python
calculate_duration(price, face_value, coupon_rate, years_to_maturity, ytm, coupon_freq=2) -> float
```

Модифицированная дюрация в годах (через дюрацию Маколея по заданной YTM).

---

### add_bond_metrics

```python
add_bond_metrics(df, params) -> pd.DataFrame
```

Добавляет `years_to_maturity`, `ytm`, `duration` к ряду цен. `params` — строка из `get_moex_bond_params` (нужны `FACEVALUE`, `COUPONPERCENT`, `MATDATE`). Цена берётся из `CLOSE`, при отсутствии — из `WAPRICE`. Если `MATDATE` отсутствует, метрики заполняются NaN.

---

## Скрипт update_data.py

Выполняет по порядку: обновление котировок → расчёт adj_close → расчёт market_cap.

**Командная строка:**

```bash
python update_data.py [--no-update] [--no-adj] [--no-cap] [--div-folder PATH] [--data-folder PATH] [--metadata-file PATH]
```

| Опция | Описание |
|-------|----------|
| `--no-update` | Не обновлять котировки с MOEX |
| `--no-adj` | Не пересчитывать adj_close |
| `--no-cap` | Не пересчитывать market_cap |
| `--no-index` | Не обновлять индексы |
| `--indexes` | Индексы через запятую (по умолчанию `IMOEX`) |
| `--rebuild` | Перескачать историю всех тикеров целиком (после смены методики данных) |
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
