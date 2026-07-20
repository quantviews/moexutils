# Справочник API

## Константы (moex_utils)

| Константа | По умолчанию | Описание |
|-----------|--------------|----------|
| `DATA_FOLDER` | `"data"` | Каталог с подпапками по тикерам и Parquet-файлами |
| `METADATA_FILE` | `"metadata/stock-index-base.xlsx"` | Excel с количеством акций по датам |

---

## Загрузка с MOEX

### get_moex_stock

```python
get_moex_stock(ticker, start='2023-01-01', end=None, session=None, frequency=24) -> pd.DataFrame
```

Котировки бумаги. Параметры: `ticker`, `start`, `end` (YYYY-MM-DD), `session`, `frequency` (1/10/60/**24**/7/31/4 — минуты/час/день/неделя/месяц/квартал).

**Возвращает:** DataFrame с индексом `date`, колонками `value_rub`, `close`, `volume`, `ticker`.

---

### get_moex_index

```python
get_moex_index(ticker, start='2023-01-01', end=None, session=None) -> pd.DataFrame
```

История индекса (например IMOEX, RGBI). **Возвращает:** DataFrame с индексом `date`, колонками `volume`, `close`.

---

## Сохранение и чтение

### save_moex_stock

```python
save_moex_stock(ticker, start='2023-01-01', end=None, session=None, frequency=24,
                out_dir=DATA_FOLDER, calculate_market_cap_flag=True,
                metadata_file=METADATA_FILE) -> Optional[str]
```

Скачивает данные и сохраняет в `out_dir/<TICKER>/<TICKER>.parquet`. При `calculate_market_cap_flag=True` добавляет `shares`, `market_cap`.

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
                  metadata_file=METADATA_FILE) -> None
```

Дозагрузка с последней даты в файле до текущей даты. Пересчёт market_cap при `calculate_market_cap_flag=True`.

---

### update_all_stocks

```python
update_all_stocks() -> None
```

Обновляет все тикеры, для которых есть `data/<TICKER>/<TICKER>.parquet`.

---

### combine_moex_stocks

```python
combine_moex_stocks() -> pd.DataFrame
```

Объединяет все Parquet из `DATA_FOLDER` в один DataFrame.

---

## Метаданные и капитализация

### load_shares_data

```python
load_shares_data(metadata_file=METADATA_FILE) -> pd.DataFrame
```

Читает из Excel количество акций. Листы с датами в формате DD.MM.YYYY, колонки: `Code`, `Number of issued shares`.

---

### calculate_market_cap

```python
calculate_market_cap(df, ticker, metadata_file=METADATA_FILE) -> pd.DataFrame
```

Добавляет колонки `shares` и `market_cap`. В `df` нужны индекс-даты и колонка `close` или `value_rub`.

---

### add_market_cap_to_all_stocks

```python
add_market_cap_to_all_stocks(metadata_file=METADATA_FILE) -> None
```

Пересчитывает и сохраняет `shares` и `market_cap` для всех Parquet в `DATA_FOLDER`.

---

## Скорректированная цена (adj close)

### calculate_adj_close

```python
calculate_adj_close(df, div_folder) -> pd.DataFrame
```

Считает adj_close по дивидендам. В `df` — `ticker`, `close`, индекс-даты. В `div_folder` — CSV `<TICKER>.csv` с колонками `closing_date`, `dividend_value`.

---

### add_adj_close_to_all_stocks

```python
add_adj_close_to_all_stocks(div_folder) -> None
```

Для всех тикеров в `DATA_FOLDER` вычисляет `adj_close` и перезаписывает Parquet.

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
