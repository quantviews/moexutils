# Данные и файлы

## Структура каталогов

```
moexutils/
├── moex_utils.py       # ядро библиотеки
├── update_data.py      # пайплайн обновления данных (CLI)
├── nb/                 # Jupyter-ноутбуки
├── scripts/            # обычные аналитические скрипты
├── marimo/             # marimo-ноутбуки (аналитика и преподавание)
├── tests/              # pytest-тесты
├── data/
│   ├── SBER/
│   │   └── SBER.parquet
│   ├── GAZP/
│   │   └── GAZP.parquet
│   └── ...
├── bonds/
│   └── <SECID>.parquet
├── indexes/
│   └── IMOEX.parquet       # локальный кэш индексов (update_data.py, шаг 1b)
├── metadata/
│   ├── stock-index-base.xlsx
│   ├── sectors.csv         # справочник тикер→сектор (для секторного разреза)
│   ├── splits.csv          # реестр сплитов: ticker,date,ratio,kind
│   └── renames.csv         # реестр переименований: old,new,date (склейка историй, источник — в source_ticker)
└── (опционально) ../dividends/data/   # CSV дивидендов для adj_close
    ├── SBER.csv
    └── ...
```

- **data/** — локальные котировки: подпапка на тикер, один Parquet на тикер.
- **bonds/** — локальные данные облигаций, один Parquet на SECID.
- **metadata/** — Excel с количеством акций по датам (для market_cap).
- Пути `data/`, `bonds/`, `metadata/` привязаны к папке модуля `moex_utils.py` и не зависят от рабочего каталога.
- Папка дивидендов задаётся параметром `div_folder` (в `update_data.py` по умолчанию `../dividends/data`).

---

## Формат Parquet (акции)

Файл: `data/<TICKER>/<TICKER>.parquet`.

| Колонка | Описание |
|---------|----------|
| **Индекс** | `date` (datetime64) |
| `close` | Цена закрытия, руб. |
| `value_rub` | Оборот торгов за период, руб. (поле `value` свечей ISS) — не цена |
| `volume` | Объём торгов |
| `ticker` | Тикер |
| `shares` | Количество акций (если считался market_cap) |
| `market_cap` | close × shares (если считался) |
| `adj_close` | Цена, скорректированная на дивиденды и сплиты, в текущей базе (если считалась) |

---

## Формат Parquet (облигации)

Файл: `bonds/<SECID>.parquet`.

| Колонка | Описание |
|---------|----------|
| **Индекс** | `TRADEDATE` (datetime64) |
| `CLOSE`, `WAPRICE` | Цена закрытия / средневзвешенная, % от номинала |
| `secid` | Идентификатор бумаги |
| `years_to_maturity`, `ytm`, `duration` | Метрики (если рассчитывались через `add_bond_metrics`) |

Прочие колонки — как в таблице history ISS MOEX.

---

## Метаданные: stock-index-base.xlsx

Используется для расчёта капитализации.

- Листы с именами в формате даты **DD.MM.YYYY**.
- На листе: колонки **Code** (тикер), **Number of issued shares**. При чтении используется `skiprows=3`.
- Между срезами дат — forward fill; до первого среза — backward fill.

---

## Дивиденды (CSV для adj_close)

- Один файл на тикер: `<TICKER>.csv` в папке `div_folder`.
- Обязательные колонки: **closing_date** (дата закрытия реестра), **dividend_value** (руб. на акцию, > 0).
- Расчёт adj_close идёт от последних дивидендов к ранним.
