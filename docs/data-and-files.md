# Данные и файлы

## Структура каталогов

Код и реестры живут в папке проекта (Яндекс.Диск, git), рыночные данные — в корне `MOEX_DATA_ROOT` (на рабочей машине `F:\moex-data`, вне облачной синхронизации).

```
moexutils/                   # проект: F:\Yandex.Disk\FINANCE\moexutils
├── moexutils/               # пакет (pip install -e .): stocks, history, rates, cashflows, contracts,
│                            # quality, lake, iss, bondmath, backup, notify (docs/README.md)
├── pyproject.toml           # пакет, зависимости, extras notebooks/dev, настройки pytest и ruff
├── update_data.py           # пайплайн обновления (CLI)
├── update_data.bat          # запуск на Windows (выбор интерпретатора)
├── scheduled_update.cmd     # обертка для планировщика задач (лог в logs/)
├── marimo/bond-market.py    # ноутбук обзора данных (аналитические — в ../moex-analytics)
├── legacy/                  # архив: Jupyter-ноутбуки и скрипты на pandas (не запускаются)
├── scripts/                 # gen_iss_columns.py — справочник колонок ISS; reader_role.sql — роль moex_reader
├── tests/                   # офлайн pytest-тесты
├── docs/                    # документация
├── metadata/
│   ├── stock-index-base.xlsx  # количество акций по датам (капитализация)
│   ├── splits.csv             # реестр сплитов: ticker, date, ratio, kind
│   ├── renames.csv            # реестр переименований: old, new, date
│   ├── delisted.csv           # снятые с торгов: ticker, last_date, note
│   ├── key_rate.csv           # ключевая ставка ЦБ: date, rate
│   └── sectors.csv            # ticker, sector — секторный разрез
└── logs/                    # логи ночного обновления и выгрузок (не в git)
    ├── update.log
    └── backfill_*.log

moex-data/                   # данные: MOEX_DATA_ROOT = F:\moex-data (не в git, не в облаке)
├── lake/                    # файлы данных хранилища DuckLake (каталог — PostgreSQL moex_lake)
│   └── main/<таблица>/...   # stocks, stocks_adjusted, indexes, indexes_all, shares, shares_securities,
│                            # bonds, bonds_securities, bond_coupons, bond_amortizations, bond_offers,
│                            # futures, futures_contracts, futures_continuous, currency, currency_fixings,
│                            # ruonia, zcyc_params, zcyc_yields, zcyc_bonds, ref_splits, ref_renames,
│                            # ref_delisted, ref_key_rate, ref_sectors, empty_dates, update_runs, quality_log
├── backups/catalog/         # копии каталога хранилища moex_lake-ГГГГММДД-ччммсс.dump (pg_dump), 14 последних
├── data/, indexes/, bonds/, futures/  # прежние Parquet-файлы — пакетом не используются; удалить после
│                            # перевода проекта vectorbt на пакет (он еще читает их)
dividends/                   # соседний проект: F:\Yandex.Disk\FINANCE\dividends
├── data/<TICKER>.csv        # приведены к текущей акции — их читает moexutils
├── data/raw/<TICKER>.csv    # сырые значения с сайта
└── metadata/splits.json     # реестр сплитов проекта dividends (внешний реестр для moexutils)
```

**Почему данные вне Яндекс.Диска.** При многочасовых выгрузках и частой перезаписи файлов клиент Яндекс.Диска создавал конфликтные копии и подменял файлы старыми серверными версиями — терялись даты (так пострадала история фьючерсов, восстановлена объединением версий). По той же причине `.git` исключен из синхронизации. Данные можно заново скачать с биржи (часы), поэтому облачная копия им не обязательна — копия каталога хранилища делается каждую ночь в `backups/catalog`; код и реестры остаются в Яндекс.Диске и git.

Без `MOEX_DATA_ROOT` данные ищутся в папке проекта. Переменная задана для пользователя Windows постоянно; ее видят новые процессы, включая ночную задачу.

**Хранилище DuckLake.** Все рыночные данные — акции, индексы, облигации и их денежные потоки, фьючерсы, валюта, ставки, кривая ОФЗ — живут в таблицах DuckLake: каталог — база `moex_lake` в локальном PostgreSQL 17 (служба `postgresql-x64-17`), файлы данных — Parquet (zstd) в `F:\moex-data\lake`. Файлы хранилища вручную не трогать: какие из них актуальны, знает только каталог. Читать — через `lake.query(...)` или функции пакета (`stocks`, `history`, `rates`, `cashflows`, `contracts`), другим проектам — под ролью `moex_reader` ([контракт данных](data-contract.md)); снимки старше 30 дней удаляются ночным обслуживанием, более свежие позволяют откатиться (`SELECT ... FROM lake.bonds AT (VERSION => n)`).

---

## Таблицы, реестры, дивиденды

Таблицы и представления хранилища, их ключи, колонки и связи, реестры `metadata/` и формат CSV дивидендов описаны в [модели данных](data-model.md). Все поля, которые отдает биржа по каждому рынку, — в [справочнике колонок ISS](iss-columns.md).

## Логи

`logs/update.log` — ночное обновление (`scheduled_update.cmd`, ротация после 5 МБ в `update.old.log`); `logs/backfill_*.log` — первичные выгрузки (`--history-init`). В начале каждого прогона — строка `Данные: <корень>`, в конце — итог проверки данных.
