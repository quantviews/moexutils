# Данные и файлы

## Структура каталогов

Код и реестры живут в папке проекта (Яндекс.Диск, git), рыночные данные — в корне `MOEX_DATA_ROOT` (на рабочей машине `F:\moex-data`, вне облачной синхронизации).

```
moexutils/                   # проект: F:\Yandex.Disk\FINANCE\moexutils
├── stocks.py, history.py, quality.py, lake.py, iss.py  # модули библиотеки (docs/README.md)
├── moex_utils.py            # фасад: реэкспорт функций модулей, облигации, фьючерсы
├── backup.py, notify.py     # копия каталога хранилища, уведомления Windows
├── update_data.py           # пайплайн обновления (CLI)
├── update_data.bat          # запуск на Windows (выбор интерпретатора)
├── scheduled_update.cmd     # обертка для планировщика задач (лог в logs/)
├── marimo/                  # marimo-ноутбуки (аналитика и преподавание)
├── legacy/                  # архив: Jupyter-ноутбуки и скрипты на pandas (не запускаются)
├── scripts/                 # служебные: перенос в хранилище, генератор справочника колонок ISS
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
│   └── main/<таблица>/...   # stocks, indexes, bonds, bonds_securities, futures, empty_dates,
│                            # update_runs, quality_log
├── backups/catalog/        # копии каталога хранилища (pg_dump), 14 последних
├── data/, indexes/, bonds/, futures/  # прежние Parquet-файлы — не используются, подлежат удалению
dividends/                   # соседний проект: F:\Yandex.Disk\FINANCE\dividends
├── data/<TICKER>.csv        # приведены к текущей акции — их читает moexutils
├── data/raw/<TICKER>.csv    # сырые значения с сайта
└── metadata/splits.json     # реестр сплитов проекта dividends (внешний реестр для moexutils)
```

**Почему данные вне Яндекс.Диска.** При многочасовых выгрузках и частой перезаписи файлов клиент Яндекс.Диска создавал конфликтные копии и подменял файлы старыми серверными версиями — терялись даты (так пострадала история фьючерсов, восстановлена объединением версий). По той же причине `.git` исключен из синхронизации. Данные можно заново скачать с биржи (часы), поэтому облачная копия им не обязательна — копия каталога хранилища делается каждую ночь в `backups/catalog`; код и реестры остаются в Яндекс.Диске и git.

Без `MOEX_DATA_ROOT` данные ищутся в папке проекта. Переменная задана для пользователя Windows постоянно; ее видят новые процессы, включая ночную задачу.

**Хранилище DuckLake.** Все рыночные данные — акции, индексы, облигации, фьючерсы — живут в таблицах DuckLake: каталог — база `moex_lake` в локальном PostgreSQL 17 (служба `postgresql-x64-17`), файлы данных — Parquet (zstd) в `F:\moex-data\lake`. Файлы хранилища вручную не трогать: какие из них актуальны, знает только каталог. Читать — через `lake.query(...)` или функции `moex_utils`/`history`; снимки старше 30 дней удаляются ночным обслуживанием, более свежие позволяют откатиться (`SELECT ... FROM lake.bonds AT (VERSION => n)`).

---

## Таблицы, реестры, дивиденды

Таблицы хранилища (`stocks`, `indexes`, `bonds`, `bonds_securities`, `futures`, `empty_dates`), их ключи, колонки и связи, реестры `metadata/` и формат CSV дивидендов описаны в [модели данных](data-model.md). Все поля, которые отдает биржа по каждому рынку, — в [справочнике колонок ISS](iss-columns.md).

## Логи

`logs/update.log` — ночное обновление (`scheduled_update.cmd`, ротация после 5 МБ в `update.old.log`); `logs/backfill_*.log` — первичные выгрузки. В начале каждого прогона — строка `Данные: <корень>`, в конце — итог проверки данных.
