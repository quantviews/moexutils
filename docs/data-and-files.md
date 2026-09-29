# Данные и файлы

## Структура каталогов

Код и реестры живут в папке проекта (Яндекс.Диск, git), рыночные данные — в корне `MOEX_DATA_ROOT` (на рабочей машине `F:\moex-data`, вне облачной синхронизации).

```
moexutils/                   # проект: F:\Yandex.Disk\FINANCE\moexutils
├── moex_utils.py            # ядро библиотеки
├── update_data.py           # пайплайн обновления (CLI)
├── update_data.bat          # запуск на Windows (выбор интерпретатора)
├── scheduled_update.cmd     # обертка для планировщика задач (лог в logs/)
├── marimo/                  # marimo-ноутбуки (аналитика и преподавание)
├── nb/                      # Jupyter-ноутбуки (исследования, примеры)
├── scripts/                 # аналитические скрипты
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
│   └── main/<таблица>/...   # bonds, futures, bonds_securities, empty_dates, stocks, indexes
├── data/                    # акции — рабочий источник до перевода в хранилище
│   ├── SBER/SBER.parquet
│   └── ...
├── indexes/                 # кэш индексов IMOEX, MCFTR, RGBITR — рабочий источник до перевода
├── bonds/, futures/         # прежнее файловое хранение — заменено хранилищем, не обновляется
dividends/                   # соседний проект: F:\Yandex.Disk\FINANCE\dividends
├── data/<TICKER>.csv        # приведены к текущей акции — их читает moexutils
├── data/raw/<TICKER>.csv    # сырые значения с сайта
└── metadata/splits.json     # реестр сплитов проекта dividends (внешний реестр для moexutils)
```

**Почему данные вне Яндекс.Диска.** При многочасовых выгрузках и частой перезаписи файлов клиент Яндекс.Диска создавал конфликтные копии и подменял файлы старыми серверными версиями — терялись даты (так пострадала история фьючерсов, восстановлена объединением версий). По той же причине `.git` исключен из синхронизации. Данные можно заново скачать с биржи, поэтому облачная копия им не нужна; код и реестры остаются в Яндекс.Диске и git.

Без `MOEX_DATA_ROOT` данные ищутся в папке проекта. Переменная задана для пользователя Windows постоянно; ее видят новые процессы, включая ночную задачу.

**Хранилище DuckLake.** Облигации и фьючерсы (а после следующей части миграции — и акции с индексами) живут в таблицах DuckLake: каталог — база `moex_lake` в локальном PostgreSQL 17 (служба `postgresql-x64-17`), файлы данных — Parquet (zstd) в `F:\moex-data\lake`. Файлы хранилища вручную не трогать: какие из них актуальны, знает только каталог. Читать — через `lake.query(...)` или функции `moex_utils`/`history`; снимки старше 30 дней удаляются ночным обслуживанием, более свежие позволяют откатиться (`SELECT ... FROM lake.bonds AT (VERSION => n)`).

---

## Акции: `data/<TICKER>/<TICKER>.parquet`

| Колонка | Описание |
|---------|----------|
| **индекс** `date` | Торговая дата (datetime64) |
| `open`, `low`, `high`, `close` | Цены основной сессии, руб.; `close` — официальное закрытие из ISS `/history` |
| `waprice` | Средневзвешенная цена дня |
| `volume` | Объем, шт. |
| `value_rub` | Оборот, руб. — не цена |
| `ticker` | Тикер |
| `adj_close` | Цена, скорректированная на дивиденды и сплиты, в текущей базе |
| `shares`, `market_cap` | Количество акций и капитализация (`close × shares`) |

Цены в файле — как их отдает ISS (без поправки на сплиты, если ISS ее не сделал); для доходностей применяйте `adjust_for_splits` или берите `adj_close`.

## Индексы: `indexes/<TICKER>.parquet`

Индекс `date`, колонки `close`, `volume`, `ticker`. Даты IMOEX (будни) — торговый календарь для проверок и докачки пропусков.

---

## Облигации: `lake.bonds`

Строка — выпуск на доске за торговую дату; ключ `date` + `SECID` + `BOARDID`. Разбита по годам. Все колонки истории ISS (набор со временем расширялся — в ранних годах часть колонок пустая):

| Группа | Колонки |
|--------|---------|
| Идентификация | `date`, `BOARDID`, `SECID`, `SHORTNAME` |
| Цены (% от номинала) | `OPEN`, `LOW`, `HIGH`, `CLOSE`, `WAPRICE`, `LEGALCLOSEPRICE`, `MARKETPRICE2`, `MARKETPRICE3`, `ADMITTEDQUOTE` |
| Доходность и риск | `YIELDCLOSE`, `YIELDATWAP`, `YIELDTOOFFER`, `YIELDLASTCOUPON`, `DURATION` (дни), `ZSPREAD`, `ZSPREADATWAPRICE`, `CALLOPTIONYIELD`, `CALLOPTIONDURATION` |
| ОФЗ-ИН | `BEICLOSE` (breakeven-инфляция), `IRICPICLOSE` (индекс потребительских цен) |
| Купон и номинал | `COUPONPERCENT`, `COUPONVALUE`, `ACCINT` (НКД), `FACEVALUE`, `FACEUNIT`, `FACEVALUE_TYPE`, `CURRENCYID`, `COUPON_DETAILS` |
| Даты (строки `YYYY-MM-DD`) | `MATDATE`, `OFFERDATE`, `BUYBACKDATE`, `CALLOPTIONDATE`, `PUTOPTIONDATE`, `LASTTRADEDATE`, `DATEYIELDFROMISSUER` |
| Обороты | `VALUE`, `VOLUME`, `NUMTRADES`, `MP2VALTRD`, `MARKETPRICE3TRADESVALUE`, `ADMITTEDVALUE` |
| Тип | `BONDTYPE`, `BONDSUBTYPE` |

Числовые колонки — DOUBLE, остальные — строки. Доски по периодам: до перехода на режим Т+ основные торги шли на EQOB, EQNB, EQOS, EQNO и др.; TQOB — с середины 2010-х, TQCB — примерно с 2019–2020 (точные даты: `SELECT BOARDID, min(date) FROM lake.bonds GROUP BY 1`).

### Реестр выпусков: `lake.bonds_securities`

Строка на `SECID`: карточка ISS, включая погашенные выпуски — `ISIN`, `NAME`, `SHORTNAME`, `EMITTER_ID`, `REGNUMBER`, `ISSUEDATE`, `MATDATE`, `ISSUESIZE`, `FACEVALUE`, `INITIALFACEVALUE`, `FACEUNIT`, `COUPONFREQUENCY`, `COUPONPERCENT`, `TYPE`, `TYPENAME`, `BOND_TYPE`, `BOND_SUBTYPE`, `HASDEFAULT`, `HASTECHNICALDEFAULT`, `LISTLEVEL` и др. (поля зависят от выпуска) и `FETCHED` — дата запроса. Флаги дефолта — текущий статус на уровне эмитента, не история.

---

## Фьючерсы: `lake.futures`

Строка — контракт за торговую дату; ключ `date` + `SECID` + `BOARDID`. Разбита по годам.

| Колонка | Описание |
|---------|----------|
| `date`, `BOARDID`, `SECID`, `SHORTNAME` | Дата, режим, код контракта (`SiZ5`), краткое имя с месяцем и годом (`Si-12.25`) |
| `ASSETCODE` | Базовый актив (`Si`, `RTS`, `BR`, `GD`, `MIX`, ...) |
| `OPEN`, `LOW`, `HIGH`, `CLOSE`, `WAPRICE` | Цены |
| `SETTLEPRICE` | Расчетная цена (для вариационной маржи) |
| `OPENPOSITION`, `OPENPOSITIONVALUE` | Открытый интерес: контракты и руб. |
| `VOLUME`, `VALUE`, `NUMTRADES`, `QTY` | Объемы и число сделок |
| `SWAPRATE`, `SWAPRATE_CURR`, `CHANGE` | Своп-ставка (для вечных фьючерсов), изменение |

Коды контрактов повторяются раз в 10 лет, год берите из `SHORTNAME`.

---

## Служебная таблица `lake.empty_dates`

Торговые (по IMOEX) даты, за которые ISS подтвержденно не вернул строк (`dataset, date`): при докачке пропусков они больше не запрашиваются.

---

## Метаданные

### stock-index-base.xlsx

Листы с датами `DD.MM.YYYY`, на листе колонки `Code` (тикер) и `Number of issued shares`, шапка на 4-й строке (`skiprows=3`). Между срезами — forward fill, до первого среза — backward fill.

### Реестры

| Файл | Колонки | Назначение |
|------|---------|------------|
| `splits.csv` | `ticker, date, ratio, kind` | Сплиты и консолидации (`price` / `shares` / `auto`) |
| `renames.csv` | `old, new, date` | Склейка историй переименованных тикеров |
| `delisted.csv` | `ticker, last_date, note` | Снятые с торгов — не опрашиваются при обновлении (статус подтвержден ISS) |
| `key_rate.csv` | `date, rate` | Ключевая ставка ЦБ; дописывается автоматически с cbr.ru |
| `sectors.csv` | `ticker, sector` | Секторный разрез в ноутбуках |

Корпоративные события вносятся только через реестры, данные руками не правятся. Папка `metadata/` в `.gitignore`: новый файл реестра добавляется в git принудительно (`git add -f`).

---

## Дивиденды: `../dividends/data/<TICKER>.csv`

Соседний проект `F:\Yandex.Disk\FINANCE\dividends` собирает историю выплат с сайта закрытияреестров.рф (`parse_all_dividends.py`) и приводит значения к текущей акции с учетом сплитов. moexutils читает `data/<TICKER>.csv`:

- **closing_date** — дата закрытия реестра; экс-дата выводится из нее по режиму расчетов (T+1 с 31.07.2023, раньше T+2);
- **dividend_value** — руб. на акцию (> 0); `year`, `period_type` — за какой период.

Привилегированные акции — в файле `<TICKER>P.csv` по правилам проекта dividends. Объявленные, но еще не наступившие выплаты в файле бывают — adj_close их игнорирует до экс-даты. Если сайт еще не внес выплату, проверка данных покажет ее как `dividend_gap`.
