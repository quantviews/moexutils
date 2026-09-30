# MOEX Utils

Загрузка, хранение и проверка данных Московской биржи (MOEX ISS) и ставок Банка России: акции, индексы, облигации всего рынка с денежными потоками, фьючерсы FORTS, валютный рынок, RUONIA, кривая бескупонной доходности. Данные обновляются каждую ночь и проверяются на качество. Проект — слой данных: аналитика живет в отдельных проектах и читает данные отсюда (пакетом `moexutils` или SQL).

## Что есть в данных

| Рынок | Что хранится | Глубина |
|-------|--------------|---------|
| Акции (рабочий набор) | Дневные OHLC основной сессии, обороты, `adj_close` (дивиденды + сплиты), капитализация; ~100 тикеров, включая снятые с торгов; `stocks_adjusted` — со склейкой переименований и поправкой на сплиты | с 2003 (у большинства — с 2011) |
| Акции и фонды (весь рынок) | История **всех** бумаг рынка акций (акции, депозитарные расписки, паи и ETF) по всем доскам со всеми полями ISS; реестр карточек бумаг | с 1997 |
| Индексы | Рабочие IMOEX, MCFTR, RGBITR; **все** индексы MOEX (акций, облигаций, RUSFAR и др.) со всеми полями ISS | с 1995 (рабочие — с 2000–2010) |
| Облигации | История **всех** выпусков всех досок (гос-, корпоративные, валютные, погашенные) со всеми полями ISS: цены, доходности, дюрация, НКД, Z-спред; реестр параметров каждого выпуска; представления ОФЗ и корпоративных | с 1997 |
| Денежные потоки облигаций | Купоны, амортизации, оферты всех выпусков, включая погашенные и будущие выплаты | с 1997 |
| Кривая ОФЗ | Кривая бескупонной доходности MOEX (КБД): параметры, доходности на сроки 0,25–20 лет, ОФЗ кривой | с 06.01.2014 |
| Фьючерсы | История **всех** контрактов FORTS: цены, расчетная цена, открытый интерес, объемы; реестр контрактов с датами экспирации; непрерывные ряды по 12 основным активам | с 2002 |
| Валюта | Валютный рынок (selt, строки со сделками); валютные фиксинги MOEX | с 1997; фиксинги — с 2019 |
| Ставки | Ключевая ставка ЦБ (до 13.09.2013 — ставка рефинансирования); RUONIA с объемом, числом сделок и процентилями | с 2003; RUONIA — с 2010 |

Дивиденды берутся из соседнего проекта `../dividends` (сайт закрытияреестров.рф). Опционы пока не выгружаются.

**Хранение.** Все рыночные данные — в хранилище DuckLake (каталог PostgreSQL, файлы Parquet в `F:\moex-data\lake`), читаются как polars DataFrame или SQL; реестры корпоративных событий копируются туда же (`ref_*`). Модель таблиц — [docs/data-model.md](docs/data-model.md), гарантии для потребителей — [docs/data-contract.md](docs/data-contract.md), все поля биржи по рынкам — [docs/iss-columns.md](docs/iss-columns.md). Весь код — на polars, pandas в проекте не используется; прежние Jupyter-ноутбуки на pandas — в архиве [legacy/](legacy/README.md).

## Установка

Python 3.11+ (CI проверяет 3.11 и 3.12; рабочее окружение — conda `py312`). Пакет ставится в режиме разработки из папки проекта:

```bash
pip install -e .                 # пакет moexutils
pip install -e ".[notebooks]"    # + marimo и plotly для ноутбука обзора данных
pip install -e ".[dev]"          # + pytest и ruff
```

Другие проекты ставят его так же: `pip install -e F:\Yandex.Disk\FINANCE\moexutils`.

## Быстрый старт

```python
import polars as pl
from moexutils import stocks, history, rates, cashflows, contracts, lake, quality

# Акции и индексы
sber = stocks.read_stocks('SBER', start='2024-01-01')      # close, adj_close, market_cap, ...
all_stocks = stocks.read_stocks(split_adjusted=True)        # все тикеры: склейка переименований + сплиты
imoex = stocks.read_index('IMOEX')

# Рынки «все инструменты за дату»: bonds, futures, shares, indexes_all, currency, currency_fixings
ofz_corp = history.read('bonds', boards=['TQOB', 'TQCB'], start='2026-01-01')
crisis = history.read('bonds', start='2008-01-01', end='2009-12-31')
cards = history.read_securities('bonds')                    # карточки выпусков, включая погашенные
si = history.read('futures', start='2024-01-01').filter(pl.col('ASSETCODE') == 'Si')
rgbi = history.read('indexes_all', secids='RGBI', start='2024-01-01')

# Фьючерсы: реестр контрактов и непрерывные ряды
si_contracts = contracts.read_contracts('Si')
si_cont = contracts.read_continuous('Si', start='2020-01-01')   # settle_adj — склейка по отношению цен

# Ставки, кривая, денежные потоки облигаций
ruonia = rates.read_ruonia(start='2025-01-01')
curve = rates.read_zcyc('yields', start='2026-09-01')
coupons = cashflows.read_cashflows('coupons', secids='SU26238RMFS4')

# SQL поверх хранилища; as_of — данные на момент снимка
zspread = lake.query("SELECT date, median(ZSPREAD) AS z FROM lake.bonds_corporate "
                     "WHERE BOARDID = 'TQCB' GROUP BY date ORDER BY date")
sber_then = stocks.read_stocks('SBER', as_of='2026-09-25')

# Качество данных
print(quality.quality_summary(quality.data_quality_report()))
```

## Обновление данных

```bash
python update_data.py          # полный цикл: акции → индексы → облигации → ставка ЦБ → фьючерсы → прочие рынки → RUONIA, КБД, потоки → пересчет → копии для SQL → проверка → обслуживание → копия каталога
python update_data.py --check  # только проверка данных за год
```

На Windows — `update_data.bat` (сам находит нужный интерпретатор). Каждую ночь вт–сб в 00:30 задача планировщика `MOEX data nightly` запускает `scheduled_update.cmd`; лог — `logs/update.log`, в конце — строка «Проверка данных: …». Сбой шага или новые замечания к данным — уведомление Windows; итоги прогонов — в таблице `lake.update_runs`, копии каталога хранилища — в `F:\moex-data\backups\catalog`.

```powershell
Start-ScheduledTask -TaskName 'MOEX data nightly'     # запустить вручную
Get-ScheduledTaskInfo -TaskName 'MOEX data nightly'   # последний запуск и результат
```

Первичные многочасовые выгрузки (`--history-init bonds,futures,shares,indexes_all,currency,currency_fixings,zcyc,cashflows`) и все опции — в [справочнике API](docs/api-reference.md#скрипт-update_datapy).

## Ноутбук (marimo)

Здесь остается только обзор данных — `marimo/bond-market.py`: кривая ОФЗ, G-спреды корпоративных облигаций, RGBITR против IMOEX.

```bash
pip install -e ".[notebooks]"
marimo edit marimo/bond-market.py
```

Аналитические ноутбуки (`stocks-performance`, `ticker-analysis`, `portfolio-analysis`, `momentum-strategy`, `arima-analysis`) перенесены в отдельный проект `../moex-analytics` и читают данные через пакет `moexutils`.

## Структура проекта

```
moexutils/                   # проект: F:\Yandex.Disk\FINANCE\moexutils
├── moexutils/               # пакет
│   ├── stocks.py            # акции и индексы: загрузка, сплиты, переименования, adj_close, капитализация, ставка ЦБ, копии реестров для SQL
│   ├── history.py           # рынки «все инструменты за дату»: облигации, фьючерсы, все акции, все индексы, валюта, фиксинги; реестры бумаг
│   ├── rates.py             # RUONIA (cbr.ru), кривая бескупонной доходности (КБД)
│   ├── cashflows.py         # денежные потоки облигаций: купоны, амортизации, оферты
│   ├── contracts.py         # реестр фьючерсных контрактов, коды после повторного листинга, непрерывные ряды
│   ├── quality.py           # проверка качества данных, история прогонов
│   ├── lake.py              # хранилище DuckLake (каталог Postgres, результаты — polars), снимки, представления
│   ├── iss.py               # доступ к MOEX ISS: HTTP-сессия, разбор ответов в polars
│   ├── bondmath.py          # доходность, дюрация, выпуклость облигации (упрощенная модель)
│   └── backup.py, notify.py # копия каталога хранилища (pg_dump), уведомления Windows
├── pyproject.toml           # пакет, зависимости, extras notebooks/dev, настройки pytest и ruff
├── update_data.py           # пайплайн обновления (CLI)
├── update_data.bat          # запуск на Windows
├── scheduled_update.cmd     # обертка для планировщика задач
├── marimo/bond-market.py    # обзор данных
├── scripts/                 # генератор справочника колонок ISS, роль moex_reader (SQL)
├── legacy/                  # архив: Jupyter-ноутбуки и скрипты на pandas (не запускаются)
├── tests/                   # офлайн pytest-тесты
├── docs/                    # документация
├── metadata/                # реестры: сплиты, переименования, снятые с торгов, ставка ЦБ, сектора, число акций
└── logs/                    # логи обновлений (не в git)

F:\moex-data/                # данные (MOEX_DATA_ROOT), вне git и облака: lake/ (хранилище), backups/catalog/ (копии каталога)
```

Рыночные данные лежат вне проекта, в папке из переменной окружения `MOEX_DATA_ROOT` (на рабочей машине `F:\moex-data`): облачная синхронизация частых перезаписей портила файлы. Без переменной данные ищутся в папке проекта.

Подробно о файлах и форматах — [docs/data-and-files.md](docs/data-and-files.md).

## Документация

- [Модель данных](docs/data-model.md) — таблицы хранилища, ключи, колонки, связи, реестры, какие шаги что пишут.
- [Контракт данных](docs/data-contract.md) — как читать, уровни стабильности таблиц, данные на момент в прошлом.
- [Колонки данных MOEX](docs/iss-columns.md) — все поля ISS по рынкам (акции, облигации, индексы, фьючерсы, опционы, валюта, фиксинги) и карточкам бумаг, с отметкой, что мы храним.
- [Справочник API](docs/api-reference.md) — функции модулей пакета, проверка качества, `update_data.py`, ночной запуск.
- [Данные и файлы](docs/data-and-files.md) — каталоги, хранилище, логи.
- [План развития](development-plan.md) — что сделано и что дальше.
- [Изменения](CHANGELOG.md) — версии пакета.

## Тесты

```bash
ruff check .
pytest -q
```

Тесты полностью офлайновые (ISS и cbr.ru замоканы): разбор ответов ISS; хранилище на временном файловом каталоге DuckLake (Postgres не нужен) — дозапись по ключу, удаление, синхронизация, снимки, представления; сплиты, переименования, дивиденды и экс-даты, капитализация, реестры `ref_*` и `stocks_adjusted`; рынки «все инструменты за дату» (пагинация, бэкфилл, досчет пропусков); RUONIA, КБД, денежные потоки облигаций, реестр фьючерсов, перекодировка контрактов, непрерывные ряды; проверка качества данных, история прогонов, копия каталога и уведомления, метрики облигаций. CI (GitHub Actions, Python 3.11 и 3.12): `pip install -e ".[dev]"`, `ruff check .`, `pytest -q`.

## Принципы

- Корпоративные события — только через реестры в `metadata/`, данные руками не правятся.
- Загрузчики не теряют данные молча: сбой — остановка и докачка в следующем прогоне.
- В хранилище пишутся только новые и изменившиеся строки, одной транзакцией на запись.

## Лицензия

MIT.
