# MOEX Utils

Загрузка, хранение и проверка данных Московской биржи (MOEX ISS): акции, индексы, облигации всего рынка, фьючерсы FORTS — плюс marimo-ноутбуки с аналитикой поверх локальных данных. Данные обновляются каждую ночь и проверяются на качество.

## Что есть в данных

| Рынок | Что хранится | Глубина |
|-------|--------------|---------|
| Акции | Дневные OHLC основной сессии, обороты, `adj_close` (дивиденды + сплиты), капитализация; ~100 тикеров, включая снятые с торгов | с 2003 (у большинства — с 2011) |
| Индексы | IMOEX, MCFTR, RGBITR | с 2000–2010 |
| Облигации | История **всех** выпусков всех досок (гос-, корпоративные, валютные, погашенные) со всеми полями ISS: цены, доходности, дюрация, НКД, Z-спред, оферты; реестр параметров каждого выпуска | с 1997 |
| Фьючерсы | История **всех** контрактов FORTS: цены, расчетная цена, открытый интерес, объемы, базовый актив | с 2002 |
| Ставки | Ключевая ставка ЦБ (безрисковая для Sharpe); до 13.09.2013 — ставка рефинансирования | с 2003 |

Дивиденды берутся из соседнего проекта `../dividends` (сайт закрытияреестров.рф). Опционы пока не выгружаются.

**Хранение.** Облигации и фьючерсы — в хранилище DuckLake (каталог PostgreSQL, файлы Parquet в `F:\moex-data\lake`), читаются как polars DataFrame или SQL. Акции и индексы пока в Parquet-файлах и pandas — идет поэтапная миграция всего проекта на polars и DuckLake.

## Установка

Python 3.11+ (CI проверяет 3.11 и 3.12; рабочее окружение — conda `py312`).

```bash
pip install -r requirements.txt            # ядро и тесты
pip install -r requirements-notebooks.txt  # + marimo-ноутбуки
```

## Быстрый старт

```python
import moex_utils as moex

# Акции и индексы (локальные файлы; при отсутствии — загрузка с MOEX)
sber = moex.read_moex_stock('SBER')                  # close, adj_close, market_cap, ...
imoex = moex.read_moex_index('IMOEX')
stocks = moex.adjust_for_splits(moex.combine_moex_stocks())  # все тикеры, склейка переименований

# Облигации (polars): весь рынок с 1997 года, все поля ISS
ofz_corp = moex.read_bonds_market(boards=['TQOB', 'TQCB'], start='2026-01-01')
crisis = moex.read_bonds_market(start='2008-01-01', end='2009-12-31')
issues = moex.read_bonds_securities()                # карточки выпусков, включая погашенные

# Фьючерсы (polars)
si = moex.read_futures_history(assets='Si', start='2024-01-01')

# SQL поверх хранилища
import lake
zspread = lake.query("SELECT date, median(ZSPREAD) AS z FROM lake.bonds "
                     "WHERE BOARDID = 'TQCB' GROUP BY date ORDER BY date")

# Качество данных
print(moex.quality_summary(moex.data_quality_report()))
```

## Обновление данных

```bash
python update_data.py          # полный цикл: акции → индексы → облигации → ставка ЦБ → фьючерсы → adj_close → market_cap → проверка
python update_data.py --check  # только проверка данных за год
```

На Windows — `update_data.bat` (сам находит нужный интерпретатор). Каждую ночь вт–сб в 00:30 задача планировщика `MOEX data nightly` запускает `scheduled_update.cmd`; лог — `logs/update.log`, в конце — строка «Проверка данных: …».

```powershell
Start-ScheduledTask -TaskName 'MOEX data nightly'     # запустить вручную
Get-ScheduledTaskInfo -TaskName 'MOEX data nightly'   # последний запуск и результат
```

Первичные многочасовые выгрузки (облигации всего рынка, фьючерсы, реестр выпусков) и все опции — в [справочнике API](docs/api-reference.md#скрипт-update_datapy).

## Ноутбуки (marimo)

| Ноутбук | Содержание |
|---------|------------|
| `stocks-performance.py` | Обзор рынка акций за период: доходности, капитализация, сектора |
| `ticker-analysis.py` | Анализ отдельной бумаги против индекса |
| `portfolio-analysis.py` | Портфельный анализ и оптимизация |
| `momentum-strategy.py` | Моментум-стратегия: walk-forward, vol scaling, trend filter |
| `arima-analysis.py` | ARIMA-модели доходностей |
| `bond-market.py` | Кривая ОФЗ, G-спреды корпоративных облигаций, RGBITR против IMOEX |

```bash
marimo edit marimo/bond-market.py
```

## Структура проекта

```
moexutils/
├── moex_utils.py        # основной интерфейс: акции, индексы, корп. события, качество данных
├── lake.py              # хранилище DuckLake (каталог Postgres, результаты — polars)
├── history.py           # история рынков в хранилище: облигации, фьючерсы, реестры бумаг
├── iss.py               # доступ к MOEX ISS: HTTP-сессия, разбор ответов в polars
├── update_data.py       # пайплайн обновления (CLI)
├── update_data.bat      # запуск на Windows
├── scheduled_update.cmd # обертка для планировщика задач
├── marimo/  nb/  scripts/
├── tests/               # офлайн pytest-тесты
├── docs/                # документация
├── metadata/            # реестры: сплиты, переименования, снятые с торгов, ставка ЦБ
└── logs/                # логи обновлений (не в git)

F:\moex-data/           # данные (MOEX_DATA_ROOT), вне git и облака: lake/ (хранилище), data/ и indexes/ (акции, индексы)
```

Рыночные данные лежат вне проекта, в папке из переменной окружения `MOEX_DATA_ROOT` (на рабочей машине `F:\moex-data`): облачная синхронизация частых перезаписей портила файлы. Без переменной данные ищутся в папке проекта.

Подробно о файлах и форматах — [docs/data-and-files.md](docs/data-and-files.md).

## Документация

- [Справочник API](docs/api-reference.md) — функции `moex_utils`, хранилища истории, проверка качества, `update_data.py`, ночной запуск.
- [Данные и файлы](docs/data-and-files.md) — каталоги, форматы Parquet, реестры, дивиденды.
- [План развития](development-plan.md) — что сделано и что дальше.

## Тесты

```bash
pytest -q
```

Тесты полностью офлайновые (ISS замокан): парсинг ответов, хранение и инкрементальное обновление, сплиты, переименования, дивиденды и экс-даты, капитализация, облигации и фьючерсы (пагинация, бэкфилл, блокировки, досчет пропусков), проверка качества данных, метрики облигаций. Запускаются в CI (GitHub Actions).

## Принципы

- Корпоративные события — только через реестры в `metadata/`, данные руками не правятся.
- Загрузчики не теряют данные молча: сбой — остановка и докачка в следующем прогоне.
- Файлы пишутся атомарно и только при изменениях (папка синхронизируется облаком).

## Лицензия

MIT.
