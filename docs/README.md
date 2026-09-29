# Документация moexutils

Библиотека и пайплайн данных Московской биржи (MOEX ISS): акции и индексы, облигации всего рынка с 1997 года (включая корпоративные, валютные и погашенные выпуски), все фьючерсы FORTS с 2002 года, ключевая ставка ЦБ; скорректированная цена (adj close), капитализация, проверка качества данных. Обзор проекта — в [README](../README.md).

## Разделы

- [Справочник API](api-reference.md) — функции `moex_utils` по рынкам, хранилища истории по датам, проверка качества, скрипт `update_data.py` и ночной запуск.
- [Данные и файлы](data-and-files.md) — структура каталогов, форматы Parquet (акции, облигации, фьючерсы), реестры `metadata/`, дивиденды соседнего проекта.
- [План развития](../development-plan.md) — состояние и ближайшие шаги.

## Состав проекта

| Файл | Назначение |
|------|------------|
| `moex_utils.py` | Основной интерфейс: акции и индексы (Parquet, pandas — до перевода), корпоративные события, adj_close, капитализация, ставка ЦБ, проверка качества; облигации и фьючерсы — обертки над `history` |
| `lake.py` | Хранилище DuckLake: каталог PostgreSQL `moex_lake`, файлы Parquet в `<MOEX_DATA_ROOT>/lake`; запросы → polars, запись по ключу, обслуживание |
| `history.py` | История рынков «все инструменты за дату» в хранилище: облигации с 1997, фьючерсы с 2002; реестр карточек бумаг |
| `iss.py` | Доступ к MOEX ISS: HTTP-сессия с таймаутом и повторами, разбор ответов в polars |
| `scripts/migrate_to_lake.py` | Разовый перенос Parquet-файлов в хранилище со сверкой |
| `update_data.py` | Пайплайн обновления: акции → индексы → облигации → ставка ЦБ → фьючерсы → adj_close → market_cap → проверка |
| `update_data.bat` | Запуск на Windows: выбор интерпретатора (`MOEX_PYTHON` → conda `py312` → `python`), проверка `apimoex` |
| `scheduled_update.cmd` | Обертка для задачи планировщика `MOEX data nightly` (вт–сб 00:30), лог в `logs/update.log` |
| `tests/` | Офлайн pytest-тесты (ISS замокан), гоняются в CI |
| `requirements.txt` | Зависимости ядра и тестов (`numpy<2` — для совместимости со сборками pandas под NumPy 1.x) |
| `requirements-notebooks.txt` | Дополнительно для marimo-ноутбуков (marimo, statsmodels, arch, PyPortfolioOpt и др.) |

## Установка и запуск

```bash
pip install -r requirements.txt
python update_data.py            # полный цикл обновления
python update_data.py --check    # только проверка данных
pytest -q                        # тесты
```

Рабочее окружение — conda `H:\conda\envs\py312`; системный `python` без `apimoex` не подходит. Данные лежат в `MOEX_DATA_ROOT` (`F:\moex-data`), реестры `metadata/` — в проекте. Хранилищу нужен локальный PostgreSQL (служба `postgresql-x64-17`) и пароль роли `moex` в `%APPDATA%\postgresql\pgpass.conf`; без них облигации и фьючерсы недоступны, остальное работает.

## Зависимости

- **requests**, **apimoex** — ISS MOEX; **lxml** — таблица ключевой ставки с cbr.ru
- **polars**, **duckdb** (с расширениями ducklake, postgres) — данные и хранилище
- **pandas**, **pyarrow** — акции и индексы до перевода на polars
- **openpyxl** — Excel с количеством акций
- **plotly** — графики в ноутбуках
