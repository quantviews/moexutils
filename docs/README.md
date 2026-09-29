# Документация moexutils

Библиотека и пайплайн данных Московской биржи (MOEX ISS): акции и индексы, облигации всего рынка с 1997 года (включая корпоративные, валютные и погашенные выпуски), все фьючерсы FORTS с 2002 года, ключевая ставка ЦБ; скорректированная цена (adj close), капитализация, проверка качества данных. Обзор проекта — в [README](../README.md).

## Разделы

- [Справочник API](api-reference.md) — функции `moex_utils` по рынкам, хранилища истории по датам, проверка качества, скрипт `update_data.py` и ночной запуск.
- [Данные и файлы](data-and-files.md) — структура каталогов, форматы Parquet (акции, облигации, фьючерсы), реестры `metadata/`, дивиденды соседнего проекта.
- [План развития](../development-plan.md) — состояние и ближайшие шаги.

## Состав проекта

| Файл | Назначение |
|------|------------|
| `moex_utils.py` | Ядро: загрузка из ISS, хранение в Parquet, корпоративные события, adj_close, капитализация, облигации, фьючерсы, проверка качества |
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

Рабочее окружение — conda `H:\conda\envs\py312`; системный `python` без `apimoex` не подходит. Данные лежат в `MOEX_DATA_ROOT` (`F:\moex-data`), реестры `metadata/` — в проекте.

## Зависимости

- **requests**, **apimoex** — ISS MOEX; **lxml** — таблица ключевой ставки с cbr.ru
- **pandas**, **pyarrow** — данные и Parquet
- **openpyxl** — Excel с количеством акций
- **plotly** — графики в ноутбуках
