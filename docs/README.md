# Документация moexutils

Библиотека для загрузки и обработки данных Московской биржи (MOEX): котировки, индексы, расчёт скорректированной цены (adj close) и капитализации.

**Требования:** Python 3.9+.

## Установка

```bash
pip install -r requirements.txt
```

Или вручную:

```bash
pip install requests apimoex pandas pyarrow openpyxl "numpy<2"
```

Ограничение `numpy<2` нужно для совместимости с pandas/numexpr/bottleneck в окружениях, где они собраны под NumPy 1.x. Если используете свежие версии pandas и зависимостей, можно ставить NumPy 2.x.

## Быстрый старт

```python
import moex_utils as moex

# Чтение данных (при отсутствии файла — загрузка с MOEX)
df = moex.read_moex_stock('SBER')

# Обновление всех данных + adj_close + market_cap — из командной строки:
# python update_data.py
# python update_data.py --div-folder "F:/path/to/dividends/data"
```

## Состав проекта

| Файл | Назначение |
|------|------------|
| **moex_utils.py** | Ядро: запросы к MOEX ISS, сохранение в Parquet, расчёт adj_close и market_cap |
| **update_data.py** | Скрипт: обновление котировок, пересчёт adj_close и капитализации |
| **requirements.txt** | Зависимости (в т.ч. numpy<2 для совместимости) |

## Разделы

- [Справочник API](api-reference.md) — функции `moex_utils` и скрипт `update_data`.
- [Данные и файлы](data-and-files.md) — структура каталогов, формат Parquet, метаданные и дивиденды.

## Зависимости

- **requests**, **apimoex** — работа с API MOEX  
- **pandas**, **pyarrow** — данные и Parquet  
- **openpyxl** — чтение Excel (метаданные по акциям)  
- **numpy<2** — в requirements.txt для совместимости со старыми сборками pandas/numexpr/bottleneck
