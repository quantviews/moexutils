# Legacy: архив до перехода на polars и DuckLake

Здесь лежат Jupyter-ноутбуки и скрипты ранних этапов проекта. Они написаны на pandas против прежнего интерфейса `moex_utils` (Parquet-файлы по тикерам) и **в текущем виде не запускаются**: функции, которые они вызывают, удалены при миграции (сентябрь 2026). Сохранены как история исследований и примеров, в работе не используются и не поддерживаются.

| Файл | Что было |
|------|----------|
| `jupyter/example-notebook.ipynb` | Примеры первых функций библиотеки |
| `jupyter/market-analysis.ipynb` | Обзор рынка акций (предшественник `marimo/stocks-performance.py`) |
| `jupyter/market-index-analysis.ipynb` | Анализ индексов |
| `jupyter/portfolio-analysis.ipynb` | Портфельный анализ (предшественник `marimo/portfolio-analysis.py`) |
| `jupyter/trading-strategy.ipynb` | Торговая стратегия (предшественник `marimo/momentum-strategy.py`) |
| `jupyter/trading-strategy-arima.ipynb` | ARIMA-стратегия (предшественник `marimo/arima-analysis.py`) |
| `jupyter/dividends-adjustment.ipynb` | Отладка корректировки цен на дивиденды |
| `scripts/examples.py` | Примеры вызовов прежнего API |
| `scripts/adj-dividends-calc.py` | Пересчет adj_close по всем тикерам прежним API |

Замены в текущем коде (polars, хранилище DuckLake):

| Было | Стало |
|------|-------|
| `moex.get_moex_stock`, `save_moex_stock`, `update_moex_stock`, `update_all_stocks` | `stocks.fetch_stock`, `stocks.add_stock`, `stocks.update_stocks` |
| `moex.read_moex_stock`, `combine_moex_stocks` | `stocks.read_stocks` |
| `moex.calculate_adj_close`, `add_adj_close_to_all_stocks`, `calculate_market_cap` | `stocks.adj_close`, `stocks.market_cap`, `stocks.recompute_stocks` |
| `moex.get_moex_index`, `read_moex_index` | `stocks.fetch_index`, `stocks.read_index` |
| чтение CSV дивидендов через pandas | `stocks.load_dividends` |

Актуальные ноутбуки — `marimo/`, описание данных — `docs/data-model.md`.
