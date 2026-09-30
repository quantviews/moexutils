# Legacy: архив до перехода на polars и DuckLake

Здесь лежат Jupyter-ноутбуки и скрипты ранних этапов проекта. Они написаны на pandas против прежнего интерфейса `moex_utils` (Parquet-файлы по тикерам) и **в текущем виде не запускаются**: функции, которые они вызывают, удалены при миграции (сентябрь 2026), фасад `moex_utils` удален при переходе на пакет `moexutils` (30.09.2026). Сохранены как история исследований и примеров, в работе не используются и не поддерживаются.

| Файл | Что было |
|------|----------|
| `jupyter/example-notebook.ipynb` | Примеры первых функций библиотеки |
| `jupyter/market-analysis.ipynb` | Обзор рынка акций (предшественник `stocks-performance.py`, теперь в `../moex-analytics`) |
| `jupyter/market-index-analysis.ipynb` | Анализ индексов |
| `jupyter/portfolio-analysis.ipynb` | Портфельный анализ (предшественник `portfolio-analysis.py`, теперь в `../moex-analytics`) |
| `jupyter/trading-strategy.ipynb` | Торговая стратегия (предшественник `momentum-strategy.py`, теперь в `../moex-analytics`) |
| `jupyter/trading-strategy-arima.ipynb` | ARIMA-стратегия (предшественник `arima-analysis.py`, теперь в `../moex-analytics`) |
| `jupyter/dividends-adjustment.ipynb` | Отладка корректировки цен на дивиденды |
| `scripts/examples.py` | Примеры вызовов прежнего API |
| `scripts/adj-dividends-calc.py` | Пересчет adj_close по всем тикерам прежним API |
| `scripts/migrate_to_lake.py` | Разовый перенос Parquet-файлов в хранилище DuckLake со сверкой (выполнен 29.09.2026) |

Замены в текущем коде (пакет `moexutils`, polars, хранилище DuckLake):

| Было | Стало |
|------|-------|
| `moex.get_moex_stock`, `save_moex_stock`, `update_moex_stock`, `update_all_stocks` | `moexutils.stocks.fetch_stock`, `moexutils.stocks.add_stock`, `moexutils.stocks.update_stocks` |
| `moex.read_moex_stock`, `combine_moex_stocks` | `moexutils.stocks.read_stocks` |
| `moex.calculate_adj_close`, `add_adj_close_to_all_stocks`, `calculate_market_cap` | `moexutils.stocks.adj_close`, `moexutils.stocks.market_cap`, `moexutils.stocks.recompute_stocks` |
| `moex.get_moex_index`, `read_moex_index` | `moexutils.stocks.fetch_index`, `moexutils.stocks.read_index` |
| чтение CSV дивидендов через pandas | `moexutils.stocks.load_dividends` |
| `moex.read_bonds_market`, `update_bonds_market`, `repair_bonds_market`, `read_bonds_securities`, `update_bonds_securities` | `moexutils.history.read('bonds')`, `history.update('bonds')`, `history.repair('bonds')`, `history.read_securities('bonds')`, `history.update_securities('bonds')` |
| `moex.read_futures_history`, `update_futures_history`, `repair_futures_history` | `moexutils.history.read('futures')`, `history.update('futures')`, `history.repair('futures')`; реестр и непрерывные ряды — `moexutils.contracts` |
| `moex.get_security_description` | `moexutils.iss.security_description` |
| `moex.calculate_ytm`, `calculate_duration`, `calculate_convexity` | `moexutils.bondmath.calculate_ytm`, `calculate_duration`, `calculate_convexity` |

Актуальный ноутбук обзора данных — `marimo/bond-market.py`, аналитические — в проекте `../moex-analytics`; описание данных — `docs/data-model.md`.
