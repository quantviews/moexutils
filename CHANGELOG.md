# Изменения

Формат версий — в [контракте данных](docs/data-contract.md#как-меняется-контракт).

## 1.0.0 — 30.09.2026

Первая версия как устанавливаемого пакета.

- Пакет `moexutils` (`pip install -e .`): модули `stocks`, `history`, `quality`, `lake`, `iss`, `bondmath`, `backup`, `notify`. Прежний фасад `moex_utils` удален — импортируйте модули пакета.
- Хранилище DuckLake с каталогом PostgreSQL: `stocks`, `indexes`, `bonds`, `bonds_securities`, `futures`, служебные `empty_dates`, `update_runs`, `quality_log`.
- Для SQL-потребителей: реестры `ref_splits`, `ref_renames`, `ref_delisted`, `ref_key_rate`, `ref_sectors`; `stocks_adjusted` (склейка переименований, поправка на сплиты); представления `bonds_ofz`, `bonds_corporate`; роль `moex_reader`.
- Чтение на момент снимка: параметр `as_of`, `lake.ref`, `lake.snapshots`.
- Аналитические ноутбуки перенесены в проект moex-analytics.
