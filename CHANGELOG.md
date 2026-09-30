# Изменения

Формат версий — в [контракте данных](docs/data-contract.md#как-меняется-контракт).

## 1.0.1 — 30.09.2026

- **Рабочие субботы.** Торговый календарь (`history.trading_calendar`) отбрасывал все субботы, хотя биржа торговала в 50 перенесенных рабочих суббот с 1995 года (последняя — 01.11.2025); наборы «все инструменты за дату» и КБД эти дни пропускали. Календарь теперь — все даты IMOEX, `repair` не отбрасывает выходные; догружено: облигации 41 суббота (31,6 тыс. строк), акции и фонды 41 (17,7 тыс.), фьючерсы 34 (5,1 тыс.), валюта 40, фиксинги 5, КБД 9 (`rates.repair_zcyc`, ночной шаг 1g).
- **Индексы по воскресеньям.** Индексы доски AGRO (например, `SGCFOOTC`) публикуются по воскресеньям: набор `indexes_all` запрашивается и в выходные (`Dataset.weekends`); догружено 4,9 тыс. строк за 298 дат. Сверка со справочником ISS: из 904 индексов нет 26 — у них нет истории на бирже.

## 1.0.0 — 30.09.2026

Первая версия как устанавливаемого пакета.

- Пакет `moexutils` (`pip install -e .`, extras `notebooks`, `dev`; `requirements*.txt` и `conftest.py` удалены): модули `stocks`, `history`, `rates`, `refdata`, `cashflows`, `contracts`, `quality`, `lake`, `iss`, `bondmath`, `backup`, `notify`. Прежний фасад `moex_utils` удален — импортируйте модули пакета (`from moexutils import stocks, history, lake`); обертки облигаций и фьючерсов заменены на `history.read` / `update` / `repair` / `read_securities` / `update_securities`, математика облигаций — `moexutils.bondmath`. Логгер — `moexutils`. CI: `ruff check` и `pytest`.
- Хранилище DuckLake с каталогом PostgreSQL: `stocks`, `indexes`, `bonds`, `bonds_securities`, `futures`, служебные `empty_dates`, `update_runs`, `quality_log`.
- Новые данные:
  - весь рынок акций и фондов с 1997 года (`shares`) и реестр карточек его бумаг (`shares_securities`); все индексы MOEX с 1995 года (`indexes_all`); валютный рынок с 1997 года (`currency`, только строки со сделками) и валютные фиксинги с 2019 года (`currency_fixings`) — наборы `history.DATASETS`, ночной шаг 1f;
  - RUONIA с cbr.ru с 2010 года (`ruonia`); кривая бескупонной доходности MOEX с 06.01.2014 (`zcyc_params`, `zcyc_yields`, `zcyc_bonds`) — модуль `rates`, шаг 1g;
  - денежные потоки облигаций: купоны, амортизации, оферты (`bond_coupons`, `bond_amortizations`, `bond_offers`) — модуль `cashflows`, шаг 1g;
  - реестр фьючерсных контрактов (`futures_contracts`) и непрерывные ряды по 12 основным активам с переходом по открытому интересу и склейкой по отношению цен (`futures_continuous`) — модуль `contracts`, шаг 1e.
- Фьючерсы: при повторном листинге кода ISS задним числом переименовывает старый контракт (`SiZ5` 2015 года → `SiZ5_2015`); строки `futures`, загруженные до переименования, переведены на новые коды (`contracts.remap_futures_secids`, ночной шаг 1e). Прежнее описание «коды повторяются, уникальна пара даты и кода» неверно.
- Для SQL-потребителей: реестры `ref_splits`, `ref_renames`, `ref_delisted`, `ref_key_rate`, `ref_sectors`; `stocks_adjusted` (склейка переименований, поправка на сплиты); представления `bonds_ofz`, `bonds_corporate`; роль `moex_reader` (`scripts/reader_role.sql`); `lake.sync`, `lake.write(delete=...)`, `lake.ensure_views`.
- Чтение на момент снимка: параметр `as_of` у функций чтения, `lake.ref`, `lake.snapshots`.
- Проверка качества охватывает все наборы `history.DATASETS` (`<набор>_stale`, `<набор>_gaps`).
- `stocks.load_sectors` (`metadata/sectors.csv`).
- Параметры бумаг по датам с 01.04.2024 (`refdata`, `stock_refdata` — только изменения; служебная `load_state`; НКД и флаги `hasprospectus`, `hastechnicaldefault` не хранятся) — ночной шаг 1g, первичная выгрузка `--history-init refdata`.
- Роль `moex_reader` создана; ночное обслуживание удаляет файлы-сироты (`ducklake_delete_orphaned_files`).
- Проект vectorbt читает данные через пакет; прежние Parquet-папки `data/`, `indexes/`, `bonds/`, `futures/` в `F:\moex-data` больше не используются.
- Аналитические ноутбуки (`stocks-performance`, `ticker-analysis`, `portfolio-analysis`, `momentum-strategy`, `arima-analysis`) перенесены в проект `moex-analytics`; здесь остался обзор данных `marimo/bond-market.py`.
