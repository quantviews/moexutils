"""
Обновление данных MOEX: акции, индексы, облигации, ключевая ставка ЦБ, фьючерсы,
adj_close и капитализация, проверка качества и обслуживание хранилища.

Шаги: 1 акции → 1b индексы → 1c облигации → 1d ключевая ставка → 1e фьючерсы →
2 пересчет adj_close и капитализации → 3 проверка данных → 4 обслуживание хранилища.
Все данные — в хранилище DuckLake (lake.py). Облигации и фьючерсы ночью только
дообновляются — первичная выгрузка запускается явно (--history-init).

Запуск: python update_data.py [--no-update] [--no-index] [--no-bonds] [--no-key-rate]
        [--no-futures] [--no-adj] [--no-cap] [--no-check] [--no-maintenance]
Первичная выгрузка: python update_data.py --history-init bonds,futures (многочасовая)
Только проверка (без обновления, окно — год, со статусом ISS): python update_data.py --check
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import history
import lake
import moex_utils as moex
import quality
import stocks


def _lake_tables() -> Optional[list]:
    """Таблицы хранилища или None, если хранилище недоступно (Postgres не запущен и т.п.)."""
    try:
        return lake.tables()
    except Exception as e:
        print(f"[WARN] Хранилище DuckLake недоступно — {e}")
        return None


def _update_dataset(dataset: str, title: str) -> None:
    tables = _lake_tables()
    if tables is None:
        return
    if dataset not in tables:
        print(f"[INFO] {title}: в хранилище нет истории — первичная выгрузка: "
              f"python update_data.py --history-init {dataset}")
        return
    try:
        history.update(dataset)
        history.repair(dataset)
    except Exception as e:
        print(f"[WARN] {title}: не удалось обновить — {e}")


def main(
    do_update: bool = True,
    do_adj_close: bool = True,
    do_market_cap: bool = True,
    do_indexes: bool = True,
    do_bonds: bool = True,
    do_key_rate: bool = True,
    do_futures: bool = True,
    do_check: bool = True,
    do_maintenance: bool = True,
    history_init: Optional[str] = None,
    history_start: Optional[str] = None,
    rebuild: bool = False,
    div_folder: Optional[str] = None,
    metadata_file: Optional[str] = None,
    index_tickers: Optional[str] = "IMOEX,MCFTR,RGBITR",
    check_days: Optional[int] = 30,
    check_div_days: Optional[int] = 120,
    check_iss: bool = False,
) -> None:
    print(f"Данные: {moex.DATA_ROOT}"
          + ("" if os.environ.get("MOEX_DATA_ROOT") else " (MOEX_DATA_ROOT не задана — папка проекта)"))
    if metadata_file is not None:
        stocks.METADATA_FILE = metadata_file

    if div_folder is None:
        base = os.path.dirname(os.path.abspath(__file__))
        div_folder = os.path.normpath(os.path.join(base, "..", "dividends", "data"))
    div_ok = do_adj_close and os.path.isdir(div_folder)

    if history_init:
        # Первичная (многочасовая) выгрузка истории наборов в хранилище
        for dataset in [d.strip().lower() for d in history_init.split(",") if d.strip()]:
            print(f"=== Первичная выгрузка истории: {dataset} с {history_start or 'начала истории ISS'} ===")
            n = history.update(dataset, start=history_start, max_days=20000)
            print(f"{dataset}: +{n} строк")
            if dataset == 'bonds':
                history.update_securities('bonds', max_new=None)

    if do_update:
        print("=== 1. Обновление данных с MOEX ===" + (" (полное перескачивание)" if rebuild else ""))
        # adj_close и капитализация считаются сразу по всей истории тикера; в
        # хранилище пишутся только новые и изменившиеся строки
        try:
            stocks.update_stocks(div_folder=div_folder if div_ok else None, rebuild=rebuild)
        except Exception as e:
            print(f"[WARN] Акции: не удалось обновить — {e}")
    else:
        print("=== 1. Обновление данных — пропуск (--no-update) ===")

    if do_indexes and index_tickers:
        print("=== 1b. Обновление индексов ===")
        try:
            stocks.update_indexes([t.strip() for t in index_tickers.split(",") if t.strip()])
        except Exception as e:
            print(f"[WARN] Индексы: не удалось обновить — {e}")
    else:
        print("=== 1b. Индексы — пропуск (--no-index) ===")

    if do_bonds:
        print("=== 1c. Облигации (весь рынок) ===")
        _update_dataset('bonds', 'Облигации')
        try:
            history.update_securities('bonds')  # новые выпуски в реестр, до 500 за ночь
        except Exception as e:
            print(f"[WARN] Реестр облигаций: не удалось обновить — {e}")
    else:
        print("=== 1c. Облигации — пропуск (--no-bonds) ===")

    if do_key_rate:
        print("=== 1d. Ключевая ставка ЦБ ===")
        try:
            stocks.update_key_rate()
        except Exception as e:
            print(f"[WARN] Ключевая ставка: не удалось обновить — {e}")
    else:
        print("=== 1d. Ключевая ставка — пропуск (--no-key-rate) ===")

    if do_futures:
        print("=== 1e. Фьючерсы FORTS ===")
        _update_dataset('futures', 'Фьючерсы')
    else:
        print("=== 1e. Фьючерсы — пропуск (--no-futures) ===")

    if do_adj_close and do_market_cap:
        if not div_ok:
            print(f"[WARN] Папка дивидендов не найдена: {div_folder}. Пересчет adj_close пропущен.")
        else:
            # Сверка: после обновления дивидендов, метаданных или реестра сплитов
            # меняется история прошлых дат — пишутся только изменившиеся строки
            print("=== 2. Пересчет adj_close и капитализации ===")
            try:
                stocks.recompute_stocks(div_folder=div_folder)
            except Exception as e:
                print(f"[WARN] Пересчет не выполнен — {e}")
    else:
        print("=== 2. Пересчет adj_close и капитализации — пропуск (--no-adj/--no-cap) ===")

    if do_check:
        print(f"=== 3. Проверка данных (окно {check_days or 'вся история'} торг. дн., "
              f"дивиденды — {check_div_days or 'вся история'}) ===")
        try:
            issues = quality.data_quality_report(days=check_days, div_folder=div_folder,
                                                 div_days=check_div_days, check_iss=check_iss)
            for check, obj, detail in issues.head(60).iter_rows():
                print(f"  [{check}] {obj}: {detail}")
            if issues.height > 60:
                print(f"  ... и еще {issues.height - 60}")
            print(quality.quality_summary(issues))
        except Exception as e:
            print(f"[WARN] Проверка данных не выполнена — {e}")
    else:
        print("=== 3. Проверка данных — пропуск (--no-check) ===")

    if do_maintenance:
        print("=== 4. Обслуживание хранилища (слияние файлов, снимки старше "
              f"{lake.SNAPSHOT_RETENTION_DAYS} дней) ===")
        if _lake_tables() is not None:
            try:
                lake.maintenance()
            except Exception as e:
                print(f"[WARN] Обслуживание хранилища не выполнено — {e}")
    else:
        print("=== 4. Обслуживание хранилища — пропуск ===")

    print("\nГотово.")


if __name__ == "__main__":
    # При перенаправлении вывода в файл (планировщик задач) Windows отдает
    # stdout в кодировке ANSI (cp1252), и первый же print кириллицы роняет прогон
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Обновление данных MOEX")
    ap.add_argument("--no-update", action="store_true", help="Не обновлять котировки акций")
    ap.add_argument("--no-index", action="store_true", help="Не обновлять индексы")
    ap.add_argument("--no-bonds", action="store_true", help="Не обновлять облигации")
    ap.add_argument("--no-key-rate", action="store_true", help="Не обновлять ключевую ставку ЦБ")
    ap.add_argument("--no-futures", action="store_true", help="Не обновлять фьючерсы")
    ap.add_argument("--no-adj", action="store_true", help="Не пересчитывать adj_close и капитализацию (шаг 2)")
    ap.add_argument("--no-cap", action="store_true", help="То же, что --no-adj")
    ap.add_argument("--no-check", action="store_true", help="Не выполнять проверку данных")
    ap.add_argument("--no-maintenance", action="store_true", help="Не обслуживать хранилище")
    ap.add_argument("--check", action="store_true",
                    help="Только проверка данных: без обновления, окно — год, статус ISS")
    ap.add_argument("--history-init", type=str, default=None,
                    help="Первичная выгрузка истории наборов через запятую: bonds,futures")
    ap.add_argument("--history-start", type=str, default=None,
                    help="Начальная дата первичной выгрузки (по умолчанию — начало истории ISS)")
    # синонимы прежних флагов первичной выгрузки
    ap.add_argument("--bonds-market-init", type=str, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--bonds-market-start", type=str, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--futures-init", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--futures-start", type=str, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--rebuild", action="store_true",
                    help="Перескачать историю всех акций целиком (после смены методики данных)")
    ap.add_argument("--indexes", type=str, default="IMOEX,MCFTR,RGBITR",
                    help="Индексы через запятую (по умолчанию IMOEX,MCFTR,RGBITR)")
    ap.add_argument("--div-folder", type=str, default=None,
                    help="Папка CSV дивидендов (по умолчанию ../dividends/data)")
    ap.add_argument("--metadata-file", type=str, default=None,
                    help="Excel с количеством акций (metadata/stock-index-base.xlsx)")
    args = ap.parse_args()

    init = [d for d in (args.history_init or "").split(",") if d.strip()]
    if args.bonds_market_init and 'bonds' not in init:
        init.append('bonds')
    if args.futures_init and 'futures' not in init:
        init.append('futures')
    start = args.history_start or args.bonds_market_start or args.futures_start
    if args.check:
        args.no_update = args.no_index = args.no_bonds = args.no_key_rate = args.no_futures = True
        args.no_adj = args.no_cap = args.no_maintenance = True

    main(
        do_update=not args.no_update,
        do_adj_close=not args.no_adj,
        do_market_cap=not args.no_cap,
        do_indexes=not args.no_index,
        do_bonds=not args.no_bonds,
        do_key_rate=not args.no_key_rate,
        do_futures=not args.no_futures,
        do_check=not args.no_check,
        do_maintenance=not args.no_maintenance,
        history_init=",".join(init) or None,
        history_start=start,
        rebuild=args.rebuild,
        div_folder=args.div_folder,
        metadata_file=args.metadata_file,
        index_tickers=args.indexes,
        check_days=250 if args.check else 30,
        check_div_days=250 if args.check else 120,
        check_iss=args.check,
    )
