"""
Обновление данных MOEX: акции, индексы, облигации, ключевая ставка ЦБ, фьючерсы,
прочие рынки, ставки, КБД, денежные потоки облигаций, параметры бумаг,
adj_close и капитализация, проверка качества, обслуживание и копия хранилища.

Шаги: 1 акции → 1b индексы → 1c облигации → 1d ключевая ставка → 1e фьючерсы →
1f прочие рынки (все акции и фонды, все индексы, валюта, фиксинги) →
1g ставки и справочники (RUONIA, КБД, денежные потоки облигаций, параметры бумаг) →
2 пересчет adj_close и капитализации → 2b копии для SQL (реестры, stocks_adjusted,
представления) → 3 проверка данных → 4 обслуживание хранилища →
5 копия каталога. Все данные — в хранилище DuckLake (lake.py). Наборы истории
(облигации, фьючерсы, прочие рынки) ночью только дообновляются — первичная
выгрузка запускается явно (--history-init).

Итог прогона и замечания проверки пишутся в lake.update_runs / lake.quality_log.
Сбой шага или новые замечания (которых не было в прошлом прогоне) — уведомление
Windows (notify.py); сбой шага — код выхода 1.

Запуск: python update_data.py [--no-update] [--no-index] [--no-bonds] [--no-key-rate]
        [--no-futures] [--no-adj] [--no-cap] [--no-check] [--no-maintenance] [--no-backup]
        [--no-derived] [--no-markets] [--no-rates]
Первичная выгрузка: python update_data.py --history-init bonds,futures,shares,indexes_all,currency,
        currency_fixings,zcyc,cashflows,refdata [--history-start <начало>] (многочасовая)
Только проверка (без обновления, окно — год, со статусом ISS): python update_data.py --check
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from typing import Optional

import polars as pl

from moexutils import backup, cashflows, contracts, futures_audit, futures_params, futures_rms, history, indices, lake, notify, openpositions, options, quality, rates, refdata, stocks


def _warn(warnings: list, msg: str) -> None:
    """Сбой шага: в лог и в итог прогона (код выхода, оповещение)."""
    print(f"[WARN] {msg}")
    warnings.append(msg)


def _lake_tables(warnings: list) -> Optional[list]:
    """Таблицы хранилища или None, если хранилище недоступно (Postgres не запущен и т.п.)."""
    try:
        return lake.tables()
    except Exception as e:
        _warn(warnings, f"Хранилище DuckLake недоступно — {e}")
        return None


def _update_dataset(dataset: str, title: str, warnings: list) -> None:
    tables = _lake_tables(warnings)
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
        _warn(warnings, f"{title}: не удалось обновить — {e}")


def _finish(run_id: dt.datetime, mode: str, warnings: list,
            issues: Optional[pl.DataFrame], notify_on: bool) -> None:
    """Итог прогона в хранилище и оповещение о сбоях и новых замечаниях."""
    fresh = issues
    if issues is not None:
        try:
            fresh = quality.new_issues(issues, quality.previous_issues(mode, run_id))
        except Exception as e:
            print(f"[WARN] История замечаний недоступна — {e}")
        if not fresh.is_empty():
            print(f"Новых замечаний (не было в прошлом прогоне): {fresh.height}")
    try:
        quality.record_run(run_id, mode, warnings, issues, 0 if fresh is None else fresh.height)
    except Exception as e:
        print(f"[WARN] Итог прогона не записан в хранилище — {e}")
    if not notify_on:
        return
    if warnings:
        notify.toast(f"MOEX: сбой обновления ({len(warnings)})", "\n".join(warnings))
    elif fresh is not None and not fresh.is_empty():
        notify.toast(f"MOEX: новые замечания к данным ({fresh.height})",
                     "\n".join(f"[{c}] {o}: {d}" for c, o, d in fresh.head(5).iter_rows()))


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
    do_backup: bool = True,
    do_derived: bool = True,
    do_markets: bool = True,
    do_rates: bool = True,
    history_init: Optional[str] = None,
    history_start: Optional[str] = None,
    rebuild: bool = False,
    div_folder: Optional[str] = None,
    metadata_file: Optional[str] = None,
    index_tickers: Optional[str] = "IMOEX,MCFTR,RGBITR",
    check_days: Optional[int] = 30,
    check_div_days: Optional[int] = 120,
    check_iss: bool = False,
    mode: str = "update",
    notify_on: bool = True,
) -> int:
    """Прогон обновления; Returns: код выхода (1 — сбой хотя бы одного шага)."""
    run_id = dt.datetime.now()
    warnings: list[str] = []
    issues = None
    print(f"Данные: {lake.DATA_ROOT}"
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
            if dataset == 'zcyc':
                rates.update_zcyc(start=history_start or rates.ZCYC_START.isoformat())
                continue
            if dataset == 'cashflows':
                cashflows.update_cashflows('full')
                continue
            if dataset == 'open_positions':
                openpositions.update(start=history_start)
                continue
            if dataset == 'refdata':
                refdata.update_refdata(start=history_start)
                continue
            if dataset == 'index_weights':
                indices.update_index_weights(start=history_start, max_days=None)
                continue
            n = history.update(dataset, start=history_start, max_days=20000)
            print(f"{dataset}: +{n} строк")
            if dataset in ('bonds', 'shares'):
                history.update_securities(dataset, max_new=None)

    if do_update:
        print("=== 1. Обновление данных с MOEX ===" + (" (полное перескачивание)" if rebuild else ""))
        # adj_close и капитализация считаются сразу по всей истории тикера; в
        # хранилище пишутся только новые и изменившиеся строки
        try:
            stocks.update_stocks(div_folder=div_folder if div_ok else None, rebuild=rebuild)
        except Exception as e:
            _warn(warnings, f"Акции: не удалось обновить — {e}")
    else:
        print("=== 1. Обновление данных — пропуск (--no-update) ===")

    if do_indexes and index_tickers:
        print("=== 1b. Обновление индексов ===")
        try:
            stocks.update_indexes([t.strip() for t in index_tickers.split(",") if t.strip()])
        except Exception as e:
            _warn(warnings, f"Индексы: не удалось обновить — {e}")
    else:
        print("=== 1b. Индексы — пропуск (--no-index) ===")

    if do_bonds:
        print("=== 1c. Облигации (весь рынок) ===")
        _update_dataset('bonds', 'Облигации', warnings)
        try:
            history.update_securities('bonds')  # новые выпуски в реестр, до 500 за ночь
        except Exception as e:
            _warn(warnings, f"Реестр облигаций: не удалось обновить — {e}")
    else:
        print("=== 1c. Облигации — пропуск (--no-bonds) ===")

    if do_key_rate:
        print("=== 1d. Ключевая ставка ЦБ ===")
        try:
            stocks.update_key_rate()
        except Exception as e:
            _warn(warnings, f"Ключевая ставка: не удалось обновить — {e}")
    else:
        print("=== 1d. Ключевая ставка — пропуск (--no-key-rate) ===")

    if do_futures:
        print("=== 1e. Фьючерсы FORTS ===")
        _update_dataset('futures', 'Фьючерсы', warnings)
        if 'futures' in (_lake_tables(warnings) or []):
            try:
                # реестр контрактов, коды после повторного листинга, непрерывные ряды
                contracts.update_contracts()
                contracts.remap_futures_secids()
                contracts.update_continuous()
            except Exception as e:
                _warn(warnings, f"Реестр и непрерывные ряды фьючерсов не обновлены — {e}")
        try:
            futures_params.update()
        except Exception as e:
            _warn(warnings, f"Параметры фьючерсов и ставки риска не обновлены — {e}")
        try:
            futures_rms.update()
        except Exception as e:
            _warn(warnings, f"Архивы риск-параметров FORTS не обновлены — {e}")
        try:
            print(f"[OK] Покрытие параметров и аудит FORTS: {futures_audit.export()}")
        except Exception as e:
            _warn(warnings, f"Отчет покрытия параметров FORTS не создан — {e}")
    else:
        print("=== 1e. Фьючерсы — пропуск (--no-futures) ===")

    if do_markets:
        print("=== 1f. Прочие рынки: все акции и фонды, все индексы, валюта, фиксинги, опционы ===")
        for dataset in ('shares', 'indexes_all', 'currency', 'currency_fixings', 'options'):
            _update_dataset(dataset, history.DATASETS[dataset].label, warnings)
        if 'open_position_assets' in (_lake_tables(warnings) or []):
            try:
                openpositions.update()
            except Exception as e:
                _warn(warnings, f"Открытые позиции физлиц/юрлиц: не удалось обновить — {e}")
        if 'options' in (_lake_tables(warnings) or []):
            try:
                options.update_registry()
            except Exception as e:
                _warn(warnings, f"Реестр опционов: не удалось обновить — {e}")
        if 'shares' in (_lake_tables(warnings) or []):
            try:
                history.update_securities('shares')  # новые бумаги в реестр, до 500 за ночь
            except Exception as e:
                _warn(warnings, f"Реестр бумаг рынка акций: не удалось обновить — {e}")
    else:
        print("=== 1f. Прочие рынки — пропуск (--no-markets) ===")

    if do_rates:
        print("=== 1g. RUONIA, КБД, денежные потоки облигаций, параметры бумаг ===")
        if _lake_tables(warnings) is not None:
            for title, step in (("RUONIA", rates.update_ruonia),
                                ("КБД", rates.update_zcyc),
                ("КБД: пропуски", rates.repair_zcyc),
                ("Параметры бумаг (объем выпуска, листинг)", refdata.update_refdata),
                ("Состав и веса индексов", indices.update_index_weights),
                                # по субботам — все будущие потоки, в остальные ночи — окно ±дни
                                ("Денежные потоки облигаций", lambda: cashflows.update_cashflows(
                                    'future' if dt.date.today().weekday() == 5 else 'window'))):
                try:
                    step()
                except Exception as e:
                    _warn(warnings, f"{title}: не удалось обновить — {e}")
    else:
        print("=== 1g. RUONIA, КБД, денежные потоки — пропуск (--no-rates) ===")

    if do_adj_close and do_market_cap:
        if not div_ok:
            _warn(warnings, f"Папка дивидендов не найдена: {div_folder}. Пересчет adj_close пропущен.")
        else:
            # Сверка: после обновления дивидендов, метаданных или реестра сплитов
            # меняется история прошлых дат — пишутся только изменившиеся строки
            print("=== 2. Пересчет adj_close и капитализации ===")
            try:
                stocks.recompute_stocks(div_folder=div_folder)
            except Exception as e:
                _warn(warnings, f"Пересчет не выполнен — {e}")
    else:
        print("=== 2. Пересчет adj_close и капитализации — пропуск (--no-adj/--no-cap) ===")

    if do_derived:
        print("=== 2b. Копии для SQL: реестры, stocks_adjusted, представления ===")
        if _lake_tables(warnings) is not None:
            try:
                stocks.sync_registries()
                if 'stocks' in lake.tables():
                    stocks.update_adjusted()
                made = lake.ensure_views()
                if made:
                    print(f"[OK] Представления: {', '.join(made)}")
            except Exception as e:
                _warn(warnings, f"Копии для SQL не обновлены — {e}")
    else:
        print("=== 2b. Копии для SQL — пропуск ===")

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
            _warn(warnings, f"Проверка данных не выполнена — {e}")
    else:
        print("=== 3. Проверка данных — пропуск (--no-check) ===")

    if do_maintenance:
        print("=== 4. Обслуживание хранилища (слияние файлов, файлы-сироты, снимки старше "
              f"{lake.SNAPSHOT_RETENTION_DAYS} дней) ===")
        if _lake_tables(warnings) is not None:
            try:
                lake.maintenance()
            except Exception as e:
                _warn(warnings, f"Обслуживание хранилища не выполнено — {e}")
    else:
        print("=== 4. Обслуживание хранилища — пропуск ===")

    if do_backup:
        # после обслуживания: копия ссылается на файлы, которые остаются на диске
        print(f"=== 5. Копия каталога хранилища (хранятся последние {backup.KEEP}) ===")
        try:
            backup.backup_catalog()
        except Exception as e:
            _warn(warnings, f"Копия каталога не сделана — {e}")
    else:
        print("=== 5. Копия каталога — пропуск ===")

    if do_check:
        _finish(run_id, mode, warnings, issues, notify_on)
    print("\nГотово." if not warnings else f"\nГотово со сбоями: {len(warnings)}.")
    return 1 if warnings else 0


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
    ap.add_argument("--no-rates", action="store_true",
                    help="Не обновлять RUONIA, КБД, денежные потоки облигаций, параметры бумаг и состав индексов")
    ap.add_argument("--no-markets", action="store_true",
                    help="Не обновлять прочие рынки (все акции, все индексы, валюта, фиксинги)")
    ap.add_argument("--no-adj", action="store_true", help="Не пересчитывать adj_close и капитализацию (шаг 2)")
    ap.add_argument("--no-cap", action="store_true", help="То же, что --no-adj")
    ap.add_argument("--no-check", action="store_true", help="Не выполнять проверку данных")
    ap.add_argument("--no-maintenance", action="store_true", help="Не обслуживать хранилище")
    ap.add_argument("--no-backup", action="store_true", help="Не делать копию каталога хранилища")
    ap.add_argument("--no-derived", action="store_true",
                    help="Не обновлять копии для SQL (реестры ref_*, stocks_adjusted, представления)")
    ap.add_argument("--check", action="store_true",
                    help="Только проверка данных: без обновления, окно — год, статус ISS")
    ap.add_argument("--history-init", type=str, default=None,
                    help="Первичная выгрузка истории наборов через запятую: bonds,futures,options,shares,indexes_all,currency,currency_fixings,zcyc,cashflows,refdata,index_weights,open_positions")
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
        args.no_markets = args.no_rates = True
        args.no_adj = args.no_cap = args.no_maintenance = args.no_backup = args.no_derived = True

    try:
        rc = main(
            do_update=not args.no_update,
            do_adj_close=not args.no_adj,
            do_market_cap=not args.no_cap,
            do_indexes=not args.no_index,
            do_bonds=not args.no_bonds,
            do_key_rate=not args.no_key_rate,
            do_futures=not args.no_futures,
            do_check=not args.no_check,
            do_maintenance=not args.no_maintenance,
            do_backup=not args.no_backup,
            do_derived=not args.no_derived,
            do_markets=not args.no_markets,
            do_rates=not args.no_rates,
            history_init=",".join(init) or None,
            history_start=start,
            rebuild=args.rebuild,
            div_folder=args.div_folder,
            metadata_file=args.metadata_file,
            index_tickers=args.indexes,
            check_days=250 if args.check else 30,
            check_div_days=250 if args.check else 120,
            check_iss=args.check,
            mode="check" if args.check else "update",
            notify_on=not args.check,
        )
    except Exception as e:
        # непредвиденное падение вне шагов — тоже оповещение, трассировка в лог
        notify.toast("MOEX: обновление упало", f"{type(e).__name__}: {e}")
        raise
    sys.exit(rc)
