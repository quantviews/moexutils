"""
Обновление данных: загрузка с MOEX, расчёт adj_close и капитализации (market_cap).
Использует moex_utils.

Шаги: 1 акции → 1b индексы → 1c облигации → 1d ключевая ставка → 2 adj_close → 3 market_cap.
Запуск: python update_data.py [--no-update] [--no-index] [--no-bonds] [--no-key-rate] [--no-adj] [--no-cap]
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import moex_utils as moex


def main(
    do_update: bool = True,
    do_adj_close: bool = True,
    do_market_cap: bool = True,
    do_indexes: bool = True,
    do_bonds: bool = True,
    do_key_rate: bool = True,
    bonds_init: Optional[str] = None,
    bonds_min_issue: Optional[float] = None,
    bonds_market_init: Optional[str] = None,
    bonds_market_start: str = "2024-01-01",
    rebuild: bool = False,
    div_folder: Optional[str] = None,
    data_folder: Optional[str] = None,
    metadata_file: Optional[str] = None,
    index_tickers: Optional[str] = "IMOEX,MCFTR,RGBITR",
) -> None:
    if data_folder is not None:
        moex.DATA_FOLDER = data_folder
    if metadata_file is not None:
        moex.METADATA_FILE = metadata_file

    if div_folder is None:
        base = os.path.dirname(os.path.abspath(__file__))
        div_folder = os.path.normpath(os.path.join(base, "..", "dividends", "data"))
    div_ok = do_adj_close and os.path.isdir(div_folder)

    if do_update:
        print("=== 1. Обновление данных с MOEX ===" + (" (полное перескачивание)" if rebuild else ""))
        # adj_close и market cap считаются сразу при обновлении тикера: файл пишется
        # один раз (папка синхронизируется облаком, серия быстрых перезаписей
        # одного файла порождает конфликтные копии). Шаги 2-3 ниже — сверка:
        # они пишут только файлы, у которых поменялись дивиденды или метаданные
        moex.update_all_stocks(calculate_market_cap_flag=do_market_cap, rebuild=rebuild,
                               div_folder=div_folder if div_ok else None)
    else:
        print("=== 1. Обновление данных — пропуск (--no-update) ===")

    if do_indexes and index_tickers:
        print("=== 1b. Обновление индексов ===")
        for idx_ticker in [t.strip() for t in index_tickers.split(",") if t.strip()]:
            try:
                moex.update_moex_index(idx_ticker)
            except Exception as e:
                print(f"[WARN] {idx_ticker}: не удалось обновить индекс — {e}")
    else:
        print("=== 1b. Индексы — пропуск (--no-index) ===")

    if bonds_market_init:
        print(f"=== 1c. Облигации: инициализация мониторинга досок {bonds_market_init} ===")
        for seg in [s.strip().upper() for s in bonds_market_init.split(",") if s.strip()]:
            n = moex.update_bonds_market(seg, start=bonds_market_start)
            print(f"{seg}: +{n} строк")
    elif bonds_init:
        print(f"=== 1c. Облигации: первичная выгрузка вселенной {bonds_init} ===")
        n = moex.download_bonds_universe(
            bonds_init,
            min_issue_size=bonds_min_issue * 1e9 if bonds_min_issue else None)
        print(f"Выгружено выпусков: {n}")
    elif do_bonds:
        print("=== 1c. Обновление облигаций ===")
        moex.update_all_bonds()
        moex.update_bonds_market_all()
    else:
        print("=== 1c. Облигации — пропуск (--no-bonds) ===")

    if do_key_rate:
        print("=== 1d. Ключевая ставка ЦБ ===")
        try:
            moex.update_key_rate()
        except Exception as e:
            print(f"[WARN] Ключевая ставка: не удалось обновить — {e}")
    else:
        print("=== 1d. Ключевая ставка — пропуск (--no-key-rate) ===")

    if do_adj_close:
        if not div_ok:
            print(f"[WARN] Папка дивидендов не найдена: {div_folder}. Adj close пропущен.")
        else:
            print("=== 2. Расчёт adjusted close (дивиденды) ===")
            moex.add_adj_close_to_all_stocks(div_folder)
    else:
        print("=== 2. Adj close — пропуск (--no-adj) ===")

    if do_market_cap:
        print("=== 3. Расчёт капитализации (market_cap) ===")
        moex.add_market_cap_to_all_stocks()
    else:
        print("=== 3. Market cap — пропуск (--no-cap) ===")

    print("\nГотово.")


if __name__ == "__main__":
    # При перенаправлении вывода в файл (планировщик задач) Windows отдает
    # stdout в кодировке ANSI (cp1252), и первый же print кириллицы роняет прогон
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Обновление данных, adj close и капитализации")
    ap.add_argument("--no-update", action="store_true", help="Не обновлять котировки с MOEX")
    ap.add_argument("--no-adj", action="store_true", help="Не пересчитывать adj_close")
    ap.add_argument("--no-cap", action="store_true", help="Не пересчитывать market_cap")
    ap.add_argument("--no-index", action="store_true", help="Не обновлять индексы")
    ap.add_argument("--rebuild", action="store_true",
                    help="Перескачать историю всех тикеров целиком (после смены методики данных)")
    ap.add_argument("--indexes", type=str, default="IMOEX,MCFTR,RGBITR",
                    help="Индексы через запятую (по умолчанию IMOEX,MCFTR,RGBITR)")
    ap.add_argument("--no-bonds", action="store_true", help="Не обновлять облигации")
    ap.add_argument("--no-key-rate", action="store_true", help="Не обновлять ключевую ставку ЦБ")
    ap.add_argument("--bonds-init", type=str, default=None,
                    help="Первичная выгрузка вселенной облигаций доски (например TQOB)")
    ap.add_argument("--bonds-min-issue", type=float, default=None,
                    help="Мин. объем выпуска в млрд руб при --bonds-init (для TQCB рекомендуется 10)")
    ap.add_argument("--bonds-market-init", type=str, default=None,
                    help="Инициализация мониторинга ВСЕХ выпусков досок через запятую (например TQOB,TQCB)")
    ap.add_argument("--bonds-market-start", type=str, default="2024-01-01",
                    help="Начальная дата мониторинга при --bonds-market-init (по умолчанию 2024-01-01)")
    ap.add_argument("--div-folder", type=str, default=None, help="Папка с CSV дивидендов (по умолчанию ../dividends/data)")
    ap.add_argument("--data-folder", type=str, default=None, help="Папка с parquet (по умолчанию data)")
    ap.add_argument("--metadata-file", type=str, default=None, help="Путь к Excel с метаданными (metadata/stock-index-base.xlsx)")
    args = ap.parse_args()

    main(
        do_update=not args.no_update,
        do_adj_close=not args.no_adj,
        do_market_cap=not args.no_cap,
        do_indexes=not args.no_index,
        do_bonds=not args.no_bonds,
        do_key_rate=not args.no_key_rate,
        bonds_init=args.bonds_init,
        bonds_min_issue=args.bonds_min_issue,
        bonds_market_init=args.bonds_market_init,
        bonds_market_start=args.bonds_market_start,
        rebuild=args.rebuild,
        div_folder=args.div_folder,
        data_folder=args.data_folder,
        metadata_file=args.metadata_file,
        index_tickers=args.indexes,
    )
