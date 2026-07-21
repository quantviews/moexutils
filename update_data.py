"""
Обновление данных: загрузка с MOEX, расчёт adj_close и капитализации (market_cap).
Использует moex_utils.

Запуск: python update_data.py [--no-update] [--no-adj] [--no-cap] [--div-folder PATH]
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

import moex_utils as moex


def main(
    do_update: bool = True,
    do_adj_close: bool = True,
    do_market_cap: bool = True,
    do_indexes: bool = True,
    div_folder: Optional[str] = None,
    data_folder: Optional[str] = None,
    metadata_file: Optional[str] = None,
    index_tickers: Optional[str] = "IMOEX",
) -> None:
    if data_folder is not None:
        moex.DATA_FOLDER = data_folder
    if metadata_file is not None:
        moex.METADATA_FILE = metadata_file

    if do_update:
        print("=== 1. Обновление данных с MOEX ===")
        # Если этап 3 всё равно пересчитает market cap для всех данных,
        # не тратим время на пересчет для каждого тикера здесь
        moex.update_all_stocks(calculate_market_cap_flag=not do_market_cap)
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

    if do_adj_close:
        if div_folder is None:
            base = os.path.dirname(os.path.abspath(__file__))
            div_folder = os.path.normpath(os.path.join(base, "..", "dividends", "data"))
        if not os.path.isdir(div_folder):
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
    ap = argparse.ArgumentParser(description="Обновление данных, adj close и капитализации")
    ap.add_argument("--no-update", action="store_true", help="Не обновлять котировки с MOEX")
    ap.add_argument("--no-adj", action="store_true", help="Не пересчитывать adj_close")
    ap.add_argument("--no-cap", action="store_true", help="Не пересчитывать market_cap")
    ap.add_argument("--no-index", action="store_true", help="Не обновлять индексы")
    ap.add_argument("--indexes", type=str, default="IMOEX", help="Индексы через запятую (по умолчанию IMOEX)")
    ap.add_argument("--div-folder", type=str, default=None, help="Папка с CSV дивидендов (по умолчанию ../dividends/data)")
    ap.add_argument("--data-folder", type=str, default=None, help="Папка с parquet (по умолчанию data)")
    ap.add_argument("--metadata-file", type=str, default=None, help="Путь к Excel с метаданными (metadata/stock-index-base.xlsx)")
    args = ap.parse_args()

    main(
        do_update=not args.no_update,
        do_adj_close=not args.no_adj,
        do_market_cap=not args.no_cap,
        do_indexes=not args.no_index,
        div_folder=args.div_folder,
        data_folder=args.data_folder,
        metadata_file=args.metadata_file,
        index_tickers=args.indexes,
    )
