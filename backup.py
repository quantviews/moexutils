"""
Резервная копия каталога хранилища DuckLake (база PostgreSQL moex_lake).

Каталог — это схема таблиц, снимки и список файлов данных: без него Parquet-файлы
в `<MOEX_DATA_ROOT>/lake` не собрать обратно в таблицы. Копия — `pg_dump` в формате
custom (сжатый), по файлу на прогон в `<MOEX_DATA_ROOT>/backups/catalog`
(или `MOEX_BACKUP_DIR`); хранятся последние KEEP копий.

Пароль pg_dump берет сам из файла паролей PostgreSQL (`%APPDATA%\\postgresql\\pgpass.conf`),
в командной строке и логах его нет. Восстановление — docs/api-reference.md
(раздел «Резервная копия каталога»).
"""
from __future__ import annotations

import glob
import logging
import os
import shutil
import subprocess
from datetime import datetime
from typing import Optional

import lake

logger = logging.getLogger("moex_utils")

BACKUP_DIR = os.environ.get("MOEX_BACKUP_DIR") or os.path.join(lake.DATA_ROOT, "backups", "catalog")
KEEP = 14
PREFIX = "moex_lake-"


class BackupError(RuntimeError):
    pass


def pg_tool(name: str) -> str:
    """Путь к утилите PostgreSQL: MOEX_PG_BIN, PATH, затем самая новая версия в Program Files."""
    exe = name + (".exe" if os.name == "nt" else "")
    env_bin = os.environ.get("MOEX_PG_BIN")
    if env_bin and os.path.isfile(os.path.join(env_bin, exe)):
        return os.path.join(env_bin, exe)
    found = shutil.which(name)
    if found:
        return found
    candidates = glob.glob(os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                                        "PostgreSQL", "*", "bin", exe))
    if candidates:
        # .../PostgreSQL/<версия>/bin/<exe> — берем самую новую версию
        def _version(p: str) -> int:
            v = os.path.basename(os.path.dirname(os.path.dirname(p)))
            return int(v) if v.isdigit() else 0
        return max(candidates, key=_version)
    raise BackupError(f"{name} не найден: задайте MOEX_PG_BIN (папка bin PostgreSQL)")


def _conn_args() -> list[str]:
    # -w: никогда не спрашивать пароль (ночной запуск без консоли)
    return ["-h", lake.PG_HOST, "-p", str(lake.PG_PORT), "-U", lake.PG_USER, "-w"]


def list_backups(folder: Optional[str] = None) -> list[str]:
    """Копии каталога, от старых к новым."""
    folder = folder or BACKUP_DIR
    return sorted(glob.glob(os.path.join(folder, PREFIX + "*.dump")))


def verify(path: str) -> int:
    """Проверка копии через pg_restore --list; Returns: число таблиц с данными в копии."""
    res = subprocess.run([pg_tool("pg_restore"), "--list", path],
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        raise BackupError(f"pg_restore --list {path}: {res.stderr.strip()}")
    n = sum(1 for line in res.stdout.splitlines() if " TABLE DATA " in line)
    if n == 0:
        raise BackupError(f"{path}: в копии нет данных таблиц")
    return n


def backup_catalog(folder: Optional[str] = None, keep: int = KEEP) -> str:
    """
    Копия каталога: pg_dump во временный файл, проверка, переименование,
    удаление копий сверх keep. Returns: путь к новой копии.
    """
    folder = folder or BACKUP_DIR
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{PREFIX}{datetime.now():%Y%m%d-%H%M%S}.dump")
    tmp = path + ".tmp"
    res = subprocess.run([pg_tool("pg_dump"), *_conn_args(), "-d", lake.PG_DATABASE,
                          "-Fc", "-f", tmp],
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise BackupError(f"pg_dump: {res.stderr.strip()}")
    try:
        n = verify(tmp)
    except BackupError:
        os.remove(tmp)
        raise
    os.replace(tmp, path)
    old = list_backups(folder)[:-keep] if keep > 0 else []
    for p in old:
        os.remove(p)
    logger.info(f"Копия каталога: {path} ({os.path.getsize(path) / 1e6:.1f} МБ, "
                f"таблиц с данными {n}; удалено старых {len(old)})")
    return path
