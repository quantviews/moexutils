"""
Оповещения о ночном обновлении: уведомление Windows (центр уведомлений).

Задача планировщика запускается от пользователя в интерактивном сеансе, поэтому
уведомление видно на рабочем столе или позже в центре уведомлений. Сбой показа
не роняет обновление. MOEX_NO_NOTIFY=1 отключает оповещения (тесты, ручные прогоны).
"""
from __future__ import annotations

import logging
import os
import subprocess

logger = logging.getLogger("moexutils")

# AppUserModelID Windows PowerShell — зарегистрирован в системе, уведомления от него показываются
_APP_ID = r"{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe"

_PS_TOAST = r"""
$ErrorActionPreference = 'Stop'
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null
$xml = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent(
    [Windows.UI.Notifications.ToastTemplateType]::ToastText02)
$t = $xml.GetElementsByTagName('text')
$t.Item(0).AppendChild($xml.CreateTextNode($env:MOEX_TOAST_TITLE)) > $null
$t.Item(1).AppendChild($xml.CreateTextNode($env:MOEX_TOAST_TEXT)) > $null
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($env:MOEX_TOAST_APP).Show(
    [Windows.UI.Notifications.ToastNotification]::new($xml))
"""


def toast(title: str, text: str) -> bool:
    """Уведомление Windows; Returns: показано ли (False — не Windows, отключено или сбой)."""
    if os.name != "nt" or os.environ.get("MOEX_NO_NOTIFY"):
        return False
    env = dict(os.environ, MOEX_TOAST_TITLE=title, MOEX_TOAST_TEXT=text[:500], MOEX_TOAST_APP=_APP_ID)
    try:
        res = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", _PS_TOAST],
                             env=env, capture_output=True, text=True, timeout=60,
                             encoding="utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError) as e:
        logger.info(f"[WARN] Уведомление не показано — {e}")
        return False
    if res.returncode != 0:
        logger.info(f"[WARN] Уведомление не показано — {res.stderr.strip()[:300]}")
        return False
    return True
