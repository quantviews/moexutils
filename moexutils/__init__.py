"""
moexutils — данные Московской биржи: акции, индексы, облигации, фьючерсы, ставки.

Модули:
- stocks   — акции и индексы: загрузка, корпоративные события, adj_close, капитализация, ставки;
- history  — история рынков «все инструменты за дату» (облигации, фьючерсы, ...), реестры бумаг;
- rates    — RUONIA, кривая бескупонной доходности (КБД);
- refdata  — параметры бумаг по датам (объем выпуска, листинг) с 01.04.2024;
- cashflows — денежные потоки облигаций: купоны, амортизации, оферты;
- contracts — реестр фьючерсных контрактов, непрерывные ряды;
- options — реестр опционных серий и контрактов, включая истекшие;
- openpositions — дневные позиции физлиц/юрлиц по фьючерсам и опционам;
- quality  — проверка качества данных, история прогонов;
- lake     — хранилище DuckLake (каталог PostgreSQL), запросы и запись (polars);
- iss      — доступ к MOEX ISS;
- indices  — состав и веса индексов по датам;
- bondmath — доходность, дюрация, выпуклость облигаций;
- backup, notify — копия каталога хранилища, уведомления ночного обновления.

    from moexutils import stocks, history, lake
    sber = stocks.read_stocks('SBER', start='2024-01-01')
"""
import logging
import sys

__version__ = "1.1.0"

_logger = logging.getLogger("moexutils")
# Если логирование в приложении не настроено — сообщения в stdout (прогресс в
# ноутбуках и update_data.bat). Любая внешняя настройка logging имеет приоритет.
if not _logger.handlers and not logging.getLogger().handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)
