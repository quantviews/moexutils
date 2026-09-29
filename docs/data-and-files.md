# Данные и файлы

## Структура каталогов

Код и реестры живут в папке проекта (Яндекс.Диск, git), рыночные данные — в корне `MOEX_DATA_ROOT` (на рабочей машине `F:\moex-data`, вне облачной синхронизации).

```
moexutils/                   # проект: F:\Yandex.Disk\FINANCE\moexutils
├── moex_utils.py            # ядро библиотеки
├── update_data.py           # пайплайн обновления (CLI)
├── update_data.bat          # запуск на Windows (выбор интерпретатора)
├── scheduled_update.cmd     # обертка для планировщика задач (лог в logs/)
├── marimo/                  # marimo-ноутбуки (аналитика и преподавание)
├── nb/                      # Jupyter-ноутбуки (исследования, примеры)
├── scripts/                 # аналитические скрипты
├── tests/                   # офлайн pytest-тесты
├── docs/                    # документация
├── metadata/
│   ├── stock-index-base.xlsx  # количество акций по датам (капитализация)
│   ├── splits.csv             # реестр сплитов: ticker, date, ratio, kind
│   ├── renames.csv            # реестр переименований: old, new, date
│   ├── delisted.csv           # снятые с торгов: ticker, last_date, note
│   ├── key_rate.csv           # ключевая ставка ЦБ: date, rate
│   └── sectors.csv            # ticker, sector — секторный разрез
└── logs/                    # логи ночного обновления и выгрузок (не в git)
    ├── update.log
    └── backfill_*.log

moex-data/                   # данные: MOEX_DATA_ROOT = F:\moex-data (не в git, не в облаке)
├── data/                    # акции
│   ├── SBER/SBER.parquet
│   └── ...
├── indexes/                 # IMOEX.parquet, MCFTR.parquet, RGBITR.parquet
├── bonds/
│   ├── market_ALL/<YYYY>.parquet    # весь рынок облигаций с 1997 года, все доски и колонки
│   ├── market_TQOB/<YYYY>.parquet   # доска гособлигаций с 2021 года (рабочие колонки)
│   ├── market_TQCB/<YYYY>.parquet   # доска корпоративных облигаций с 2021 года
│   ├── securities.parquet           # реестр параметров всех выпусков, включая погашенные
│   ├── params.parquet               # снапшот параметров торгуемых выпусков досок
│   └── <SECID>.parquet              # истории отдельных выпусков (ранний механизм)
└── futures/
    └── history/<YYYY>.parquet       # все контракты FORTS с 2002 года

dividends/                   # соседний проект: F:\Yandex.Disk\FINANCE\dividends
├── data/<TICKER>.csv        # приведены к текущей акции — их читает moexutils
├── data/raw/<TICKER>.csv    # сырые значения с сайта
└── metadata/splits.json     # реестр сплитов проекта dividends (внешний реестр для moexutils)
```

**Почему данные вне Яндекс.Диска.** При многочасовых выгрузках и частой перезаписи годовых файлов клиент Яндекс.Диска создавал конфликтные копии и подменял файлы старыми серверными версиями — терялись даты (так пострадала история фьючерсов, восстановлена объединением версий). По той же причине `.git` исключен из синхронизации. Данные можно заново скачать с биржи, поэтому облачная копия им не нужна; код и реестры остаются в Яндекс.Диске и git.

Без `MOEX_DATA_ROOT` данные ищутся в папке проекта (`moexutils/data`, `moexutils/bonds`, ...). Переменная задана для пользователя Windows постоянно; ее видят новые процессы, включая ночную задачу.

---

## Акции: `data/<TICKER>/<TICKER>.parquet`

| Колонка | Описание |
|---------|----------|
| **индекс** `date` | Торговая дата (datetime64) |
| `open`, `low`, `high`, `close` | Цены основной сессии, руб.; `close` — официальное закрытие из ISS `/history` |
| `waprice` | Средневзвешенная цена дня |
| `volume` | Объем, шт. |
| `value_rub` | Оборот, руб. — не цена |
| `ticker` | Тикер |
| `adj_close` | Цена, скорректированная на дивиденды и сплиты, в текущей базе |
| `shares`, `market_cap` | Количество акций и капитализация (`close × shares`) |

Цены в файле — как их отдает ISS (без поправки на сплиты, если ISS ее не сделал); для доходностей применяйте `adjust_for_splits` или берите `adj_close`.

## Индексы: `indexes/<TICKER>.parquet`

Индекс `date`, колонки `close`, `volume`, `ticker`. Даты IMOEX (будни) — торговый календарь для проверок и докачки пропусков.

---

## Облигации

### Весь рынок: `bonds/market_ALL/<YYYY>.parquet`

Строка — выпуск на доске за торговую дату; ключ `date` + `SECID` + `BOARDID`. Все колонки истории ISS (набор со временем расширялся — в ранних годах часть колонок пустая):

| Группа | Колонки |
|--------|---------|
| Идентификация | `date`, `BOARDID`, `SECID`, `SHORTNAME`, `segment` (=`ALL`) |
| Цены (% от номинала) | `OPEN`, `LOW`, `HIGH`, `CLOSE`, `WAPRICE`, `LEGALCLOSEPRICE`, `MARKETPRICE2`, `MARKETPRICE3`, `ADMITTEDQUOTE` |
| Доходность и риск | `YIELDCLOSE`, `YIELDATWAP`, `YIELDTOOFFER`, `YIELDLASTCOUPON`, `DURATION` (дни), `ZSPREAD`, `ZSPREADATWAPRICE`, `CALLOPTIONYIELD`, `CALLOPTIONDURATION` |
| ОФЗ-ИН | `BEICLOSE` (breakeven-инфляция), `IRICPICLOSE` (индекс потребительских цен) |
| Купон и номинал | `COUPONPERCENT`, `COUPONVALUE`, `ACCINT` (НКД), `FACEVALUE`, `FACEUNIT`, `FACEVALUE_TYPE`, `CURRENCYID`, `COUPON_DETAILS` |
| Даты | `MATDATE`, `OFFERDATE`, `BUYBACKDATE`, `CALLOPTIONDATE`, `PUTOPTIONDATE`, `LASTTRADEDATE`, `DATEYIELDFROMISSUER` |
| Обороты | `VALUE`, `VOLUME`, `NUMTRADES`, `MP2VALTRD`, `MARKETPRICE3TRADESVALUE`, `ADMITTEDVALUE` |
| Тип | `BONDTYPE`, `BONDSUBTYPE` |

Доски по периодам: до перехода на режим Т+ основные торги шли на EQOB, EQNB, EQOS, EQNO и др.; TQOB работает с середины 2010-х, TQCB — примерно с 2019–2020 (точные даты видны в самих данных: `groupby('BOARDID')['date'].min()`); валютные — TQOD (USD), TQOE (EUR), TQOY (CNY), TQUD; TQRD. Типы колонок приведены: числовые — float, текстовые — string.

### Доски: `bonds/market_TQOB/`, `bonds/market_TQCB/`

Рабочий набор колонок: `date`, `SECID`, `SHORTNAME`, `CLOSE`, `LEGALCLOSEPRICE`, `YIELDCLOSE` (биржевая YTM, %), `DURATION` (дни), `VALUE`, `VOLUME`, `MATDATE`, `FACEVALUE`, `FACEUNIT`, `COUPONPERCENT`, `segment`; ключ `date` + `SECID`. История с 2021 года; их читает ноутбук `bond-market.py` (`read_bonds_market()` без аргумента).

### Реестр выпусков: `bonds/securities.parquet`

Строка на `SECID`: карточка ISS, включая погашенные выпуски — `ISIN`, `NAME`, `SHORTNAME`, `EMITTER_ID`, `REGNUMBER`, `ISSUEDATE`, `MATDATE`, `ISSUESIZE`, `FACEVALUE`, `INITIALFACEVALUE`, `FACEUNIT`, `COUPONFREQUENCY`, `COUPONPERCENT`, `TYPE`, `TYPENAME`, `BOND_TYPE`, `BOND_SUBTYPE`, `HASDEFAULT`, `HASTECHNICALDEFAULT`, `LISTLEVEL` и др. (поля зависят от выпуска) и `FETCHED` — дата запроса.

### Ранний механизм: `bonds/params.parquet`, `bonds/<SECID>.parquet`

`params.parquet` — снапшот параметров торгуемых выпусков досок (колонка `segment`). `<SECID>.parquet` — история одного выпуска (индекс `TRADEDATE`, колонки истории ISS, `secid`).

---

## Фьючерсы: `futures/history/<YYYY>.parquet`

Строка — контракт за торговую дату; ключ `date` + `SECID` + `BOARDID`.

| Колонка | Описание |
|---------|----------|
| `date`, `BOARDID`, `SECID`, `SHORTNAME` | Дата, режим, код контракта (`SiZ5`), краткое имя с месяцем и годом (`Si-12.25`) |
| `ASSETCODE` | Базовый актив (`Si`, `RTS`, `BR`, `GD`, `MIX`, ...) |
| `OPEN`, `LOW`, `HIGH`, `CLOSE`, `WAPRICE` | Цены |
| `SETTLEPRICE` | Расчетная цена (для вариационной маржи) |
| `OPENPOSITION`, `OPENPOSITIONVALUE` | Открытый интерес: контракты и руб. |
| `VOLUME`, `VALUE`, `NUMTRADES`, `QTY` | Объемы и число сделок |
| `SWAPRATE`, `SWAPRATE_CURR`, `CHANGE` | Своп-ставка (для вечных фьючерсов), изменение |

Коды контрактов повторяются раз в 10 лет, год берите из `SHORTNAME`.

---

## Служебные файлы хранилищ

В каждой папке хранилища истории (`bonds/market_*`, `futures/history`):

- `_empty_dates.csv` — торговые (по IMOEX) даты, за которые ISS подтвержденно не вернул строк; больше не запрашиваются.
- `.lock` — блокировка на время записи; ночное обновление пропускает занятое хранилище. Блокировка старше 12 часов считается брошенной.

---

## Метаданные

### stock-index-base.xlsx

Листы с датами `DD.MM.YYYY`, на листе колонки `Code` (тикер) и `Number of issued shares`, шапка на 4-й строке (`skiprows=3`). Между срезами — forward fill, до первого среза — backward fill.

### Реестры

| Файл | Колонки | Назначение |
|------|---------|------------|
| `splits.csv` | `ticker, date, ratio, kind` | Сплиты и консолидации (`price` / `shares` / `auto`) |
| `renames.csv` | `old, new, date` | Склейка историй переименованных тикеров |
| `delisted.csv` | `ticker, last_date, note` | Снятые с торгов — не опрашиваются при обновлении (статус подтвержден ISS) |
| `key_rate.csv` | `date, rate` | Ключевая ставка ЦБ; дописывается автоматически с cbr.ru |
| `sectors.csv` | `ticker, sector` | Секторный разрез в ноутбуках |

Корпоративные события вносятся только через реестры, данные руками не правятся. Папка `metadata/` в `.gitignore`: новый файл реестра добавляется в git принудительно (`git add -f`).

---

## Дивиденды: `../dividends/data/<TICKER>.csv`

Соседний проект `F:\Yandex.Disk\FINANCE\dividends` собирает историю выплат с сайта закрытияреестров.рф (`parse_all_dividends.py`) и приводит значения к текущей акции с учетом сплитов. moexutils читает `data/<TICKER>.csv`:

- **closing_date** — дата закрытия реестра; экс-дата выводится из нее по режиму расчетов (T+1 с 31.07.2023, раньше T+2);
- **dividend_value** — руб. на акцию (> 0); `year`, `period_type` — за какой период.

Привилегированные акции — в файле `<TICKER>P.csv` по правилам проекта dividends. Объявленные, но еще не наступившие выплаты в файле бывают — adj_close их игнорирует до экс-даты. Если сайт еще не внес выплату, проверка данных покажет ее как `dividend_gap`.
