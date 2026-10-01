"""Daily futures/options positions by individuals and legal entities.

ISS aggregates contracts by underlying asset; options retain their type/style
dimensions. Counts of people on opposite sides are not disjoint populations.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import logging
from urllib.parse import quote

import polars as pl

from moexutils import iss, lake

logger = logging.getLogger('moexutils')
TABLES = {'forts': 'futures_open_positions', 'options': 'options_open_positions'}
START = dt.date(2025, 1, 1)
OPTION_DIMS = ['asset_type', 'option_type', 'margin_style', 'exec_type', 'settle_type']
COUNTERS = ['persons_long', 'persons_short', 'open_position_long', 'open_position_short',
            'oichange_long', 'oichange_short']


def _url(market):
    if market not in TABLES:
        raise ValueError(f'Unknown market: {market}')
    return f'{iss.ISS_URL}/statistics/engines/futures/markets/{market}/openpositions'


def _block(response, name):
    response.raise_for_status()
    payload = response.json().get(name)
    if (not isinstance(payload, dict) or not isinstance(payload.get('columns'), list)
            or not isinstance(payload.get('data'), list)
            or any(not isinstance(row, list) or len(row) != len(payload['columns'])
                   for row in payload['data'])):
        raise ValueError(f'ISS: invalid {name} block')
    return iss.to_frame(payload)


def fetch_assets(market, session=None):
    session = session or iss.make_session()
    frame = _block(session.get(_url(market) + '.json'), 'assets')
    if frame.is_empty() or not {'asset_code', 'date_from', 'date_till'} <= set(frame.columns):
        raise ValueError(f'ISS: empty/incomplete {market} positions assets')
    return (frame.with_columns(pl.col('date_from', 'date_till').str.to_date('%Y-%m-%d'))
            .group_by('asset_code').agg(pl.col('date_from').min(), pl.col('date_till').max())
            .rename({'asset_code': 'asset'}).with_columns(pl.lit(market).alias('market')))


def fetch_positions(market, asset, start, end):
    """The ISS asset endpoint returns the complete requested range without a cursor."""
    with iss.make_session() as session:
        frame = _block(session.get(f'{_url(market)}/{quote(asset, safe="")}.json',
                                   params={'from': str(start), 'till': str(end)}), 'open_positions')
    if frame.is_empty():
        return frame
    required = {'tradedate', 'asset', 'is_fiz', *COUNTERS}
    if market == 'options':
        required.update(OPTION_DIMS)
    if not required <= set(frame.columns):
        raise ValueError('ISS: incomplete open positions columns')
    for col in ['is_fiz', *COUNTERS]:
        if frame.filter(pl.col(col) != pl.col(col).floor()).height:
            raise ValueError(f'ISS: invalid integral positions field {col}')
    frame = (frame.with_columns(pl.col('tradedate').str.to_date('%Y-%m-%d').alias('date'),
                                pl.col('is_fiz').cast(pl.Int8),
                                pl.col(COUNTERS).cast(pl.Int64)).drop('tradedate'))
    if frame.filter((pl.col('asset') != asset) | (pl.col('date') < start)
                    | (pl.col('date') > end) | ~pl.col('is_fiz').is_in([0, 1])).height:
        raise ValueError('ISS: positions asset/date/group mismatch')
    key = lake.TABLE_KEYS[TABLES[market]]
    if frame.select(key).null_count().sum_horizontal()[0] or frame.select(key).is_duplicated().any():
        raise ValueError('ISS: missing/duplicate positions key')
    if frame.filter(pl.any_horizontal(pl.col(COUNTERS[:4]) < 0)).height:
        raise ValueError('ISS: negative positions/person count')
    return frame


def diagnostics(market='forts', start=None, end=None):
    """Published group count and aggregate long-minus-short; missing is not zero."""
    frame = read(market, start=start, end=end)
    dims = [c for c in lake.TABLE_KEYS[TABLES[market]] if c != 'is_fiz']
    return frame.group_by(dims).agg(
        pl.col('is_fiz').n_unique().alias('groups_published'),
        pl.when(pl.col('open_position_long').null_count() == 0)
          .then(pl.col('open_position_long').sum()).otherwise(None).alias('long'),
        pl.when(pl.col('open_position_short').null_count() == 0)
          .then(pl.col('open_position_short').sum()).otherwise(None).alias('short')).with_columns(
            (pl.col('long') - pl.col('short')).alias('balance_difference'))


def update(start=None, end=None, workers=4, force=False):
    """Resumable 2025+ backfill; nightly refresh includes a seven-day overlap.

    Each asset is checkpointed only after its complete range has been saved.
    Failures preserve successful assets and raise an ExceptionGroup.
    """
    if not 1 <= workers <= 8:
        raise ValueError('workers must be 1..8')
    first = dt.date.fromisoformat(str(start)[:10]) if start is not None else START
    end = dt.date.fromisoformat(str(end)[:10]) if end is not None else dt.date.today() - dt.timedelta(days=1)
    if first > end or end >= dt.date.today():
        raise ValueError('Require start <= end < today')
    assets = pl.concat([fetch_assets(market) for market in TABLES])
    if assets.select('market', 'asset').null_count().sum_horizontal()[0]:
        raise ValueError('ISS: missing positions asset identity')
    tables = set(lake.tables())
    old_assets = (lake.query('SELECT * FROM lake.open_position_assets')
                  if 'open_position_assets' in tables else assets.clear())
    lake.write('open_position_assets', lake.changed_rows(old_assets, assets, ['market', 'asset']))
    state = (dict(lake.query('SELECT name,date FROM lake.load_state').iter_rows())
             if 'load_state' in tables else {})
    jobs = []
    for row in assets.to_dicts():
        begin, till = max(first, row['date_from']), min(end, row['date_till'])
        name = f"{TABLES[row['market']]}:{row['asset']}"
        done = state.get(name)
        if begin > till or (not force and done is not None and done >= till):
            continue
        if done is not None and not force:
            begin = max(begin, done - dt.timedelta(days=6))
        jobs.append((row['market'], row['asset'], begin, till, name))
    written, deleted, completed, errors = 0, 0, 0, []
    logger.info(f'[INFO] Open positions: {len(jobs)} pending assets')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(fetch_positions, *job[:4]): job for job in jobs}
        for future in as_completed(pending):
            market, asset, begin, till, name = pending[future]
            try:
                frame = future.result()
                table = TABLES[market]
                stale = None
                if table in lake.tables():
                    old = lake.query(f'SELECT * FROM lake.{table} WHERE asset=? AND date BETWEEN ? AND ?',
                                     [asset, begin, till])
                    if frame.is_empty() and old.height:
                        raise ValueError('ISS: unexpectedly empty previously populated positions range')
                    if frame.height:
                        key = lake.TABLE_KEYS[table]
                        stale = old.select(key).join(frame.select(key), on=key, how='anti')
                        frame = lake.changed_rows(old, frame, lake.TABLE_KEYS[table])
                written += lake.write(table, frame, delete=stale)
                deleted += 0 if stale is None else stale.height
                lake.write('load_state', pl.DataFrame({'name': [name], 'date': [till]}))
                completed += 1
                if completed % 20 == 0 or completed == len(jobs):
                    logger.info(f'[INFO] Open positions: {completed}/{len(jobs)} assets, {written} rows, {deleted} deleted')
            except Exception as error:
                logger.warning(f'[WARN] Open positions {market}/{asset}: {error}')
                errors.append(error)
    if errors:
        raise ExceptionGroup('Open positions incomplete; rerun to resume', errors)
    return {'assets_processed': completed, 'rows_written': written, 'rows_deleted': deleted}


def read(market='forts', assets=None, start=None, end=None, is_fiz=None, as_of=None):
    """Read daily client-group positions; market='forts'/'options', is_fiz=1/0."""
    _url(market)
    table = TABLES[market]
    if table not in lake.tables():
        raise FileNotFoundError('Run openpositions.update() first')
    where, params = [], []
    if assets is not None:
        assets = [assets] if isinstance(assets, str) else list(assets)
        where.append(f"asset IN ({','.join('?' * len(assets))})" if assets else 'FALSE')
        params.extend(assets)
    for op, value in (('>=', start), ('<=', end)):
        if value is not None:
            where.append(f'date {op} CAST(? AS DATE)')
            params.append(str(value)[:10])
    if is_fiz is not None:
        if is_fiz not in (0, 1):
            raise ValueError('is_fiz must be 0 or 1')
        where.append('is_fiz=?')
        params.append(int(is_fiz))
    return lake.query(f'SELECT * FROM {lake.ref(table, as_of)}'
                      + (' WHERE ' + ' AND '.join(where) if where else '') + ' ORDER BY date,asset,is_fiz', params)
