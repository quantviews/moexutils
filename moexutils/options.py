"""Options series and contracts, including expired series covering stored history.

Contract identity is (secid, series_name): short codes may be reused.
Series lot size comes from an ISS security card belonging to that exact series;
the source secid is retained. It is never inferred from a ticker code.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import logging
import math
import re
from urllib.parse import quote

import polars as pl

from moexutils import iss, lake

logger = logging.getLogger('moexutils')
URL = f'{iss.ISS_URL}/statistics/engines/futures/markets/options/series'
LOT_FIELDS = {'lot_size': pl.Float64, 'lot_size_source_secid': pl.String,
              'unit': pl.String, 'faceunit': pl.String}
STATE_PREFIX = 'options_series:'


def _write_changes(table, frame):
    if frame.is_empty():
        return 0
    if table in lake.tables():
        if table == 'options_contracts':
            values = frame['series_name'].unique().to_list()
            old = lake.query(f"SELECT * FROM lake.{table} WHERE series_name IN "
                             f"({','.join('?' * len(values))})", values)
        else:
            old = lake.query(f'SELECT * FROM lake.{table}')
        frame = lake.changed_rows(old, frame, lake.TABLE_KEYS[table])
    return lake.write(table, frame)


def _block(response, name):
    response.raise_for_status()
    payload = response.json().get(name)
    if (not isinstance(payload, dict) or not isinstance(payload.get('columns'), list)
            or not isinstance(payload.get('data'), list)
            or any(not isinstance(row, list) or len(row) != len(payload['columns'])
                   for row in payload['data'])):
        raise ValueError(f'ISS: invalid options {name} block')
    return iss.to_frame(payload)


def fetch_series(session=None) -> pl.DataFrame:
    session = session or iss.make_session()
    frame = _block(session.get(URL + '.json', params={'show_expired': 1}), 'series')
    required = {'name', 'start_date', 'expiration_date', 'asset_code', 'underlying_asset'}
    if frame.is_empty() or not required <= set(frame.columns):
        raise ValueError('ISS: empty or incomplete options series registry')
    # ISS includes one anonymous placeholder, not an identifiable series.
    placeholder = (pl.col('name') == '') & pl.col('start_date').is_null() & pl.col('expiration_date').is_null() & pl.col('asset_code').is_null()
    frame = frame.filter(~placeholder.fill_null(False))
    if frame.is_empty():
        raise ValueError('ISS: options registry contains only placeholders')
    frame = frame.with_columns(
        pl.col('start_date', 'expiration_date').str.to_date('%Y-%m-%d', strict=True))
    if (frame['name'].null_count() or frame['name'].n_unique() != frame.height
            or frame.filter(pl.col('name') == '').height):
        raise ValueError('ISS: options series identity is missing or duplicated')
    if frame['expiration_date'].null_count():
        raise ValueError('ISS: options series expiration date is missing')
    return frame


def fetch_series_contracts(series: dict) -> tuple[pl.DataFrame, dict]:
    """Fetch one complete, unpaginated series and its verified lot-size source."""
    with iss.make_session() as session:
        frame = _block(session.get(f"{URL}/{quote(series['name'], safe='')}/securities.json"),
                       'securities')
        required = {'secid', 'shortname', 'option_type', 'strike', 'history_from', 'history_till'}
        if not required <= set(frame.columns):
            raise ValueError(f"ISS: incomplete options contracts for {series['name']}")
        if frame.height and (frame['secid'].null_count()
                             or frame['secid'].n_unique() != frame.height
                             or frame.filter(~pl.col('option_type').is_in(['C', 'P'])
                                             | pl.col('option_type').is_null()).height):
            raise ValueError(f"ISS: invalid contract identity/type/strike for {series['name']}")
        records = frame.to_dicts()
        # A small set of archived ISS contracts has a null numeric strike, but
        # retains its full exchange code. Recover only with an exact series
        # prefix and the exchange's documented type/style/strike encoding.
        pattern = re.compile(re.escape(series['name'][:-2]) + r'([CP])'
                             + re.escape(series['name'][-1]) + r'\s*([+-]?\d+(?:\.\d+)?)$')
        for row in records:
            row['strike_source'] = 'ISS'
            if row['strike'] is None:
                match = pattern.fullmatch(row['shortname'] or '')
                if not match or match[1] != row['option_type']:
                    raise ValueError(f"ISS: cannot recover missing strike for {row['secid']}")
                row['strike'] = float(match[2])
                row['strike_source'] = 'full_code'
        if records:
            frame = pl.DataFrame(records, schema={**frame.schema, 'strike': pl.Float64,
                                                  'strike_source': pl.String})
        else:
            frame = frame.with_columns(pl.lit('ISS').alias('strike_source'))
        frame = frame.with_columns(
            pl.col('history_from', 'history_till').str.to_date('%Y-%m-%d', strict=True),
            pl.col('strike').cast(pl.Float64), pl.lit(series['name']).alias('series_name'))
        metadata = {**series, **{k: series.get(k) for k in LOT_FIELDS}}
        if metadata['lot_size'] is None:
            candidates = frame.sort(pl.col('history_till') - pl.col('history_from'),
                                    descending=True, nulls_last=True)
            for secid in candidates['secid'].head(10):
                desc = iss.security_description(secid, session)
                # A short code can now describe a different expiration year.
                if desc.get('SERIES_NAME') != series['name']:
                    continue
                if desc.get('LOTSIZE') is not None:
                    size = float(desc['LOTSIZE'])
                    if not math.isfinite(size) or size <= 0:
                        raise ValueError(f'ISS: invalid option lot size for {secid}')
                    metadata.update(lot_size=size, lot_size_source_secid=secid,
                                    unit=desc.get('UNIT'), faceunit=desc.get('FACEUNIT'))
                    break
        return frame, metadata


def _recover_missing_contracts(workers: int) -> int:
    """ISS omits some expired series; recover their contracts from security cards."""
    if 'options' not in lake.tables():
        return 0
    missing = lake.query('SELECT h.SECID, min(h.date) AS first, max(h.date) AS last '
                         'FROM lake.options h ANTI JOIN lake.options_contracts c '
                         'ON h.SECID=c.secid AND h.date BETWEEN c.history_from AND c.history_till '
                         'GROUP BY h.SECID').to_dicts()
    if not missing:
        return 0
    logger.info(f'[INFO] Options registry: {len(missing)} missing codes; checking ISS cards')
    schema = lake.query('SELECT * FROM lake.options_series LIMIT 0').schema
    known = set(lake.query('SELECT name FROM lake.options_series')['name'])

    def fetch(item):
        with iss.make_session() as session:
            desc = iss.security_description(item['SECID'], session)
        required = ('SERIES_NAME', 'STRIKE', 'OPTIONTYPE', 'FRSTTRADE', 'LSTDELDATE',
                    'ASSETCODE', 'UNDERLYINGASSET', 'SHORTNAME')
        if any(not desc.get(k) for k in required) or desc['OPTIONTYPE'] not in ('C', 'P'):
            raise ValueError(f"ISS: incomplete fallback card for {item['SECID']}")
        first = dt.date.fromisoformat(desc['FRSTTRADE'])
        expiry = dt.date.fromisoformat(desc['LSTDELDATE'])
        if first > item['first'] or expiry < item['last']:
            raise ValueError(f"ISS: fallback card date mismatch for {item['SECID']}")
        contract = {'secid': item['SECID'], 'shortname': desc['SHORTNAME'],
                    'option_type': desc['OPTIONTYPE'], 'strike': float(desc['STRIKE']),
                    'history_from': first, 'history_till': item['last'],
                    'is_traded': None, 'series_name': desc['SERIES_NAME'],
                    'strike_source': 'description'}
        series = {key: None for key in schema}
        series.update(name=desc['SERIES_NAME'], start_date=first, expiration_date=expiry,
                      asset_code=desc['ASSETCODE'], underlying_asset=desc['UNDERLYINGASSET'],
                      exec_type={'Американский': 'A', 'Европейский': 'E'}.get(desc.get('EXECTYPE')),
                      margin_style={'Маржируемый': 'M', 'Премиальный': 'P'}.get(desc.get('MARGINSTYLE')),
                      lot_size=float(desc['LOTSIZE']) if desc.get('LOTSIZE') else None,
                      lot_size_source_secid=item['SECID'] if desc.get('LOTSIZE') else None,
                      unit=desc.get('UNIT'), faceunit=desc.get('FACEUNIT'))
        return contract, series

    contracts, series, errors = [], {}, []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(fetch, item) for item in missing]
        for future in as_completed(jobs):
            try:
                contract, row = future.result()
                contracts.append(contract)
                if row['name'] not in known:
                    if row['name'] in series:
                        row['start_date'] = min(row['start_date'], series[row['name']]['start_date'])
                    series[row['name']] = row
            except Exception as error:
                errors.append(error)
    if series:
        _write_changes('options_series', pl.DataFrame(list(series.values()), schema=schema))
    if contracts:
        frame = pl.DataFrame(contracts).with_columns(pl.col('is_traded').cast(pl.Float64))
        _write_changes('options_contracts', frame)
    if errors:
        raise ExceptionGroup('Options fallback cards incomplete; rerun to resume', errors)
    return len(contracts)


def update_registry(start=None, end=None, workers: int = 4, flush_every: int = 50) -> dict:
    """Resume expired-series backfill and refresh active series once per day.

    Scope defaults to the date range of stored options history. Completed series
    are checkpointed only after both contracts and metadata have been saved.
    A failed series remains pending; successful series are saved before raising.
    """
    if not 1 <= workers <= 8 or flush_every <= 0:
        raise ValueError('workers must be 1..8 and flush_every must be positive')
    tables = set(lake.tables())
    if 'options' not in tables and (start is None or end is None):
        raise FileNotFoundError('Load options history or specify start and end')
    if start is None or end is None:
        bounds = lake.query('SELECT min(date) AS start, max(date) AS end FROM lake.options').row(0)
        start, end = start or bounds[0], end or bounds[1]
    start, end = dt.date.fromisoformat(str(start)[:10]), dt.date.fromisoformat(str(end)[:10])
    if start > end:
        raise ValueError('start must not exceed end')
    today = dt.date.today()
    series = fetch_series()
    if 'options_series' in tables:
        old = lake.query('SELECT * FROM lake.options_series')
        series = series.join(old.select('name', *LOT_FIELDS), on='name', how='left')
    else:
        series = series.with_columns([pl.lit(None, dtype=t).alias(k) for k, t in LOT_FIELDS.items()])
    # Preserve existing archive rows if the upstream registry ever contracts.
    _write_changes('options_series', series)
    state = (dict(lake.query('SELECT name,date FROM lake.load_state').iter_rows())
             if 'load_state' in tables else {})
    relevant = series.filter(pl.col('expiration_date') >= start,
                             pl.col('start_date').fill_null(start) <= end)
    todo = [row for row in relevant.to_dicts()
            if state.get(STATE_PREFIX + row['name'], dt.date.min)
            < min(today, row['expiration_date']) or row['lot_size'] is None]
    logger.info(f'[INFO] Options registry: {len(todo)} pending / {relevant.height} relevant series')
    frames, metadata, errors = [], [], []
    completed, written = 0, 0

    def flush():
        nonlocal frames, metadata, written
        if not metadata:
            return
        nonempty = [frame for frame in frames if frame.height]
        if nonempty:
            written += _write_changes('options_contracts', pl.concat(nonempty, how='diagonal_relaxed'))
        rows = pl.DataFrame(metadata, schema=series.schema)
        _write_changes('options_series', rows)
        lake.write('load_state', pl.DataFrame({'name': [STATE_PREFIX + r['name'] for r in metadata],
                                             'date': [today] * len(metadata)}))
        logger.info(f'[INFO] Options registry: saved {completed}/{len(todo)} series, {written} contract rows')
        frames, metadata = [], []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(fetch_series_contracts, row): row['name'] for row in todo}
        for future in as_completed(jobs):
            try:
                frame, row = future.result()
            except Exception as error:
                logger.warning(f'[WARN] Options series {jobs[future]}: {error}')
                errors.append(error)
                continue
            frames.append(frame)
            metadata.append(row)
            completed += 1
            if len(metadata) >= flush_every:
                flush()
        flush()
    if errors:
        raise ExceptionGroup('Options registry incomplete; rerun to resume', errors)
    recovered = _recover_missing_contracts(workers)
    lake.write('load_state', pl.DataFrame({'name': ['options_registry'], 'date': [today]}))
    return {'series_processed': completed, 'contract_rows_written': written,
            'contracts_recovered_from_cards': recovered}


def read_contracts(secids=None, assets=None, as_of: lake.AsOf = None) -> pl.DataFrame:
    """Contract parameters joined to series; lot_size is the series lot size.

    Join history on SECID=secid and date within history_from..history_till.
    Do not join on the short code alone: it may have multiple series identities.
    """
    if not {'options_contracts', 'options_series'} <= set(lake.tables()):
        raise FileNotFoundError('Run options.update_registry() first')
    where, params = [], []
    for column, values in (('c.secid', secids), ('s.asset_code', assets)):
        if values is not None:
            values = [values] if isinstance(values, str) else list(values)
            where.append(f"{column} IN ({','.join('?' * len(values))})" if values else 'FALSE')
            params.extend(values)
    sql = (f"SELECT c.*, s.* EXCLUDE(name,is_traded) FROM {lake.ref('options_contracts', as_of)} c "
           f"JOIN {lake.ref('options_series', as_of)} s ON c.series_name=s.name")
    return lake.query(sql + (' WHERE ' + ' AND '.join(where) if where else '')
                      + ' ORDER BY c.secid, s.expiration_date', params)
