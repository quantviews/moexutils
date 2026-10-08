"""Observed public foreign reference bars, isolated from MOEX market history.

These provider front-series/spot/ETF bars are research proxies, NOT contract-level
settlements or a historical MOEX price reconstruction. No pandas dependency.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

import polars as pl

from moexutils import iss, lake

SOURCES = {
    'COFFEE': ('KC=F', 'USX', 'provider_front_futures', 'https://www.moex.com/ru/fyuchers-na-kofe'),
    'TTF': ('TTF=F', 'EUR', 'provider_front_futures', 'https://www.moex.com/ru/gas-datch'),
    'ORANGE': ('OJ=F', 'USX', 'provider_front_futures', 'https://www.moex.com/a113'),
    'ED': ('EURUSD=X', 'USD', 'spot_not_forward', 'https://www.moex.com/a3873'),
    'UCNY': ('CNY=X', 'CNY', 'onshore_spot_not_forward', 'https://www.moex.com/a3873'),
    'NIKK': ('1329.T', 'JPY', 'ETF_market_price_not_NAV', 'https://www.moex.com/n55315?nt=112'),
}


def parse_chart(payload, asset, start, end):
    symbol, currency, kind, _ = SOURCES[asset]
    chart = payload.get('chart', {})
    if chart.get('error') or not chart.get('result') or len(chart['result']) != 1:
        raise ValueError(f'{asset}: empty/invalid chart')
    result = chart['result'][0]
    meta = result['meta']
    if meta.get('symbol') != symbol or meta.get('currency') != currency:
        raise ValueError(f'{asset}: unexpected symbol/currency')
    if result.get('events', {}).get('splits'):
        raise ValueError(f'{asset}: split events need an explicit mapping before research')
    zone = ZoneInfo(meta['exchangeTimezoneName'])
    timestamps = result.get('timestamp', [])
    quote_data = result['indicators']['quote'][0]
    if any(len(quote_data.get(c, [])) != len(timestamps) for c in ('open', 'high', 'low', 'close', 'volume')):
        raise ValueError('Inconsistent bar lengths')
    data = {
        'source_timestamp': [dt.datetime.fromtimestamp(t, dt.timezone.utc) for t in timestamps],
        'date': [dt.datetime.fromtimestamp(t, zone).date() for t in timestamps],
        **{c: quote_data[c] for c in ('open', 'high', 'low', 'close', 'volume')},
    }
    frame = pl.DataFrame(data, schema_overrides={'date': pl.Date, 'source_timestamp': pl.Datetime('us', 'UTC'),
        **{c: pl.Float64 for c in ('open', 'high', 'low', 'close', 'volume')}}).filter(
            pl.col('date').is_between(start, end))
    if frame.is_empty() or frame['date'].n_unique() != frame.height:
        raise ValueError(f'{asset}: empty or duplicate daily dates')
    return frame.with_columns(pl.lit(asset).alias('asset'), pl.lit(symbol).alias('source_symbol'),
        pl.lit(currency).alias('source_currency'), pl.lit(kind).alias('source_kind'),
        pl.lit(False).alias('publication_time_verified'), pl.lit(False).alias('roll_method_verified'),
        pl.lit(False).alias('historical_vintage_verified')).sort('date')


def download(start=dt.date(1970, 1, 1), end=None, output=None):
    now = dt.datetime.now(dt.timezone.utc)
    end = end or (now.astimezone(dt.timezone(dt.timedelta(hours=3))).date() - dt.timedelta(days=1))
    if start > end:
        raise ValueError('start after end')
    root = Path(output) if output else Path(lake.DATA_ROOT) / 'reference/foreign' / now.strftime('%Y%m%dT%H%M%S%fZ')
    root.mkdir(parents=True, exist_ok=False)
    session = iss.make_session()
    records, frames = [], []
    for asset, (symbol, currency, kind, mapping_url) in SOURCES.items():
        url = f'https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe="")}'
        params = {'period1': int(dt.datetime.combine(start, dt.time(), dt.timezone.utc).timestamp()),
                  'period2': int(dt.datetime.combine(end + dt.timedelta(days=2), dt.time(), dt.timezone.utc).timestamp()),
                  'interval': '1d', 'events': 'div,splits'}
        try:
            response = session.get(url, params=params, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            payload = response.json()
            envelope = {'url': response.url, 'observed_at': dt.datetime.now(dt.timezone.utc).isoformat(),
                        'requested_start': str(start), 'requested_end': str(end), 'payload': payload}
            (root / f'{asset}-raw.json').write_text(json.dumps(envelope, ensure_ascii=False), encoding='utf-8')
            frame = parse_chart(payload, asset, start, end)
            if frame.filter(pl.col('close').is_finite()).height < 250:
                raise ValueError('Fewer than 250 usable daily bars; not a long-history input')
            frame.write_parquet(root / f'{asset}.parquet')
            frames.append(frame)
            records.append({'asset': asset, 'symbol': symbol, 'currency': currency, 'kind': kind,
                'mapping_source': mapping_url, 'rows': frame.height, 'first': str(frame['date'].min()),
                'last': str(frame['date'].max()), 'null_close': frame['close'].null_count(), 'status': 'downloaded'})
        except Exception as exc:
            records.append({'asset': asset, 'symbol': symbol, 'status': 'error', 'error': str(exc)})
        print(records[-1], flush=True)
    if frames:
        pl.concat(frames).write_parquet(root / 'reference.parquet')
    meta = {'schema_version': 1, 'generated_at': now.isoformat(), 'provider': 'Yahoo Finance public chart',
            'status': 'complete' if len(frames) == len(SOURCES) else 'partial', 'sources': records,
            'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'limitations': ['not_exchange_contract_archive', 'unknown_front_roll', 'unknown_historical_publication',
                            'ETF_close_not_NAV', 'FX_spot_not_futures', 'no_redistribution_license_inferred'],
            'artifact_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()}}
    (root / 'summary.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    return root


def read_reference(folder):
    root = Path(folder)
    meta = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    if meta.get('schema_version') != 1 or meta.get('status') != 'complete':
        raise ValueError('Complete reference package version 1 required')
    p = root / 'reference.parquet'
    if hashlib.sha256(p.read_bytes()).hexdigest() != meta['artifact_sha256'].get(p.name):
        raise ValueError('Reference checksum mismatch')
    return pl.read_parquet(p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=dt.date.fromisoformat, default=dt.date(1970, 1, 1))
    parser.add_argument('--end', type=dt.date.fromisoformat)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = download(args.start, args.end, args.output)
    print(result)
    raise SystemExit(int(json.loads((result / 'summary.json').read_text(encoding='utf-8'))['status'] != 'complete'))
