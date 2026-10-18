"""Read-only positioning audit: bounded source probes, no lake updates."""
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta, datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import polars as pl
from moexutils import iss, lake

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://iss.moex.com/iss/statistics/engines/futures/markets/forts/openpositions'
INPUT = ROOT.parent / 'futures-analytics/outputs/positioning-audit-20261016-accepted'
OUT = Path('F:/moex-data/reports/positioning-availability-20261016-v2')
KEY = ['date', 'asset', 'is_fiz']


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str)+'\n', encoding='utf-8')


def fetch(asset, start, end, label):
    url = BASE + (f'/{asset}' if asset else '') + '.json'
    params = {'from': str(start), 'till': str(end)} if asset else {}
    with iss.make_session() as session:
        response = session.get(url, params=params)
        response.raise_for_status()
        data = response.json()
    write(f'sources/{label}.json', data)
    write(f'sources/{label}.provenance.json', dict(url=response.url,
          observed_at=datetime.now(timezone.utc).isoformat(), published_at=None))
    block = data['open_positions' if asset else 'assets']
    frame = iss.to_frame(block)
    if asset and frame.height:
        frame = frame.with_columns(pl.col('tradedate').str.to_date().alias('date')).drop('tradedate')
    return frame


def differences(old, new):
    cols = sorted(set(old.columns) & set(new.columns) - set(KEY))
    a = {tuple(r[k] for k in KEY): tuple(r[c] for c in cols) for r in old.to_dicts()}
    b = {tuple(r[k] for k in KEY): tuple(r[c] for c in cols) for r in new.to_dicts()}
    common = a.keys() & b.keys()
    changed = [k for k in common if a[k] != b[k]]
    return dict(common=len(common), changed=len(changed), added=len(b.keys()-a.keys()),
                removed=len(a.keys()-b.keys()), changed_key_examples=changed[:20])


def run():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'sources').mkdir()
    cfg = json.loads((ROOT.parent / 'futures-analytics/configs/universe24.json').read_text())
    assets = cfg['original18'] + cfg['additions']
    write('universe24.json', cfg)
    for p in INPUT.iterdir():
        if p.is_file():
            shutil.copyfile(p, OUT / ('consumer_' + p.name))
    expected = json.loads((INPUT / 'manifest.json').read_text())
    checks = []
    for name in ['positions.parquet', 'registry.parquet', 'quality.parquet']:
        matching = [h for n, h in expected.items() if n.replace('\\', '/').endswith('/'+name)]
        actual = hashlib.sha256((INPUT / name).read_bytes()).hexdigest()
        if matching != [actual]:
            raise ValueError('Consumer manifest mismatch: ' + name)
        checks.append(dict(file=name, sha256=actual, verified=True))
    write('consumer_input_checks.json', checks)
    old = pl.read_parquet(INPUT/'positions.parquet').with_columns(pl.col('date').cast(pl.Date))
    registry = fetch(None, None, None, 'assets')
    registry.write_parquet(OUT/'source_registry.parquet')
    bounds = registry.group_by('asset_code').agg(pl.col('date_from').min(), pl.col('date_till').max())
    bound_map = {r['asset_code']: r for r in bounds.to_dicts()}
    jobs = []
    for a in assets:
        d = date.fromisoformat(bound_map[a]['date_from'])
        jobs.append((a, d, d+timedelta(days=10), 'first-'+a))
        if d.year < 2025:
            jobs.append((a, date(2024,12,16), date(2024,12,30), 'pre2025-'+a))
    jobs.append(('ORANGE', date(2025,6,10), date(2026,10,5), 'orange-full'))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda args: fetch(*args), jobs))
    probes = []
    for job, frame in zip(jobs, results):
        probes.append(dict(asset=job[0], label=job[3], rows=frame.height,
                           first=frame['date'].min() if frame.height else None,
                           last=frame['date'].max() if frame.height else None,
                           registry_first=bound_map[job[0]]['date_from']))
    write('history_probes.json', probes)
    orange = results[-1]
    orange.write_parquet(OUT/'orange_source.parquet')
    orange_old = old.filter((pl.col('asset')=='ORANGE') & (pl.col('date')<=date(2026,10,5)))
    write('orange_comparison.json', differences(orange_old, orange))
    incomplete = orange.group_by('date').agg(pl.col('is_fiz').alias('groups')).filter(pl.col('groups').list.len()!=2).sort('date')
    write('orange_incomplete.json', incomplete.to_dicts())
    with lake.session(read_only=True) as con:
        current = con.execute('SELECT * FROM lake.futures_open_positions').pl()
        current.write_parquet(OUT/'local_positions.parquet')
        con.execute('SELECT * FROM lake.futures_contracts').pl().write_parquet(OUT/'contracts.parquet')
        # Source ASSETCODE avoids SECID reuse and current-registry join duplication.
        market_oi = con.execute("""SELECT date,ASSETCODE AS asset,sum(OPENPOSITION) AS market_oi,
            count(*) AS contract_rows FROM lake.futures WHERE BOARDID='RFUD'
            AND date IN ('2025-01-03','2025-06-10','2026-10-05') GROUP BY date,ASSETCODE""").pl()
        groups = current.filter(pl.col('asset').is_in(assets)).group_by('date','asset').agg(
            pl.col('is_fiz').n_unique().alias('groups'),
            pl.col('open_position_long').sum().alias('long'),
            pl.col('open_position_short').sum().alias('short'))
        comparison = groups.join(market_oi,on=['date','asset']).with_columns(
            (pl.col('long')+pl.col('short')-pl.col('market_oi')).alias('difference'))
        comparison.write_parquet(OUT/'contract_oi_comparison.parquet')
        write('contract_oi_summary.json',dict(rows=comparison.height,
             equal=comparison.filter(pl.col('difference')==0).height,
             differences=comparison.filter(pl.col('difference')!=0).to_dicts()))
        snaps = con.execute("SELECT * FROM ducklake_snapshots('lake') ORDER BY snapshot_id").pl()
        info = con.execute("SELECT * FROM ducklake_table_info('lake') WHERE table_name='futures_open_positions'").pl().row(0,named=True)
        sid = str(info['table_id'])
        selected = [r for r in snaps.to_dicts() if any(sid in c['value'] or 'main.futures_open_positions' in c['value'] for c in r['changes'])]
        write('table_snapshots.json', selected)
        daily = {}
        for r in selected:
            daily[str(r['snapshot_time'].date())] = r
        comparisons = []
        previous = None
        for day, r in daily.items():
            f = con.execute(f"SELECT * FROM lake.futures_open_positions AT (VERSION => {int(r['snapshot_id'])})").pl().filter(pl.col('asset').is_in(assets))
            f.write_parquet(OUT/f'snapshot-{r["snapshot_id"]}.parquet')
            comparisons.append(dict(day=day, snapshot_id=r['snapshot_id'], snapshot_time=r['snapshot_time'],
                                    rows=f.height, comparison_to_previous=None if previous is None else differences(previous, f)))
            previous = f
        write('daily_snapshot_comparisons.json', comparisons)
    local_bounds = current.filter(pl.col('asset').is_in(assets)).group_by('asset').agg(pl.col('date').min().alias('first'),pl.col('date').max().alias('last'),pl.len().alias('rows')).sort('asset')
    write('local_bounds.json', local_bounds.to_dicts())
    common_current = current.filter(pl.col('date')<=old['date'].max())
    write('consumer_vs_current.json', differences(old.filter(pl.col('asset').is_in(assets)),common_current.filter(pl.col('asset').is_in(assets))))
    shutil.copyfile(__file__, OUT/'audit_script.py')
    write('manifest.json', dict(artifacts={p.relative_to(OUT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.rglob('*')) if p.is_file()}))
    print('Saved', OUT)


if __name__ == '__main__':
    run()
