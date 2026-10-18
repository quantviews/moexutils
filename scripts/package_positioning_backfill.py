"""Verify and package a completed universe24 positioning refresh."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import polars as pl
from moexutils import lake
from moexutils.position_versions import save_json


def run(version, output, universe):
    version, output, universe = map(Path, (version, output, universe))
    manifest = json.loads((version/'manifest.json').read_text())
    for name, expected in manifest['artifacts'].items():
        if hashlib.sha256((version/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Checksum mismatch: '+name)
    if not json.loads((version/'result.json').read_text())['complete']:
        raise ValueError('Incomplete update')
    cfg = json.loads(universe.read_text())
    assets = cfg['original18']+cfg['additions']
    positions = pl.read_parquet(version/'forts-snapshot.parquet').filter(pl.col('asset').is_in(assets))
    if positions.select('date','asset','is_fiz').is_duplicated().any():
        raise ValueError('Duplicate key')
    current = lake.query('SELECT * FROM lake.futures_open_positions').filter(pl.col('asset').is_in(assets))
    if not positions.sort('date','asset','is_fiz').equals(current.select(positions.columns).sort('date','asset','is_fiz')):
        raise ValueError('Saved version differs from current lake')
    counts = {'insert':0, 'update':0, 'delete':0}
    for p in version.glob('observation-*/changes.json'):
        for row in json.loads(p.read_text(encoding='utf-8')):
            counts[row['operation']] += 1
    for p in version.glob('observation-*/after.parquet'):
        if not (p.parent/'applied.json').exists():
            raise ValueError('Observation was not confirmed applied: '+str(p.parent))
        after = pl.read_parquet(p)
        meta = json.loads((p.parent/'observation.json').read_text())
        saved = positions.filter((pl.col('asset')==meta['asset']) &
            pl.col('date').is_between(pl.lit(meta['start']).str.to_date(), pl.lit(meta['end']).str.to_date()))
        if not after.sort('date','is_fiz').equals(saved.select(after.columns).sort('date','is_fiz')):
            raise ValueError('Source observation differs from saved data')
    quality = positions.group_by('asset','date').agg(
        pl.col('is_fiz').n_unique().alias('groups'),
        pl.when(pl.col('open_position_long').null_count() == 0)
          .then(pl.col('open_position_long').sum()).otherwise(None).alias('long'),
        pl.when(pl.col('open_position_short').null_count() == 0)
          .then(pl.col('open_position_short').sum()).otherwise(None).alias('short'),
        pl.any_horizontal(pl.all().exclude('asset','date').is_null()).any().alias('has_null')
    ).with_columns((pl.col('long')-pl.col('short')).alias('balance_difference'))
    bounds = positions.group_by('asset').agg(pl.col('date').min().alias('first'),
        pl.col('date').max().alias('last'),pl.len().alias('rows')).sort('asset')
    output.mkdir(parents=True,exist_ok=False)
    shutil.copytree(version,output/'version')
    shutil.copyfile(universe,output/'universe24.json')
    positions.write_parquet(output/'positions.parquet')
    quality.write_parquet(output/'quality.parquet')
    bounds.write_parquet(output/'coverage.parquet')
    save_json(output/'coverage.json',bounds.to_dicts())
    save_json(output/'summary.json',dict(assets=positions['asset'].n_unique(),rows=positions.height,
        first=positions['date'].min(),last=positions['date'].max(),changes=counts,
        incomplete_groups=quality.filter(pl.col('groups')!=2).height,
        null_dates=quality.filter(pl.col('has_null')).height,
        unbalanced_complete_dates=quality.filter((pl.col('groups')==2)&(pl.col('balance_difference')!=0)).height,
        original_version=str(version),publication_times_unknown=True))
    shutil.copyfile(__file__,output/'package_script.py')
    shutil.copyfile(Path(__file__).resolve().parents[1]/'docs/positioning-history.md', output/'README.md')
    save_json(output/'manifest.json',dict(artifacts={p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                                                    for p in sorted(output.rglob('*')) if p.is_file()}))
    print((output/'summary.json').read_text())


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['version','output','universe']:
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    run(args.version,args.output,args.universe)
