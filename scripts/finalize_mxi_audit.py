"""Add cross-response revision checks and coverage to an immutable MXI audit."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import polars as pl
from scripts.audit_mxi_versions import availability, compare, dump, normalized, sha, timestamp, MSK
from urllib.parse import parse_qs, urlparse


def run(source, output):
    manifest = json.loads((source/'manifest.json').read_text())['artifacts']
    for name, expected in manifest.items():
        if sha(source/name) != expected:
            raise ValueError('Audit file changed: '+name)
    shutil.copytree(source, output)
    shutil.copyfile(source/'manifest.json', output/'initial_manifest.json')
    inputs = json.loads((output/'inputs.json').read_text(encoding='utf-8'))
    lookup = {r['path']: r['copy'] for r in inputs}
    observations = json.loads((output/'source_observations.json').read_text())
    # Include earlier probes in the preserved interim audit, not only its final edition.
    for root in sorted(source.parent.glob('positioning-*')):
        for meta in sorted(root.glob('sources/*MXI.provenance.json')):
            raw_path = meta.with_name(meta.name.replace('.provenance.json', '.json'))
            m = json.loads((root/'manifest.json').read_text())['artifacts']
            info = json.loads(meta.read_text())
            stamp = timestamp(info['observed_at'])
            digest = sha(raw_path)
            if any(o['source_sha256'] == digest and o['observed_at_utc'] == stamp.isoformat() for o in observations):
                continue
            for path in [root/'manifest.json', meta, raw_path]:
                h = sha(path)
                expected = None if path.name == 'manifest.json' else m[path.relative_to(root).as_posix()]
                if expected is not None and h != expected:
                    raise ValueError('Earlier source changed')
                dest = output/'sources'/(h+path.suffix)
                shutil.copyfile(path, dest)
                lookup[str(path)] = dest.relative_to(output).as_posix()
                inputs.append(dict(path=str(path), sha256=h, expected_sha256=expected, copy=lookup[str(path)]))
            params = parse_qs(urlparse(info['url']).query)
            raw = json.loads(raw_path.read_text())['open_positions']
            observations.append(dict(version=root.name+'-'+raw_path.stem, source_path=str(raw_path),
                source_sha256=digest, observed_at_utc=stamp.isoformat(), observed_at_msk=stamp.astimezone(MSK).isoformat(),
                start=params['from'][0], end=params['till'][0], applied_at=None, rows=len(raw['data']), url=info['url']))
    observations.sort(key=lambda r: r['observed_at_utc'])
    dump(output/'source_observations.json', observations)
    dump(output/'inputs.json', inputs)
    response_frames = []
    latest, changes, windows = {}, [], []
    for obs in observations:
        raw = json.loads((output/lookup[obs['source_path']]).read_text())['open_positions']
        f = normalized(pl.DataFrame(raw['data'], schema=raw['columns'], orient='row'))
        response_frames.append((obs, f))
        relevant = [r for (d, g), r in latest.items() if obs['start'] <= d <= obs['end']]
        before = f.clear() if not relevant else normalized(pl.from_dicts(relevant))
        delta = compare(before, f)
        changes.extend(dict(r, version=obs['version'], observed_at=obs['observed_at_utc'],
            interpretation='first_source_observation_not_revision' if r['operation'] == 'insert' else 'repeated_source_revision') for r in delta)
        previous_keys = {(str(r['date']), r['is_fiz']) for r in relevant}
        current = {(str(r['date']), r['is_fiz']): r for r in f.to_dicts()}
        for key in previous_keys-current.keys():
            latest.pop(key)
        latest.update(current)
        windows.append(dict(version=obs['version'], observed_at=obs['observed_at_utc'],
            start=obs['start'], end=obs['end'], rows=f.height,
            rows_reobserved=len(previous_keys & current.keys()),
            older_rows_not_rechecked=sum(d < obs['start'] for d, g in latest),
            comparison_scope='previous raw responses only; source absence requires requested range'))
    pl.from_dicts(changes, infer_schema_length=None).write_parquet(output/'repeated_source_changes.parquet')
    dump(output/'reread_windows.json', windows)
    # Verify every archived table copy against any recovered local DB state;
    # timestamp attribution comes only from the DuckLake catalog, never file mtime.
    catalog = pl.read_parquet(output/'versions.parquet')
    state_map = {}
    for p in catalog['state_path'].unique():
        f = normalized(pl.read_parquet(output/p))
        state_map[json.dumps(f.to_dicts(), default=str, sort_keys=True)] = p
    checks = []
    for row in json.loads((output/'input_inventory.json').read_text()):
        f = normalized(pl.read_parquet(output/lookup[row['path']]))
        state = state_map.get(json.dumps(f.to_dicts(), default=str, sort_keys=True))
        checks.append(dict(row, matches_available_db_state=state is not None, matching_state_path=state))
    dump(output/'archive_state_checks.json', checks)
    availability_frame = pl.from_dicts(availability(response_frames), infer_schema_length=None)
    availability_frame.write_parquet(output/'availability.parquet')
    coverage = availability_frame.group_by('tradedate', 'is_fiz').agg(
        pl.col('observed_at_utc').filter(pl.col('present')).min().alias('first_positive_source_at'),
        pl.col('observed_at_utc').filter(pl.col('present')).max().alias('last_positive_source_at'),
        pl.col('present').sum().alias('positive_observations'),
        (~pl.col('present')).sum().alias('negative_observations'))
    first_db = pl.read_parquet(output/'first_local_availability.parquet')
    coverage = coverage.join(first_db.select('tradedate','is_fiz','first_local_db_at'), on=['tradedate','is_fiz'], how='full', coalesce=True)
    coverage.write_parquet(output/'coverage.parquet')
    summary = json.loads((output/'summary.json').read_text())
    summary.update(source_observations=len(observations), repeated_raw_changes={op: sum(r['operation'] == op for r in changes) for op in ['insert','update','delete']},
        repeated_raw_physical_sign_changes=sum(r['physical_sign_changed'] for r in changes),
        archived_table_copies=len(checks), archived_copies_unmatched=sum(not r['matches_available_db_state'] for r in checks),
        first_raw_observed_at=observations[0]['observed_at_utc'], last_raw_observed_at=observations[-1]['observed_at_utc'],
        generated_at=datetime.now(timezone.utc).isoformat())
    dump(output/'summary.json', summary)
    shutil.copyfile(__file__, output/'finalize_script.py')
    for name, dest in [('mxi-publication-audit.md','README.md'), ('mxi-publication-moex-questions.md','moex-questions.md')]:
        shutil.copyfile(Path(__file__).resolve().parents[1]/'docs'/name, output/dest)
    readme = output/'README.md'
    readme.write_text(readme.read_text(encoding='utf-8').replace('(mxi-publication-moex-questions.md)', '(moex-questions.md)'), encoding='utf-8')
    dump(output/'manifest.json', dict(artifacts={p.relative_to(output).as_posix(): sha(p)
         for p in sorted(output.rglob('*')) if p.is_file() and p != output/'manifest.json'}))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    run(args.source, args.output)
