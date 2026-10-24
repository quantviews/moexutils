"""Offline MXI observations and all available DuckLake table-changing snapshots."""
import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import polars as pl
from moexutils import lake

KEY = ['date', 'asset', 'is_fiz']
VALUES = ['persons_long', 'persons_short', 'open_position_long', 'open_position_short', 'oichange_long', 'oichange_short']
MSK = ZoneInfo('Europe/Moscow')


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, default=str, indent=2)+'\n', encoding='utf-8')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def timestamp(value):
    value = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if value.tzinfo is None:
        raise ValueError('Timezone missing')
    return value.astimezone(timezone.utc)


def normalized(frame):
    if 'tradedate' in frame.columns:
        frame = frame.rename({'tradedate': 'date'})
    frame = frame.filter(pl.col('asset') == 'MXI').select(KEY+VALUES).with_columns(
        pl.col('date').cast(pl.String).str.slice(0, 10).str.to_date(),
        pl.col('is_fiz').cast(pl.Int8), pl.col(VALUES).cast(pl.Int64)).sort(KEY)
    if frame.select(KEY).is_duplicated().any():
        raise ValueError('Duplicate MXI key')
    return frame


def net(row):
    if row is None or row['open_position_long'] is None or row['open_position_short'] is None:
        return None
    return row['open_position_long']-row['open_position_short']


def compare(before, after):
    old = {(str(r['date']), r['is_fiz']): r for r in before.to_dicts()}
    new = {(str(r['date']), r['is_fiz']): r for r in after.to_dicts()}
    changes = []
    for key in sorted(old.keys() | new.keys()):
        a, b = old.get(key), new.get(key)
        if a == b:
            continue
        na, nb = net(a), net(b)
        op = 'insert' if a is None else 'delete' if b is None else 'update'
        changes.append(dict(tradedate=key[0], asset='MXI', is_fiz=key[1], operation=op,
            before_long=None if a is None else a['open_position_long'],
            before_short=None if a is None else a['open_position_short'],
            after_long=None if b is None else b['open_position_long'],
            after_short=None if b is None else b['open_position_short'],
            before_net=na, after_net=nb,
            long_short_changed=op == 'update' and any(a[c] != b[c] for c in ['open_position_long', 'open_position_short']),
            physical_sign_changed=key[1] == 1 and op == 'update' and na is not None and nb is not None
                and (na > 0)-(na < 0) != (nb > 0)-(nb < 0),
            before=json.dumps(a, default=str, sort_keys=True), after=json.dumps(b, default=str, sort_keys=True)))
    return changes


def availability(observations):
    """Absence only from an actual response whose requested range covers the date."""
    first, absent, rows = {}, {}, []
    for obs, frame in sorted(observations, key=lambda pair: pair[0]['observed_at_utc']):
        present = {(str(r['date']), r['is_fiz']): r for r in frame.to_dicts()}
        day, end = date.fromisoformat(obs['start']), date.fromisoformat(obs['end'])
        while day <= end:
            for group in (0, 1):
                key = (str(day), group)
                row = present.get(key)
                if row is not None:
                    first.setdefault(key, obs['observed_at_utc'])
                rows.append(dict(tradedate=str(day), asset='MXI', is_fiz=group,
                    observed_at_utc=obs['observed_at_utc'], observed_at_msk=obs['observed_at_msk'],
                    applied_at=obs.get('applied_at'), version=obs['version'], source_path=obs['source_path'],
                    source_sha256=obs['source_sha256'], present=row is not None,
                    first_seen_at=first.get(key), last_seen_absent_at=absent.get(key),
                    open_position_long=None if row is None else row['open_position_long'],
                    open_position_short=None if row is None else row['open_position_short'],
                    net_position=net(row), published_at=None,
                    calendar_publication_required_unknown=True))
                if row is None:
                    absent[key] = obs['observed_at_utc']
            day += timedelta(days=1)
    return rows


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    (output/'sources').mkdir()
    (output/'states').mkdir()
    inputs, observations, source_changes, inventory = [], [], [], []
    manifests = {}

    def capture(path, root=None):
        digest = sha(path)
        expected = None
        if root is not None:
            if root not in manifests:
                mpath = root/'manifest.json'
                m = json.loads(mpath.read_text(encoding='utf-8'))
                manifests[root] = m.get('artifacts', m.get('artifact_sha256', {}))
                capture(mpath)
            expected = manifests[root].get(path.relative_to(root).as_posix())
            if expected != digest:
                raise ValueError('Input hash mismatch: '+str(path))
        dest = output/'sources'/(digest+path.suffix)
        if not dest.exists():
            shutil.copyfile(path, dest)
        inputs.append(dict(path=str(path), sha256=digest, expected_sha256=expected,
                           copy=dest.relative_to(output).as_posix()))
        return digest

    def add_response(raw, meta, root, version, applied=None):
        digest = capture(raw, root)
        capture(meta, root)
        info = json.loads(meta.read_text(encoding='utf-8'))
        data = json.loads(raw.read_text(encoding='utf-8'))['open_positions']
        frame = normalized(pl.DataFrame(data['data'], schema=data['columns'], orient='row'))
        params = parse_qs(urlparse(info['url']).query)
        observed = timestamp(info['observed_at'])
        obs = dict(version=version, source_path=str(raw), source_sha256=digest,
            observed_at_utc=observed.isoformat(), observed_at_msk=observed.astimezone(MSK).isoformat(),
            start=info.get('start', params.get('from', [None])[0]),
            end=info.get('end', params.get('till', [None])[0]), applied_at=applied,
            rows=frame.height, url=info['url'])
        # Identical copied observations are evidence copies, not extra observations.
        if not any(o['source_sha256'] == digest and o['observed_at_utc'] == obs['observed_at_utc'] for o, _ in observations):
            observations.append((obs, frame))

    data_root = Path(lake.DATA_ROOT)
    roots = sorted((data_root/'versions/openpositions').glob('*'))
    archive = data_root/'reports/positioning-availability-20261016-v2'
    for root in roots:
        if not root.is_dir():
            continue
        for meta in sorted(root.glob('observation-*/observation.json')):
            info = json.loads(meta.read_text(encoding='utf-8'))
            if info.get('asset') != 'MXI' or info.get('market') != 'forts':
                continue
            p = meta.parent
            applied = None
            if (p/'applied.json').exists():
                capture(p/'applied.json', root)
                applied = json.loads((p/'applied.json').read_text())['applied_at']
            add_response(p/'response.json', meta, root, root.name, applied)
            for name in ['before.parquet', 'after.parquet', 'changes.json']:
                capture(p/name, root)
            old, new = [normalized(pl.read_parquet(p/n)) for n in ['before.parquet', 'after.parquet']]
            delta = compare(old, new)
            recorded = json.loads((p/'changes.json').read_text(encoding='utf-8'))
            def signature(rows):
                return sorted((r['operation'], str(r['key']['date']), r['key']['is_fiz']) for r in rows)
            actual = [dict(operation=r['operation'], key={'date': r['tradedate'], 'is_fiz': r['is_fiz']}) for r in delta]
            if signature(actual) != signature(recorded):
                raise ValueError('Journal differs')
            if not new.equals(observations[-1][1]):
                raise ValueError('Raw/normalized response differs')
            source_changes.extend(dict(r, stream='source_refresh', version=root.name,
                observed_at=info['observed_at'], interpretation='historical_backfill' if r['operation'] == 'insert' and r['tradedate'] < '2025-01-01' else 'observed_local_change') for r in delta)
        for p in sorted(root.glob('forts-*.parquet')):
            capture(p, root)
            f = normalized(pl.read_parquet(p))
            inventory.append(dict(path=str(p), rows=f.height, kind='permanent_table_copy', source_time_unknown=True))

    for root in sorted((data_root/'reports').glob('positioning-*')):
        for meta in sorted(root.glob('sources/*MXI.provenance.json')):
            raw = meta.with_name(meta.name.replace('.provenance.json', '.json'))
            add_response(raw, meta, root, root.name+'-'+raw.stem)
    archived_frames = []
    for root in sorted((data_root/'reports').glob('positioning-*')):
        if root == output or not (root/'manifest.json').exists():
            continue
        for p in sorted(root.glob('*.parquet')):
            if p.name not in ['positions.parquet', 'local_positions.parquet', 'consumer_positions.parquet'] and not p.name.startswith('snapshot-'):
                continue
            capture(p, root)
            f = normalized(pl.read_parquet(p))
            archived_frames.append((p, f))
            inventory.append(dict(path=str(p), rows=f.height, kind='archived_table_copy', source_time_unknown=True))
    # Original audit maps archived snapshot IDs to database transaction times.
    capture(archive/'daily_snapshot_comparisons.json', archive)
    capture(archive/'table_snapshots.json', archive)
    project = Path(__file__).resolve().parents[1]
    for p in [project/'docs/positioning-availability-20261016.md', project/'docs/positioning-history.md',
              project.parent/'futures-analytics/docs/mxi-publication-audit.md',
              project.parent/'futures-analytics/docs/moexutils-mxi-publication-request.md',
              project.parent/'futures-analytics/outputs/reports/mxi-publication-audit-20261021.json']:
        capture(p)

    dump(output/'source_observations.json', [o for o, _ in observations])
    dump(output/'input_inventory.json', inventory)
    pl.from_dicts(availability(observations), infer_schema_length=None).write_parquet(output/'availability.parquet')
    snapshots, changes, errors, first_db, state_hashes = [], [], [], {}, {}
    previous = None
    with lake.session(read_only=True) as con:
        all_snaps = con.execute("SELECT * FROM ducklake_snapshots('lake') ORDER BY snapshot_id").pl()
        all_snaps.write_parquet(output/'catalog_snapshots.parquet')
        info = con.execute("SELECT * FROM ducklake_table_info('lake') WHERE table_name='futures_open_positions'").pl().row(0, named=True)
        dump(output/'table_info.json', info)
        sid = str(info['table_id'])
        selected = [r for r in all_snaps.to_dicts() if any(sid in c['value'] or 'main.futures_open_positions' in c['value'] for c in r['changes'])]
        print('Table-changing snapshots:', len(selected), flush=True)
        for i, r in enumerate(selected):
            version = r['snapshot_id']
            try:
                f = normalized(con.execute(f"SELECT * FROM lake.futures_open_positions AT (VERSION => {version}) WHERE asset='MXI'").pl())
                serialized = json.dumps(f.to_dicts(), default=str, sort_keys=True).encode()
                digest = hashlib.sha256(serialized).hexdigest()
                dest = output/'states'/f'{digest}.parquet'
                if not dest.exists():
                    f.write_parquet(dest)
                state_hashes[digest] = f
                stamp = timestamp(r['snapshot_time']).isoformat()
                for row in f.to_dicts():
                    first_db.setdefault((str(row['date']), row['is_fiz']), stamp)
                delta = compare(f.clear() if previous is None else previous, f)
                changes.extend(dict(d, stream='ducklake', version=str(version), observed_at=stamp,
                    interpretation='initial_local_state' if previous is None else 'local_state_transition') for d in delta)
                snapshots.append(dict(snapshot_id=version, available_at_utc=stamp, rows=f.height,
                    state_path=dest.relative_to(output).as_posix(), state_sha256=sha(dest),
                    changes=len(delta), predecessor_read_failed=bool(errors)))
                previous = f
            except Exception as error:
                errors.append(dict(snapshot_id=version, error=str(error)))
            if (i+1) % 100 == 0:
                dump(output/'progress.json', dict(checked=i+1, total=len(selected), errors=len(errors)))
                print('Snapshots', i+1, '/', len(selected), flush=True)
    # Compare archived snapshot copies independently with recovered database states.
    archive_checks = []
    for p, f in archived_frames:
        digest = hashlib.sha256(json.dumps(f.to_dicts(), default=str, sort_keys=True).encode()).hexdigest()
        archive_checks.append(dict(path=str(p), matches_available_db_state=digest in state_hashes))
    dump(output/'archive_state_checks.json', archive_checks)
    rows = [dict(tradedate=d, asset='MXI', is_fiz=g, first_local_db_at=t,
                 historical_published_at=None) for (d, g), t in sorted(first_db.items())]
    pl.from_dicts(rows, infer_schema_length=None).write_parquet(output/'first_local_availability.parquet')
    pl.from_dicts(snapshots, infer_schema_length=None).write_parquet(output/'versions.parquet')
    pl.from_dicts(changes+source_changes, infer_schema_length=None).write_parquet(output/'changes.parquet')
    dump(output/'snapshot_errors.json', errors)
    dump(output/'inputs.json', inputs)
    dump(output/'summary.json', dict(source_observations=len(observations), catalog_snapshots=all_snaps.height,
        table_changing_snapshots=len(selected), read_snapshots=len(snapshots), snapshot_errors=len(errors),
        unique_mxi_states=len(state_hashes), first_local_db_at=min(first_db.values()),
        database_changes={op: sum(r['operation'] == op for r in changes) for op in ['insert', 'update', 'delete']},
        source_refresh_changes={op: sum(r['operation'] == op for r in source_changes) for op in ['insert', 'update', 'delete']},
        physical_sign_changes=sum(r['physical_sign_changed'] for r in changes+source_changes),
        first_tradedate=min(d for d, _ in first_db), last_tradedate=max(d for d, _ in first_db),
        historical_publication_proven=False, monitoring_started=False))
    shutil.copyfile(__file__, output/'audit_script.py')
    dump(output/'manifest.json', dict(artifacts={p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*')) if p.is_file()}))
    print((output/'summary.json').read_text(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args().output)
