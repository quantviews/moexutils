"""Capture and audit CNY premium option listings without inventing PIT metadata.

Run with --output NEW_DIRECTORY. Does not modify the lake or older packages.
"""
import argparse
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import polars as pl

from moexutils import iss, lake

BASE = iss.ISS_URL + '/statistics/engines/futures/markets/options/series'
TARGET = 'CNYP051126XE'
FILTER = dict(asset_code='CNY', exec_type='E', margin_style='P',
              settle_type='S', underlying_asset='CNYRUB_TOM')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(payload, block):
    data = payload[block]
    if any(len(row) != len(data['columns']) for row in data['data']):
        raise ValueError('Malformed ISS response')
    return [dict(zip(data['columns'], row)) for row in data['data']]


def classify_series(signal, start, observed_before=False):
    # start_date is a retrospective registry assertion, not a known_at timestamp.
    if start and start > signal:
        if observed_before:
            return 'unknown', 'registry_start_conflicts_with_history'
        return 'not_yet_listed', 'declared_series_start_after_signal'
    return 'unknown', 'series_start_does_not_prove_individual_strike_listing'


def wing(side, strike):
    return (side == 'P' and strike <= 11.4961382) or (side == 'C' and strike >= 13.4051447)


def run(output, coverage):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    raw = root / 'raw'
    raw.mkdir()
    checked = datetime.now(timezone.utc).isoformat()
    sources = []
    session = iss.make_session()

    def get(name, url, params=None):
        response = session.get(url, params=params)
        path = raw / name
        path.write_bytes(response.content)
        sources.append(dict(file=path.relative_to(root).as_posix(), url=response.url,
                            checked_at=datetime.now(timezone.utc).isoformat(),
                            http_status=response.status_code, sha256=digest(path)))
        response.raise_for_status()
        return response.json()

    shutil.copyfile(coverage, root / 'consumer_coverage.json')
    shutil.copyfile(__file__, root / 'build_script.py')
    shutil.copyfile(Path(__file__).resolve().parents[1] / 'docs/futures-analytics-options-listing-request.md',
                    root / 'request.md')
    signals = [r['signal'] for r in json.loads(Path(coverage).read_text())['rows']]
    archive = records(get('series.json', BASE + '.json', {'show_expired': 1}), 'series')
    series = [r for r in archive if all(r.get(k) == v for k, v in FILTER.items())]
    if not series or len({r['name'] for r in series}) != len(series):
        raise ValueError('Empty or duplicated series registry')
    pl.DataFrame(series).write_parquet(root / 'series.parquet')
    selected = {r['name']: r for r in series if any(
        28 <= (date.fromisoformat(r['expiration_date']) - date.fromisoformat(s)).days <= 42
        for s in signals)}
    if TARGET not in selected:
        raise ValueError('Target series absent: investigate before building')
    contracts = []
    for name in sorted(selected):
        rows = records(get(name + '.json', f'{BASE}/{name}/securities.json'), 'securities')
        for row in rows:
            row['series_name'] = name
        contracts.extend(rows)
    cf = pl.DataFrame(contracts, infer_schema_length=None)
    if cf.unique(['secid', 'series_name']).height != cf.height:
        raise ValueError('Duplicate contract identity')
    cf.write_parquet(root / 'contracts.parquet')
    # Retain local input versions and joined observations in one read transaction.
    with lake.session(read_only=True) as con:
        con.execute('BEGIN TRANSACTION')
        con.register('audit_contracts', cf.to_arrow())
        local_series = con.execute("SELECT * FROM lake.options_series WHERE asset_code='CNY' AND exec_type='E' AND margin_style='P' AND settle_type='S' AND underlying_asset='CNYRUB_TOM'").pl()
        local_series.write_parquet(root / 'local_series.parquet')
        local_contracts = con.execute('SELECT c.* FROM lake.options_contracts c JOIN (SELECT DISTINCT series_name FROM audit_contracts) s USING(series_name)').pl()
        local_contracts.write_parquet(root / 'local_contracts.parquet')
        history = con.execute("""SELECT h.*, c.series_name, c.option_type, c.strike
            FROM lake.options h JOIN audit_contracts c ON h.SECID=c.secid
            AND h.date BETWEEN CAST(c.history_from AS DATE) AND CAST(c.history_till AS DATE)
            WHERE h.date BETWEEN DATE '2025-01-03' AND DATE '2026-10-14'""").pl()
        con.execute('COMMIT')
    if history.unique(['date', 'BOARDID', 'SECID']).height != history.height:
        raise ValueError('Ambiguous short-code history interval')
    history.write_parquet(root / 'history.parquet')
    # Fresh current trading registry can reveal contracts with no history at all.
    current = []
    offset = 0
    previous = None
    while True:
        p = get(f'current-{offset}.json', iss.ISS_URL + '/engines/futures/markets/options/securities.json',
                {'iss.only': 'securities', 'start': offset})
        rows = records(p, 'securities')
        current.extend(rows)
        if not rows:
            break
        if offset and rows == previous:
            # Some ISS reference endpoints are unpaginated; repeated page is terminal.
            current = current[:-len(rows)]
            break
        previous = rows
        offset += len(rows)
    target_current = [r for r in current if str(r.get('SHORTNAME', '')).startswith(TARGET[:-2])]
    (root / 'current_target.json').write_text(json.dumps(target_current, ensure_ascii=False, indent=2), encoding='utf-8')
    # Independent fresh per-security EOD check, including zero-trade rows.
    fresh_day = []
    for row in contracts:
        if row['series_name'] != TARGET:
            continue
        offset = 0
        while True:
            url = iss.ISS_URL + f"/history/engines/futures/markets/options/securities/{row['secid']}.json"
            payload = get(f"day-{row['secid']}-{offset}.json", url,
                          {'from': '2026-10-01', 'till': '2026-10-01', 'start': offset})
            page = records(payload, 'history')
            if not page:
                break
            for observation in page:
                if observation['SECID'] != row['secid'] or observation['TRADEDATE'] != '2026-10-01':
                    raise ValueError('Unexpected history response')
                if not (row['history_from'] and row['history_till']
                        and row['history_from'] <= '2026-10-01' <= row['history_till']):
                    raise ValueError('Signal history outside series code interval')
                fresh_day.append({**observation, 'series_name': TARGET})
            offset += len(page)
            if offset > 1000:
                raise ValueError('Unexpected single-day pagination')
    fd = pl.DataFrame(fresh_day, infer_schema_length=None)
    if fd.height:
        if fd.unique(['TRADEDATE', 'SECID', 'BOARDID']).height != fd.height:
            raise ValueError('Duplicate live history observations')
        fd.write_parquet(root / 'fresh_signal_history.parquet')
    # Save the launch notice as context only; it is NOT a 2025-26 rule edition.
    response = session.get('https://www.moex.com/n56990')
    (raw / 'launch-notice.html').write_bytes(response.content)
    sources.append(dict(file='raw/launch-notice.html', url=response.url,
                        checked_at=datetime.now(timezone.utc).isoformat(), http_status=response.status_code,
                        use='historical_context_only_not_current_rules'))
    matrix, results, wings = [], [], []
    for signal in signals:
        sd = date.fromisoformat(signal)
        candidates = [r for r in series if 28 <= (date.fromisoformat(r['expiration_date']) - sd).days <= 42]
        active = 0
        for row in candidates:
            h = history.filter(pl.col('series_name') == row['name'], pl.col('date') <= sd)
            status, reason = classify_series(signal, row['start_date'], h.height > 0)
            active += status != 'not_yet_listed'
            matrix.append(dict(signal_date=signal, expiry=row['expiration_date'], series_name=row['name'],
                               secid=None, option_type=None, strike=None,
                               listed_at=None, delisted_at=None, known_at=None,
                               declared_series_start_date=row['start_date'],
                               source_url=BASE + '.json?show_expired=1', checked_at=checked,
                               status=status, reason=reason, evidence_scope='current_archived_registry',
                               historical_publication_proven=False))
        results.append(dict(signal_date=signal, candidate_count=len(candidates),
                            eligible_by_declared_start=active,
                            status='no_eligible_series_in_registry' if candidates and not active else 'requires_contract_check',
                            exhaustive_historical_listing_proven=False))
    target = [r for r in contracts if r['series_name'] == TARGET]
    signal = date(2026, 10, 1)
    for row in target:
        h = history.filter(pl.col('series_name') == TARGET, pl.col('SECID') == row['secid'])
        day = h.filter(pl.col('date') == signal)
        live = [x for x in fresh_day if x['SECID'] == row['secid']]
        local_values = sorted((x['BOARDID'], x['NUMTRADES']) for x in day.to_dicts())
        live_values = sorted((x['BOARDID'], x['NUMTRADES']) for x in live)
        status = 'unknown'
        if day.height and day['NUMTRADES'].null_count() == 0:
            status = 'listed_with_trades' if day['NUMTRADES'].sum() > 0 else 'listed_no_trades'
        wings.append(dict(signal_date=str(signal), expiry=selected[TARGET]['expiration_date'],
                          series_name=TARGET, secid=row['secid'], option_type=row['option_type'], strike=row['strike'],
                          requested_wing=wing(row['option_type'], row['strike']),
                          listed_at=None, delisted_at=None, known_at=None,
                          history_from=row['history_from'], history_till=row['history_till'],
                          first_observed_date=str(h['date'].min()) if h.height else None,
                          signal_numtrades=day['NUMTRADES'].sum() if day.height else None,
                          live_signal_rows=len(live), live_matches_local=local_values == live_values,
                          source_url=f'{BASE}/{TARGET}/securities.json', checked_at=checked,
                          status=status, reason='daily_history_observation' if day.height else 'no_signal_day_observation',
                          historical_publication_proven=False))
    for side, limit in [('P', 11.4961382), ('C', 13.4051447)]:
        found = [r for r in wings if r['option_type'] == side and r['requested_wing']]
        if not found:
            wings.append(dict(signal_date=str(signal), expiry=selected[TARGET]['expiration_date'],
                              series_name=TARGET, secid=None, option_type=side, strike=None,
                              requested_wing=True, listed_at=None, delisted_at=None, known_at=None,
                              history_from=None, history_till=None, first_observed_date=None,
                              signal_numtrades=None, source_url=f'{BASE}/{TARGET}/securities.json',
                              live_signal_rows=None, live_matches_local=None,
                              checked_at=checked, status='unknown',
                              reason=f'no_contract_in_archived_registry_for_{side}_threshold_{limit}',
                              historical_publication_proven=False))
    pl.DataFrame(matrix, infer_schema_length=None).write_parquet(root / 'listing_matrix.parquet')
    pl.DataFrame(results).write_parquet(root / 'coverage.parquet')
    pl.DataFrame(wings, infer_schema_length=None).write_parquet(root / 'strikes.parquet')
    gaps = [dict(scope='all', field='known_at/listed_at/delisted_at', status='unknown',
                 reason='No historical publication timestamps or strike assignment log',
                 recovery='Requires historical instrument assignment archive from MOEX'),
            dict(scope='2025-01-03/2026-09-01', field='exhaustive_listing', status='unknown',
                 reason='Current expired-series registry is not an archived daily universe',
                 recovery='Historical daily instrument directories or exchange confirmation'),
            dict(scope=TARGET, field='outer_put/outer_call', status='unknown',
                 reason='No requested wing in archived series contracts; absence is not proof of never listing',
                 recovery='Strike assignment log for 2026-10-01 and later changes'),
            dict(scope='all', field='intraday_execution', status='out_of_scope',
                 reason='Daily records cannot prove simultaneous executable quotes',
                 recovery='Priorities 2 and 3 deferred')]
    pl.DataFrame(gaps).write_parquet(root / 'gaps.parquet')
    summary = dict(checked_at=checked, history_as_of='2026-10-14', filters=FILTER,
                   signal_dates=len(signals), candidate_series=len(selected),
                   signals_without_eligible_series=sum(r['status']=='no_eligible_series_in_registry' for r in results),
                   target_contracts=len(target), target_current_contracts=len(target_current),
                   requested_wing_contracts=sum(r['requested_wing'] and r['secid'] is not None for r in wings),
                   fresh_signal_rows=len(fresh_day),
                   live_local_mismatches=sum(r.get('live_matches_local') is False for r in wings),
                   history_rows=history.height, acceptance_status='research_with_limitations',
                   exhaustive_historical_listing_proven=False, execution_ready=False)
    (root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (root / 'sources.json').write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding='utf-8')
    shutil.copyfile(Path(__file__).resolve().parents[1] / 'docs/options-listing.md', root / 'README.md')
    manifest = dict(created_at=checked, artifacts={p.relative_to(root).as_posix(): digest(p)
                    for p in sorted(root.rglob('*')) if p.is_file()})
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('manifest_sha256=' + digest(root / 'manifest.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--coverage', default='F:/Yandex.Disk/FINANCE/futures-analytics/outputs/options-coverage-20261015-final/coverage.json')
    args = parser.parse_args()
    run(args.output, args.coverage)
