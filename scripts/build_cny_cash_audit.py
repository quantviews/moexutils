"""Targeted cash audit. Existing sources are pinned; unknown fees stay null."""
import argparse
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from lxml import etree
import polars as pl

from moexutils import iss, lake

D = Decimal
ROOT = Path(__file__).resolve().parents[1]
BASE = iss.ISS_URL + '/statistics/engines/futures/markets/options/series'


def round_money(value):
    return D(str(value)).quantize(D('.01'), rounding=ROUND_HALF_UP)


def payoff(kind, strike, fixing, multiplier=D('100')):
    if kind not in ('C', 'P'):
        raise ValueError('Unknown option kind')
    difference = D(str(fixing)) - D(str(strike))
    intrinsic = max(difference if kind == 'C' else -difference, D(0))
    return intrinsic, round_money(intrinsic * multiplier)


def opening_fee(theoretical, underlying, venue, mode='taker'):
    if venue not in ('moex', 'ncc') or mode not in ('taker', 'negotiated'):
        raise ValueError('Unsupported fee scenario')
    if theoretical is None or underlying is None:
        return None
    rate = {'moex': {'taker': '.0069', 'negotiated': '.0023'},
            'ncc': {'taker': '.0051', 'negotiated': '.0017'}}[venue][mode]
    cap = D('.00006' if mode == 'taker' else '.00002') * 100 * D(str(underlying))
    return max(D('.01'), round_money(min(cap, round_money(D(str(theoretical))*100)*D(rate))))


def exercise_fee(strike, itm):
    # Standard participant, not category A; no execution obligation for OTM/ATM.
    return max(D('.01'), round_money(round_money(abs(D(str(strike)))*100)*D('.00004'))) if itm else D('0.00')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str)+'\n', encoding='utf-8')


def run(evidence, output, consumer):
    evidence, out, consumer = Path(evidence), Path(output), Path(consumer)
    out.mkdir(parents=True, exist_ok=False)
    raw = out / 'sources'
    raw.mkdir()
    observed = datetime.now(timezone.utc).isoformat()
    urls = json.loads((ROOT/'scripts/cny_cash_sources.json').read_text(encoding='utf-8'))
    source_rows = []
    for name, url in urls.items():
        path = evidence / name
        if name == 'moex-tariff-index.html':
            path = evidence / 'moex-tariff.pdf'  # Returned HTML index, not a PDF.
        if path.exists():
            shutil.copyfile(path, raw/name)
            status = 'original_saved'
            if name.endswith('.docx'):
                tree = etree.fromstring(zipfile.ZipFile(path).read('word/document.xml'))
                ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
                text = '\n'.join(''.join(x.itertext()) for x in tree.xpath('//w:p', namespaces=ns))
                (raw/(name+'.txt')).write_text(text, encoding='utf-8')
        else:
            status = 'reviewed_via_web_reader_original_download_403'
        source_rows.append(dict(source_id=name, primary_source=url, observed_at=observed,
                                published_at=None, status=status,
                                sha256=sha(raw/name) if (raw/name).exists() else None))
    spec = (raw/'spec.docx.txt').read_text(encoding='utf-8')
    assert all(x in spec for x in ['CNYFIXME', '100 CNY', '0,001 RUB', '14:00 МСК', 'Round(W/R; 5)'])
    shutil.copyfile(consumer/'mechanics.json', out/'consumer_mechanics.json')
    shutil.copyfile(consumer/'manifest.json', out/'consumer_manifest.json')
    shutil.copyfile(ROOT/'docs/futures-analytics-cny-cash-request.md', out/'request.md')
    shutil.copyfile(__file__, out/'build_script.py')
    shutil.copyfile(ROOT/'scripts/cny_cash_sources.json', out/'source_urls.json')
    inputs = json.loads((consumer/'mechanics.json').read_text(encoding='utf-8'))
    if len(inputs) != 2 or sum(len(c['legs']) for c in inputs) != 8:
        raise ValueError('Expected exactly two four-leg structures')
    session = iss.make_session()

    def get(name, url, params=None):
        response = session.get(url, params=params)
        (raw/name).write_bytes(response.content)
        response.raise_for_status()
        source_rows.append(dict(source_id=name, primary_source=response.url, observed_at=datetime.now(timezone.utc).isoformat(),
                                published_at=None, status='original_saved', sha256=sha(raw/name)))
        return response.json()

    def block(payload, name):
        b = payload[name]
        if any(len(r) != len(b['columns']) for r in b['data']):
            raise ValueError('Malformed ISS block')
        return [dict(zip(b['columns'], r)) for r in b['data']]

    contracts, settlements, fees, aggregates, histories, fixings = [], [], [], [], [], []
    with lake.session(read_only=True) as con:
        con.execute('BEGIN TRANSACTION')
        for case in inputs:
            series = case['series']
            registry = block(get(series+'.json', f'{BASE}/{series}/securities.json'), 'securities')
            fixed = []
            for code in ('CNYFIXME', 'CNYFIX'):
                url = iss.ISS_URL+f'/history/engines/currency/markets/index/securities/{code}.json'
                data = get(f"{code}-{case['expiry']}.json", url, {'from': case['expiry'], 'till': case['expiry']})
                rows = block(data, 'history')
                rows = [r for r in rows if r['SECID']==code and r['TRADEDATE']==case['expiry'] and r['BOARDID']=='FIXI']
                if len(rows) != 1:
                    raise ValueError('Ambiguous/missing official fixing')
                fixed.extend(rows)
            fixings.extend(fixed)
            sstar = next(r['CLOSE'] for r in fixed if r['SECID']=='CNYFIXME')
            net_premium, net_settlement, exp_fees = D(0), D(0), D(0)
            for leg in case['legs']:
                secid = leg['secid']
                candidates = [r for r in registry if r['secid']==secid]
                if len(candidates)!=1:
                    raise ValueError('Contract absent or ambiguous in exact series')
                cr = candidates[0]
                card = block(get(secid+'.json', iss.ISS_URL+f'/securities/{secid}.json'), 'description')
                desc = {r['name']:r['value'] for r in card}
                kind = 'P' if leg['role'].endswith('put') else 'C'
                if (desc.get('SERIES_NAME')!=series or desc.get('UNDERLYINGASSET')!='CNYRUB_TOM'
                        or desc.get('OPTIONTYPE')!=kind or D(desc['STRIKE'])!=D(str(leg['strike']))
                        or D(desc['LOTSIZE'])!=100 or desc['LSTDELDATE']!=case['expiry']
                        or not cr['history_from'] <= case['entry'] <= cr['history_till']):
                    raise ValueError('Identity/specification mismatch')
                history = con.execute("SELECT * FROM lake.options WHERE SECID=? AND date BETWEEN ?::DATE AND ?::DATE ORDER BY date,BOARDID",
                                      [secid, cr['history_from'], case['expiry']]).pl()
                histories.append(history.with_columns(pl.lit(series).alias('series_name')))
                entry = history.filter(pl.col('date')==date.fromisoformat(case['entry']))
                if entry.height!=1 or D(str(entry['CLOSE'][0]))!=D(str(leg['close'])):
                    raise ValueError('Entry CLOSE disagrees with consumer')
                sign = D(1) if leg['role'].startswith('long') else D(-1)
                intrinsic, rubles = payoff(kind, leg['strike'], sstar)
                premium = -sign*round_money(D(str(leg['close']))*100)
                ef = exercise_fee(leg['strike'], intrinsic>0)
                common = dict(secid=secid, series_name=series, role=leg['role'],
                              valid_from=desc['FRSTTRADE'], valid_to=case['expiry'],
                              published_at=None, observed_at=observed)
                contracts.append(dict(**common, strike=str(leg['strike']), option_type=kind,
                    lot_size=100, lot_coeff=1, multiplier=100, minstep='0.001', stepprice='0.1',
                    premium_unit='RUB/CNY', strike_unit='RUB/CNY', currency='RUB',
                    signal_date=case['signal'], entry_date=case['entry'], expiry=case['expiry'],
                    spec_valid_from='2026-03-23', spec_valid_to=None,
                    source_id='spec.docx', primary_source=urls['spec.docx'],
                    confirmation='historical_spec_and_exact_series_card',
                    rounding='ROUND_HALF_UP: multiplier 5 decimals, rubles 2 decimals',
                    exceptions_exhaustively_checked=False))
                settlements.append(dict(**common, expiry=case['expiry'], fixing_secid='CNYFIXME', fixing=str(sstar),
                    fixing_reference_at=case['expiry']+'T12:30:00+03:00', fixing_published_at=None,
                    trading_end_at=case['expiry']+'T14:00:00+03:00',
                    intrinsic=str(intrinsic), unsigned_rubles=str(rubles), signed_rubles=str(sign*rubles),
                    automatic_exercise=intrinsic>0, exercise_basis='spec_2.2.4_2.2.5',
                    payment_date=None, payment_rule='next_settlement_day_after_last_trading_day',
                    primary_source=urls['spec.docx'], confirmation='calculated_from_official_fixing_and_spec',
                    actual_clearing_report_verified=False, entry_premium_rubles=str(premium)))
                ncc_entry = 'ncc-march.pdf' if case['entry']<'2026-06-01' else 'ncc-july.pdf'
                ncc_exp = 'ncc-june.pdf' if case['expiry']<'2026-07-01' else 'ncc-july.pdf'
                for venue, source, rate in [('moex','moex-march.docx','0.69'),('ncc',ncc_entry,'0.51')]:
                    fees.append(dict(**common, event='opening', venue=venue, date=case['entry'],
                        scenario='standard_taker_non_A', rate_percent=rate, cap_percent='0.006', minimum_rubles='0.01',
                        model_fee_rubles=None, actual_fee_rubles=None, debit_date=None,
                        primary_source=urls[source], source_id=source,
                        formula='max(0.01,round(min(K*100*PriceRub,round(Premium*100,2)*rate),2))',
                        confirmation='formula_verified_amount_unresolved',
                        reason='Previous main-session theoretical price, risk PriceRub, order role and participant category not fully verified'))
                fees.append(dict(**common, event='exercise', venue='ncc', date=case['expiry'],
                    scenario='standard_participant_non_A', rate_percent='0.004', cap_percent=None, minimum_rubles='0.01',
                    model_fee_rubles=str(ef), actual_fee_rubles=None, debit_date=None,
                    primary_source=urls[ncc_exp], source_id=ncc_exp,
                    formula='ITM: max(0.01,round(round(abs(strike)*100,2)*0.00004,2)); OTM/ATM: no exercise obligation',
                    confirmation='historical_tariff_formula_conditional_participant', reason='Broker invoice and category not verified'))
                net_premium += premium
                net_settlement += sign*rubles
                exp_fees += ef
            if not -D(20)<=net_settlement<=0:
                raise ValueError('Condor payout exceeds width bound')
            aggregates.append(dict(series_name=series, entry_date=case['entry'], expiry=case['expiry'],
                premium_credit=str(net_premium), signed_settlement=str(net_settlement),
                standard_exercise_fee=str(exp_fees), result_before_all_fees=str(net_premium+net_settlement),
                result_after_standard_exercise_before_opening_broker=str(net_premium+net_settlement-exp_fees),
                opening_fee=None, final_net_result=None, max_settlement_obligation='20.00',
                execution_ready=False))
        con.execute('COMMIT')
    pl.concat(histories).write_parquet(out/'history.parquet')
    for name, rows in [('contracts',contracts),('settlements',settlements),('fees',fees),('structures',aggregates),('fixings',fixings)]:
        pl.DataFrame(rows,infer_schema_length=None).write_parquet(out/(name+'.parquet'))
    gaps = [dict(field=k, status='unknown', reason=v) for k,v in {
        'opening_fee_amount':'Missing verified historical PriceRub, required Premium and order/participant classification; CLOSE is not Premium',
        'fee_originals_ncc':'Three historical PDFs and publication dates reviewed with web reader; original download returned HTTP 403; local originals absent',
        'payment_date_and_debit_date':'Next settlement day rule verified; futures settlement calendar/exceptions and actual debit not audited',
        'broker_fee':'broker_confirmation_required; no zero commission assumed',
        'published_at':'Precise historical publication of contract cards/fixings unavailable; current observation is not historical knowledge',
        'individual_exceptions':'No exhaustive per-series decisions audit; source specification formula is standard-rule reconstruction',
        'margin_cash_events':'Detailed cash ledger, margin, LQDT, RUONIA and execution quotes deferred by request'
    }.items()]
    pl.DataFrame(gaps).write_parquet(out/'gaps.parquet')
    coverage = [dict(secid=r['secid'],series_name=r['series_name'],spec=True,fixing=True,
                     payoff=True,opening_fee_amount=False,exercise_fee_standard_scenario=True,
                     actual_cash_verified=False) for r in contracts]
    pl.DataFrame(coverage).write_parquet(out/'coverage.parquet')
    write_json(out/'sources.json',source_rows)
    write_json(out/'summary.json',dict(observed_at=observed,consumer_as_of='2026-10-14',
        contracts=8,status='partial_cash_acceptance',spec_and_fixing_confirmed=True,
        opening_fees_confirmed=False,execution_ready=False,structures=aggregates))
    shutil.copyfile(ROOT/'docs/cny-cash-audit.md',out/'README.md')
    shutil.copyfile(ROOT/'scripts/cny_cash_review.json',out/'review.json')
    write_json(out/'manifest.json',dict(artifacts={p.relative_to(out).as_posix():sha(p)
               for p in sorted(out.rglob('*')) if p.is_file()},observed_at=observed))
    print(json.dumps(aggregates,indent=2))
    print('manifest_sha256='+sha(out/'manifest.json'))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--consumer',default='F:/Yandex.Disk/FINANCE/futures-analytics/outputs/cny-cash-mechanics-20261015')
    a=p.parse_args()
    run(a.evidence,a.output,a.consumer)
