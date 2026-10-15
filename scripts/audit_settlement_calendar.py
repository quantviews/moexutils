"""Compare NCC Excel exports with ISS and audit conditional SPEQ settlement dates.

Does not equate a missing calendar exception with a verified settlement day.
Reads Excel only; all normalized tables use Polars. Output packages are immutable.
"""

import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

from openpyxl import load_workbook
import polars as pl
import requests


def digest(body):
    return hashlib.sha256(body).hexdigest()


def excel_rows(path):
    book = load_workbook(path, read_only=True, data_only=False)
    try:
        sheet = book["settle_dates"]
        rows = list(sheet.values)
        expected = ("Код валюты", "Краткое наименование", "Дата",
                    "Расчётный день по календарю ВР", "Расчётный день по календарю ФР")
        if tuple(rows[2]) != expected:
            raise ValueError("Unexpected calendar headers")
        result = []
        for number, (currency, title, day, fx, stock) in enumerate(rows[3:], 4):
            if not isinstance(day, datetime) or fx not in ("Да", "Нет", None) or stock not in ("Да", "Нет", None):
                raise ValueError(f"Invalid calendar row {number}")
            result.append(dict(tradedate=day.date().isoformat(), currencyid=currency,
                               currency_title=title, currency_market={"Да": 1, "Нет": 0, None: None}[fx],
                               stock_market={"Да": 1, "Нет": 0, None: None}[stock],
                               source_row=number))
        keys = [(r["tradedate"], r["currencyid"]) for r in result]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate currency/date in Excel")
        return result
    finally:
        book.close()


def explicit_day(day, rows):
    values = [r["stock_market"] for r in rows
              if r["tradedate"] == day.isoformat() and r["currencyid"] in ("RUB", "-")]
    if not values:
        return None, "no_explicit_entry"
    if all(v is None for v in values):
        return None, "explicit_null"
    known = {v for v in values if v is not None}
    if len(known) != 1:
        return None, "conflicting_entries"
    return bool(next(iter(known))), "explicit_RUB_or_all_currencies"


def conditional_settlement(trade, lag, rows, covered_years):
    """Weekday scenario only; unknown explicit values or missing years stop it."""
    trace = []
    day = trade
    for _ in range(20):
        day += timedelta(days=1)
        if day.year not in covered_years:
            return None, trace, "year_not_covered"
        value, reason = explicit_day(day, rows)
        if value is None and reason != "no_explicit_entry":
            return None, trace, reason
        assumed = value is None
        if assumed:
            value = day.weekday() < 5
        trace.append(dict(date=day.isoformat(), is_settlement_day=value,
                          basis="unverified_weekday_baseline" if assumed else reason))
        lag -= int(value)
        if lag == 0:
            return day, trace, "conditional_weekday_baseline"
    raise ValueError("No candidate settlement within 20 days")


def audit(incoming, series, output, iss_cache=None):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "originals").mkdir()
    (out / "iss").mkdir()
    observations, comparisons, all_rows, differences = [], [], [], []
    with requests.Session() as session:
        for path in sorted(Path(incoming).glob("trading-calendar-*.xlsx")):
            year = int(path.stem[-4:])
            raw = path.read_bytes()
            rows = excel_rows(path)
            if any(date.fromisoformat(r["tradedate"]).year != year for r in rows):
                raise ValueError("Filename year differs from data")
            (out / "originals" / path.name).write_bytes(raw)
            if iss_cache:
                content = (Path(iss_cache) / "iss" / f"{year}.json").read_bytes()
                metadata = next(r for r in json.loads((Path(iss_cache) / "observations.json").read_text(encoding="utf-8"))
                                if r["artifact"] == f"iss/{year}.json")
                if digest(content) != metadata["sha256"]:
                    raise ValueError("Cached ISS checksum mismatch")
            else:
                response = session.get(
                    "https://iss.moex.com/iss/rms/engines/stock/objects/settlementscalendar.json",
                    params={"year": year}, timeout=(10, 60))
                response.raise_for_status()
                content = response.content
                metadata = dict(url=response.url, observed_at=datetime.now(timezone.utc).isoformat(),
                                artifact=f"iss/{year}.json", sha256=digest(content))
            block = json.loads(content)["settlements_calendar"]
            remote = [dict(zip(block["columns"], r)) for r in block["data"]]
            (out / "iss" / f"{year}.json").write_bytes(content)
            observations.append(metadata)
            def canonical(items):
                return sorted(json.dumps({k: r[k] for k in block["columns"]}, sort_keys=True,
                                         ensure_ascii=False) for r in items)
            equal = canonical(rows) == canonical(remote)
            local_keys = {(r["tradedate"], r["currencyid"]): r for r in rows}
            remote_keys = {(r["tradedate"], r["currencyid"]): r for r in remote}
            if len(remote_keys) != len(remote):
                raise ValueError("Duplicate ISS currency/date")
            local_subset = all(k in remote_keys and all(r[f] == remote_keys[k][f] for f in block["columns"])
                               for k, r in local_keys.items())
            for k in sorted(local_keys.keys() | remote_keys.keys()):
                left, right = local_keys.get(k), remote_keys.get(k)
                if left is None or right is None or any(left[f] != right[f] for f in block["columns"]):
                    differences.append(dict(year=year, key=k, excel=left, iss=right))
            comparisons.append(dict(year=year, excel_rows=len(rows), iss_rows=len(remote), equal=equal,
                                    excel_is_iss_subset=local_subset,
                                    source_sha256=digest(raw)))
            if not local_subset:
                raise ValueError("Conflicting calendars require explicit review")
            all_rows.extend(r | dict(source_file=f"iss/{year}.json", source_sha256=digest(content),
                                    present_in_excel=(r["tradedate"], r["currencyid"]) in local_keys)
                            for r in remote)
        # Preserve the access limitation as evidence; do not accept subscriber HTML as JSON.
        response = session.get("https://iss.moex.com/iss/calendars/stock/session/settlecodes.json",
                               params={"date": "2023-06-16"}, timeout=(10, 60))
        (out / "iss" / "settlecodes-response.bin").write_bytes(response.content)
        observations.append(dict(url=response.url, http_status=response.status_code,
                                 observed_at=datetime.now(timezone.utc).isoformat(),
                                 artifact="iss/settlecodes-response.bin", sha256=digest(response.content),
                                 subscriber_notice="only to subscribers" in response.text))
    series_path = Path(series)
    (out / "input_series.parquet").write_bytes(series_path.read_bytes())
    frame = pl.read_parquet(series_path)
    # Empty or partial early exports do not qualify as year coverage.
    years = {r["year"] for r in comparisons if r["excel_is_iss_subset"] and r["excel_rows"] and r["year"] >= 2023}
    results, traces = [], []
    for r in frame.to_dicts():
        trade = r["delivery_trade_date"]
        lag = 2 if trade < date(2023, 7, 31) else 1
        candidate, trace, status = conditional_settlement(trade, lag, all_rows, years)
        results.append(dict(secid=r["secid"], asset=r["asset"], last_trade_date=r["last_trade_date"],
                            delivery_trade_date=trade, matured_by_snapshot=r["matured_by_snapshot"],
                            standard_speq_lag=lag, standard_speq_code=f"Y{lag}",
                            conditional_cash_settlement_date=candidate, cash_settlement_date=None,
                            settlement_calendar_verified=False, historical_known_at_verified=False,
                            status=status, actual_execution_verified=False))
        traces.extend(t | dict(secid=r["secid"]) for t in trace)
    pl.DataFrame(all_rows).write_parquet(out / "calendar_entries.parquet")
    pl.DataFrame(results, schema_overrides={"cash_settlement_date": pl.Date,
                                          "conditional_cash_settlement_date": pl.Date}).write_parquet(out / "series_settlement_audit.parquet")
    pl.DataFrame(traces).write_parquet(out / "calculation_trace.parquet")
    summary = dict(excel_files=len(comparisons), entries=len(all_rows), comparisons=comparisons,
                   contracts=len(results), conditional_dates=sum(r["conditional_cash_settlement_date"] is not None for r in results),
                   verified_cash_settlement_dates=0, request_fully_closed=False,
                   limits=["Weekday baseline for absent entries not independently verified",
                           "Standard SPEQ Y2/Y1 cycle does not prove actual participant settlement or Nn code",
                           "Historical publication timestamps and complete series exceptions unverified",
                           "No 2027 Excel calendar supplied"])
    sources = [dict(url="https://www.moex.com/s3511", accessed_on="2026-10-15",
                    finding="SPEQ listed among boards changing Y2 to Y1 on 2023-07-31; earlier trades retain dates",
                    evidence_type="official_web_page_review"),
               dict(url="https://www.moex.com/files/4rkd3yjkghfhqz4h7g8rewetx4", accessed_on="2026-10-15",
                    pages=[5, 17, 18, 21],
                    finding="Dash denotes all currencies in currency_settlements; W distinguishes weekend sessions; settlecodes exposes settledate",
                    limitation="This documents a related ISS calendar; absent RMS entries are not explicitly defined",
                    evidence_type="official_document_review"),
               dict(url="https://www.nationalclearingcentre.ru/catalog/0219", accessed_on="2026-10-15",
                    finding="Legacy FX calendar lists non-settlement dates by currency",
                    limitation="Not sufficient to infer complete stock-market calendar",
                    evidence_type="official_web_page_review")]
    for name, data in [("summary.json", summary), ("observations.json", observations),
                       ("calendar_differences.json", differences), ("rule_sources.json", sources)]:
        (out / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "auditor.py").write_bytes(Path(__file__).read_bytes())
    manifest = {p.relative_to(out).as_posix(): digest(p.read_bytes()) for p in out.rglob("*") if p.is_file()}
    (out / "manifest.json").write_text(json.dumps(dict(artifact_sha256=manifest), indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("incoming", "series", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--iss-cache")
    print(json.dumps(audit(**vars(parser.parse_args())), indent=2))
