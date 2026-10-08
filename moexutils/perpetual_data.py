"""Versioned offline CNY input bundle for consumers; no portfolio/VM simulation."""
from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
import hashlib
import json
from pathlib import Path

import polars as pl


POLICY = {
    'version': 1,
    'mode': 'estimated_vm',
    'funding_source': 'FORTS.SWAPRATE',
    'funding_unit_cnyrubf': 'RUB_per_CNY',
    'assumed_cnyrubf_lot': 1000,
    'historical_lot_applicability': 'assumption_not_verified',
    'positive_funding_payer': 'long',
    'missing_funding': 'unknown_never_zero_or_forward_fill',
    'disputed_funding': 'retain_FORTS_and_flag',
    'swaprate_curr': 'diagnostic_not_additional_payment',
    'daily_mark': 'SETTLEPRICE_no_CLOSE_fallback',
    'publication_time': 'unknown_not_available_for_same_day_signal',
    'calendar': 'observed_FORTS_dates_no_synthetic_weekend_accrual',
    'position_eligibility': 'consumer_explicit_before_or_after_daily_mark',
    'rounding': 'consumer_decimal_half_up_to_kopecks_per_contract_per_daily_mark',
    'commissions_slippage_margin': 'consumer_explicit_not_included',
}


def _unique(df, keys, label):
    if any(df[c].null_count() for c in keys):
        raise ValueError(f'Null {label}: {keys}')
    if df.select(keys).is_duplicated().any():
        raise ValueError(f'Duplicate {label}: {keys}')


def _quality(history, comparisons):
    _unique(history, ['SECID', 'BOARDID', 'date'], 'history keys')
    result = history
    for board in ('FUSR', 'FUSC'):
        name = board.lower() + '_status'
        if board not in comparisons:
            result = result.with_columns(pl.lit('not_checked').alias(name))
            continue
        check = comparisons[board].filter(pl.col('forts_present').fill_null(False))
        _unique(check, ['date', 'BOARDID'], board)
        check = check.select('date', 'BOARDID', pl.lit('CNYRUBF').alias('SECID'),
                             pl.col('status').alias(name))
        result = result.join(check, on=['date', 'BOARDID', 'SECID'], how='left').with_columns(
            pl.col(name).fill_null('not_checked'))
    return result.with_columns(
        pl.col('SETTLEPRICE').is_null().alias('settle_missing'),
        (~pl.col('registry_present').fill_null(False)).alias('registry_missing'),
        (pl.col('VOLUME') == 0).fill_null(False).alias('volume_zero'),
        pl.col('VOLUME').is_null().alias('volume_missing'),
        pl.when(pl.col('SECID') != 'CNYRUBF').then(pl.lit('not_applicable'))
        .when(pl.col('SWAPRATE').is_null()).then(pl.lit('missing'))
        .when(pl.col('SWAPRATE') == 0).then(pl.lit('zero'))
        .otherwise(pl.lit('observed')).alias('funding_state'),
        pl.lit(False).alias('historical_parameters_verified'),
        pl.lit(False).alias('publication_time_verified'),
    )


@dataclass(frozen=True)
class CnyBundle:
    """Tables share one audit snapshot. Observations are NOT historical parameters."""

    history: pl.DataFrame
    descriptions: pl.DataFrame
    parameter_observations: pl.DataFrame
    candidate_gaps: pl.DataFrame
    metadata: dict

    def pair(self, dated_secid: str, *, board: str = 'RFUD', start=None, end=None) -> pl.DataFrame:
        """Full date join of an explicit ordinary CNY and CNYRUBF, all raw columns.

        Prefixes dated_/perpetual_; missing legs retained, never filled.
        No roll rule, normalized quote, execution price or P&L is inferred.
        """
        dated = self.history.filter((pl.col('SECID') == dated_secid) & (pl.col('BOARDID') == board))
        perp = self.history.filter((pl.col('SECID') == 'CNYRUBF') & (pl.col('BOARDID') == board))
        if dated.is_empty() or perp.is_empty():
            raise ValueError('Both instruments must exist on the explicit board')
        valid = (pl.col('source_type') == 'futures') & (pl.col('registry_asset') == 'CNY')
        if dated_secid == 'CNYRUBF' or not dated.select(valid.fill_null(False).all()).item():
            raise ValueError('Expected an ordinary CNY contract, not a spread or unknown type')
        legs = []
        for prefix, frame in [('dated_', dated), ('perpetual_', perp)]:
            frame = frame.with_columns(pl.lit(True).alias('present'))
            legs.append(frame.rename({c: prefix + c for c in frame.columns if c != 'date'}))
        result = legs[0].join(legs[1], on='date', how='full', coalesce=True).with_columns(
            pl.col('dated_present').fill_null(False), pl.col('perpetual_present').fill_null(False),
        ).with_columns((pl.col('dated_present') & pl.col('perpetual_present')).alias('both_legs_present'))
        begin = dt.date.fromisoformat(start) if isinstance(start, str) else start
        finish = dt.date.fromisoformat(end) if isinstance(end, str) else end
        if begin and finish and begin > finish:
            raise ValueError('start must not be after end')
        if begin:
            result = result.filter(pl.col('date') >= begin)
        if finish:
            result = result.filter(pl.col('date') <= finish)
        return result.sort('date')


def read_cny_bundle(report_dir: str | Path) -> CnyBundle:
    """Read an explicit completed audit directory, offline, schema version 1.

    Missing/malformed declared comparison files fail; an audit without comparison
    exposes not_checked, never an implied pass. No automatic latest selection.
    """
    root = Path(report_dir)
    metadata = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    if metadata.get('bundle_schema_version') != 1:
        raise ValueError('Expected bundle schema 1; rerun python -m moexutils.perpetual_audit')
    required = ['history.parquet', 'descriptions.parquet', 'parameter_observations.parquet', 'candidate_gaps.parquet']
    required += [f'{board}-comparison.parquet' for board in ('FUSR', 'FUSC')
                 if board in metadata.get('comparisons', {})]
    for name in required:
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if metadata.get('artifact_sha256', {}).get(name) != digest:
            raise ValueError(f'Missing or mismatched artifact checksum: {name}')
    history = pl.read_parquet(root / 'history.parquet')
    comparisons = {board: pl.read_parquet(root / f'{board}-comparison.parquet')
                   for board in metadata.get('comparisons', {}) if board in ('FUSR', 'FUSC')}
    return CnyBundle(
        _quality(history, comparisons),
        pl.read_parquet(root / 'descriptions.parquet'),
        pl.read_parquet(root / 'parameter_observations.parquet'),
        pl.read_parquet(root / 'candidate_gaps.parquet'),
        {**metadata, 'estimation_policy': dict(POLICY)},
    )
