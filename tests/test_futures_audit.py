import duckdb
import pytest

from moexutils import futures_audit as audit


@pytest.fixture
def con():
    c = duckdb.connect()
    c.execute("ATTACH ':memory:' AS lake")
    schemas = {
        'futures': 'SECID VARCHAR,date DATE,BOARDID VARCHAR,VOLUME DOUBLE,OPEN DOUBLE,HIGH DOUBLE,LOW DOUBLE,CLOSE DOUBLE,SETTLEPRICE DOUBLE',
        'futures_contracts': 'secid VARCHAR,asset_code VARCHAR,start_date DATE,expiration_date DATE',
        'futures_description_observations': '''secid VARCHAR,ASSETCODE VARCHAR,TYPE VARCHAR,TYPENAME VARCHAR,
            "GROUP" VARCHAR,FRSTTRADE VARCHAR,LSTTRADE VARCHAR,LSTDELDATE VARCHAR,
            observed_at TIMESTAMPTZ,source_url VARCHAR''',
        'futures_parameter_observations': 'SECID VARCHAR,MINSTEP DOUBLE,STEPPRICE DOUBLE,INITIALMARGIN DOUBLE,observed_at TIMESTAMPTZ',
        'futures_risk_limits': 'date DATE,assetcode VARCHAR,updatetime VARCHAR',
        'futures_specification_editions': '''asset_candidate VARCHAR,source_url VARCHAR,
            source_valid_from DATE,source_valid_to DATE,reviewed_on DATE,
            contract_applicability_verified BOOLEAN,document_downloaded BOOLEAN''',
        'futures_continuous': 'SECID VARCHAR,date DATE,asset VARCHAR,settle DOUBLE,roll BOOLEAN',
    }
    for name, schema in schemas.items():
        c.execute(f'CREATE TABLE lake.{name} ({schema})')
    yield c
    c.close()


def test_observations_never_backfill_historical_parameters(con):
    con.execute("INSERT INTO lake.futures (SECID,date,VOLUME) VALUES ('A','2020-01-02',1)")
    con.execute("INSERT INTO lake.futures_contracts VALUES ('A','Si','2019-01-01','2020-03-01')")
    con.execute("INSERT INTO lake.futures_parameter_observations VALUES ('A',1,1,100,'2026-10-07 00:00:00+00')")
    con.execute("INSERT INTO lake.futures_risk_limits VALUES ('2020-01-02','Si','1'),('2020-01-02','Si','2')")
    row = audit._reports(con)['coverage'].row(0, named=True)
    assert row['history_rows'] == row['risk_archive_matched_rows'] == 1
    assert row['point_value_observations'] == row['margin_observations'] == 1
    assert row['historical_point_value_status'] == row['historical_ruble_margin_status'] == 'unavailable'


def test_source_types_do_not_invent_registry_assets(con):
    con.execute("INSERT INTO lake.futures (SECID,date) VALUES ('NTB','2015-01-01'),('RSK','2021-01-01')")
    con.execute('''INSERT INTO lake.futures_description_observations (secid,TYPE,observed_at)
        VALUES ('NTB','commodity_futures','2026-10-07 00:00:00+00'),('RSK','currency','2026-10-07 00:00:00+00')''')
    rows = audit._reports(con)['missing_registry'].to_dicts()
    assert [r['instrument_class'] for r in rows] == ['ntb_standard_observed', 'non_forts_description_unresolved']
    assert all(r['registry_asset'] is None for r in rows)


def test_only_exact_confirmed_negative_settlement_is_explained(con):
    con.execute('''INSERT INTO lake.futures (SECID,date,VOLUME,SETTLEPRICE) VALUES
        ('CLJ0','2020-04-21',1,-37.63),('OTHER','2020-04-21',1,-37.63)''')
    rows = audit._reports(con)['nonpositive_settlement'].to_dicts()
    assert rows[0]['status'] == 'confirmed_exchange_settlement'
    assert rows[1]['status'] == 'unresolved'
    assert rows[1]['confirmation_url'] is None


@pytest.mark.parametrize('new_previous,old_today,expected,ratio', [
    (120, 100, 'previous_date', 1.2),
    (None, 100, 'roll_date', 1.3),
    (None, None, 'missing_pair_ratio_one', 1.0),
])
def test_roll_pair_provenance(con, new_previous, old_today, expected, ratio):
    con.execute('''INSERT INTO lake.futures_continuous VALUES
        ('A','2020-01-01','Si',100,false),('B','2020-01-02','Si',130,true)''')
    con.executemany('INSERT INTO lake.futures (SECID,date,SETTLEPRICE) VALUES (?,?,?)', [
        ('A', '2020-01-01', 100), ('B', '2020-01-01', new_previous),
        ('A', '2020-01-02', old_today), ('B', '2020-01-02', 130)])
    row = audit._reports(con)['rolls'].row(0, named=True)
    assert row['ratio_source'] == expected
    assert row['ratio'] == pytest.approx(ratio)


def test_latest_description_is_unique_and_conflicts_are_reported(con):
    con.execute("INSERT INTO lake.futures (SECID,date) VALUES ('A','2020-01-01')")
    con.execute("INSERT INTO lake.futures_contracts VALUES ('A','Si','2019-01-01','2020-03-01')")
    con.execute('''INSERT INTO lake.futures_description_observations (secid,ASSETCODE,observed_at)
        VALUES ('A','Si','2026-10-06 00:00:00+00'),('A','BR','2026-10-07 00:00:00+00')''')
    con.execute("INSERT INTO lake.futures_risk_limits VALUES ('2020-01-01','Si','1')")
    row = audit._reports(con)['coverage'].row(0, named=True)
    assert row['history_rows'] == 1 and row['asset_conflict']
    assert row['risk_archive_matched_rows'] == 0
