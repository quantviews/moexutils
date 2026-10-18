from decimal import Decimal as D
import pytest
from scripts.build_cny_cash_audit import payoff, exercise_fee, opening_fee, round_money


def test_correct_fixing_materially_changes_june_payment():
    assert payoff('C', '10.8', '10.9718')[1] == D('17.18')
    assert payoff('C', '10.8', '10.8307')[1] == D('3.07')


def test_september_individual_rounding_and_net():
    short = payoff('C', '12.6', '12.9001')[1]
    long = payoff('C', '12.8', '12.9001')[1]
    assert (short, long, long-short) == (D('30.01'), D('10.01'), D('-20'))


def test_condor_bound_across_strikes():
    for s in map(D, ['0','10.39999','10.4','10.5','10.6','10.7','10.8','10.9','11','100']):
        net = payoff('P','10.4',s)[1]-payoff('P','10.6',s)[1]-payoff('C','10.8',s)[1]+payoff('C','11',s)[1]
        assert -20 <= net <= 0


def test_fees_are_separate_and_unknown_is_not_zero():
    assert opening_fee(None,'11','moex') is None
    assert opening_fee('.2',None,'ncc') is None
    assert opening_fee('.2','11','moex') == D('.07')
    assert opening_fee('.2','11','ncc') == D('.07')
    assert exercise_fee('10.8', True) == D('.04')
    assert exercise_fee('12.6', True) == D('.05')
    assert exercise_fee('12.2', False) == 0


def test_rounding_and_invalid_type():
    assert round_money('1.005') == D('1.01')
    assert payoff('C',12,12) == (0,0)
    with pytest.raises(ValueError):
        payoff('bad',12,12)
