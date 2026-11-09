import math

import numpy as np
import pytest

from cashflow_hedge import Cashflows, ZeroCurve, match_risk, net_cashflows, risk, stress
from cashflow_hedge.reference import decimal_reference


@pytest.fixture
def curve():
    return ZeroCurve([2, 5, 10], [0.025, 0.032, 0.038])


def test_basis_partition_and_linear_reproduction(curve):
    times = np.linspace(2, 10, 101)
    basis = curve.basis(times)
    assert np.min(basis) >= 0
    np.testing.assert_allclose(basis.sum(axis=1), 1, atol=2e-16)
    np.testing.assert_allclose(basis @ curve.knots, times, atol=2e-15)
    np.testing.assert_array_equal(curve.basis(curve.knots), np.eye(3))


def test_zero_coupon_closed_form(curve):
    r = risk(Cashflows([6], [100]), curve)
    value = 100 * math.exp(-6 * 0.0332)
    assert r.value == pytest.approx(value)
    np.testing.assert_allclose(r.gradient, -value*np.array([0, 4.8, 1.2]))
    assert r.parallel_dollar_duration == pytest.approx(6*value)
    assert r.parallel_dollar_convexity == pytest.approx(36*value)
    assert r.dollar_dv01.sum() == pytest.approx(6e-4*value)


@pytest.mark.parametrize("seed", range(8))
def test_price_and_change_against_independent_decimal(curve, seed):
    rng = np.random.default_rng(seed)
    times = rng.uniform(2, 10, 20)
    amounts = rng.integers(-500, 501, 20).astype(float)
    shifts = rng.uniform(-250, 250, 20)
    flows = Cashflows(times, amounts)
    ref = decimal_reference(curve.knots, curve.rates, times, amounts, shifts)
    assert risk(flows, curve).value == pytest.approx(float(ref["price"]), abs=1e-11)
    report = stress(flows, curve, cashflow_shifts_bp=shifts)
    assert report["full_repricing_change"] == pytest.approx(float(ref["change"]), abs=1e-11)
    assert abs(report["second_order_error"]) <= report["taylor_remainder_bound"] + 1e-11


def test_gradient_and_hessian_against_finite_differences(curve):
    flows = Cashflows([2, 3.5, 6, 8, 10], [4, 8, -30, 20, 70])
    result = risk(flows, curve)
    eps = 1e-5
    for j in range(3):
        step = np.eye(3)[j]*eps
        plus = risk(flows, ZeroCurve(curve.knots, np.array(curve.rates)+step))
        minus = risk(flows, ZeroCurve(curve.knots, np.array(curve.rates)-step))
        assert (plus.value-minus.value)/(2*eps) == pytest.approx(result.gradient[j], rel=2e-8, abs=1e-8)
        np.testing.assert_allclose((plus.gradient-minus.gradient)/(2*eps), result.hessian[:, j], rtol=2e-8, atol=1e-7)


def test_hessian_positive_for_positive_cashflows(curve):
    r = risk(Cashflows([2, 4, 7, 10], [1, 2, 3, 4]), curve)
    assert np.linalg.eigvalsh(r.hessian).min() > 0
    np.testing.assert_array_equal(r.hessian, r.hessian.T)


def test_linear_basis_identity_links_keyrates_to_parallel_convexity(curve):
    # A structural identity: sum_j k_j B_j(t) = t.
    r = risk(Cashflows([2.1, 4.6, 7.4, 9.9], [4, -5, 11, 8]), curve)
    assert r.parallel_dollar_convexity == pytest.approx(-np.dot(curve.knots, r.gradient))


def test_taylor_terms_and_cubic_remainder(curve):
    flows = Cashflows([6], [100])
    r = risk(flows, curve)
    direction = np.array([70, -100, 50])
    a = stress(flows, curve, knot_shifts_bp=direction)
    b = stress(flows, curve, knot_shifts_bp=direction/2)
    dx = direction*1e-4
    assert a["linear_change"] == pytest.approx(r.gradient @ dx)
    assert a["quadratic_term"] == pytest.approx(0.5*dx @ r.hessian @ dx)
    assert abs(a["second_order_error"] / b["second_order_error"]) == pytest.approx(8, rel=.03)
    assert abs(a["second_order_error"]) <= a["taylor_remainder_bound"]


def test_tiny_shock_retains_change(curve):
    flows = Cashflows([6], [100])
    out = stress(flows, curve, cashflow_shifts_bp=[1e-14])
    reference = decimal_reference(curve.knots, curve.rates, flows.times, flows.amounts, [1e-14])
    assert out["full_repricing_change"] != 0
    assert out["full_repricing_change"] == pytest.approx(float(reference["change"]), rel=1e-14, abs=0)


def test_zero_shock_is_exact_zero(curve):
    report = stress(Cashflows([5], [10]), curve, knot_shifts_bp=[0, 0, 0])
    for key in ["full_repricing_change", "linear_change", "quadratic_term", "second_order_error", "taylor_remainder_bound"]:
        assert report[key] == 0


def test_exact_hedge_allocations_and_risk(curve):
    liability = Cashflows([6], [100])
    instruments = [Cashflows([t], [1]) for t in [2, 4, 7, 10]]
    match = match_risk(instruments, liability, curve)
    expected = np.array([4/13, -6/13, 56/39, -11/39])
    assert match["status"] == "matched"
    assert match["matrix_rank"] == 4
    np.testing.assert_allclose(match["pv_weights_relative_to_liability"], expected, atol=2e-14)
    assert match["gross_pv_multiple"] == pytest.approx(97/39)
    assert match["short_pv_multiple"] == pytest.approx(29/39)
    net = risk(net_cashflows(instruments, match["holdings"], liability), curve)
    assert abs(net.value) < 1e-11
    np.testing.assert_allclose(net.gradient, 0, atol=1e-10)
    assert abs(net.parallel_dollar_convexity) < 1e-9


def test_parallel_match_leaves_twist_exposure(curve):
    liability = Cashflows([6], [100])
    instruments = [Cashflows([t], [1]) for t in [2, 10]]
    match = match_risk(instruments, liability, curve, mode="parallel")
    np.testing.assert_allclose(match["pv_weights_relative_to_liability"], [.5, .5], atol=1e-14)
    net = risk(net_cashflows(instruments, match["holdings"], liability), curve)
    np.testing.assert_allclose(net.gradient/risk(liability, curve).value, [-1, 4.8, -3.8], atol=1e-14)
    assert abs(net.parallel_dollar_duration) < 1e-10
    assert abs(net.gradient @ [0.01, 0, -0.01]) > 1


def test_unit_rescaling_does_not_change_pv_allocations(curve):
    liability = Cashflows([6], [100])
    times = [2, 4, 7, 10]
    units = np.array([10, 100, .2, 17])
    a = match_risk([Cashflows([t], [1]) for t in times], liability, curve)
    b = match_risk([Cashflows([t], [u]) for t, u in zip(times, units)], liability, curve)
    np.testing.assert_allclose(a["holdings"], np.array(b["holdings"])*units, rtol=1e-13)
    np.testing.assert_allclose(a["pv_weights_relative_to_liability"], b["pv_weights_relative_to_liability"], rtol=1e-13)


def test_infeasible_span_is_not_labeled_matched(curve):
    out = match_risk([Cashflows([2], [1])], Cashflows([6], [100]), curve)
    assert out["status"] == "outside_instrument_span"
    assert max(map(abs, out["scaled_constraint_residual"])) > .1


def test_duplicate_instruments_report_rank_and_split_pv(curve):
    out = match_risk([Cashflows([6], [1])]*2, Cashflows([6], [100]), curve)
    assert out["status"] == "matched"
    assert out["matrix_rank"] == 1
    np.testing.assert_allclose(out["holdings"], [50, 50], atol=1e-12)


def test_net_cashflows_aggregate_duplicates(curve):
    net = net_cashflows([Cashflows([5, 5, 7], [1, 2, 4])], [2], Cashflows([5, 7], [6, 8]))
    assert net.times == (5, 7)
    assert net.amounts == (0, 0)
    assert risk(net, curve).value == 0


def test_off_grid_risk_is_invisible_at_knots_and_assets(curve):
    liability = Cashflows([6], [100])
    instruments = [Cashflows([t], [1]) for t in [2, 4, 7, 10]]
    hedge = match_risk(instruments, liability, curve)
    net = net_cashflows(instruments, hedge["holdings"], liability)
    bump = lambda t: 100*max(0, 1-abs(t-6)/.5)
    assert all(bump(t) == 0 for t in (*curve.knots, 2, 4, 7, 10))
    report = stress(net, curve, cashflow_shifts_bp=[bump(t) for t in net.times])
    expected = risk(liability, curve).value * (-math.expm1(-.06))
    assert report["full_repricing_change"] == pytest.approx(expected)
    assert expected > 4


@pytest.mark.parametrize("times,amounts", [([], []), ([0], [1]), ([51], [1]), ([3], [True]), ([3], [float("nan")]), ([3], [1e10]), ([3, 4], [1]), ([[3]], [1]), ([3], ["1"])])
def test_invalid_cashflows(times, amounts):
    with pytest.raises(ValueError):
        Cashflows(times, amounts)


@pytest.mark.parametrize("knots,rates", [([2], [.03]), ([2, 2], [.03, .04]), ([5, 2], [.03, .04]), ([0, 5], [.03, .04]), ([2, 5], [.03]), ([2, 5], [.03, .3]), ([2, 5], [.03, True])])
def test_invalid_curve(knots, rates):
    with pytest.raises(ValueError):
        ZeroCurve(knots, rates)


@pytest.mark.parametrize("time", [1.99, 10.01])
def test_no_curve_extrapolation(curve, time):
    with pytest.raises(ValueError, match="extrapolation"):
        risk(Cashflows([time], [1]), curve)


@pytest.mark.parametrize("shocks", [{}, {"knot_shifts_bp": [1]}, {"knot_shifts_bp": [0, 0, 2001]}, {"cashflow_shifts_bp": [1, 2]}, {"knot_shifts_bp": [0, 0, 0], "cashflow_shifts_bp": [0]}])
def test_invalid_shock(curve, shocks):
    with pytest.raises(ValueError):
        stress(Cashflows([6], [100]), curve, **shocks)


@pytest.mark.parametrize("mode", ["long_only", "", None])
def test_invalid_hedge_mode(curve, mode):
    with pytest.raises(ValueError):
        match_risk([Cashflows([6], [1])], Cashflows([6], [100]), curve, mode=mode)


def test_nonpositive_pv_cannot_be_used_as_allocation_unit(curve):
    with pytest.raises(ValueError, match="positive present value"):
        match_risk([Cashflows([6], [-1])], Cashflows([6], [100]), curve)
