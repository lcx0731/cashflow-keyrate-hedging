"""Continuous-compounding cashflow valuation under a stated nodal zero-curve model.

No bond calendar, market quote, spread, option, financing, or trading model is
implied. All sensitivities are with respect to continuously compounded zero rates.
"""

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np


def _vector(values, name, maximum, bound):
    if not isinstance(values, (list, tuple, np.ndarray)) or np.ndim(values) != 1 or not 1 <= len(values) <= maximum:
        raise ValueError(f"{name} must be a one-dimensional vector of length 1 to {maximum}")
    result = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{name} must contain real numbers, not booleans")
        try:
            number = float(value)
        except OverflowError as exc:
            raise ValueError(f"{name} contains an oversized number") from exc
        if not math.isfinite(number) or abs(number) > bound:
            raise ValueError(f"{name} must be finite with absolute values <= {bound}")
        result.append(number)
    return tuple(result)


@dataclass(frozen=True)
class Cashflows:
    times: tuple
    amounts: tuple

    def __post_init__(self):
        times = _vector(self.times, "times", 500, 50)
        amounts = _vector(self.amounts, "amounts", 500, 1e9)
        if len(times) != len(amounts) or any(t <= 0 for t in times):
            raise ValueError("times must be positive and match the number of amounts")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "amounts", amounts)


@dataclass(frozen=True)
class ZeroCurve:
    knots: tuple
    rates: tuple

    def __post_init__(self):
        knots = _vector(self.knots, "knots", 12, 50)
        rates = _vector(self.rates, "rates", 12, 0.2)
        if len(knots) < 2 or len(knots) != len(rates) or knots[0] <= 0 or any(a >= b for a,b in zip(knots, knots[1:])):
            raise ValueError("need 2 to 12 positive strictly increasing knots, with one rate per knot")
        object.__setattr__(self, "knots", knots)
        object.__setattr__(self, "rates", rates)

    def basis(self, times):
        times = _vector(times, "times", 500, 50)
        knots = np.asarray(self.knots)
        result = np.zeros((len(times), len(knots)))
        for i, t in enumerate(times):
            if not knots[0] <= t <= knots[-1]:
                raise ValueError("cashflow time outside curve knot range; extrapolation is not supported")
            j = min(max(int(np.searchsorted(knots, t, side="right")) - 1, 0), len(knots)-2)
            fraction = (t-knots[j]) / (knots[j+1]-knots[j])
            result[i, j] = 1-fraction
            result[i, j+1] = fraction
        return result


@dataclass
class Risk:
    value: float
    gradient: np.ndarray
    hessian: np.ndarray

    @property
    def dollar_dv01(self):
        """Minus the price derivative times one basis point; a positive loss convention."""
        return -1e-4 * self.gradient

    @property
    def parallel_dollar_duration(self):
        return -float(np.sum(self.gradient))

    @property
    def parallel_dollar_convexity(self):
        return float(np.sum(self.hessian))


def _discounted(flows, curve):
    if not isinstance(flows, Cashflows) or not isinstance(curve, ZeroCurve):
        raise ValueError("provide Cashflows and ZeroCurve objects")
    basis = curve.basis(flows.times)
    times = np.asarray(flows.times)
    pv = np.asarray(flows.amounts) * np.exp(-times * (basis @ np.asarray(curve.rates)))
    return basis, times, pv


def risk(flows, curve):
    basis, times, pv = _discounted(flows, curve)
    exposure = times[:, None] * basis
    gradient = -np.sum(pv[:, None] * exposure, axis=0)
    hessian = exposure.T @ (pv[:, None] * exposure)
    return Risk(math.fsum(pv), gradient, (hessian+hessian.T)/2)


def stress(flows, curve, *, knot_shifts_bp=None, cashflow_shifts_bp=None):
    """Full finite-shock repricing and a second-order Taylor remainder estimate.

    Exactly one shift vector is required. Arbitrary cashflow shifts may be outside
    the nodal model and are useful for an explicit model-scope audit.
    """
    if (knot_shifts_bp is None) == (cashflow_shifts_bp is None):
        raise ValueError("provide exactly one of knot_shifts_bp or cashflow_shifts_bp")
    basis, times, pv = _discounted(flows, curve)
    if knot_shifts_bp is not None:
        bumps = _vector(knot_shifts_bp, "knot_shifts_bp", 12, 2000)
        if len(bumps) != len(curve.knots):
            raise ValueError("one shift is needed per curve knot")
        shifts = basis @ np.asarray(bumps) * 1e-4
        kind = "linear nodal zero-rate shift"
    else:
        bumps = _vector(cashflow_shifts_bp, "cashflow_shifts_bp", 500, 2000)
        if len(bumps) != len(flows.times):
            raise ValueError("one shift is needed per cashflow")
        shifts = np.asarray(bumps) * 1e-4
        kind = "explicit cashflow-time zero-rate shift; not necessarily in nodal span"
    x = times * shifts
    linear = math.fsum(-pv*x)
    quadratic = math.fsum(0.5*pv*x*x)
    full = math.fsum(pv * np.expm1(-x))
    bound = math.fsum(np.abs(pv) * np.exp(np.abs(x)) * np.abs(x)**3 / 6)
    return {"shock_kind": kind, "base_value": math.fsum(pv),
            "full_repricing_change": full, "linear_change": linear,
            "quadratic_term": quadratic, "second_order_change": linear+quadratic,
            "second_order_error": full-linear-quadratic,
            "taylor_remainder_bound": bound,
            "cashflow_shifts_bp": (shifts*1e4).tolist()}


def match_risk(instruments, liability, curve, mode="keyrate"):
    """Unconstrained minimum-norm PV allocations with numerical span diagnostics.

    Negative holdings are allowed and reported. This is not a long-only,
    transaction-cost-aware, capital-constrained, or optimal trading strategy.
    """
    instruments = tuple(instruments)
    if not 1 <= len(instruments) <= 20 or not all(isinstance(x, Cashflows) for x in instruments):
        raise ValueError("provide 1 to 20 instruments")
    if mode not in ("parallel", "keyrate"):
        raise ValueError("mode must be parallel or keyrate")
    target = risk(liability, curve)
    measures = [risk(instrument, curve) for instrument in instruments]
    prices = np.array([r.value for r in measures])
    if target.value <= 0 or np.any(prices <= 0):
        raise ValueError("liability and each hedge instrument must have positive present value")
    horizon = curve.knots[-1]
    if mode == "parallel":
        sensitivities = np.array([[r.parallel_dollar_duration] for r in measures]).T
        target_sensitivities = np.array([target.parallel_dollar_duration])
    else:
        sensitivities = np.column_stack([-r.gradient for r in measures])
        target_sensitivities = -target.gradient
    matrix = np.vstack([np.ones(len(instruments)), sensitivities/prices[None, :]/horizon])
    rhs = np.r_[1.0, target_sensitivities/target.value/horizon]
    weights, _, rank_value, singular = np.linalg.lstsq(matrix, rhs, rcond=1e-12)
    residual = matrix @ weights - rhs
    matched = bool(np.max(np.abs(residual)) <= 1e-10)
    holdings = weights * target.value / prices
    return {"mode": mode, "status": "matched" if matched else "outside_instrument_span",
            "holdings": holdings.tolist(), "pv_weights_relative_to_liability": weights.tolist(),
            "gross_pv_multiple": float(np.sum(np.abs(weights))),
            "short_pv_multiple": float(np.sum(np.maximum(-weights, 0))),
            "liability_value": target.value, "instrument_values": prices.tolist(),
            "scaled_constraint_residual": residual.tolist(), "matrix_rank": int(rank_value),
            "constraint_count": len(rhs), "instrument_count": len(instruments),
            "singular_values": singular.tolist(), "scaled_residual_tolerance": 1e-10}


def net_cashflows(instruments, holdings, liability):
    instruments = tuple(instruments)
    holdings = _vector(holdings, "holdings", 20, 1e9)
    if len(instruments) != len(holdings) or not isinstance(liability, Cashflows) or not all(isinstance(c, Cashflows) for c in instruments):
        raise ValueError("holdings must match valid instruments and liability")
    by_time = {}
    for instrument, quantity in zip(instruments, holdings):
        for t, amount in zip(instrument.times, instrument.amounts):
            by_time.setdefault(t, []).append(quantity*amount)
    for t, amount in zip(liability.times, liability.amounts):
        by_time.setdefault(t, []).append(-amount)
    times = sorted(by_time)
    return Cashflows(times, [math.fsum(by_time[t]) for t in times])
