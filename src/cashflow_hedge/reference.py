"""Independent 60-digit decimal price and finite-shock change from raw arrays."""

from decimal import Decimal, localcontext


def decimal_reference(knots, rates, times, amounts, cashflow_shifts_bp=None):
    """Interpret each input as its exact binary float, then interpolate in Decimal.

    This is a high-precision check, not a directed-rounding proof. Inputs are
    assumed validated; unlike the core it uses scalar interpolation and exp.
    """
    def d(x):
        return Decimal.from_float(float(x))
    with localcontext() as context:
        context.prec = 60
        k, r = list(map(d, knots)), list(map(d, rates))
        total = Decimal(0)
        change = Decimal(0)
        shifts = [0]*len(times) if cashflow_shifts_bp is None else cashflow_shifts_bp
        for raw_t, raw_amount, raw_shift in zip(times, amounts, shifts):
            t = d(raw_t)
            j = next(j for j in range(len(k)-1) if k[j] <= t <= k[j+1])
            rate = r[j] + (r[j+1]-r[j]) * (t-k[j]) / (k[j+1]-k[j])
            pv = d(raw_amount) * (-t*rate).exp()
            total += pv
            change += pv * ((-t*d(raw_shift)/Decimal(10000)).exp() - 1)
        return {"price": str(total), "change": str(change)}
