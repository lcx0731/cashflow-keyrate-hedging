"""Price and stress explicit signed cashflows; educational, not investment advice."""

import argparse
import json
from pathlib import Path
import sys

from .core import Cashflows, ZeroCurve, risk, stress


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _fields(value, allowed, required):
    if not isinstance(value, dict) or set(value)-set(allowed) or set(required)-set(value):
        raise ValueError("missing or unknown object fields")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args(argv)
    try:
        with args.input.open("rb") as handle:
            data = handle.read(1024*1024+1)
        if len(data) > 1024*1024:
            raise ValueError("input exceeds 1 MiB")
        request = _fields(json.loads(data, object_pairs_hook=_pairs),
                          ["curve", "cashflows", "knot_shifts_bp", "cashflow_shifts_bp"], ["curve", "cashflows"])
        curve = ZeroCurve(**_fields(request["curve"], ["knots", "rates"], ["knots", "rates"]))
        flows = Cashflows(**_fields(request["cashflows"], ["times", "amounts"], ["times", "amounts"]))
        r = risk(flows, curve)
        result = {"present_value": r.value, "gradient_per_decimal_zero_rate": r.gradient.tolist(),
                  "hessian_per_decimal_zero_rate_squared": r.hessian.tolist(),
                  "dollar_dv01": r.dollar_dv01.tolist(),
                  "parallel_dollar_duration": r.parallel_dollar_duration,
                  "parallel_dollar_convexity": r.parallel_dollar_convexity,
                  "use": "Educational only. Not investment advice."}
        bumps = {key: request[key] for key in ["knot_shifts_bp", "cashflow_shifts_bp"] if key in request}
        if len(bumps) == 2:
            raise ValueError("provide only one shock field")
        if bumps:
            result["stress"] = stress(flows, curve, **bumps)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError, ArithmeticError) as exc:
        print(f"cashflow-risk: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
