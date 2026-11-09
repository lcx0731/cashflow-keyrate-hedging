"""Reproduce an entirely synthetic liability-hedging and model-scope audit."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cashflow_hedge import Cashflows, ZeroCurve, match_risk, net_cashflows, risk, stress
from cashflow_hedge.reference import decimal_reference


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    curve = ZeroCurve([2, 5, 10], [.025, .032, .038])
    liability = Cashflows([6], [100])
    instrument_times = [2, 4, 7, 10]
    instruments = [Cashflows([t], [1]) for t in instrument_times]
    definitions = {"parallel": [instruments[0], instruments[-1]], "keyrate": instruments}
    hedges, nets = {}, {}
    for name, assets in definitions.items():
        hedge = match_risk(assets, liability, curve, mode=name)
        if hedge["status"] != "matched":
            raise RuntimeError(f"Unexpected infeasible example: {name}")
        net = net_cashflows(assets, hedge["holdings"], liability)
        result = risk(net, curve)
        hedge.update({"asset_maturities": [a.times[0] for a in assets],
                      "net_present_value": result.value,
                      "net_dollar_dv01": result.dollar_dv01.tolist(),
                      "net_parallel_dollar_convexity": result.parallel_dollar_convexity})
        hedges[name], nets[name] = hedge, net
    scenarios = {"parallel_up": [100, 100, 100], "parallel_down": [-100, -100, -100],
                 "twist": [100, 0, -100], "reverse_twist": [-100, 0, 100],
                 "belly_up": [0, 100, 0], "belly_down": [0, -100, 0]}
    results = {}
    reference_differences = []
    for name, net in nets.items():
        results[name] = {}
        for label, bumps in scenarios.items():
            report = stress(net, curve, knot_shifts_bp=bumps)
            reference = decimal_reference(curve.knots, curve.rates, net.times, net.amounts, report["cashflow_shifts_bp"])
            report["decimal_reference_change"] = reference["change"]
            error = abs(report["full_repricing_change"] - float(reference["change"]))
            report["decimal_reference_absolute_difference"] = error
            reference_differences.append(error)
            results[name][label] = report
    # A triangular zero-rate perturbation outside the three-knot model.
    bump = lambda t: 100*max(0, 1-abs(t-6)/.5)
    off_grid = {}
    for name, net in nets.items():
        off_grid[name] = {"up": stress(net, curve, cashflow_shifts_bp=[bump(t) for t in net.times]),
                          "down": stress(net, curve, cashflow_shifts_bp=[-bump(t) for t in net.times])}
    scales = np.linspace(-300, 300, 121)
    sweep = {name: [stress(net, curve, knot_shifts_bp=[s, 0, -s]) for s in scales] for name, net in nets.items()}
    summary = {
        "data": "Deterministic synthetic cashflows and zero rates; no market observations.",
        "use": "Educational only. Not investment advice.",
        "curve": {"knots_years": curve.knots, "continuous_zero_rates": curve.rates},
        "liability": {"time_years": 6, "amount": 100, "present_value": risk(liability, curve).value},
        "hedges": hedges, "nodal_scenarios_bp": scenarios, "nodal_results": results,
        "off_grid": {"definition": "h(t) = 100 max(0, 1 - |t-6|/0.5) bp", "results": off_grid},
        "max_decimal_change_difference": max(reference_differences),
        "diagnostic": "Bounds are real-arithmetic Taylor formulas evaluated in float64, not directed-rounding certificates."
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    (output / "twist_sweep.json").write_text(json.dumps({"twist_size_bp": scales.tolist(), "results": sweep}, indent=2, allow_nan=False)+"\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"parallel": "#b45822", "keyrate": "#176885"}
    names = {"parallel": "PV + parallel duration", "keyrate": "PV + all key rates"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6), layout="constrained")
    fig.suptitle("Key-rate hedging: matched sensitivities, remaining risks", fontsize=17, fontweight="bold")
    ax = axes[0, 0]
    for i, name in enumerate(nets):
        ax.bar(np.arange(3)+(i-.5)*.35, hedges[name]["net_dollar_dv01"], .35, color=colors[name], label=names[name])
    ax.axhline(0, color="#777777", lw=.7)
    ax.set(xticks=range(3), xticklabels=["2 years", "5 years", "10 years"], ylabel="Net dollar DV01 (loss convention)", title="A. First-order risk of assets minus liability")
    ax.legend(fontsize=9)
    ax = axes[0, 1]
    for name in nets:
        ax.plot(scales, [r["full_repricing_change"] for r in sweep[name]], color=colors[name], lw=2, label=names[name])
    ax.axhline(0, color="#777777", lw=.7)
    ax.set(xlabel="Twist s: nodal shifts [s, 0, −s] (bp)", ylabel="Change in net present value", title="B. Full finite-shock repricing")
    ax.legend(fontsize=9)
    ax = axes[1, 0]
    positive = scales > 0
    errors = np.array([abs(r["second_order_error"]) for r in sweep["keyrate"]])
    bounds = np.array([r["taylor_remainder_bound"] for r in sweep["keyrate"]])
    ax.loglog(scales[positive], errors[positive], color=colors["keyrate"], lw=2, label="Actual second-order error")
    ax.loglog(scales[positive], bounds[positive], "--", color="#7660a4", lw=2, label="Taylor remainder formula")
    ax.set(xlabel="Positive twist size s (bp)", ylabel="Absolute net-value difference", title="C. Approximation error for the key-rate hedge")
    ax.legend(fontsize=9)
    ax = axes[1, 1]
    times = np.linspace(2, 10, 321)
    ax.plot(times, [bump(t) for t in times], color="#7660a4", lw=2)
    ax.scatter(curve.knots, [0]*3, s=110, facecolors="none", edgecolors="#333333", label="Curve knots", zorder=5)
    ax.scatter(instrument_times, [0]*4, marker="x", color=colors["keyrate"], s=45, label="Asset maturities", zorder=6)
    ax.scatter([6], [100], marker="D", color=colors["parallel"], s=50, label="Liability maturity")
    ax.set(xlabel="Cashflow time (years)", ylabel="Zero-rate change (bp)", title="D. A shock outside the nodal model", ylim=(-12, 125))
    ax.legend(fontsize=9, loc="upper left")
    fig.savefig(output / "hedging_audit.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"liability_present_value": summary["liability"]["present_value"],
                      "gross_pv_multiples": {k: v["gross_pv_multiple"] for k, v in hedges.items()},
                      "twist_changes": {k: v["twist"]["full_repricing_change"] for k, v in results.items()},
                      "off_grid_up_change": off_grid["keyrate"]["up"]["full_repricing_change"],
                      "max_decimal_change_difference": max(reference_differences)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    run(parser.parse_args().output)
