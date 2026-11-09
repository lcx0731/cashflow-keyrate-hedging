"""Synthetic deterministic cashflow analysis. Educational; not investment advice."""

from .core import Cashflows, ZeroCurve, match_risk, net_cashflows, risk, stress

__all__ = ["Cashflows", "ZeroCurve", "match_risk", "net_cashflows", "risk", "stress"]
