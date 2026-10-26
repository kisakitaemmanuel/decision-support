"""
rules/referral.py - decide manage_here / refer_routine / refer_urgent.

Inputs: red-flag hits, the disease's levels_of_care string from the guideline ("HC2, HC4"), facility level.
The guideline's LOC column tells the *lowest level at which each treatment may be given*; we use it to say
which parts of the management need a higher level than this facility.
"""
from __future__ import annotations

import re

from schemas import ReferralAdvice, RuleHit


def level_ranks(con) -> dict[str, int]:
    return {r["code"]: r["rank"] for r in con.execute("SELECT code, rank FROM referral_levels")}


def advise(con, facility_level: str, red_flags: list[RuleHit], levels_of_care: str | None) -> ReferralAdvice:
    ranks = level_ranks(con)
    here = ranks.get(facility_level, ranks["HC3"])
    reasons: list[str] = []
    decision = "manage_here"

    for h in red_flags:
        if h.refer:
            reasons.append(f"{h.message} [{h.rule_id}]")
            decision = "refer_urgent"

    codes = [c for c in re.split(r"[,\s]+", levels_of_care or "") if c in ranks]
    higher = sorted({c for c in codes if ranks[c] > here}, key=lambda c: ranks[c])
    if codes and min(ranks[c] for c in codes) > here:
        reasons.append(f"Guideline treatment starts at {min(codes, key=lambda c: ranks[c])}, above this facility ({facility_level}).")
        if decision == "manage_here":
            decision = "refer_routine"
    elif higher:
        reasons.append("Some management steps are only listed for higher levels: " + ", ".join(higher) + ".")
    return ReferralAdvice(decision, reasons, facility_level, codes, higher)
