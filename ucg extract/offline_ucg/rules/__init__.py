"""rules package: seed_all(con) loads every seeded rule table (called by the ETL), signoff.json marks reviewed rules."""
from __future__ import annotations

import json
from pathlib import Path

from rules import contraindications, dosing, red_flags

SIGNOFF = Path(__file__).with_name("signoff.json")


def load_signoff() -> dict:
    return json.loads(SIGNOFF.read_text(encoding="utf-8")) if SIGNOFF.exists() else {}


def seed_all(con) -> dict[str, int]:
    so = load_signoff()
    counts = {"red_flag_rules": red_flags.seed(con, so),
              "contraindication_rules": contraindications.seed(con, so),
              "dose_rules": dosing.seed(con, so)}
    con.commit()
    return counts
