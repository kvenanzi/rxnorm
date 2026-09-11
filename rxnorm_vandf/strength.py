"""Deterministic strength arithmetic the embedding model can't do on its own.

RxNorm writes concentrations per 1 mL (or per 1 mg for gels and ointments);
the VA writes them as they appear on the label: "0.125MG/5ML", "0.25%".
`normalize_strength` appends the RxNorm-style value after the original string:

    ACETIC ACID 0.25% IRRG SOLN            -> ACETIC ACID 0.25% IRRG SOLN | 2.5 mg/ml
    HYOSCYAMINE SO4 0.125MG/5ML ELIXIR     -> HYOSCYAMINE SO4 0.125MG/5ML ELIXIR | 0.025 mg/ml
    POVIDONE 0.5% OPH GEL                  -> POVIDONE 0.5% OPH GEL | 0.005 mg/mg

The original text is kept, so everything the model learned from it still applies.
Strings already in RxNorm shape ("50MCG/ML", "25MG") are left alone.
"""

import re

_NUM = r"(\d+(?:\.\d+)?)"
# "0.125MG/5ML", "2.5 MG/3 ML", "16,800UNIT/5ML" (commas stripped first)
_PER_VOLUME = re.compile(_NUM + r"\s*(MG|MCG|GM|G|UNITS?|UNT|MEQ|MMOL)/" + _NUM + r"\s*ML\b", re.I)
# "0.25%", "10-15%"
_PERCENT = re.compile(_NUM + r"(?:-" + _NUM + r")?\s*%")
# Gels and ointments are per mg of product in RxNorm; everything else per mL.
_PER_MG_FORMS = re.compile(r"\b(OINT|GEL)\b", re.I)

_TO_MG = {"MG": 1.0, "MCG": 0.001, "GM": 1000.0, "G": 1000.0}


def _fmt(v: float) -> str:
    s = f"{v:.4g}"
    return f"{v:.0f}" if "e" in s else s


def _unit(u: str) -> tuple[float, str]:
    u = u.upper()
    if u in _TO_MG:
        return _TO_MG[u], "mg"
    if u in ("UNIT", "UNITS", "UNT"):
        return 1.0, "unt"
    return 1.0, u.lower()


def canonical_strengths(s: str) -> list[str]:
    text = s.replace(",", "")
    out = []
    for value, unit, volume in _PER_VOLUME.findall(text):
        scale, name = _unit(unit)
        out.append(f"{_fmt(float(value) * scale / float(volume))} {name}/ml")
    per_mg = bool(_PER_MG_FORMS.search(text))
    for lo, hi in _PERCENT.findall(text):
        for v in (lo, hi):
            if v:
                pct = float(v)
                out.append(f"{_fmt(pct / 100)} mg/mg" if per_mg else f"{_fmt(pct * 10)} mg/ml")
    return out


def normalize_strength(s: str) -> str:
    parts = canonical_strengths(s)
    return f"{s} | {' / '.join(parts)}" if parts else s
