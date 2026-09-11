import pytest

from rxnorm_vandf.strength import normalize_strength


@pytest.mark.parametrize("va, suffix", [
    # per-volume -> per mL
    ("HYOSCYAMINE SO4 0.125MG/5ML ELIXIR", "0.025 mg/ml"),
    ("ACYCLOVIR 200MG/5ML SUSP,ORAL", "40 mg/ml"),
    ("MEPERIDINE HCL 50MG/5ML SYRUP", "10 mg/ml"),
    ("ALBUTEROL 2.5MG/3ML INHL SOLN", "0.8333 mg/ml"),
    ("FENTANYL CITRATE 250MCG/5ML INJ", "0.05 mg/ml"),          # mcg -> mg
    ("CEFAZOLIN 1GM/50ML INJ,BAG", "20 mg/ml"),                 # gm -> mg
    ("HEPARIN NA 5,000UNIT/0.5ML INJ SYR", "10000 unt/ml"),     # comma, unit
    # percent -> per mL, or per mg for gels/ointments
    ("ACETIC ACID 0.25% IRRG SOLN", "2.5 mg/ml"),
    ("BENZOYL PEROXIDE 10% WASH", "100 mg/ml"),
    ("POVIDONE 0.5% OPH GEL", "0.005 mg/mg"),
    ("SULFACETAMIDE NA 10% OPH OINT", "0.1 mg/mg"),
    ("MENTHOL/M-SALICYLATE 10-15% TOP OINT", "0.1 mg/mg / 0.15 mg/mg"),
    ("HYDROCORT 2/LIDOCAINE 2% RTL CREAM W/APP", "20 mg/ml"),
])
def test_appends_canonical_strength(va, suffix):
    assert normalize_strength(va) == f"{va} | {suffix}"


@pytest.mark.parametrize("va", [
    "METOPROLOL TARTRATE 12.5MG TAB",
    "FENTANYL CITRATE 50MCG/ML INJ",          # already per mL
    "ESTRADIOL 0.0375MG/DAY (EQV-VIVELLE-DOT)",
    "CATHETER,FOLEY SILICONE 22FR 5CC",
])
def test_leaves_rxnorm_shaped_strings_alone(va):
    assert normalize_strength(va) == va
