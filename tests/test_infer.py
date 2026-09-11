from pathlib import Path

import pytest

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "sapbert-ingredient-strength-final" / "best"

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "calibration.json").exists(), reason="final model not present locally")


@pytest.fixture(scope="module")
def mapper():
    from rxnorm_vandf.infer import Mapper
    return Mapper.from_pretrained(str(MODEL_DIR))


def test_maps_known_strings(mapper):
    preds = mapper.map(["METOPROLOL TARTRATE 12.5MG TAB", "DOXEPIN HCL 10MG CAP"])
    assert preds[0].name == "metoprolol tartrate 12.5 MG Oral Tablet"
    assert preds[1].name == "doxepin 10 MG Oral Capsule"
    assert all(p.accept for p in preds)
    assert all(0.0 <= p.confidence <= 1.0 for p in preds)
    assert len(preds[0].alternatives) == 4


def test_abstains_when_unsure_and_keeps_truth_in_alternatives(mapper):
    # The model ranks the 0.125 MG/ML product first (the raw number wins over
    # the appended 0.025 mg/ml), but its calibrated confidence is below the
    # threshold, so the string is routed to review with the right answer listed.
    (p,) = mapper.map(["HYOSCYAMINE SO4 0.125MG/5ML ELIXIR"])
    assert not p.accept
    assert any(name == "hyoscyamine sulfate 0.025 MG/ML Oral Solution" for _, name, _ in p.alternatives)


def test_abstains_on_supplies(mapper):
    (p,) = mapper.map(["CATHETER,FOLEY SILICONE 22FR 5CC"])
    assert not p.accept
    assert p.confidence < 0.2


def test_threshold_override(mapper):
    (p,) = mapper.map(["CATHETER,FOLEY SILICONE 22FR 5CC"], threshold=0.0)
    assert p.accept
