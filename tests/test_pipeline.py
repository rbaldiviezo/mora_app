"""Pipeline tests (no web server needed): valid case, hard case, invalid inputs, full flow."""
from pathlib import Path

import pytest

from app import pipeline

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "registro_clases_asistencia.csv"


@pytest.fixture(scope="module")
def bundle():
    return pipeline.train(CSV_PATH.read_text(encoding="utf-8-sig"))


def test_valid_case(bundle):
    result = pipeline.predict(bundle, "Marco Antonio", "15/05/2026")
    assert 0 <= result["probabilities"]["late_payment"] <= 1
    assert 0 <= result["probabilities"]["absence"] <= 1
    assert result["warnings"] == []
    assert result["cutoff_used"] == "2026-05-15"


def test_hard_case_short_history(bundle):
    result = pipeline.predict(bundle, "Lucas Garrón", "26/02/2026")
    assert any("poco confiable" in w for w in result["warnings"])


def test_predictions_are_computed_not_fixed(bundle):
    a = pipeline.predict(bundle, "Marco Antonio", "15/05/2026")["probabilities"]
    b = pipeline.predict(bundle, "Hugo", "20/05/2026")["probabilities"]
    assert a != b


@pytest.mark.parametrize("student,date,status", [
    ("Pedro", "15/05/2026", 404),
    ("", "15/05/2026", 422),
    ("Marco Antonio", "31/02/2026", 422),
    ("Marco Antonio", "hola", 422),
])
def test_invalid_inputs(bundle, student, date, status):
    with pytest.raises(pipeline.ApiError) as error:
        pipeline.predict(bundle, student, date)
    assert error.value.status_code == status


def test_typo_suggestion(bundle):
    with pytest.raises(pipeline.ApiError) as error:
        pipeline.predict(bundle, "Marcos Antonio", "15/05/2026")
    assert "Marco Antonio" in error.value.suggestions


def test_bad_csv_is_rejected():
    with pytest.raises(pipeline.ApiError) as error:
        pipeline.train("a,b\n1,2\n")
    assert error.value.status_code == 400


def test_full_flow_matches_notebook(bundle):
    """CSV -> training -> metrics -> prediction. Metrics must match notebook v4."""
    by_event = {m["event"]: m for m in bundle.metrics}
    assert by_event["late_payment"]["ap_model"] == pytest.approx(0.514, abs=0.001)
    assert by_event["absence"]["ap_model"] == pytest.approx(0.166, abs=0.001)
    result = pipeline.predict(bundle, "marco antonio", "2026-05-15")
    assert result["probabilities"]["late_payment"] == pytest.approx(0.201, abs=0.001)
