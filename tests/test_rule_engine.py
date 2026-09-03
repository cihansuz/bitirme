import pytest
from src.rule_engine.safety_engine import RuleEngine


def test_rule_engine_kidney_and_drugs():
    engine = RuleEngine()

    biomarkers = [
        {"name": "Glukoz", "normalized_name": "fasting_glucose", "value": 140.0},
        {"name": "HbA1c", "normalized_name": "hba1c", "value": 7.2},
        {"name": "eGFR", "normalized_name": "egfr", "value": 25.0} # Ağır böbrek yetmezliği
    ]
    medications = ["Metformin 1000mg", "Levotiroksin 50mcg"]
    allergies = ["Gluten"]

    eval_res = engine.evaluate(biomarkers, medications, allergies)

    assert "t2dm" in eval_res["triggered_diseases"]
    assert eval_res["active_constraints"]["carbohydrate_ratio_max"] <= 0.45
    assert eval_res["active_constraints"]["protein_ratio_max"] <= 0.12 # Böbrek kısıtı

    all_contra_text = " ".join(eval_res["contraindications"])
    assert "eGFR" in all_contra_text
    assert "B12" in all_contra_text # Metformin
    assert "Levotiroksin" in all_contra_text # Tiroid
    assert "Gluten" in all_contra_text # Alerji
