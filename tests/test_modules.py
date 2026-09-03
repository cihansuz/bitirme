import pytest
from modules.t2dm.evaluator import T2DMEvaluator
from modules.anemia.evaluator import AnemiaEvaluator
from modules.registry import DiseaseRegistry


def test_t2dm_triggers():
    evaluator = T2DMEvaluator()
    
    # Triggered by HbA1c
    assert evaluator.evaluate_triggers({"HbA1c": 6.8, "glucose": 95}) is True

    # Triggered by Fasting Glucose
    assert evaluator.evaluate_triggers({"hba1c": 5.4, "fasting_glucose": 130}) is True

    # Not triggered
    assert evaluator.evaluate_triggers({"HbA1c": 5.2, "fasting_glucose": 88}) is False


def test_anemia_triggers():
    evaluator = AnemiaEvaluator()

    # Triggered by Ferritin
    assert evaluator.evaluate_triggers({"Ferritin": 14.0, "hemoglobin": 13.5}) is True

    # Triggered by Hemoglobin
    assert evaluator.evaluate_triggers({"ferritin": 45.0, "hemoglobin": 11.2}) is True

    # Not triggered
    assert evaluator.evaluate_triggers({"ferritin": 65.0, "hemoglobin": 14.0}) is False


def test_conflict_resolution_engine():
    registry = DiseaseRegistry(load_defaults=True)

    # İki farklı modülden gelen kısıtlar
    constraints_t2dm = {
        "carbohydrate_ratio_max": 0.45,
        "added_sugar_max_g": 0.0,
        "minimum_fiber_g": 30.0,
        "sodium_mg_max": 2300.0
    }
    constraints_hypertension_or_anemia = {
        "carbohydrate_ratio_max": 0.50,
        "minimum_fiber_g": 25.0,
        "sodium_mg_max": 1500.0, # Daha kısıtlayıcı üst sınır
        "min_iron_mg": 18.0
    }

    resolved = registry.resolve_conflicts([constraints_t2dm, constraints_hypertension_or_anemia])

    # Min of Max: En düşük üst sınır seçilmeli
    assert resolved["carbohydrate_ratio_max"] == 0.45
    assert resolved["added_sugar_max_g"] == 0.0
    assert resolved["sodium_mg_max"] == 1500.0

    # Max of Min: En yüksek alt sınır seçilmeli
    assert resolved["minimum_fiber_g"] == 30.0
    assert resolved["min_iron_mg"] == 18.0
