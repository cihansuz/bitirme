import pytest
from src.parser.lab_parser import LabParser


def test_unit_conversion():
    parser = LabParser()
    # Fasting glucose mmol/L -> mg/dL (7.0 mmol/L = 126.1 mg/dL)
    val, unit = parser.convert_unit("fasting_glucose", 7.0, "mmol/L")
    assert unit == "mg/dL"
    assert val >= 126.0

    # HbA1c IFCC mmol/mol -> NGSP % (48 mmol/mol = 6.5%)
    val_a1c, unit_a1c = parser.convert_unit("hba1c", 48.0, "mmol/mol")
    assert unit_a1c == "%"
    assert round(val_a1c, 1) == 6.5


def test_reference_range_and_status():
    parser = LabParser()
    
    # Normal Glucose
    res_normal = parser.process_biomarker("Glukoz", 85.0, "mg/dL", "70-100")
    assert res_normal["status"] == "NORMAL"

    # High Glucose (T2DM eşiği)
    res_high = parser.process_biomarker("Açlık Kan Şekeri", 134.0, "mg/dL", "70-100")
    assert res_high["status"] == "HIGH"

    # Critical High Glucose (> 300)
    res_crit_high = parser.process_biomarker("Glucose", 320.0, "mg/dL", "70-100")
    assert res_crit_high["status"] == "CRITICAL"

    # Critical Low Glucose (< 54)
    res_crit_low = parser.process_biomarker("Glucose", 48.0, "mg/dL", "70-100")
    assert res_crit_low["status"] == "CRITICAL"

    # Ferritin Low (Anemi)
    res_ferritin = parser.process_biomarker("Ferritin", 12.0, "ng/mL", "30-200")
    assert res_ferritin["status"] == "LOW"

    # eGFR Critical (< 30 mL/min)
    res_egfr = parser.process_biomarker("eGFR", 24.0, "mL/min/1.73m2", "> 60")
    assert res_egfr["status"] == "CRITICAL"
