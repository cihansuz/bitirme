import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_api_root_and_health():
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert res_root.json()["status"] == "online"
    assert "t2dm" in res_root.json()["registered_modules"]

    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"
    assert res_health.json()["chromadb_chunks_count"] > 0


def test_api_modules_and_guidelines():
    res_mod = client.get("/modules")
    assert res_mod.status_code == 200
    mods = res_mod.json()["modules"]
    mod_ids = [m["disease_id"] for m in mods]
    assert "t2dm" in mod_ids
    assert "anemia" in mod_ids

    res_guide = client.get("/guidelines?query=diabetes")
    assert res_guide.status_code == 200
    assert res_guide.json()["count"] > 0


def test_api_end_to_end_analyze_and_review():
    patient_payload = {
        "patient_id": "P-TEST-9001",
        "age": 52,
        "gender": "female",
        "biomarkers": [
            {"name": "Açlık Kan Şekeri", "value": 142.0, "unit": "mg/dL", "reference_range": "70-100"},
            {"name": "HbA1c", "value": 7.3, "unit": "%", "reference_range": "4.0-5.6"},
            {"name": "Ferritin", "value": 11.5, "unit": "ng/mL", "reference_range": "30-200"}
        ],
        "medications": ["Metformin 1000mg"],
        "allergies": ["Fındık"],
        "target_calories": 1850.0
    }

    # 1. /analyze çağrısı
    res_analyze = client.post("/analyze", json=patient_payload)
    assert res_analyze.status_code == 200
    data = res_analyze.json()
    assert data["patient_id"] == "P-TEST-9001"
    assert data["status"] == "PENDING_DIETITIAN_REVIEW"
    assert len(data["findings"]) >= 0
    assert data["macro_plan"] is not None
    assert len(data["contraindications_flagged"]) > 0

    # 2. /review çağrısı (Diyetisyen onayı)
    review_payload = {
        "patient_id": "P-TEST-9001",
        "approved": True,
        "dietitian_notes": "Klinik değerlendirme doğrulandı, 3 ay sonra HbA1c ve ferritin kontrolü planlandı."
    }
    res_review = client.post("/review", json=review_payload)
    assert res_review.status_code == 200
    rev_data = res_review.json()
    assert rev_data["status"] == "APPROVED"
    assert "Diyetisyen Klinik Notu" in rev_data["clinical_summary"]
