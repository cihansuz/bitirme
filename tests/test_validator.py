import pytest
from src.validator.citation_verifier import CitationAndSafetyValidator
from src.validator.schemas import RetrievedEvidence


def test_citation_verifier_removes_hallucinations():
    validator = CitationAndSafetyValidator()

    evidence_list = [
        RetrievedEvidence(chunk_id="ADA_2024_P14", title="Carbs", text="Carbohydrate intake quality..."),
        RetrievedEvidence(chunk_id="ADA_2024_P15", title="Fiber", text="Fiber targets...")
    ]

    mock_json = """
    {
        "clinical_summary": "Hasta açlık kan şekeri ve HbA1c yüksek.",
        "findings": [
            {
                "biomarker": "HbA1c",
                "interpretation": "Diyabetik aralıkta.",
                "citation_ids": ["ADA_2024_P14", "HALLUCINATED_CHUNK_999"]
            }
        ],
        "dietary_guidelines_suggested": [
            {
                "parameter": "Lif",
                "target": ">= 30g",
                "rationale": "Glisemik kontrol",
                "citation_ids": ["FAKE_GUIDELINE_XYZ"]
            }
        ],
        "contraindications_flagged": []
    }
    """

    output = validator.validate_and_repair(
        raw_json_str=mock_json,
        evidence_list=evidence_list,
        rule_contraindications=[]
    )

    # 1. İlk bulgudaki sahte atıf silinmiş olmalı, geçerli olan kalmalı
    assert "ADA_2024_P14" in output.findings[0].citation_ids
    assert "HALLUCINATED_CHUNK_999" not in output.findings[0].citation_ids

    # 2. İkinci önerideki tek sahte atıf silinip context'teki ilk geçerli chunk atanmalı
    assert "FAKE_GUIDELINE_XYZ" not in output.dietary_guidelines_suggested[0].citation_ids
    assert len(output.dietary_guidelines_suggested[0].citation_ids) > 0


def test_safety_cross_check():
    validator = CitationAndSafetyValidator()

    unsafe_json = """
    {
        "clinical_summary": "Diyet listesine serbestçe sofra şekeri ve rafine şeker ilave edilebilir.",
        "findings": [],
        "dietary_guidelines_suggested": [],
        "contraindications_flagged": []
    }
    """

    output = validator.validate_and_repair(
        raw_json_str=unsafe_json,
        evidence_list=[],
        rule_contraindications=["Rafine şeker yasak"]
    )

    all_contra = " ".join(output.contraindications_flagged)
    assert "GÜVENLİK İHLALİ UYARISI" in all_contra or "Rafine şeker" in all_contra
