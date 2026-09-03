import json
import re
from typing import Dict, Any, List, Tuple

try:
    from src.validator.schemas import DietitianDecisionSupportOutput, RetrievedEvidence
except (ImportError, ModuleNotFoundError):
    from .schemas import DietitianDecisionSupportOutput, RetrievedEvidence


class CitationAndSafetyValidator:
    """
    Şema, Kaynak ve Güvenlik Doğrulayıcı (Post-Processing Validator).
    1. Pydantic şema doğrulaması
    2. Citation Verifier: Halüsinatif chunk_id'leri filtreleme
    3. Güvenlik Çapraz Kontrolü: Yasaklı besin / kontrendikasyon taraması
    """

    FORBIDDEN_PATTERNS = [
        (r"(rafine\s*şeker|sofra\s*şekeri|ilave\s*şeker|gazlı\s*içecek)", "Rafine ve ilave şeker tüketimi Tip 2 Diyabette kesinlikle kontrendikedir!"),
        (r"(ketojenik\s*diyet|keto\s*diyet|< ?50\s*g\s*karbonhidrat)", "SGLT2 inhibitörü kullanan hastalarda ketojenik diyet öglisemik DKA riski nedeniyle yasaktır!"),
        (r"(demir.*(çay|kahve|süt)|(çay|kahve|süt).*demir)", "Demir preparatları veya demir zengini öğünler çay/kahve/süt ile eşzamanlı tüketilmemelidir!"),
    ]

    def verify_citations(
        self, 
        output_data: Dict[str, Any], 
        evidence_list: List[RetrievedEvidence]
    ) -> Tuple[Dict[str, Any], bool, List[str]]:
        """
        Modelin atıf yaptığı chunk ID'lerinin gerçekten context içerisinde yer alıp almadığını doğrular.
        """
        valid_chunk_ids = {e.chunk_id for e in evidence_list}
        all_valid = True
        citation_warnings = []

        # 1. Findings atıf kontrolü
        if "findings" in output_data and isinstance(output_data["findings"], list):
            for item in output_data["findings"]:
                cids = item.get("citation_ids", [])
                verified_cids = []
                for cid in cids:
                    if cid in valid_chunk_ids:
                        verified_cids.append(cid)
                    else:
                        all_valid = False
                        citation_warnings.append(f"Geçersiz/Halüsinatif atıf silindi: '{cid}' (bulgu: {item.get('biomarker')})")
                
                # Hiç geçerli atıf kalmadıysa ve context varsa, en alakalı ilk chunk'ı ekle
                if not verified_cids and evidence_list:
                    verified_cids = [evidence_list[0].chunk_id]
                item["citation_ids"] = verified_cids

        # 2. Dietary Guidelines atıf kontrolü
        if "dietary_guidelines_suggested" in output_data and isinstance(output_data["dietary_guidelines_suggested"], list):
            for item in output_data["dietary_guidelines_suggested"]:
                cids = item.get("citation_ids", [])
                verified_cids = []
                for cid in cids:
                    if cid in valid_chunk_ids:
                        verified_cids.append(cid)
                    else:
                        all_valid = False
                        citation_warnings.append(f"Geçersiz/Halüsinatif atıf silindi: '{cid}' (öneri: {item.get('parameter')})")

                if not verified_cids and evidence_list:
                    verified_cids = [evidence_list[0].chunk_id]
                item["citation_ids"] = verified_cids

        return output_data, all_valid, citation_warnings

    def safety_cross_check(
        self, 
        output_data: Dict[str, Any], 
        rule_contraindications: List[str]
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Model metninde kural motorunun yasakladığı bir önerinin yer alıp almadığını kontrol eder.
        """
        safety_alerts = []
        
        # Tüm model metnini birleştir
        full_text = output_data.get("clinical_summary", "")
        for f in output_data.get("findings", []):
            full_text += " " + f.get("interpretation", "")
        for d in output_data.get("dietary_guidelines_suggested", []):
            full_text += " " + d.get("parameter", "") + " " + d.get("rationale", "") + " " + d.get("target", "")

        lower_text = full_text.lower()

        # Regex kontrolleri
        for pattern, msg in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, lower_text, re.IGNORECASE):
                # Eğer metinde "tüketilmemelidir", "kaçınılmalıdır", "kısıtlanmalıdır" gibi olumsuzlama yoksa uyar
                if not any(neg in lower_text for neg in ["tüketilmemeli", "kaçınılmalı", "yasak", "kısıtlanmalı", "kontrendike", "önerilmez"]):
                    safety_alerts.append(f"GÜVENLİK İHLALİ UYARISI: {msg}")

        # Kural motorunun kontrendikasyonlarını da ekle
        current_contra = output_data.get("contraindications_flagged", [])
        combined_contra = list(dict.fromkeys(current_contra + rule_contraindications + safety_alerts))
        output_data["contraindications_flagged"] = combined_contra

        return output_data, safety_alerts

    def validate_and_repair(
        self, 
        raw_json_str: str, 
        evidence_list: List[RetrievedEvidence],
        rule_contraindications: List[str] = None
    ) -> DietitianDecisionSupportOutput:
        """
        Ham JSON metnini ayrıştırır, atıfları ve güvenlik kurallarını doğrular ve
        Pydantic nesnesi olarak döndürür.
        """
        rule_contraindications = rule_contraindications or []

        # 1. JSON ayrıştırma ve markdown temizliği
        clean_json = raw_json_str.strip()
        if "```json" in clean_json:
            clean_json = clean_json.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_json:
            clean_json = clean_json.split("```")[1].split("```")[0].strip()

        # JSON bloğu bul
        match = re.search(r"\{.*\}", clean_json, re.DOTALL)
        if match:
            clean_json = match.group(0)

        try:
            parsed_data = json.loads(clean_json)
        except Exception as e:
            # Fallback onarım: temel yapı oluştur
            parsed_data = {
                "clinical_summary": f"Model çıktısı JSON ayrıştırma hatası verdi: {str(e)[:100]}. Tahliller ve kısıtlar doğrultusunda diyetisyen incelemesi gereklidir.",
                "findings": [],
                "dietary_guidelines_suggested": [],
                "contraindications_flagged": rule_contraindications,
                "status": "PENDING_DIETITIAN_REVIEW"
            }

        # 2. Atıf doğrulama
        verified_data, citations_ok, cit_warnings = self.verify_citations(parsed_data, evidence_list)
        verified_data["citations_verified"] = citations_ok

        # 3. Güvenlik çapraz denetimi
        safe_data, safety_alerts = self.safety_cross_check(verified_data, rule_contraindications)

        # 4. Pydantic şema denetimi
        return DietitianDecisionSupportOutput(**safe_data)
