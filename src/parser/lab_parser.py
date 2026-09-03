"""
Laboratuvar ve Biyobelirteç Ayrıştırıcısı (Grup Arkadaşı Modülü)
Bu modül kan tahlili birim dönüşümleri ve referans aralığı kontrolleri ile görevli ekip üyesi için ayrılmıştır.
"""
from typing import Dict, Any


class LabParser:
    def process_biomarker(
        self, 
        name: str, 
        value: float, 
        unit: str, 
        reference_range: str = ""
    ) -> Dict[str, Any]:
        return {
            "name": name,
            "value": value,
            "unit": unit,
            "reference_range": reference_range,
            "status": "NORMAL"
        }
