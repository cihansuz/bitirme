import os
import yaml
from typing import List, Dict, Any
try:
    from modules.base import BaseDiseaseModule
except (ImportError, ModuleNotFoundError):
    from ..base import BaseDiseaseModule


class AnemiaEvaluator(BaseDiseaseModule):
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "rules.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    @property
    def disease_id(self) -> str:
        return self.config.get("disease_id", "anemia")

    @property
    def disease_name(self) -> str:
        return self.config.get("name", "Demir Eksikliği Anemisi & Demir Yetersizliği")

    def _normalize_name(self, name: str) -> str:
        clean = name.strip().lower().replace(" ", "_").replace("-", "_")
        aliases = {
            "ferritin": "ferritin",
            "serum_ferritin": "ferritin",
            "hemoglobin": "hemoglobin",
            "hb": "hemoglobin",
            "hgb": "hemoglobin",
            "demir": "serum_iron",
            "iron": "serum_iron",
            "serum_iron": "serum_iron",
            "fe": "serum_iron"
        }
        return aliases.get(clean, clean)

    def evaluate_triggers(self, lab_data: Dict[str, Any]) -> bool:
        normalized_labs = {}
        if isinstance(lab_data, dict):
            for k, v in lab_data.items():
                if isinstance(v, dict) and "value" in v:
                    normalized_labs[self._normalize_name(k)] = float(v["value"])
                elif isinstance(v, (int, float)):
                    normalized_labs[self._normalize_name(k)] = float(v)
        elif isinstance(lab_data, list):
            for item in lab_data:
                if isinstance(item, dict) and "name" in item and "value" in item:
                    normalized_labs[self._normalize_name(item["name"])] = float(item["value"])

        triggers = self.config.get("triggers", [])
        for trig in triggers:
            bio = self._normalize_name(trig["biomarker"])
            op = trig.get("operator", "<=")
            thresh = float(trig.get("threshold", 0))

            if bio in normalized_labs:
                val = normalized_labs[bio]
                if op == "<=" and val <= thresh:
                    return True
                elif op == "<" and val < thresh:
                    return True
                elif op == ">=" and val >= thresh:
                    return True
                elif op == ">" and val > thresh:
                    return True

        return False

    def get_safety_constraints(self) -> List[Dict[str, Any]]:
        return [self.config.get("constraints", {})]

    def get_rag_filter(self) -> Dict[str, Any]:
        return self.config.get("rag_metadata_filter", {})

    def get_contraindications(self) -> List[str]:
        return [
            "Demir emilimini inhibe eden polifenol ve tanen içerikli içecekler (çay, kahve, kakao) demir içeren ana öğünlerle eş zamanlı tüketilmemeli, en az 1.5 - 2 saat aralık bırakılmalıdır.",
            "Kalsiyum (süt, peynir, kalsiyum takviyeleri) non-hem demir emilimiyle yarışacağından, demir zengini öğünler veya demir takviyeleriyle birlikte tüketilmemelidir."
        ]
