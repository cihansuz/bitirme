from typing import Dict, List, Any

try:
    from modules.base import BaseDiseaseModule
    from modules.t2dm.evaluator import T2DMEvaluator
    from modules.anemia.evaluator import AnemiaEvaluator
except (ImportError, ModuleNotFoundError):
    from .base import BaseDiseaseModule
    from .t2dm.evaluator import T2DMEvaluator
    from .anemia.evaluator import AnemiaEvaluator


class DiseaseRegistry:
    """
    Modüler Hastalık Kayıt ve Çakışma Çözümleme Motoru (Conflict Resolution Engine)
    """

    def __init__(self, load_defaults: bool = True):
        self.modules: Dict[str, BaseDiseaseModule] = {}
        if load_defaults:
            self.register("t2dm", T2DMEvaluator())
            self.register("anemia", AnemiaEvaluator())

    def register(self, disease_id: str, module: BaseDiseaseModule):
        self.modules[disease_id] = module

    def get_module(self, disease_id: str) -> BaseDiseaseModule:
        return self.modules.get(disease_id)

    def evaluate_patient(self, lab_data: Any) -> List[BaseDiseaseModule]:
        """Hastanın tahlillerine göre tetiklenen tüm hastalık modüllerini bulur."""
        triggered = []
        for mod in self.modules.values():
            if mod.evaluate_triggers(lab_data):
                triggered.append(mod)
        return triggered

    def resolve_conflicts(self, constraints_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Çoklu modül tetiklendiğinde kısıtları birleştirir (Conflict Resolution Engine).
        - Üst sınırlar ('max', 'ratio_max' vb.): En Düşük Üst Limit (Min of Max) seçilir.
        - Alt sınırlar ('min', 'minimum' vb.): En Yüksek Alt Limit (Max of Min) seçilir.
        """
        resolved = {}
        for c_dict in constraints_list:
            if not isinstance(c_dict, dict):
                continue
            for key, val in c_dict.items():
                if val is None:
                    continue
                
                # Sayısal olmayan değerler
                if not isinstance(val, (int, float)):
                    if key not in resolved:
                        resolved[key] = val
                    continue

                if key not in resolved:
                    resolved[key] = val
                else:
                    curr = resolved[key]
                    # Üst sınır anahtarları: max_*, *_max, limit_max
                    if "max" in key.lower() or "upper" in key.lower():
                        resolved[key] = min(curr, val)
                    # Alt sınır anahtarları: min_*, *_min, minimum_*
                    elif "min" in key.lower() or "lower" in key.lower():
                        resolved[key] = max(curr, val)
                    else:
                        # Varsayılan olarak en güvenli / kısıtlayıcı yaklaşım
                        resolved[key] = min(curr, val)

        return resolved
