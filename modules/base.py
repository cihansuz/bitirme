from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseDiseaseModule(ABC):
    """
    Klinik Karar Destek Sistemi Modüler Hastalık Eklenti Arayüzü (BaseDiseaseModule)
    """

    @property
    @abstractmethod
    def disease_id(self) -> str:
        """Hastalık tekil kimliği (örn: 't2dm', 'anemia')"""
        pass

    @property
    @abstractmethod
    def disease_name(self) -> str:
        """Hastalık klinik adı (örn: 'Tip 2 Diabetes Mellitus')"""
        pass

    @abstractmethod
    def evaluate_triggers(self, lab_data: Dict[str, Any]) -> bool:
        """Tahlillerde hastalığın klinik bayraklarının tetiklenip tetiklenmediğini belirler."""
        pass

    @abstractmethod
    def get_safety_constraints(self) -> List[Dict[str, Any]]:
        """Besin ve tarif planlayıcısına iletilecek kesin sınırları döndürür."""
        pass

    @abstractmethod
    def get_rag_filter(self) -> Dict[str, Any]:
        """Vektör veritabanı sorgusuna eklenecek metadata filtresini sağlar."""
        pass

    @abstractmethod
    def get_contraindications(self) -> List[str]:
        """Kesinlikle izin verilmeyen gıda bileşenleri ve ilaç etkileşim listesi."""
        pass
