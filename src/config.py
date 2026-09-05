"""
Merkezi Konfigürasyon Yükleyici
configs/settings.yaml dosyasını okur ve sistem bileşenlerine sunar.
"""
import os
import yaml
from typing import Dict, Any

CONFIG_FILE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "configs", "settings.yaml")
)


def load_config(config_path: str = CONFIG_FILE_PATH) -> Dict[str, Any]:
    """settings.yaml dosyasını yükler."""
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[Uyarı] settings.yaml okunamadı ({e}), varsayılanlar kullanılacak.")
    return {}


def get_llm_config() -> Dict[str, Any]:
    """LLM ayarlarını döndürür."""
    cfg = load_config()
    return cfg.get("llm", {})


def get_rag_config() -> Dict[str, Any]:
    """RAG ayarlarını döndürür."""
    cfg = load_config()
    return cfg.get("rag", {})
