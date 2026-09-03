import json
import urllib.request
import urllib.error
import re
from typing import Dict, Any, List, Optional
from src.rag.hybrid_retriever import HybridRetriever


class QwenAgent:
    """
    Yerel Ollama (qwen3:8b) üzerinde çalışan Kanıta Dayalı Soru-Cevap Ajanı.
    - Kullanıcının sorusunu alır.
    - HybridRetriever ile makalelerden en alakalı pasajları çeker.
    - Modele bağlam olarak verir ve kaynak atıflı (citation) Türkçe yanıt üretir.
    """

    SYSTEM_PROMPT = """Sen klinik diyetisyenlik ve tıp literatürü konusunda uzman bir RAG asistanısın.

GÖREVİN:
1. Kullanıcının sorusunu SADECE ve SADECE sana verilen 'KLİNİK BAĞLAM' içindeki bilgilere dayanarak cevapla.
2. Bilmediğin veya bağlamda geçmeyen hiçbir bilgiyi uydurma (sıfır halüsinasyon).
3. Verdiğin her önemli klinik bilginin yanına mutlaka ilgili kaynak kimliğini [Kaynak: CHUNK_ID] şeklinde ekle.
4. Yanıtını anlaşılır, profesyonel ve maddeler halinde Türkçe olarak açıkla.
"""

    def __init__(self, model_name: str = "qwen3:8b", ollama_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.retriever = HybridRetriever()

    def ask(self, question: str, top_k: int = 4) -> Dict[str, Any]:
        """
        Soruya makaleler üzerinden cevap üretir.
        """
        # 1. Makalelerden İlgili Pasajları Getir (RAG)
        retrieved_chunks = self.retriever.retrieve(query=question, top_k=top_k)
        
        if not retrieved_chunks:
            return {
                "question": question,
                "answer": "Yüklenen makalelerde bu konuyla ilgili yeterli bilgi bulunamadı.",
                "citations": [],
                "sources": []
            }

        # 2. Bağlam Metnini Oluştur
        context_parts = []
        for c in retrieved_chunks:
            context_parts.append(f"--- [KAYNAK: {c['chunk_id']}] ({c['title']}) ---\n{c['text']}")
        context_text = "\n\n".join(context_parts)

        # 3. Modele Gönderilecek Kullanıcı İstemi
        user_prompt = f"""KLİNİK BAĞLAM:
{context_text}

KULLANICININ SORUSU:
{question}

Lütfen yukarıdaki klinik bağlama sadık kalarak, kaynak atıflı ([Kaynak: CHUNK_ID]) yanıt ver:"""

        # 4. Ollama'ya İstek Gönder
        payload = {
            "model": self.model_name,
            "system": self.SYSTEM_PROMPT,
            "prompt": user_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_ctx": 8192
            }
        }

        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self.ollama_url}/api/generate",
                data=req_data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=120) as response:
                res_body = json.loads(response.read().decode("utf-8"))
                raw_answer = res_body.get("response", "").strip()
        except Exception as e:
            raw_answer = f"Ollama çıkarım hatası: {e}"

        # 5. Metin İçindeki Atıfları Tara
        valid_cids = {c["chunk_id"] for c in retrieved_chunks}
        found_citations = [cid for cid in valid_cids if cid in raw_answer]

        return {
            "question": question,
            "answer": raw_answer,
            "citations": found_citations,
            "sources": retrieved_chunks
        }
