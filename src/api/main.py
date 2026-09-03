"""
Klinik Makaleler RAG Soru-Cevap API Servisi
"""
from fastapi import FastAPI, Query
from src.llm.qwen_agent import QwenAgent

app = FastAPI(
    title="Klinik Makaleler RAG Soru-Cevap Servisi",
    description="Yüklenen makaleler ve klinik rehberler üzerinden kanıta dayalı soru-cevap",
    version="1.0.0"
)

agent = QwenAgent()


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Clinical Articles RAG Q&A",
        "model": agent.model_name,
        "indexed_chunks": len(agent.retriever.chunks)
    }


@app.get("/ask")
def ask_endpoint(q: str = Query(..., description="Makalelere sorulacak soru")):
    return agent.ask(q)
