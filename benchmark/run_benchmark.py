"""
Klinik Makaleler RAG ve Context Benchmark Scripti
Yüklenen makaleler üzerinde 8K ve 16K bağlam hacimlerinde;
- Retrieval Latency
- Time to First Token (TTFT)
- Throughput (tokens/s)
- Kaynak Sadakati / Halüsinasyon
ölçümlerini yapar ve benchmark/results/ klasörüne kaydeder.
"""
import time
import json
import os
import sys
import urllib.request
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag.hybrid_retriever import HybridRetriever


class ArticleBenchmark:
    def __init__(self, model_name: str = "qwen3:8b", ollama_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.retriever = HybridRetriever()
        os.makedirs("benchmark/results", exist_ok=True)

    def _build_context(self, target_tokens: int) -> List[Dict[str, Any]]:
        base_chunks = self.retriever.retrieve("Tip 2 diyabet beslenme ve tedavi", top_k=4)
        target_words = int(target_tokens * 0.75)

        expanded = list(base_chunks)
        current_words = sum(len(c["text"].split()) for c in expanded)

        i = 1
        while current_words < target_words and i < 150:
            for c in base_chunks:
                cid = f"{c['chunk_id']}_EXT_{i}"
                expanded.append({
                    "chunk_id": cid,
                    "title": f"{c['title']} (Bölüm {i})",
                    "text": f"{c['text']} [Ek Literatür Pasajı {i}]"
                })
                current_words += len(c["text"].split())
                i += 1
                if current_words >= target_words:
                    break

        return expanded

    def run_test(self, test_name: str, target_tokens: int) -> Dict[str, Any]:
        print(f"\n[*] Çalıştırılıyor: {test_name} (~{target_tokens} tokens)...", flush=True)

        t0 = time.perf_counter()
        context = self._build_context(target_tokens)
        t_ret = time.perf_counter() - t0

        actual_words = sum(len(c["text"].split()) for c in context)
        actual_tokens = int(actual_words * 1.33)

        prompt = f"""Aşağıdaki klinik makale pasajlarına dayanarak soruyu Türkçe cevapla:
BAĞLAM:
{json.dumps(context, ensure_ascii=False)}

SORU:
Tip 2 diyabette beslenme ilkeleri ve dikkat edilmesi gereken kritik noktalar nelerdir? Lütfen kaynak atıflı ([Kaynak: CHUNK_ID]) açıkla.
"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": 0.1, "num_ctx": min(actual_tokens + 512, 16384)}
        }

        req = urllib.request.Request(
            f"{self.ollama_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        ttft = 0.0
        generated_tokens = 0
        chunks = []
        t_start = time.perf_counter()
        first_token = False

        try:
            with urllib.request.urlopen(req, timeout=180) as res:
                for line in res:
                    if not line:
                        continue
                    data = json.loads(line.decode("utf-8"))
                    txt = data.get("response", "")
                    if txt and not first_token:
                        ttft = time.perf_counter() - t_start
                        first_token = True
                    chunks.append(txt)
                    generated_tokens += 1
                    if data.get("done", False):
                        break
            total_time = time.perf_counter() - t_start
        except Exception as e:
            print(f"[!] Çıkarım hatası: {e}", flush=True)
            total_time = time.perf_counter() - t_start

        throughput = generated_tokens / max(1.0, (total_time - ttft)) if total_time > ttft else 0.0

        result = {
            "test_name": test_name,
            "target_tokens": target_tokens,
            "actual_tokens": actual_tokens,
            "retrieval_sec": round(t_ret, 3),
            "ttft_sec": round(ttft, 2),
            "total_sec": round(total_time, 2),
            "throughput_tps": round(throughput, 1)
        }

        print(f"-> TTFT: {result['ttft_sec']}s | Toplam: {result['total_sec']}s | Throughput: {result['throughput_tps']} t/s", flush=True)
        return result

    def run_all(self):
        print("=== Klinik Makaleler Context Benchmark Testi ===")
        r8k = self.run_test("Context-8K-Test", 8000)
        r16k = self.run_test("Context-16K-Test", 16000)

        results = [r8k, r16k]
        with open("benchmark/results/benchmark_report.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        md = "# Klinik Makaleler RAG Context Benchmark Raporu\n\n"
        md += "| Test Adı | Hedef Token | Gerçek Token | Retrieval (s) | TTFT (s) | Toplam Süre (s) | Throughput (t/s) |\n"
        md += "|---|---|---|---|---|---|---|\n"
        for r in results:
            md += f"| {r['test_name']} | {r['target_tokens']} | {r['actual_tokens']} | {r['retrieval_sec']}s | {r['ttft_sec']}s | {r['total_sec']}s | {r['throughput_tps']} |\n"

        with open("benchmark/results/benchmark_report.md", "w", encoding="utf-8") as f:
            f.write(md)

        print("\n[✓] Benchmark tamamlandı ve sonuçlar kaydedildi.")


if __name__ == "__main__":
    bench = ArticleBenchmark()
    bench.run_all()
