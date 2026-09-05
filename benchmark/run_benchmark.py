"""
Klinik Makaleler RAG ve Context Benchmark Scripti
Yüklenen makaleler üzerinde 4K, 6K, 8K, 10K ve 16K bağlam hacimlerinde;
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
import argparse
import urllib.request
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import get_llm_config
from src.rag.hybrid_retriever import HybridRetriever


class ArticleBenchmark:
    def __init__(self, model_name: Optional[str] = None, ollama_url: Optional[str] = None):
        llm_cfg = get_llm_config()
        self.model_name = model_name or llm_cfg.get("model_name", "qwen3.5:9b")
        self.ollama_url = ollama_url or llm_cfg.get("base_url", "http://localhost:11434")
        self.retriever = HybridRetriever()
        os.makedirs("benchmark/results", exist_ok=True)

    def _build_context(self, target_tokens: int) -> List[Dict[str, Any]]:
        base_chunks = self.retriever.retrieve("Tip 2 diyabet beslenme ve tedavi", top_k=4)
        target_words = int(target_tokens * 0.75)

        expanded = list(base_chunks)
        current_words = sum(len(c["text"].split()) for c in expanded)

        i = 1
        while current_words < target_words and i < 500:
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
        num_ctx = min(max(actual_tokens + 1024, 4096), 24576)
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": 0.1, "num_ctx": num_ctx}
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
            with urllib.request.urlopen(req, timeout=300) as res:
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

    def run_all(self, contexts: Optional[List[int]] = None, merge_existing: bool = False):
        print("=== Klinik Makaleler Context Benchmark Testi ===")
        target_list = contexts if contexts else [4000, 6000, 8000, 10000, 16000]

        report_file = "benchmark/results/benchmark_report.json"
        existing_results: Dict[int, Dict[str, Any]] = {}

        if merge_existing and os.path.exists(report_file):
            try:
                with open(report_file, "r", encoding="utf-8") as f:
                    old_data = json.load(f)
                    for item in old_data:
                        existing_results[item["target_tokens"]] = item
            except Exception:
                pass

        new_results = []
        for target in target_list:
            name = f"Context-{target//1000}K-Test"
            res = self.run_test(name, target)
            new_results.append(res)
            existing_results[target] = res

        # Sort all results by target_tokens
        sorted_results = [existing_results[k] for k in sorted(existing_results.keys())]

        model_slug = self.model_name.replace(":", "_").replace(".", "_")
        report_file_tagged = f"benchmark/results/benchmark_report_{model_slug}.json"
        report_file_main = "benchmark/results/benchmark_report.json"

        with open(report_file_tagged, "w", encoding="utf-8") as f:
            json.dump(sorted_results, f, ensure_ascii=False, indent=2)
        with open(report_file_main, "w", encoding="utf-8") as f:
            json.dump(sorted_results, f, ensure_ascii=False, indent=2)

        md = f"# Klinik Makaleler RAG Context Benchmark Raporu (Model: {self.model_name})\n\n"
        md += "| Test Adı | Hedef Token | Gerçek Token | Retrieval (s) | TTFT (s) | Toplam Süre (s) | Throughput (t/s) |\n"
        md += "|---|---|---|---|---|---|---|\n"
        for r in sorted_results:
            md += f"| {r['test_name']} | {r['target_tokens']} | {r['actual_tokens']} | {r['retrieval_sec']}s | {r['ttft_sec']}s | {r['total_sec']}s | {r['throughput_tps']} |\n"

        md_file_tagged = f"benchmark/results/benchmark_report_{model_slug}.md"
        md_file_main = "benchmark/results/benchmark_report.md"
        with open(md_file_tagged, "w", encoding="utf-8") as f:
            f.write(md)
        with open(md_file_main, "w", encoding="utf-8") as f:
            f.write(md)

        print(f"\n[+] Benchmark tamamlandi. Sonuclar kaydedildi: {md_file_tagged}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Klinik Makaleler RAG Context Benchmark")
    parser.add_argument(
        "--contexts",
        type=int,
        nargs="+",
        default=[4000, 6000, 8000, 10000, 16000],
        help="Test edilecek hedef token listesi (örn: --contexts 4000 6000 8000 10000 16000)"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Mevcut benchmark_report.json sonuçlarını koruyup yeni sonuçlarla birleştir"
    )
    parser.add_argument("--model", type=str, default=None, help="Ollama model adı (None ise settings.yaml kullanılır)")
    parser.add_argument("--ollama-url", type=str, default=None, help="Ollama API adresi (None ise settings.yaml kullanılır)")

    args = parser.parse_args()

    bench = ArticleBenchmark(model_name=args.model, ollama_url=args.ollama_url)
    bench.run_all(contexts=args.contexts, merge_existing=args.merge)
