"""
Klinik ve Kural Doğrulama Benchmark Scripti (TC-01 - TC-13)
Klinik senaryolar üzerinde 8K ve 10K context hacimlerinde:
- Triyaj Doğruluğu (Triage Accuracy)
- Kaynak Sadakati (Citation Fidelity)
- Retrieval Latency & Chunk Recall
- TTFT (Time to First Token)
- Throughput (t/s)
ölçümlerini yapar ve sonuçları temiz JSON formatında benchmark/results/ klasörüne kaydeder.
"""
import time
import json
import os
import sys
import re
import argparse
import urllib.request
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import get_llm_config
from src.rag.hybrid_retriever import HybridRetriever


class ClinicalBenchmark:
    SYSTEM_PROMPT = """Sen klinik karar destek sistemi (CDSS) ve uzman klinik diyetisyensin.
Sana verilen klinik bağlamı ve hasta sorusunu değerlendir.

CEVAP FORMATIN MUTLAKA ŞU ŞEKİLDE OLMALIDIR:
1. TRİYAJ KODU: [RED / ORANGE / YELLOW / STANDARD_REVIEW] seçeneklerinden birini büyük harfle belirt.
2. KLİNİK DEĞERLENDİRME VE EYLEMLER: Hastanın durumu ve beslenme planı için kritik kuralları, kısıtları ve hekim uyarılarını maddeler halinde açıkla.
3. KAYNAK ATIFLARI: Verdiğin her bilgi için metin içinde mutlaka [Kaynak: CHUNK_ID] formatında ilgili kaynak kimliğini ekle.
"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        ollama_url: Optional[str] = None,
        enable_thinking: Optional[bool] = None
    ):
        llm_cfg = get_llm_config()
        self.model_name = model_name or llm_cfg.get("model_name", "qwen3:8b")
        self.ollama_url = ollama_url or llm_cfg.get("base_url", "http://localhost:11434")
        self.enable_thinking = enable_thinking if enable_thinking is not None else llm_cfg.get("enable_thinking", False)
        self.retriever = HybridRetriever(top_k=6)
        os.makedirs("benchmark/results", exist_ok=True)

    def _normalize_triage(self, text: str) -> str:
        text_upper = text.upper()
        if "RED" in text_upper or "KIRMIZI" in text_upper or "ACİL" in text_upper:
            return "RED"
        if "ORANGE" in text_upper or "TURUNCU" in text_upper:
            return "ORANGE"
        if "YELLOW" in text_upper or "SARI" in text_upper:
            return "YELLOW"
        if "STANDARD" in text_upper or "STANDART" in text_upper or "RUTİN" in text_upper:
            return "STANDARD_REVIEW"
        return "UNKNOWN"

    def _evaluate_triage_match(self, predicted: str, expected: str) -> bool:
        if expected == "YELLOW_OR_ORANGE":
            return predicted in ["YELLOW", "ORANGE"]
        return predicted == expected

    def run_suite_for_context(
        self,
        test_cases: List[Dict[str, Any]],
        context_window: int
    ) -> Dict[str, Any]:
        print(f"\n==================================================", flush=True)
        print(f"[*] Model: {self.model_name} | Context Window: {context_window} Tokens", flush=True)
        print(f"==================================================", flush=True)

        results = []
        suite_start_time = time.perf_counter()

        for i, tc in enumerate(test_cases, 1):
            tid = tc["test_id"]
            question = tc["question"]
            expected_triage = tc["expected_triage"]
            target_tags = tc.get("target_chunk_tags", [])

            print(f"\n[{i}/{len(test_cases)}] {tid} ({tc['category']})...", flush=True)

            # 1. Retrieval Latency & Recall
            t_ret_0 = time.perf_counter()
            retrieved = self.retriever.retrieve(question, top_k=6)
            t_ret = time.perf_counter() - t_ret_0

            retrieved_cids = [c["chunk_id"] for c in retrieved]
            matched_tags = [tag for tag in target_tags if any(tag in cid for cid in retrieved_cids)]
            tag_recall = len(matched_tags) / len(target_tags) if target_tags else 1.0
            print(f"  [RAG] {len(retrieved)} pasaj çekildi ({round(t_ret, 2)}s). Model çıkarımı başlatılıyor...", flush=True)

            # 2. Context Construction
            context_blocks = []
            for c in retrieved:
                context_blocks.append(f"--- [KAYNAK: {c['chunk_id']}] ({c['title']}) ---\n{c['text']}")
            context_text = "\n\n".join(context_blocks)

            user_prompt = f"""KLİNİK BAĞLAM:
{context_text}

HASTA SORUSU / PROFİLİ:
{question}

Lütfen belirtilen cevap formatına uyarak yanıtla:"""

            # 3. LLM Inference via Ollama
            payload = {
                "model": self.model_name,
                "system": self.SYSTEM_PROMPT,
                "prompt": user_prompt,
                "stream": True,
                "think": self.enable_thinking,
                "options": {
                    "temperature": 0.1,
                    "num_ctx": context_window
                }
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
                print(f"  [!] Hata: {e}", flush=True)
                total_time = time.perf_counter() - t_start

            throughput = generated_tokens / max(1.0, (total_time - ttft)) if total_time > ttft else 0.0
            answer_text = "".join(chunks)
            if not self.enable_thinking:
                answer_text = re.sub(r"<think>.*?</think>", "", answer_text, flags=re.DOTALL).strip()

            # 4. Triage & Citations
            triage_match_res = re.search(r"TR[İI]YAJ\s*KODU\s*:\s*([A-Z_]+)", answer_text, re.IGNORECASE)
            predicted_triage = self._normalize_triage(triage_match_res.group(1)) if triage_match_res else self._normalize_triage(answer_text[:200])
            triage_success = self._evaluate_triage_match(predicted_triage, expected_triage)

            citations_found = [cid for cid in retrieved_cids if cid in answer_text]
            has_citation = len(citations_found) > 0

            status_mark = "OK" if triage_success else "FAIL"
            print(f"  [LLM] TTFT: {round(ttft, 2)}s | TPS: {round(throughput, 1)} | Süre: {round(total_time, 1)}s | Triyaj: {predicted_triage} [{status_mark}]", flush=True)

            results.append({
                "test_id": tid,
                "category": tc["category"],
                "question": question,
                "expected_triage": expected_triage,
                "predicted_triage": predicted_triage,
                "triage_match": triage_success,
                "retrieval_sec": round(t_ret, 3),
                "chunk_recall": round(tag_recall, 2),
                "target_chunk_tags": target_tags,
                "matched_chunk_tags": matched_tags,
                "citations_found": citations_found,
                "has_citation": has_citation,
                "ttft_sec": round(ttft, 2),
                "total_sec": round(total_time, 2),
                "throughput_tps": round(throughput, 1),
                "answer_snippet": answer_text[:200].replace("\n", " ") + "..."
            })

        total_suite_time = time.perf_counter() - suite_start_time
        n = len(results)

        summary = {
            "model": self.model_name,
            "context_window": context_window,
            "total_test_cases": n,
            "triage_accuracy_percent": round(sum(1 for r in results if r["triage_match"]) / n * 100, 2),
            "citation_fidelity_percent": round(sum(1 for r in results if r["has_citation"]) / n * 100, 2),
            "avg_chunk_recall_percent": round(sum(r["chunk_recall"] for r in results) / n * 100, 2),
            "avg_retrieval_sec": round(sum(r["retrieval_sec"] for r in results) / n, 3),
            "avg_ttft_sec": round(sum(r["ttft_sec"] for r in results) / n, 2),
            "avg_throughput_tps": round(sum(r["throughput_tps"] for r in results) / n, 2),
            "total_duration_sec": round(total_suite_time, 2)
        }

        # Clean JSON Export
        model_slug = self.model_name.replace(":", "_").replace(".", "_")
        ctx_k = context_window // 1000 if context_window % 1000 == 0 else round(context_window / 1024)
        output_filename = f"benchmark/results/{model_slug}_{ctx_k}k_results.json"

        data_to_save = {
            "summary": summary,
            "test_cases": results
        }

        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)

        print(f"\n[+] Temiz JSON Kaydedildi: {output_filename}", flush=True)
        print(f"-> Dogruluk: %{summary['triage_accuracy_percent']} | TTFT: {summary['avg_ttft_sec']}s | TPS: {summary['avg_throughput_tps']}", flush=True)

        return summary

    def run_all(
        self,
        test_cases_path: str = "benchmark/clinical_test_cases.json",
        contexts: List[int] = [8000, 10000]
    ):
        with open(test_cases_path, "r", encoding="utf-8") as f:
            test_cases = json.load(f)

        print(f"=== Model Benchmark Testi Baslatildi: {self.model_name} ===", flush=True)
        print(f"Test Vakaları: {len(test_cases)} adet | Hedef Contextler: {contexts}", flush=True)

        matrix_file = "benchmark/results/models_comparison_matrix.json"
        existing_matrix = []
        if os.path.exists(matrix_file):
            try:
                with open(matrix_file, "r", encoding="utf-8") as f:
                    existing_matrix = json.load(f)
            except Exception:
                existing_matrix = []

        for ctx in contexts:
            summary = self.run_suite_for_context(test_cases, context_window=ctx)
            # Add or update in matrix
            existing_matrix = [m for m in existing_matrix if not (m["model"] == summary["model"] and m["context_window"] == summary["context_window"])]
            existing_matrix.append(summary)

        # Save clean matrix JSON
        with open(matrix_file, "w", encoding="utf-8") as f:
            json.dump(existing_matrix, f, ensure_ascii=False, indent=2)

        print(f"\n[+] Tum sonuclar karsilastirma matrisine eklendi: {matrix_file}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Klinik Karar Destek Benchmark Testi")
    parser.add_argument("--model", type=str, default=None, help="Kullanılacak model adı (None ise settings.yaml kullanılır)")
    parser.add_argument("--ollama-url", type=str, default=None, help="Ollama API URL")
    parser.add_argument("--test-cases", type=str, default="benchmark/clinical_test_cases.json", help="Test vakaları JSON dosyası")
    parser.add_argument("--contexts", type=int, nargs="+", default=[8000, 10000], help="Test edilecek context boyutları (örn: 8000 10000)")
    parser.add_argument("--enable-thinking", action="store_true", default=None, help="Düşünme (reasoning) sürecini açar")
    parser.add_argument("--disable-thinking", action="store_true", default=None, help="Düşünme (reasoning) sürecini kapatır")

    args = parser.parse_args()
    thinking_setting = None
    if args.disable_thinking:
        thinking_setting = False
    elif args.enable_thinking:
        thinking_setting = True

    bench = ClinicalBenchmark(model_name=args.model, ollama_url=args.ollama_url, enable_thinking=thinking_setting)
    bench.run_all(test_cases_path=args.test_cases, contexts=args.contexts)
