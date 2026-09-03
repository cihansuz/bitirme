import sys
import os

# Windows konsolunda UTF-8 çıktı desteği sağla
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Proje kök dizinini sys.path'e ekle
sys.path.insert(0, os.path.abspath("."))

from src.llm.qwen_agent import QwenAgent


def print_banner():
    print("\n" + "=" * 70)
    print("      DIYETISYEN RAG SISTEMI - MAKALELERE SORU-CEVAP (Q&A)      ")
    print("=" * 70)


def answer_question(agent: QwenAgent, question: str):
    print(f"\n[?] Soru: {question}")
    print("[...] Makaleler taranıyor ve Qwen 3:8B yanıtı üretiliyor...")
    
    result = agent.ask(question)

    print("\n" + "-" * 70)
    print("TARANAN İLGİLİ MAKALE PASAJLARI (RAG KAYNAKLARI):")
    print("-" * 70)
    for i, s in enumerate(result["sources"], 1):
        print(f"[{i}] {s['chunk_id']} - {s['title']} (Dosya: {s['source']})")

    print("\n" + "-" * 70)
    print("QWEN 3:8B KLİNİK YANITI:")
    print("-" * 70)
    print(result["answer"])
    print("-" * 70 + "\n")


def main():
    print_banner()
    print("[*] Hibrit RAG motoru ve Qwen 3:8B başlatılıyor...")
    agent = QwenAgent()
    print(f"[OK] Sistem hazır! Toplam {len(agent.retriever.chunks)} makale pasajı indekslendi.")

    # 1. Komut satırından doğrudan soru verilmişse:
    # Örn: uv run python main.py "Metformin B12 ilişkisi nedir?"
    if len(sys.argv) > 1:
        user_question = " ".join(sys.argv[1:])
        answer_question(agent, user_question)
        return

    # 2. İnteraktif Soru Sorma Döngüsü:
    print("\nMakalelerinize soru sorabilirsiniz. Çıkmak için 'q' yazın.\n")
    while True:
        try:
            q = input("Sorunuz: ").strip()
            if not q:
                continue
            if q.lower() in ("q", "quit", "exit", "çıkış"):
                print("Çıkış yapıldı. İyi çalışmalar!")
                break
            answer_question(agent, q)
        except (KeyboardInterrupt, EOFError):
            print("\nÇıkış yapıldı.")
            break


if __name__ == "__main__":
    main()
