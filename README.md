# Diyetisyen Klinik Karar Destek Sistemi (CDSS)
## Hibrit RAG, Deterministik Kural Motoru ve Kısıt Tabanlı Optimizasyon Mimarisi

Bu proje; kan tahlilleri, hasta anamnezi, kullanılan ilaçlar, alerjiler ve klinik beslenme rehberlerini (ADA, WHO, ESPEN) sentezleyerek uzman diyetisyene kanıta dayalı klinik karar desteği sunan yerel bir yapay zekâ asistanıdır.

---

## 1. Mimari Genel Bakış (8 Operasyonel Katman)

Sistem, deterministik güvenlik kuralları ile olasılıksal dil modeli kabiliyetlerini birbirinden izole eder:

1. **Girdi Katmanı:** Hasta demografik verileri, kan tahlilleri, aktif ilaçlar ve alerjiler.
2. **Birim ve Referans Aralığı Doğrulama (`LabParser`):** Heterojen laboratuvar birimlerinin standardizasyonu (örn. Glukoz `mmol/L` -> `mg/dL`, HbA1c `mmol/mol` -> `%`) ve deterministik `LOW`, `NORMAL`, `HIGH`, `CRITICAL` etiketlemesi.
3. **Kural ve Güvenlik Motoru (`RuleEngine`):** LLM'den bağımsız katı Python mantığı; ilaç-besin etkileşimleri (Metformin-B12, Levotiroksin-Kalsiyum/Soya), eGFR < 30 protein kısıtları ve modüler kural çakışma çözümleme (**Conflict Resolution Engine: Min of Max, Max of Min**).
4. **İki Kanallı Paralel Çözümleme:**
   - **Kanal A (Planlayıcı - `NutritionalPlanner`):** `PuLP` doğrusal programlama (Linear Programming) ile enerji, makro (karbonhidrat/protein/yağ) ve mikro (lif/demir/C vitamini) kısıtlarına uygun optimum porsiyonlama matrisi çıkarır.
   - **Kanal B (Hibrit RAG - `HybridRetriever`):** `ChromaDB` (Dense Vektör Arama) + `BM25Okapi` (Sparse Anahtar Kelime Arama) + **Reciprocal Rank Fusion (RRF)** ile en alakalı kanıt pasajlarını filtreler.
5. **Karar Paketi Sentezi (`DecisionPacket`):** Planlayıcı kısıtları ile RAG kanıt pasajlarını Qwen modeli için standart bağlamda toplar.
6. **Yerel Qwen Çıkarımı (`QwenAgent`):** Yerel Ollama altyapısında `qwen3:8b` modeli ile çıkarım yapar. Katı JSON şemasında ve kaynak atıflı (`citation_ids`) karar taslağı üretir.
7. **Şema, Kaynak ve Güvenlik Doğrulayıcı (`CitationAndSafetyValidator`):**
   - **Pydantic Şema Doğrulama**: Eksik/hatalı JSON alanlarını onarır.
   - **Citation Verifier**: Modelin uydurma/halüsinatif `chunk_id` üretmesini engeller; sadece context'te var olan kanıtları onaylar.
   - **Güvenlik Çapraz Denetimi**: Kural motorunun yasakladığı bir gıdanın LLM tarafından önerilmesini engeller.
8. **Diyetisyen İnceleme ve Onay Katmanı (`/review`):** Sistemin ürettiği taslak diyetisyenin onayına (`APPROVED`) veya düzenlemesine (`EDITED_AND_APPROVED`) sunulur.

---

## 2. Kurulum ve Çalıştırma

### Gereksinimler
- Python 3.12+
- `uv` paket yöneticisi
- Yerel `Ollama` servisi (`qwen3:8b` modeli yüklü)

### Adım 1: Bağımlılıkları Yükleme
```powershell
uv sync
```

### Adım 2: API Servisini Başlatma
```powershell
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
API Dokümantasyonu: `http://localhost:8000/docs`

### Adım 3: Testleri Çalıştırma
```powershell
uv run pytest
```

### Adım 4: Benchmark Analizini Çalıştırma
Şartname Bölüm 5 uyarınca 8K ve 16K context pencerelerinde TTFT, Throughput ve Halüsinasyon analizini çalıştırmak için:
```powershell
uv run python benchmark/run_benchmark.py
```
Sonuçlar `benchmark/results/benchmark_report.md` ve `benchmark/results/benchmark_report.json` dosyalarına otomatik yazılır.

---

## 3. Dizin Yapısı

```
dietitian-rag-cdss/
├── configs/
│   └── settings.yaml           # Model, RAG ve Ollama konfigürasyonu
├── data/
│   ├── raw_guidelines/         # ADA, WHO, ESPEN ve ilaç etkileşim kılavuzları
│   ├── processed_chunks/       # JSON formatında indekslenen pasajlar
│   ├── chromadb/               # Vektör veritabanı deposu (Faz 1)
│   └── schema_pgvector.sql     # PostgreSQL + pgvector geçiş DDL şeması (Faz 2)
├── modules/                    # Modüler Eklenti Motoru (Plug-and-Play)
│   ├── base.py                 # BaseDiseaseModule arayüzü
│   ├── registry.py             # DiseaseRegistry & Conflict Resolution Engine
│   ├── t2dm/                   # Tip 2 Diyabet Modülü (rules.yaml + evaluator.py)
│   └── anemia/                 # Demir Anemisi Modülü (rules.yaml + evaluator.py)
├── src/
│   ├── parser/                 # LabParser (Birim standardizasyonu & LOW/NORMAL/HIGH/CRITICAL)
│   ├── rule_engine/            # RuleEngine (İlaç-besin, böbrek kısıtı & güvenlik)
│   ├── planner/                # NutritionalPlanner (PuLP doğrusal programlama optimizasyonu)
│   ├── rag/                    # HybridRetriever (ChromaDB + BM25 + RRF Reranking)
│   ├── llm/                    # QwenAgent (Ollama qwen3:8b entegrasyonu)
│   ├── validator/              # CitationAndSafetyValidator & Pydantic şemaları
│   └── api/                    # FastAPI REST API (main.py)
├── benchmark/                  # Qwen context ve gecikme test scriptleri
│   ├── run_benchmark.py
│   └── results/
└── tests/                      # Kapsamlı otomatik test paketi
```
