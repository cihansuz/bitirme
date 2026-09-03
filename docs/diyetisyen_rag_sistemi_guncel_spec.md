# Diyetisyen Klinik Karar Destek Sistemi (CDSS)
## Hibrit RAG, Kural Motoru ve Kısıt Tabanlı Optimizasyon Mimarisi

Bu doküman; kan tahlilleri, hasta anamnezi, klinik rehberler ve bilimsel literatürü sentezleyerek uzman diyetisyene kanıta dayalı karar desteği sunan yerel yapay zekâ asistanının teknik mimari ve uygulama şartnamesidir.

---

## 1. Proje Amacı ve Temel Sınırlar

### 1.1. Amaç
Sistem, klinik ortamda çalışan diyetisyenlerin hasta verilerini (tahliller, ilaçlar, alerjiler) inceleme süresini kısaltmak, potansiyel biyobelirteç-besin-ilaç etkileşimlerini gözden kaçırmamak ve kanıta dayalı bilimsel literatürü karar anında diyetisyenin önüne getirmek amacıyla tasarlanmıştır.

### 1.2. Etik ve Klinik Çalışma İlkeleri
* **Otonom Değildir:** Sistem tek başına nihai diyet yazmaz, tıbbi tanı koymaz veya doğrudan hastaya yönelik bir öneri bildirimi yapmaz.
* **Yardımcı ve Sentezleyicidir:** Üretilen her çıktı, diyetisyenin kontrolüne ve onayına sunulmak üzere bir taslak ve açıklama paketidir.
* **Kaynak Zorunluluğu:** Dil modelinin (LLM) ürettiği her klinik argüman, yerel literatür veri tabanından çekilen spesifik bir makale/kılavuz pasajına (chunk ID) dayanmak zorundadır.

### 1.3. Yapması ve Yapmaması Gerekenler (Do's & Don'ts)

| Yapması Gerekenler (Do's) | Yapmaması Gerekenler (Don'ts) |
|---|---|
| Kan tahlili parametrelerini referans aralıklarına göre deterministik doğrulamak. | Hastaya doğrudan "Şunu tüketin / tüketmeyin" şeklinde buyurgan reçete üretmek. |
| Kritik biyobelirteçleri (örn. GFR, Ferritin, HbA1c) kural motoruyla filtrelemek. | Tahlil sonuçlarından bağımsız olarak otonom tıbbi teşhis koymak. |
| İlaç-besin kontrendikasyonlarını matematiksel/kural tabanlı denetlemek. | RAG veri tabanında yer almayan varsayımsal klinik bilgiyi üretmek (halüsinasyon). |
| Literatürden kanıt seviyeli alıntılarla açıklamalı karar paketi sunmak. | Diyetisyen onayı olmaksızın nihai raporu doğrulanmış kabul etmek. |
| Tüm sistem çıktısını doğrulanabilir Pydantic JSON formatında üretmek. | Referans aralığı kontrollerini LLM'in sayısal tahminine bırakmak. |

---

## 2. Uçtan Uca Sistem Mimarisi

Sistem 8 ana operasyonel katmandan oluşur. Süreç deterministik güvenlik kuralları ile olasılıksal dil modeli kabiliyetlerini birbirinden izole eder:

```
                  [Hasta Profili, Laboratuvar, İlaç, Alerji]
                                      │
                                      ▼
                    [Birim ve Referans Aralığı Doğrulama]
                                      │
                                      ▼
                           [Kural ve Güvenlik Motoru]
                            │                      │
                   ┌────────┘                      └────────┐
                   ▼                                        ▼
            [Besin ve Tarif Verisi]                [Kılavuzlar, İlaç Etiketleri, Makaleler]
                   │                                        │
                   ▼                                        ▼
            [Kısıt Tabanlı Planlayıcı]                 [Hibrit RAG]
                   │                                        │
                   └───────────────┬────────────────────────┘
                                   ▼
                             [Karar Paketi]
                                   │
                                   ▼
                      [Qwen: Açıklama ve Kaynaklı JSON]
                                   │
                                   ▼
                      [Şema, Kaynak ve Güvenlik Doğrulayıcı]
                                   │
                                   ▼
                       [Diyetisyen Düzenleme ve Onay]
```

### 2.1. Katmanların Fonksiyonel Tanımları

1. **Hasta Profili, Laboratuvar, İlaç, Alerji (Girdi Katmanı):**
   * Hastanın demografik bilgileri (yaş, cinsiyet, boy, kilo, fiziksel aktivite seviyesi).
   * Kan tahlili verileri (sayısal değer, birim, laboratuvar referans aralığı).
   * Kullanılan aktif farmakolojik ajanlar (ilaçlar, takviyeler).
   * Tanılı gıda alerjileri ve intoleranslar.

2. **Birim ve Referans Aralığı Doğrulama:**
   * Farklı laboratuvarlardan gelen heterojen birimlerin standartlaştırılması (örn. Glukoz için mg/dL ve mmol/L dönüşümü).
   * Her tahlil parametresi için deterministik `LOW`, `NORMAL`, `HIGH`, `CRITICAL` etiketlemesi.

3. **Kural ve Güvenlik Motoru (Rule Engine):**
   * LLM'den bağımsız, Python tabanlı katı mantık filtresi.
   * Kesin tıbbi kontrendikasyonların kontrolü (örn. eGFR < 30 ise yüksek protein kısıtı; Levotiroksin kullanımında kalsiyum/soya etkileşimi uyarısı).
   * Tahlil tablosunda tespit edilen risk faktörlerini alt sistemlere parametrik kısıt olarak iletir.

4. **İki Kanallı Paralel Çözümleme:**
   * **Kanal A - Besin ve Tarif Verisi & Kısıt Tabanlı Planlayıcı:**
     * Besin içeriği veri tabanını (USDA / TürKomp) kullanır.
     * Lineer optimizasyon (Linear Programming) ile kural motorundan gelen kalori, makro besin ve mikro besin sınırlarına uygun matematiksel taslak porsiyonlama matrisi çıkarır.
   * **Kanal B - Kılavuzlar, İlaç Etiketleri, Makaleler & Hibrit RAG:**
     * Klinik beslenme kılavuzları (ESPEN, EASD, ADA), ilaç prospektüsleri ve PubMed taranmış akademik makaleleri içerir.
     * Dense (Vektörel) ve Sparse (Anahtar Kelime - BM25) yöntemlerini hibrit olarak çalıştırır; ilgili paragrafları Reranker ile sıralar.

5. **Karar Paketi (Decision Context Synthesis):**
   * Planlayıcıdan gelen sayısal kısıt matrisi ile Hibrit RAG'den çekilen en alakalı doküman pasajı bir araya getirilir.
   * Qwen modeli için standart bir bağlam (context) şablonu oluşturulur.

6. **Qwen: Açıklama ve Kaynaklı JSON (Ollama Yerel Çıkarım):**
   * Qwen modeli (3.8B/3B parametreli yerel varyant) karar paketini analiz eder.
   * Diyetisyenin hızlıca kavrayabileceği klinik gerekçelendirmeyi oluşturur.
   * Çıktıyı katı bir JSON şemasında ve her klinik çıkarımın hangi makaleden alındığını belirten kaynak id'leriyle (`citation_ids`) döndürür.

7. **Şema, Kaynak ve Güvenlik Doğrulayıcı (Post-Processing Validator):**
   * **Pydantic Şema Doğrulama:** Çıktının JSON yapısını denetler; eksik alan varsa onarım döngüsünü tetikler.
   * **Citation Verifier:** Modelin atıf yaptığı kaynak ID'lerinin gerçekten context içerisinde yer alıp almadığını kontrol eder (Halüsinasyon engelleme).
   * **Güvenlik Çapraz Kontrolü:** Model metninde kural motorunun yasakladığı bir besinin yer alıp almadığını metin aramasıyla son kez doğrular.

8. **Diyetisyen Düzenleme ve Onay:**
   * Arayüz üzerinden diyetisyen, sistemin sunduğu referanslı açıklamayı, çıkarılan riskleri ve taslak kısıtları inceler.
   * İhtiyaç halinde parametreleri düzenler ve onaylayarak hastanın klinik dosyasına işler.

---

## 3. Modüler Hastalık Eklenti Mimarisi (Plug-and-Play Engine)

Tüm hastalık tablolarını monolitik bir yapıda kodlamak yerine, sisteme sonradan dinamik olarak eklenebilecek modüler bir "Disease Plugin" deseni uygulanır.

```
                  [Hasta Verisi & Tahliller]
                              │
                              ▼
                [Disease Registry & Router]
                              │
       ┌──────────────────────┼──────────────────────┐
       ▼                      ▼                      ▼
[Tip 2 Diyabet]         [Demir Anemisi]       [Gelecek Modül...]
(Pilot Modül)           (Eklenti Modül)       (Hipertansiyon/KBY)
       │
       ├─ Laboratuvar Eşikleri (HbA1c, Açlık Kan Şekeri)
       ├─ Kural ve Güvenlik Kuralları (Glisemik Yük Limitleri)
       ├─ Planlayıcı Kısıtları (Karbonhidrat Dağılımı ve Lif Alt Limiti)
       └─ RAG Koleksiyon Filtresi (Metadata: {"disease": "t2dm"})
```

### 3.1. Modül Arayüz Sözleşmesi (`BaseDiseaseModule`)
Her hastalık paketi şu arayüzü implemente eder:
* `evaluate_triggers(lab_data: dict) -> bool`: Tahlillerde bu hastalığın klinik bayraklarının tetiklenip tetiklenmediğini belirler.
* `get_safety_constraints() -> List[Constraint]`: Besin ve tarif planlayıcısına iletilecek kesin sınırları döndürür.
* `get_rag_filter() -> dict`: Vektör veritabanı sorgusuna eklenecek metadata filtresini sağlar.
* `get_contraindications() -> List[str]`: Kesinlikle izin verilmeyen gıda bileşenleri ve ilaç etkileşim listesi.

### 3.2. YAML Tabanlı Kural Konfigürasyonu Örneği (`modules/t2dm/rules.yaml`)
```yaml
disease_id: "t2dm"
name: "Tip 2 Diabetes Mellitus & İnsülin Direnci"
triggers:
  - biomarker: "HbA1c"
    operator: ">="
    threshold: 6.5
    unit: "%"
  - biomarker: "Fasting_Glucose"
    operator: ">="
    threshold: 126
    unit: "mg/dL"
constraints:
  carbohydrate_ratio_max: 0.45
  added_sugar_max_g: 0
  minimum_fiber_g: 30
  glycemic_index_max: 55
rag_metadata_filter:
  domain: "endocrinology"
  tags: ["t2dm", "insulin_resistance", "glycemic_control"]
```

### 3.3. Çakışma Çözümleme (Conflict Resolution Engine)
Bir hastada birden fazla modül tetiklendiğinde (örneğin Tip 2 Diyabet + Hipertansiyon):
* Güvenlik motoru kısıtları birleştirirken en kısıtlayıcı kuralı (En Düşük Üst Limit - Min of Max) seçer.
* Sodyum: Hipertansiyon (< 1500mg) vs Standart (< 2300mg) -> **1500mg seçilir**.
* Şeker: Diyabet (0g eklenmiş şeker) -> **0g seçilir**.

---

## 4. Teknik Altyapı ve Kütüphane Yığını

* **Programlama Dili:** Python 3.11+
* **Paket ve Ortam Yöneticisi:** `uv` (Hızlı sanal ortam ve deterministik kilit dosyası yönetimi)
* **LLM Motoru:** Yerel Ollama altyapısı üzerinde `Qwen 2.5 (3.8B / 3B variant)`
* **Orkestrasyon:** LangChain / LangGraph (Durum yönetimli ajan ve iş akışı kontrolü)
* **Vektör Veritabanı:** ChromaDB (Şimdilik / Faz 1). İlerleyen aşamalarda ölçeklenebilirlik ve ilişkisel veri bütünlüğü için **PostgreSQL + pgvector** yapısına geçilecektir.
* **Embedding Modeli:** `BAAI/bge-m3` veya `nomic-embed-text`
* **Reranker:** `BAAI/bge-reranker-base` veya `cross-encoder/ms-marco-MiniLM-L-6-v2`
* **Arama Yöntemi:** Sparse (BM25) + Dense (Cosine Similarity) Hibrit Arama
* **Doğrusal Programlama (Kısıt Planlayıcı):** `PuLP` veya `SciPy.optimize`
* **Veri Doğrulama ve Tip Denetimi:** `Pydantic v2`
* **Servis Katmanı:** FastAPI (İstemci ve arayüz entegrasyonu için REST API)

---

## 5. Qwen Context Analizi ve Performans Ölçüm Metodolojisi

Qwen modelinin yerel donanımda farklı bağlam (context) yükleri altındaki davranışları sistematik testlerle analiz edilecektir.

### 5.1. Test Değişkenleri
* **Makale / Doküman Hacmi:** Değişken adetlerde akademik makale özeti / kılavuz pasajı.
* **Token Sayısı:** **8.000 token** ve **16.000 token** bağlam (context) pencereleri.
* **Donanım Metrikleri:** CPU/VRAM kullanımı, inference gecikmesi, bellek doygunluğu.

### 5.2. Ölçülecek Performans Metrikleri
1. **Retrieval Latency ($t_{ret}$):** Hibrit aramanın belgeleri getirip rerank etme süresi (ms).
2. **Time to First Token (TTFT):** Ollama üzerinden modelin ilk çıktı token'ını üretme anı (ms/s).
3. **Throughput (Token/s):** Üretim fazındaki token akış hızı.
4. **Factual Groundedness (Kaynak Sadakati):** Üretilen argümanların context içindeki chunk'larla örtüşme yüzdesi (RAGAS veya kural denetimi ile halüsinasyon oranı).
5. **Context Retention (Lost in the Middle):** Kritik bilginin context'in başında, ortasında veya sonunda bulunması durumunda modelin bilgiyi yakalama isabeti.

### 5.3. Benchmark Takip Tablosu Formatı

```
| Test ID | Modül | Context Hacmi | Context (Tokens) | Retrieval (s) | TTFT (s) | Toplam Süre (s) | Token/s | Halüsinasyon Var mı? |
|---|---|---|---|---|---|---|---|---|
| BM-8K-1 | T2DM | Orta Yoğunluk  | ~8.000 | | | | | |
| BM-8K-2 | T2DM | Yüksek Yoğunluk| ~8.000 | | | | | |
| BM-16K-1| T2DM | Çoklu Kaynak   | ~16.000 | | | | | |
| BM-16K-2| T2DM | Limit Testi    | ~16.000 | | | | | |
```

---

## 6. Veri Sözleşmeleri ve JSON Şemaları

### 6.1. Model Giriş Karar Paketi Şeması (`DecisionPacket`)
```json
{
  "patient_id": "P-10492",
  "abnormal_biomarkers": [
    {
      "name": "Fasting_Glucose",
      "value": 134.0,
      "unit": "mg/dL",
      "reference_range": "70-100",
      "status": "HIGH"
    },
    {
      "name": "HbA1c",
      "value": 6.8,
      "unit": "%",
      "reference_range": "4.0-5.6",
      "status": "HIGH"
    }
  ],
  "active_constraints": {
    "max_carbs_percent": 45,
    "max_sugar_g": 0,
    "min_fiber_g": 30
  },
  "retrieved_evidence": [
    {
      "chunk_id": "ADA_2024_P14",
      "title": "Standards of Care in Diabetes",
      "text": "In individuals with type 2 diabetes, carbohydrate intake should emphasize nutrient-dense carbohydrate sources with high dietary fiber..."
    }
  ]
}
```

### 6.2. Qwen Çıktı Şeması (`DietitianDecisionSupportOutput`)
```json
{
  "clinical_summary": "Hasta açlık kan şekeri ve HbA1c düzeyleri Tip 2 Diyabet eşiklerinin üzerindedir.",
  "findings": [
    {
      "biomarker": "HbA1c",
      "interpretation": "Geriye dönük 3 aylık glisemik kontrolün yetersiz olduğunu göstermektedir.",
      "citation_ids": ["ADA_2024_P14"]
    }
  ],
  "dietary_guidelines_suggested": [
    {
      "parameter": "Lif Tüketimi",
      "target": ">= 30g/gün",
      "rationale": "Postprandiyal glukoz dalgalanmasını baskılamak ve insülin duyarlılığını desteklemek.",
      "citation_ids": ["ADA_2024_P14"]
    }
  ],
  "contraindications_flagged": [
    "Yüksek glisemik indeksli basit şekerler ve rafine unlu mamuller kısıtlanmalıdır."
  ],
  "status": "PENDING_DIETITIAN_REVIEW"
}
```

---

## 7. Proje Dosya ve Dizin Yapısı

```
dietitian-rag-cdss/
├── pyproject.toml              # uv bağımlılık tanımları
├── README.md                   # Proje tanıtım ve çalıştırma rehberi
├── data/
│   ├── raw_guidelines/         # PDF kılavuzlar ve akademik makaleler
│   ├── processed_chunks/       # Parçalanmış ve temizlenmiş metinler
│   └── chromadb/               # Vektör veri tabanı kalıcı deposu (Faz 1)
├── configs/
│   └── settings.yaml           # Model parametreleri, chunk boyutları, eşikler
├── modules/                    # Modüler Hastalık Motoru
│   ├── base.py                 # BaseDiseaseModule soyut sınıfı
│   ├── registry.py             # Dinamik modül yükleyici ve çakışma çözücü
│   ├── t2dm/                   # Tip 2 Diyabet Modülü (Pilot)
│   │   ├── rules.yaml
│   │   └── evaluator.py
│   └── anemia/                 # Demir Anemisi Modülü (Gelecek)
│       ├── rules.yaml
│       └── evaluator.py
├── src/
│   ├── parser/                 # Tahlil verisi ve birim standardizasyonu
│   ├── rule_engine/            # Deterministik kural kontrolü ve güvenlik
│   ├── planner/                # PuLP kısıt tabanlı besin planlayıcı
│   ├── rag/                    # Hibrit RAG (Dense + BM25 + Reranker)
│   ├── llm/                    # Ollama Qwen entegrasyonu ve structured output
│   ├── validator/              # Şema, kaynak ve halüsinasyon kontrolü
│   └── api/                    # FastAPI endpoint'leri
└── benchmark/                  # Qwen performans ve bağlam analiz scriptleri
    ├── run_benchmark.py
    └── results/
```

---

## 8. İş Paketleri ve Geliştirme Yol Haritası

### Paket 1: Ortam Kurulumu ve Veri Tabanı İskeleti
* [ ] `uv` ile proje ortamının kurulması, Ollama üzerinden Qwen modelinin yerel olarak doğrulanması.
* [ ] Pilot hastalık (Tip 2 Diyabet) için kılavuz ve makalelerin toplanması.
* [ ] ChromaDB üzerinde (MVP aşaması) koleksiyon oluşturulması, chunking stratejisinin (512 token, %10 overlap) uygulanması ve BM25 indeksinin bağlanması.
* [ ] İlerleyen fazlar için PostgreSQL veritabanı şemalarının ve pgvector geçiş planının tasarlanması.

### Paket 2: Deterministik Katmanlar ve Pilot Modül
* [ ] Birim standardizasyonu ve laboratuvar referans aralığı doğrulayıcısının kodlanması.
* [ ] `BaseDiseaseModule` arayüzünün ve `t2dm` pilot modül kurallarının (YAML) yazılması.
* [ ] Kısıt tabanlı taslak planlayıcının (PuLP) entegre edilmesi.

### Paket 3: LLM Entegrasyonu, Hibrit RAG ve Doğrulayıcı
* [ ] Karar paketinin dinamik oluşturulması ve Qwen için strict prompt şablonunun yapılandırılması.
* [ ] Pydantic tabanlı JSON çıktı ayrıştırıcısı ve citation doğrulayıcısının yazılması.
* [ ] Modelin halüsinasyon yapmasını önleyen güvenlik filtresinin bağlanması.

### Paket 4: Qwen Bağlam ve Gecikme Analizi (Benchmark)
* [ ] 8.000 ve 16.000 token context yükleri ile otomatik latency ve token/s test scriptlerinin çalıştırılması.
* [ ] Elde edilen metriklerin tablo ve grafik haline getirilerek tez/rapor formatına dökülmesi.
* [ ] Sonraki hastalık modüllerinin (Anemi, vb.) sisteme kolayca takılabilmesi için mimari testlerin tamamlanması.
