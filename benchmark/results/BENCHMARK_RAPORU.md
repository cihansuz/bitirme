# Qwen Modelleri Reasoning Kapatma ve Benchmark Karşılaştırma Raporu

## Yapılan Değişiklikler
Mimari bozulmadan parametre bazlı düşünme (reasoning) kontrolü sağlandı:
1. **[settings.yaml](../../configs/settings.yaml)**: `llm` bloğuna `enable_thinking: false` parametresi eklendi.
2. **[qwen_agent.py](../../src/llm/qwen_agent.py)**: Ollama payload'ına `"think": self.enable_thinking` eklendi ve `<think>` etiket temizliği entegre edildi.
3. **[run_clinical_benchmark.py](../run_clinical_benchmark.py)**: Benchmark çıkarımlarına `--enable-thinking` / `--disable-thinking` desteği ve adım adım faz loglaması eklendi.

---

## 13 Klinik Vaka (TC-01 - TC-13) Karşılaştırma Sonuçları (8K Context)

Tüm modeller `enable_thinking=false` parametresiyle test edildi:

| Model | İlk Token (TTFT) | Çıkarım Hızı (TPS) | Toplam Süre (13 Soru) | Triyaj Doğruluğu | Kaynak Sadakati |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **qwen3.5:4b** | **3.59 s** | **51.28 t/s** | **216.90 s (~3.6 dk)** | **%84.62 (11/13)** | **%100** |
| **qwen3:8b** | **3.54 s** | **36.46 t/s** | **275.98 s (~4.6 dk)** | **%53.85 (7/13)** | **%100** |
| **qwen3.5:9b** | **4.04 s** | **31.15 t/s** | **309.84 s (~5.1 dk)** | **%61.54 (8/13)** | **%100** |

---

## Düşünme Açık vs Düşünme Kapalı Hız Değişimi

| Model | Düşünme Açık TTFT | Düşünme Kapalı TTFT | Hızlanma Oranı | Toplam Süre Tasarrufu |
| :--- | :--- | :--- | :--- | :--- |
| **qwen3.5:4b** | 83.56 s | **3.59 s** | 🚀 **~23.3 Kat Hızlı** | 20.7 dk ➔ **3.6 dk** (%82.6 tasarruf) |
| **qwen3.5:9b** | 157.07 s | **4.04 s** | 🚀 **~38.8 Kat Hızlı** | 37.3 dk ➔ **5.1 dk** (%86.2 tasarruf) |
| **qwen3:8b** | 16.40 s | **3.54 s** | 🚀 **~4.6 Kat Hızlı** | 6.8 dk ➔ **4.6 dk** (%32.3 tasarruf) |

---

## Klinik Değerlendirme & Tavsiye
* **En Başarılı Model: `qwen3.5:4b`**
  - **Triyaj Doğruluğu:** **%84.62** ile en yüksek doğruluğa ulaştı. Düşünme açıkken yanıt formatı karışırken, reasoning kapatıldığında doğrudan CDSS triyaj şablonuna odaklandı ve doğruluğu %46'dan %84'e fırladı.
  - **Hız:** **51.28 TPS** ile en yüksek üretim hızını sundu.
* Tüm sonuçlar kalıcı olarak [models_comparison_matrix.json](./models_comparison_matrix.json) içine işlendi.
