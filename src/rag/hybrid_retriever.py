import os
import re
import json
import zipfile
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi


class HybridRetriever:
    """
    Klinik Makaleler İçin Hibrit RAG (Retrieval-Augmented Generation) Arama Motoru.
    - Dense Search: ChromaDB Vektör Veritabanı
    - Sparse Search: BM25 Anahtar Kelime Araması
    - Sıralama: Reciprocal Rank Fusion (RRF)
    - Desteklenen Formatlar: .docx (Word), .md (Markdown), .txt (Metin)
    """

    def __init__(
        self,
        guidelines_dir: str = "data/raw_guidelines",
        chroma_dir: str = "data/chromadb",
        collection_name: str = "clinical_articles",
        top_k: int = 4
    ):
        self.guidelines_dir = guidelines_dir
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self.top_k = top_k

        self.chunks: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None

        # 1. ChromaDB İstemcisi
        os.makedirs(self.chroma_dir, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_dir)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn
        )

        # 2. Makaleleri İndeksle
        self.index_documents()

    def _tokenize(self, text: str) -> List[str]:
        """Metni küçük harfe çevirir ve temiz kelime listesine ayrıştırır."""
        clean = re.sub(r"[^\w\s]", " ", text.lower())
        return [word for word in clean.split() if len(word) > 2]

    def _extract_docx_paragraphs(self, file_path: str) -> List[str]:
        """Word (.docx) dosyasındaki paragrafları harici kütüphane gerektirmeden XML üzerinden okur."""
        try:
            with zipfile.ZipFile(file_path) as z:
                tree = ET.fromstring(z.read("word/document.xml"))
                namespaces = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                paragraphs = []
                for p in tree.iterfind(".//w:p", namespaces):
                    texts = [node.text for node in p.iterfind(".//w:t", namespaces) if node.text]
                    if texts:
                        para_text = "".join(texts).strip()
                        if len(para_text) > 15:
                            paragraphs.append(para_text)
                return paragraphs
        except Exception as e:
            print(f"[Uyarı] .docx dosyası okunamadı ({file_path}): {e}")
            return []

    def _parse_guideline_file(self, file_name: str) -> List[Dict[str, Any]]:
        """Bir dosyayı okuyup anlamlı bilgi parçalarına (chunks) dönüştürür."""
        file_path = os.path.join(self.guidelines_dir, file_name)
        chunks = []

        # A) Markdown (.md) veya Metin (.txt) Dosyaları
        if file_name.endswith(".md") or file_name.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            sections = re.split(r"\n(?=##\s+)", content)
            for i, sec in enumerate(sections):
                sec = sec.strip()
                if not sec:
                    continue
                lines = sec.split("\n", 1)
                header = lines[0].replace("#", "").strip()
                body = lines[1].strip() if len(lines) > 1 else ""

                if ":" in header:
                    chunk_id, title = header.split(":", 1)
                    chunk_id = chunk_id.strip()
                    title = title.strip()
                else:
                    clean_name = os.path.splitext(file_name)[0].upper()
                    chunk_id = f"{clean_name}_P{i+1}"
                    title = header if header else f"{file_name} Bölüm {i+1}"

                if body:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "title": title,
                        "text": body,
                        "source": file_name
                    })

        # B) Word (.docx) Dosyaları
        elif file_name.endswith(".docx"):
            paras = self._extract_docx_paragraphs(file_path)
            clean_name = os.path.splitext(file_name)[0].replace(" ", "_").upper()
            
            # Paragrafları 2-3'lü mantıksal gruplar halinde chunk yap
            chunk_size = 3
            for i in range(0, len(paras), chunk_size):
                group = paras[i:i + chunk_size]
                chunk_id = f"{clean_name[:12]}_P{(i // chunk_size) + 1}"
                title = group[0][:60]
                text = "\n".join(group)
                chunks.append({
                    "chunk_id": chunk_id,
                    "title": title,
                    "text": text,
                    "source": file_name
                })

        return chunks

    def index_documents(self):
        """Klasördeki tüm makaleleri okur, ChromaDB'ye ve BM25'e kaydeder."""
        if not os.path.exists(self.guidelines_dir):
            return

        all_chunks = []
        for fname in os.listdir(self.guidelines_dir):
            all_chunks.extend(self._parse_guideline_file(fname))

        # ID Çakışmalarını %100 Önle (Her chunk için tekil kimlik garantisi)
        seen_ids = set()
        for c in all_chunks:
            cid = c["chunk_id"]
            counter = 1
            orig_cid = cid
            while cid in seen_ids:
                cid = f"{orig_cid}_{counter}"
                counter += 1
            c["chunk_id"] = cid
            seen_ids.add(cid)

        self.chunks = all_chunks

        # ChromaDB Vektör Veritabanı İndekslemesi
        if self.chunks:
            ids = [c["chunk_id"] for c in self.chunks]
            documents = [f"{c['title']}\n{c['text']}" for c in self.chunks]
            metadatas = [{"title": c["title"], "source": c["source"]} for c in self.chunks]

            # Upsert ile ekle veya güncelle
            self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

            # BM25 Sparse İndekslemesi
            tokenized_corpus = [self._tokenize(doc) for doc in documents]
            self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Sorulan soruya en uygun makale parçalarını hibrit arama (Dense + Sparse + RRF) ile getirir.
        """
        k = top_k or self.top_k
        if not self.chunks:
            return []

        rrf_scores = {c["chunk_id"]: 0.0 for c in self.chunks}
        chunk_map = {c["chunk_id"]: c for c in self.chunks}

        # 1. Dense (Vektörel) Arama
        try:
            dense_res = self.collection.query(query_texts=[query], n_results=min(len(self.chunks), k * 2))
            if dense_res and "ids" in dense_res and dense_res["ids"]:
                for rank, cid in enumerate(dense_res["ids"][0]):
                    if cid in rrf_scores:
                        rrf_scores[cid] += 1.0 / (60.0 + rank + 1)
        except Exception as e:
            print(f"[Dense Search Hatası]: {e}")

        # 2. Sparse (BM25 Anahtar Kelime) Arama
        if self.bm25:
            tokens = self._tokenize(query)
            scores = self.bm25.get_scores(tokens)
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            for rank, idx in enumerate(sorted_indices[:k * 2]):
                cid = self.chunks[idx]["chunk_id"]
                rrf_scores[cid] += 1.0 / (60.0 + rank + 1)

        # 3. RRF Skoruna Göre Sıralama
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for cid, score in sorted_results[:k]:
            item = chunk_map[cid]
            results.append({
                "chunk_id": item["chunk_id"],
                "title": item["title"],
                "text": item["text"],
                "source": item["source"],
                "score": round(score, 4)
            })

        return results
