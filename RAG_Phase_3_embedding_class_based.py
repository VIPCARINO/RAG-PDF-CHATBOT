from sentence_transformers import SentenceTransformer
import json
import time


class EmbeddingPipeline:

    def __init__(
        self,
        input_data,
        model_path="./models/all-MiniLM-L6-v2",
        status_callback=None,
        doc_id=None
    ):

        self.model_path = model_path
        self.input_data = input_data
        self.doc_id = doc_id
        self.start_time = None

        self.model = None
        self.embedded_vectors = []

        self.vector_id = 0

        # =====================================================
        # STATUS CALLBACK
        # =====================================================
        self.status = (
            status_callback
            if status_callback
            else lambda x: None
        )

    # =========================================================
    # LOAD MODEL
    # =========================================================
    def load_model(self):

        self.status({
                        "stage": "model_loading",
                        "progress": 0.5,
                        "processed": 0,
                        "total": 1,
                        "speed": 0,
                        "eta": None,
                        "message": "Loading embedding model..."
                    })

        self.model = SentenceTransformer(
            self.model_path
        )

        self.status({
            "stage": "model_loaded",
            "progress": 1.0,
            "message": "Embedding model loaded"
        })

    # =========================================================
    # EMBED SINGLE TEXT
    # =========================================================
    def embed_text(self, text):

        return self.model.encode(
            text,
            normalize_embeddings=True
        )

    # =========================================================
    # MAIN EMBEDDING PROCESS
    # =========================================================
    def run(self):

        self.start_time = time.time()

        if self.model is None:
            self.load_model()

        chunks = self.input_data.get(
            "vector_chunks",
            []
        )

        if not chunks:

            self.status({
                "stage": "embedding_warning",
                "progress": 0.0,
                "message": "No vector chunks found!"
            })

            return None

        total = len(chunks)

        # =====================================================
        # EMBEDDING LOOP
        # =====================================================
        self.status({
                            "stage": "embedding_start",
                            "progress": 0.0,
                            "processed": 0,
                            "total": total,
                            "speed": 0,
                            "eta": None,
                            "message": f"Starting embeddings ({total} chunks)"
                        })
        for idx, item in enumerate(chunks, start=1):

            text = item["text"]
            embedding = self.embed_text(text)

            m = self.compute_metrics(self.start_time, idx, total)

            self.status({
                "stage": "embedding",
                "progress": m["progress"],
                "processed": idx,
                "total": total,
                "speed": m["speed"],
                "eta": m["eta"],
                "message": f"Embedding chunk {idx}/{total}"
            })

            self.embedded_vectors.append({
                "vector_id": f"{self.doc_id}_{self.vector_id}",
                "doc_id": self.doc_id,
                "chunk_id": item["chunk_id"],
                "subchunk_index": item["subchunk_index"],
                "title": item["title"],
                "subtitle": item["subtitle"],
                "pages": item["pages"],
                "text": text,
                "embedding": embedding.tolist()
            })

            self.vector_id += 1

        # =====================================================
        # FINISHED
        # =====================================================

        self.status({
                        "stage": "embedding_done",
                        "progress": 1.0,
                        "processed": total,
                        "total": total,
                        "speed": 0,
                        "eta": 0,
                        "message": f"Embedding complete ({len(self.embedded_vectors)} vectors)"
                    })

        return self.export()

    # =========================================================
    # EXPORT
    # =========================================================
    def export(self):

        return {
            "document": self.input_data.get(
                "document",
                "unknown"
            ),
            "embeddings": self.embedded_vectors
        }

    # =========================================================
    # SAVE TO FILE
    # =========================================================
    def save(
        self,
        output_file="./json_files/rag_embeddings.json"
    ):

        data = self.export()

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(data, f, indent=2)

        self.status({
            "stage": "save",
            "progress": 1.0,
            "message": f"Saved embeddings to {output_file}"
        })

    def compute_metrics(self, start_time, processed, total):

        elapsed = time.time() - start_time

        speed = processed / elapsed if elapsed > 0 else 0

        remaining = total - processed

        # avoid noisy early ETA
        if elapsed < 2:
            eta = None
        else:
            eta = remaining / speed if speed > 0 else None

        progress = processed / total if total > 0 else 0

        return {
            "progress": round(progress, 4),
            "speed": round(speed, 2),
            "eta": round(eta, 2) if eta is not None else None,
            "elapsed": round(elapsed, 2)
        }