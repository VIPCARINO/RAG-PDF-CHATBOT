import chromadb
import time


class ChromaIndexer:

    def __init__(
        self,
        db_path="./database/chroma_db",
        collection_name="pdf_rag",
        status_callback=None,
        doc_id=None,
        batch_size=4000   # 👈 SAFE DEFAULT (below chroma limit)
    ):

        self.doc_id = doc_id
        self.batch_size = batch_size

        # =====================================================
        # STATUS CALLBACK
        # =====================================================
        self.status = (
            status_callback
            if status_callback
            else lambda x: None
        )

        # =====================================================
        # INIT CHROMA DB
        # =====================================================
        self.status({
            "stage": "db_init",
            "progress": 0.0,
            "message": "Initializing ChromaDB..."
        })

        self.client = chromadb.PersistentClient(
            path=db_path
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        )

        self.embedded_vectors = []

        self.ids = []
        self.embeddings = []
        self.documents = []
        self.metadatas = []

        self.status({
            "stage": "db_ready",
            "progress": 1.0,
            "message": "ChromaDB ready"
        })

    # =========================================================
    # LOAD DATA FROM EMBEDDING PIPELINE
    # =========================================================
    def load_vectors(self, embedded_vectors):

        self.embedded_vectors = embedded_vectors

        self.status({
            "stage": "load_vectors",
            "progress": 0.0,
            "message": f"Loaded {len(self.embedded_vectors)} vectors"
        })

    # =========================================================
    # PREPARE DATA FOR CHROMA
    # =========================================================
    def prepare(self):

        total = len(self.embedded_vectors)

        self.status({
            "stage": "prepare_start",
            "progress": 0.0,
            "message": f"Preparing {total} vectors for indexing"
        })

        for idx, item in enumerate(self.embedded_vectors, start=1):

            self.status({
                "stage": "prepare",
                "progress": round(idx / total, 3),
                "message": f"Preparing vector {idx}/{total}"
            })

            self.ids.append(str(item["vector_id"]))
            self.embeddings.append(item["embedding"])
            self.documents.append(item["text"])

            self.metadatas.append({
                "doc_id": item["doc_id"],
                "chunk_id": item["chunk_id"],
                "subchunk_index": item["subchunk_index"],
                "title": item["title"],
                "subtitle": item["subtitle"],
                "pages": ",".join(map(str, item["pages"]))
            })

        self.status({
            "stage": "prepare_done",
            "progress": 1.0,
            "message": f"Prepared {len(self.ids)} vectors"
        })

    # =========================================================
    # 🔥 BATCH INSERT (FIX FOR YOUR ERROR)
    # =========================================================
    def index(self):

        if not self.ids:
            raise ValueError("No data prepared. Run prepare() first.")

        total = len(self.ids)
        start_time = time.time()

        self.status({
            "stage": "indexing",
            "progress": 0.0,
            "percent": 0,
            "speed": 0,
            "eta": None,
            "message": f"Indexing {total} vectors in batches..."
        })

        for i in range(0, total, self.batch_size):

            end = min(i + self.batch_size, total)

            batch_ids = self.ids[i:end]
            batch_embeddings = self.embeddings[i:end]
            batch_documents = self.documents[i:end]
            batch_metadatas = self.metadatas[i:end]

            processed = end

            m = self.compute_metrics(start_time, processed, total)

            self.status({
                "stage": "indexing_batch",
                "progress": m["progress"],
                "percent": m["percent"],
                "speed": m["speed"],
                "eta": m["eta"],
                "message": f"Batch {(i // self.batch_size) + 1} | {len(batch_ids)} items"
            })

            self.collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas
            )

        final = self.compute_metrics(start_time, total, total)

        self.status({
            "stage": "index_done",
            "progress": 1.0,
            "percent": 100,
            "speed": final["speed"],
            "eta": 0,
            "message": f"Indexed {total} vectors successfully"
        })

    # =========================================================
    # FULL PIPELINE
    # =========================================================
    def run(self, data):

        if isinstance(data, dict):
            data = data.get("embeddings", [])

        self.load_vectors(data)
        self.prepare()
        self.index()

        return self.collection
    
    def compute_metrics(self, start_time, processed, total):

        elapsed = time.time() - start_time

        speed = processed / elapsed if elapsed > 0 else 0

        remaining = total - processed

        eta = remaining / speed if speed > 0 else 0

        progress = processed / total if total > 0 else 0

        return {
            "progress": round(progress, 4),
            "percent": round(progress * 100, 2),
            "speed": round(speed, 2),
            "eta": round(eta, 2),
            "elapsed": round(elapsed, 2)
        }