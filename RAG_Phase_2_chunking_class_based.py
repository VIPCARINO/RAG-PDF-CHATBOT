import re
import time


class RAGVectorBuilder:

    def __init__(self, final_output, max_words=120, overlap=30, status_callback=None, doc_id=None):

        self.final_output = final_output
        self.max_words = max_words
        self.overlap = overlap

        self.doc_id = doc_id 

        self.vector_store = []
        self.vector_id = 0

        self.status = ( status_callback if status_callback else lambda x: None )

    # =========================================================
    # SENTENCE SPLITTER
    # =========================================================
    def split_sentences(self, text):

        return re.split(r'(?<=[.!?])\s+', text.strip())

    # =========================================================
    # SUBCHUNK CREATION (OVERLAP-AWARE)
    # =========================================================
    def create_subchunks(self, text):

        sentences = self.split_sentences(text)

        chunks = []
        current = []
        word_count = 0

        for sentence in sentences:

            words = len(sentence.split())

            # ------------------ SPLIT CONDITION ------------------
            if word_count + words > self.max_words:

                chunks.append(" ".join(current))

                # ------------------ OVERLAP ------------------
                overlap_chunk = []
                overlap_words = 0

                for s in reversed(current):

                    w = len(s.split())

                    if overlap_words + w > self.overlap:
                        break

                    overlap_chunk.insert(0, s)
                    overlap_words += w

                current = overlap_chunk
                word_count = overlap_words

            current.append(sentence)
            word_count += words

        if current:
            chunks.append(" ".join(current))

        return chunks

    # =========================================================
    # BUILD VECTOR STORE
    # =========================================================
    def build(self):

        chunks = self.final_output["chunks"]
        total = len(chunks)

        start_time = time.time()

        self.status({
            "stage": "chunking_start",
            "progress": 0.0,
            "processed": 0,
            "total": total,
            "speed": 0,
            "eta": None,
            "message": f"Starting chunking ({total} sections)"
        })

        for idx, chunk in enumerate(chunks, start=1):

            # -----------------------------
            # METRICS (CLEAN CALL)
            # -----------------------------
            m = self.compute_metrics(start_time, idx, total)

            self.status({
                "stage": "chunking",
                "progress": m["progress"],
                "processed": idx,
                "total": total,
                "speed": m["speed"],
                "eta": m["eta"],
                "message": f"Processing section {idx}/{total}"
            })

            # -----------------------------
            # SUBCHUNKS
            # -----------------------------
            subchunks = self.create_subchunks(chunk["content"])
            sub_total = len(subchunks)
            sub_start = time.time()

            for i, sub in enumerate(subchunks):

                sub_m = self.compute_metrics(sub_start, i + 1, sub_total)

                self.vector_store.append({
                    "vector_id": self.vector_id,
                    "doc_id": self.doc_id,

                    "chunk_id": chunk["chunk_id"],
                    "title": chunk["title"],
                    "subtitle": chunk["subtitle"],
                    "pages": chunk["pages"],

                    "subchunk_index": i,

                    "text": f"""
    TITLE: {chunk['title']}
    SUBTITLE: {chunk['subtitle']}

    {sub}
    """.strip()
                })

                self.vector_id += 1

                self.status({
                    "stage": "subchunking",
                    "progress": sub_m["progress"],
                    "processed": i + 1,
                    "total": sub_total,
                    "speed": sub_m["speed"],
                    "eta": sub_m["eta"],
                    "message": f"Subchunk {i+1}/{sub_total}"
                })

        # -----------------------------
        # DONE
        # -----------------------------
        final_metrics = self.compute_metrics(start_time, total, total)

        self.status({
            "stage": "chunking_done",
            "progress": 1.0,
            "processed": len(self.vector_store),
            "total": len(self.vector_store),
            "speed": final_metrics["speed"],
            "eta": 0,
            "message": f"Chunking complete ({len(self.vector_store)} vectors)"
        })

    def compute_metrics(self,start_time, processed, total):
        """
        Computes speed, ETA, and progress for pipeline tracking.
        """

        elapsed = time.time() - start_time

        speed = processed / elapsed if elapsed > 0 else 0

        remaining = total - processed
        eta = remaining / speed if speed > 0 else 0

        progress = processed / total if total > 0 else 0

        return {
                "progress": round(progress, 4),
                "speed": round(speed, 2),
                "eta": round(eta, 2),
                "elapsed": round(elapsed, 2)
            }

    # =========================================================
    # EXPORT RAG OUTPUT
    # =========================================================
    def export(self):

        return {
            "document": self.final_output.get("document", "unknown"),
            "vector_chunks": self.vector_store
        }

    # =========================================================
    # FULL PIPELINE
    # =========================================================
    def run(self):

        self.build()
        return self.export()