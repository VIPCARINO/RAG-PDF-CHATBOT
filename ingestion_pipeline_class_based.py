from RAG_Phase_1_pdf_extraction_structuring_class_based import PDFStructurer
from RAG_Phase_2_chunking_class_based import RAGVectorBuilder
from RAG_Phase_3_embedding_class_based import EmbeddingPipeline
from RAG_Phase_4_adding_to_db_class_based import ChromaIndexer
from database import MetadataDB


class PDFIngestionPipeline:

    def __init__(
        self,
        pdf_path,
        model_path="./models/all-MiniLM-L6-v2",
        db_path="./database/chroma_db",
        collection_name="pdf_rag",
        doc_id=None,
        filename=None,

        # 👇 NEW MULTI CALLBACKS
        extract_callback=None,
        structure_callback=None,
        chunk_callback=None,
        embed_callback=None,
        db_callback=None
    ):

        self.pdf_path = pdf_path
        self.model_path = model_path
        self.db_path = db_path
        self.collection_name = collection_name
        self.doc_id = doc_id
        self.filename = filename

        self.db = MetadataDB()

        # ================================
        # MULTI-STAGE CALLBACKS
        # ================================
        self.callbacks = {
            "extract": extract_callback or (lambda x: None),
            "structure": structure_callback or (lambda x: None),
            "chunk": chunk_callback or (lambda x: None),
            "embed": embed_callback or (lambda x: None),
            "db": db_callback or (lambda x: None),
        }

    # =====================================================
    # FULL INGESTION PIPELINE
    # =====================================================
    def run(self):

        doc_id = self.doc_id

        # -----------------------------
        # STEP 1: PDF STRUCTURE
        # -----------------------------
        self.callbacks["extract"]({
                                    "stage": "extract_start",
                                    "progress": 0.0,
                                    "processed": 0,
                                    "total": 1,
                                    "speed": 0,
                                    "eta": None,
                                    "message": "Starting PDF parsing..."
                                })

        parser = PDFStructurer(
            self.pdf_path,
            status_callback=self.callbacks["extract"]
        )

        parsed_data = parser.run()

        self.db.add_document(
            doc_id=doc_id,
            filename=self.filename,
            collection_name=self.collection_name,
            total_pages=len(parsed_data.get("chunks", []))
        )

        # -----------------------------
        # STEP 2: CHUNKING
        # -----------------------------
        self.callbacks["chunk"]({
                                    "stage": "chunk_start",
                                    "progress": 0.0,
                                    "processed": 0,
                                    "total": 1,
                                    "speed": 0,
                                    "eta": None,
                                    "message": "Starting chunking..."
                                })

        chunker = RAGVectorBuilder(
            parsed_data, 
            doc_id=doc_id,
            status_callback=self.callbacks["chunk"]
        )

        rag_output = chunker.run()

        # -----------------------------
        # STEP 3: EMBEDDINGS
        # -----------------------------
        self.callbacks["embed"]({
                                    "stage": "embed_start",
                                    "progress": 0.0,
                                    "processed": 0,
                                    "total": 1,
                                    "speed": 0,
                                    "eta": None,
                                    "message": "Generating embeddings..."
                                })

        embedder = EmbeddingPipeline(
            input_data=rag_output,
            doc_id=doc_id,
            status_callback=self.callbacks["embed"]
        )

        embeddings = embedder.run()

        # -----------------------------
        # STEP 4: CHROMA INDEXING
        # -----------------------------
        self.callbacks["db"]({
                                "stage": "db_start",
                                "progress": 0.0,
                                "processed": 0,
                                "total": 1,
                                "speed": 0,
                                "eta": None,
                                "message": "Indexing into ChromaDB..."
                            })

        indexer = ChromaIndexer(
            db_path=self.db_path,
            collection_name=self.collection_name,
            doc_id=doc_id,
            status_callback=self.callbacks["db"]
        )

        indexer.run(embeddings)

        self.db.link_chroma_collection(
            doc_id=doc_id,
            collection_name=self.collection_name
        )

        self.callbacks["db"]({
                                "stage": "complete",
                                "progress": 1.0,
                                "processed": 1,
                                "total": 1,
                                "speed": 0,
                                "eta": 0,
                                "message": "Pipeline complete!"
                            })

        return embeddings
    


# ---------------------------------------------------------------------
# HOW TO USE
# ---------------------------------------------------------------------

"""
pipeline = PDFIngestionPipeline(
    pdf_path="book.pdf",
    collection_name="my_book"
)

pipeline.run()
"""