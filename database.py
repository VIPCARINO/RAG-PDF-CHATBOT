import sqlite3
from datetime import datetime


class MetadataDB:

    def __init__(self, db_name="./database/metadata.db"):

        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()

        self.create_tables()

    # =====================================================
    # CREATE TABLES
    # =====================================================
    def create_tables(self):

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            filename TEXT,
            collection_name TEXT,
            upload_date TEXT,
            total_pages INTEGER,
            summary TEXT
        )
        """)

        self.conn.commit()

    # =====================================================
    # CHECKING IF FILE ALREADY EXIST
    # =====================================================
    def document_exists_by_name(self, filename):

        self.cursor.execute("""
        SELECT doc_id, filename 
        FROM documents 
        WHERE filename = ?
        """, (filename,))

        return self.cursor.fetchone()

    # =====================================================
    # ADD DOCUMENT
    # =====================================================
    def add_document(
        self,
        doc_id,
        filename,
        collection_name="pdf_rag",
        total_pages=0,
        summary=""
    ):

        upload_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.cursor.execute("""
        INSERT OR REPLACE INTO documents (
            doc_id,
            filename,
            collection_name,
            upload_date,
            total_pages,
            summary
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            doc_id,
            filename,
            collection_name,
            upload_date,
            total_pages,
            summary
        ))

        self.conn.commit()

    # =====================================================
    # GET ALL DOCUMENTS
    # =====================================================
    def get_all_documents(self):

        self.cursor.execute("""
        SELECT * FROM documents
        ORDER BY upload_date DESC
        """)

        return self.cursor.fetchall()

    # =====================================================
    # GET SINGLE DOCUMENT
    # =====================================================
    def get_document(self, doc_id):

        self.cursor.execute("""
        SELECT * FROM documents
        WHERE doc_id=?
        """, (doc_id,))

        return self.cursor.fetchone()

    # =====================================================
    # DELETE DOCUMENT
    # =====================================================
    def delete_document(self, doc_id):

        self.cursor.execute("""
        DELETE FROM documents
        WHERE doc_id=?
        """, (doc_id,))

        self.conn.commit()
    
    def link_chroma_collection(self, doc_id, collection_name):
        self.cursor.execute("""
        UPDATE documents
        SET collection_name = ?
        WHERE doc_id = ?
        """, (collection_name, doc_id))
        self.conn.commit()

    # =====================================================
    # CLOSE
    # =====================================================
    def close(self):

        self.conn.close()