import chromadb

def delete_from_chroma(db_path, collection_name, doc_id):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(collection_name)

    # delete all vectors for that doc_id
    collection.delete(where={"doc_id": doc_id})