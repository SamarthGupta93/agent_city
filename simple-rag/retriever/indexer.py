"""
Load chunked documents, generate embeddings, and index them in vector store (PGVector).
Do no index duplicate content - use metadata to check for duplicates before indexing.
"""

import os
import constants
import glob
import json
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document
from logger import log
from tqdm import tqdm


class DocumentIndexingPipeline:
    """
    Loads chunked documents from the CHUNKS_DIR, generates embeddings for each chunk, 
    and indexes them in the vector store (PGVector).
    """
    def __init__(self):
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model=os.getenv("GOOGLE_EMBEDDING_MODEL"),
        )
        self.vector_store = PGVector(
            embeddings=self.embedding_model,
            collection_name=os.getenv("COLLECTION_NAME"),
            connection=os.getenv("VECTOR_DB_URI"),
            use_jsonb=True,
        )


    def add_documents(self):
        paths = glob.glob(f"{constants.CHUNKS_DIR}/*.txt")
        log.info(f"Found {len(paths)} chunk file(s) to index.")
        documents = []
        for filepath in tqdm(paths, desc="Loading chunks", unit="chunk"):
            document = self._load_chunked_documents(filepath)
            documents.append(document)
        log.info(f"Indexing {len(documents)} chunk(s) into vector store...")
        ids = [doc.metadata["id"] for doc in documents]
        self.vector_store.add_documents(documents, ids=ids)
        log.info(f"Indexing complete. {len(documents)} chunk(s) upserted.")


    def _load_chunked_documents(self, content_path: str) -> Document:
        """Load all chunked documents from the CHUNKS_DIR."""
        metadata_path = content_path.replace(".txt", "_metadata.json")
        with open(content_path, "r") as content_file:
            content = content_file.read().replace('\x00', '')
        with open(metadata_path, "r") as metadata_file:
            metadata = json.load(metadata_file)
        return Document(page_content=content, metadata=metadata)