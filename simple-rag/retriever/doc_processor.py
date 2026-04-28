"""
This file loads and maintains documents in a vector database.
"""
import json
import os
import glob
import uuid
from logger import log
from typing import Literal
import constants
from retriever.models import Document
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunkingPipeline:
    """
    Chunk the documents under TEXT_LOADER_SAVE_DIR into smaller pieces and save them in CHUNKS_DIR.
    """
    def __init__(self, chunk_size: int=1000, chunk_overlap: int=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def run(self, paths: list[str]=None) -> list[Document]:
        """Load documents from the TEXT_LOADER_SAVE_DIR."""
        chunks = []
        if paths is None:
            paths = glob.glob(os.path.join(constants.TEXT_LOADER_SAVE_DIR, "*_content.txt"))
            log.info(f"Found {len(paths)} documents to chunk.")
        for filepath in paths:
            document = self._load_text_document(filepath)
            chunked_documents = self.recursive_character_text_splitter(document)
            chunks.extend(chunked_documents)
        self._save_chunked_documents(chunks)
        return chunks
    

    def _save_chunked_documents(self, documents: list[Document]):
        """Save the chunked documents to the CHUNKS_DIR."""
        os.makedirs(constants.CHUNKS_DIR, exist_ok=True)
        for doc in documents:
            name = doc.metadata["source"].split("/")[-1].rsplit(".")[0]
            chunk_index = doc.metadata["chunk_index"]
            chunk_path = f"{constants.CHUNKS_DIR}/{name}_chunk_{chunk_index}.txt"
            chunk_metadata_path = f"{constants.CHUNKS_DIR}/{name}_chunk_{chunk_index}_metadata.json"
            with open(chunk_path, "w") as chunk_file:
                chunk_file.write(doc.content)
            with open(chunk_metadata_path, "w") as chunk_metadata_file:
                json.dump(doc.metadata, chunk_metadata_file)

    
    def _load_text_document(self, content_path: str) -> Document:
        """Load the intermediate text document and its metadata."""
        metadata_path = content_path.replace("_content.txt", "_metadata.json")
        with open(content_path, "r") as content_file:
            content = content_file.read()
        with open(metadata_path, "r") as metadata_file:
            metadata = json.load(metadata_file)
        return Document(content, metadata)

    def recursive_character_text_splitter(self, document: Document) -> list[Document]:
        # Implementation for chunking the document content
        texts = self.text_splitter.create_documents([document.content])
        chunks = []
        for i, text in enumerate(texts):
            if text.page_content.strip() == "":
                # Skip empty chunks
                continue
            chunk_metadata = document.metadata.copy()
            chunk_metadata["chunk_index"] = i
            # Generate a unique ID for each chunk. This should be deterministic
            chunk_metadata["id"] = str(uuid.uuid5(uuid.NIL, document.metadata["source"] + str(i)))
            chunks.append(Document(content=text.page_content, metadata=chunk_metadata))
        return chunks


class DocumentLoaderPipeline:
    """
    Loads documents from the RAW_DOCUMENTS_DIR, extracts their content and metadata, 
    and saves them as text files in the TEXT_LOADER_SAVE_DIR.
    """
    def __init__(self, raw_dir: str=None, text_save_dir: str=None):
        self.raw_dir = raw_dir if raw_dir else constants.RAW_DOCUMENTS_DIR
        self.text_save_dir = text_save_dir if text_save_dir else constants.TEXT_LOADER_SAVE_DIR
        os.makedirs(self.text_save_dir, exist_ok=True)

    def run(self):
        documents = self.load()
        self.save(documents)
        return documents

    def load(self):
        documents = []

        for filename in os.listdir(self.raw_dir):
            if filename.endswith(".pdf"):
                doc_path = os.path.join(self.raw_dir, filename)
                documents.append(self._load_document(doc_path))

        return documents


    def save(self, documents: list[Document]):
        """Save the content and metadata of the documents to the specified directory."""
        for _, doc in enumerate(documents):
            name = doc.metadata["source"].split("/")[-1].rsplit(".")[0]
            content_path = f"{self.text_save_dir}/{name}_content.txt"
            metadata_path = f"{self.text_save_dir}/{name}_metadata.json"
            with open(content_path, "w") as content_file:
                content_file.write(doc.content)
            with open(metadata_path, "w") as metadata_file:
                json.dump(doc.metadata, metadata_file)


    def _load_document(self, path: str) -> Document:
        """Load the content of the document based on its file format."""
        file_format = path.split(".")[-1].lower()
        if file_format == "pdf":
            docs = self._load_pdf(path)
            return Document(content=docs[0].page_content.replace('\x00', ''), metadata=docs[0].metadata)
        else:
            # Print a warning message and skip the file if the format is unsupported
            log.warning(f"Unsupported file format. Only .pdf files are supported. Skipping file: {path}")


    def _load_pdf(self, path, loader: Literal["pymupdf", "pypdf"]=constants.PYMUPDF) -> str:
        """Load the content of the document using the specified loader."""
        if loader == "pymupdf":
            loader = PyMuPDFLoader(path, mode="single", extract_tables="markdown")
        elif loader == "pypdf":
            loader = PyPDFLoader(path)
        else:
            raise ValueError(f"Unsupported loader: {loader}")
        return loader.load()

    