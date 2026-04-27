from retriever.doc_processor import DocumentChunkingPipeline, DocumentLoaderPipeline
from retriever.indexer import DocumentIndexingPipeline
from rag.agent import SimpleRAG
from agents.simple_rag.agent import ConversationalRAG
from dotenv import load_dotenv
import uuid
load_dotenv()

def index_documents():
    """
    Load all pdf files from the RAW_DOCUMENTS_DIR, 
    extract their content and metadata, and save them as text files in the TEXT_LOADER_SAVE_DIR.
    """
    # Load documents from the RAW_DOCUMENTS_DIR
    loader = DocumentLoaderPipeline()
    loader.run()

    # Chunk documents
    chunker = DocumentChunkingPipeline()
    docs = chunker.run()

    # Index documents in vector store
    indexer = DocumentIndexingPipeline()
    indexer.add_documents()

def main(query: str):
    # Run RAG
    rag = SimpleRAG()
    prompt, response = rag.run(query)
    print("Prompt:\n", prompt)
    print("Response:\n", response)

def converse(session_id: str = None):
    session_id = session_id or str(uuid.uuid4())
    print(f"Session: {session_id}")
    print("Type 'exit' to quit.\n")
    agent = ConversationalRAG()
    try:
        while True:
            query = input("You: ").strip()
            if query.lower() == "exit":
                break
            if not query:
                continue
            response = agent.chat(session_id, query)
            print(f"Assistant: {response}\n")
    finally:
        agent.close()

if __name__ == "__main__":
    converse()