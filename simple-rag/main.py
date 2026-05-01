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

def ask(query: str):
    """
    run the RAG pipeline for a given query: retrieve relevant documents from the vector store,
    construct a prompt with the retrieved documents as context, and generate a response using the LLM
    """
    # Run RAG
    rag = SimpleRAG()
    prompt, response = rag.run(query)
    print("Prompt:\n", prompt)
    print("Response:\n", response)

def converse(session_id: str = None):
    """
    Chat with an assistant that uses the RAG pipeline to retrieve relevant documents and generate responses 
    based on the conversation history and retrieved context. 
    Each conversation is associated with a unique session ID for state management.
    Args:
        session_id (str, optional): A unique identifier for the conversation session. 
            If not provided, a new UUID will be generated.
    """
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
    import argparse

    parser = argparse.ArgumentParser(description="Simple RAG application")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("index", help="Load, chunk, and index documents into the vector store")

    converse_parser = subparsers.add_parser("converse", help="Start an interactive RAG conversation")
    converse_parser.add_argument("--session-id", default=None, help="Resume an existing session by ID")

    args = parser.parse_args()

    if args.command == "index":
        index_documents()
    elif args.command == "converse":
        converse(session_id=args.session_id)