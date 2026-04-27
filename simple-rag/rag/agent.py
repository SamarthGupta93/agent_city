from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
import json
import os



class SimpleRAG:
    """
    Simple RAG pipeline that retrieves relevant documents from the vector store based on the query,
    constructs a prompt with the retrieved documents as context, and generates a response using the LLM.
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
        self._prompt_template = open(os.path.join(os.path.dirname(__file__), "agent.md")).read()

    def run(self, query: str) -> str:
        # Retriever
        relevant_docs = self.vector_store.similarity_search(query, k=3)
        # Context
        context = json.dumps([
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "title": doc.metadata.get("title", "unknown"),
            }
            for doc in relevant_docs
        ], indent=2)
        prompt = self._prompt_template.format(context=context, query=query)

        # Generator
        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash-lite"),
            temperature=0.3,
        )
        return prompt, llm.invoke(prompt).content