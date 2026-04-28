import json
import os
import psycopg
from collections.abc import AsyncGenerator
from typing import Annotated
from typing_extensions import TypedDict
import constants
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres import PostgresSaver


class State(TypedDict):
    messages: Annotated[list, add_messages]
    context: str
    doc_ids: list[str]


class ConversationalRAG:
    """
    Conversational RAG agent backed by LangGraph with PostgreSQL session persistence.
    Each session is identified by a session_id (thread_id). Conversation history
    is stored in Postgres and automatically restored on subsequent calls.
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
        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash-lite"),
            temperature=0.3,
        )
        self._prompt_template = open(
            os.path.join(os.path.dirname(__file__), "agent.md")
        ).read()

        # PostgresSaver needs a plain postgresql:// URI (not postgresql+psycopg://)
        db_uri = os.getenv("POSTGRES_SESSION_URI")
        self._conn = psycopg.connect(db_uri, autocommit=True)
        self._checkpointer = PostgresSaver(self._conn)
        self._checkpointer.setup()
        self._graph = self._build_graph()

    def _retrieve(self, state: State) -> dict:
        query = state["messages"][-1].content
        docs = self.vector_store.similarity_search(query, k=constants.TOPK_DOCS)
        doc_ids = [doc.metadata.get("id", "") for doc in docs]
        context = json.dumps(
            [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "title": doc.metadata.get("title", "unknown"),
                }
                for doc in docs
            ],
            indent=2,
        )
        return {"context": context, "doc_ids": doc_ids}

    def _generate(self, state: State) -> dict:
        history = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in state["messages"][:-1]
        )
        prompt = self._prompt_template.format(
            context=state["context"],
            history=history or "No previous conversation.",
            query=state["messages"][-1].content,
        )
        response = self.llm.invoke(prompt)
        return {"messages": [response]}

    def _build_graph(self):
        graph = StateGraph(State)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("generate", self._generate)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)
        return graph.compile(checkpointer=self._checkpointer)

    def chat(self, session_id: str, message: str) -> dict:
        config = {"configurable": {"thread_id": session_id}}
        result = self._graph.invoke(
            {"messages": [HumanMessage(content=message)]},
            config=config,
        )
        return {
            "response": result["messages"][-1].content,
            "doc_ids": result.get("doc_ids", []),
        }

    async def astream(self, session_id: str, message: str) -> AsyncGenerator[dict, None]:
        config = {"configurable": {"thread_id": session_id}}
        doc_ids_sent = False
        async for event_type, data in self._graph.astream(
            {"messages": [HumanMessage(content=message)]},
            config=config,
            stream_mode=["values", "messages"],
        ):
            if event_type == "values" and not doc_ids_sent and data.get("doc_ids"):
                yield {"type": "doc_ids", "doc_ids": data["doc_ids"]}
                doc_ids_sent = True
            elif event_type == "messages":
                chunk, metadata = data
                if metadata.get("langgraph_node") == "generate" and chunk.content:
                    yield {"type": "token", "content": chunk.content}

    def close(self):
        self._conn.close()
