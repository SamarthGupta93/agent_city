import json
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from agents.simple_rag.agent import ConversationalRAG
from dotenv import load_dotenv

load_dotenv()


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    response: str
    doc_ids: list[str]


agent: ConversationalRAG = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    agent = ConversationalRAG()
    yield
    agent.close()


app = FastAPI(title="Simple RAG API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    result = agent.chat(request.session_id, request.message)
    return ChatResponse(response=result["response"], doc_ids=result["doc_ids"])


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    async def generate():
        async for event in agent.astream(request.session_id, request.message):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
