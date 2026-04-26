import os
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI


llm = ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash-lite"),
        temperature=0.3,
    )

def get_weather(location: str) -> str:
    """Mock weather tool that returns a fake forecast for the given location."""
    return f"The current weather in {location} is sunny with a high of 25°C."

agent = create_agent(
    model=llm,
    system_prompt="""
    You are a helpful conversational assistant.
    """,
    tools=[get_weather],
)