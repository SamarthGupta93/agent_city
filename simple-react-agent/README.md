# Simple ReAct Agent

A minimal LangGraph agent powered by Google Gemini that demonstrates the ReAct (Reason + Act) pattern using the LangGraph CLI deployment workflow.

## What it does

The agent (`weather_agent`) is a conversational assistant with a mock weather tool. Given a location, it reasons about the request and calls the tool to return a forecast. It serves as a reference implementation for deploying LangGraph agents locally and to production.

## Project structure

```
simple-react-agent/
├── agents/
│   └── simple_react/
│       └── agent.py       # Agent definition and tools
├── langgraph.json          # LangGraph config (graph, env, checkpointer)
├── pyproject.toml          # Dependencies
└── .env                    # Environment variables (not committed)
```

## Setup

**Prerequisites:** Python ≥ 3.11, [uv](https://docs.astral.sh/uv/), Docker (for `langgraph up`)

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Create a `.env` file:
   ```
   GOOGLE_API_KEY=your_key_here
   GOOGLE_MODEL=gemini-2.5-flash-lite
   ```

## Running

| Command | Description |
|---|---|
| `uv run langgraph dev` | Local dev server with in-memory checkpointer |
| `uv run langgraph up` | Local Docker stack (Redis + Postgres) — mirrors production |
| `uv run langgraph deploy` | Deploy to LangSmith (requires paid plan) |

The agent is served at `http://localhost:2025` by default.

## Configuration

`langgraph.json` registers the graph and sets a checkpointer TTL of 24 hours with hourly sweeps:

```json
{
  "graphs": { "weather_agent": "./agents/simple_react/agent.py:agent" },
  "checkpointer": {
    "ttl": { "strategy": "delete", "sweep_interval_minutes": 60, "default_ttl": 1440 }
  }
}
```
