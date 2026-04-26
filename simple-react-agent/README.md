We will deploy a basic langchain agent using langgraph cli

To deploy with langgraph, we need the following items:
1. Agent code
2. langgraph.json - lists dependencies, graphs etc
3. Dependencies file - requirements.txt, pyproject.toml
4. Environment variables (Optional) - .env file

-- Dev Server
uv run langgraph dev // Starts the agent locally. Uses InMemory checkpointer

-- Deployment Validation Server
uv run langgraph up // Starts the agent locally on Docker. Uses Redis and PSQL DB similar to a production environment.

-- Prod Deploy
uv run langgraph deploy //Deploys on langsmith. Requires Paid plan