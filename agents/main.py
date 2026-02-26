"""
Entry point for the Databricks Scientific Assistant multi-agent system.

Usage:
    python -m agents.main

Prerequisites:
    1. Copy .env.example to .env and fill in all required values
    2. Run `gcloud auth application-default login` for GCP Vertex AI auth
    3. Install: pip install -r requirements_agents.txt
"""
import asyncio
import uuid

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agents.auth import init_vertex_ai
from agents.config import validate_config
from agents.agents.root_agent import root_agent

APP_NAME = "databricks_scientific_assistant"


async def run_query(
    runner: Runner,
    user_id: str,
    session_id: str,
    query: str,
) -> str:
    """Send a single query through the agent system and return the final response text."""
    message = types.Content(
        role="user",
        parts=[types.Part(text=query)],
    )
    final_response = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=message,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_response = event.content.parts[0].text
    return final_response


async def interactive_loop() -> None:
    """Interactive CLI chat loop — maintains session across turns."""
    session_service = InMemorySessionService()
    user_id = "local_user"
    session_id = str(uuid.uuid4())

    await session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
    )

    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    print("=" * 60)
    print("  Databricks Scientific Assistant")
    print("  Multi-Agent System (Google ADK)")
    print("=" * 60)
    print("Agents: DatabricksCatalog | DatabricksJob | GoogleScholar")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print("Thinking...", flush=True)
        response = await run_query(runner, user_id, session_id, query)
        print(f"\nAssistant: {response}\n")


def main() -> None:
    # 1. Validate required environment variables
    missing = validate_config()
    if missing:
        print(f"ERROR: Missing required environment variables: {', '.join(missing)}")
        print("Copy .env.example to .env and fill in the values.")
        return

    # 2. Initialize GCP Vertex AI + register Claude with ADK
    print("Initializing Vertex AI...")
    try:
        init_vertex_ai()
    except Exception as e:
        print(f"ERROR: Failed to initialize Vertex AI: {e}")
        print("Ensure you have run: gcloud auth application-default login")
        return

    # 3. Start interactive loop
    asyncio.run(interactive_loop())


if __name__ == "__main__":
    main()
