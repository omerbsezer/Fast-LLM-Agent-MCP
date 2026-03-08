from langchain_aws import ChatBedrockConverse
from langchain_core.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv
from markdown_saver import MarkdownSaver  # custom checkpointer

load_dotenv()

"""
Each user/thread gets a persistent memory file:
  checkpoints/thread_alice.md
  checkpoints/thread_bob.md

The agent remembers:
  - Cities the user has visited
  - Their travel preferences
  - Wishlist destinations
  - Any personal notes across sessions
"""


TRAVEL_DB = {
    "paris":     {"country": "France",  "best_for": "art, food, romance",    "avg_cost": "high"},
    "lisbon":    {"country": "Portugal","best_for": "history, seafood, trams","avg_cost": "medium"},
    "tokyo":     {"country": "Japan",   "best_for": "culture, tech, food",   "avg_cost": "high"},
    "tbilisi":   {"country": "Georgia", "best_for": "wine, mountains, old town","avg_cost": "low"},
    "medellin":  {"country": "Colombia","best_for": "spring weather, nightlife","avg_cost": "low"},
    "istanbul":  {"country": "Turkey",  "best_for": "history, bazaars, food", "avg_cost": "medium"},
    "chiang mai":{"country": "Thailand","best_for": "temples, nature, food",  "avg_cost": "low"},
}


@tool
def get_city_info(city: str) -> dict:
    """Get travel details for a specific city."""
    return TRAVEL_DB.get(city.lower(), {"error": f"No data found for '{city}'"})


@tool
def suggest_destinations(budget: str, vibe: str) -> list[dict]:
    """
    Suggest travel destinations.
    budget: 'low', 'medium', or 'high'
    vibe: any keyword like 'food', 'history', 'nature', 'romance'
    """
    results = []
    for city, info in TRAVEL_DB.items():
        if info["avg_cost"] == budget.lower() and vibe.lower() in info["best_for"]:
            results.append({"city": city.title(), **info})
    return results or [{"message": "No exact matches — try broadening your vibe keyword."}]


@tool
def compare_cities(city_a: str, city_b: str) -> dict:
    """Compare two cities side by side."""
    a = TRAVEL_DB.get(city_a.lower())
    b = TRAVEL_DB.get(city_b.lower())
    if not a or not b:
        return {"error": "One or both cities not found."}
    return {
        city_a.title(): a,
        city_b.title(): b,
        "cheaper": city_a.title() if a["avg_cost"] <= b["avg_cost"] else city_b.title(),
    }

llm = ChatBedrockConverse(
    model="us.amazon.nova-pro-v1:0",
    temperature=0.3,
)

tools = [get_city_info, suggest_destinations, compare_cities]


def chat(thread_id: str, message: str, agent) -> str:
    """Send a message to the agent on a specific thread and return the reply."""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        {"configurable": {"thread_id": thread_id}},
    )
    return result["messages"][-1].content

if __name__ == "__main__":
    with MarkdownSaver() as checkpointer:
        checkpointer.setup()

        agent = create_agent(
            model=llm,
            tools=tools,
            checkpointer=checkpointer,
            system_prompt=(
                "You are a warm and knowledgeable travel journal assistant. "
                "Remember everything the user tells you about their travels, preferences, "
                "and wishlist. Use tools to enrich your answers with real destination data. "
                "Reference past context naturally — like a friend who remembers your stories."
            ),
        )

        # Alice's session
        print("\n" + "═"*65)
        print("SESSION: Alice (thread_alice)")
        print("═"*65)

        turns = [
            "Hi! I'm Alice. I've been to Lisbon and Tokyo — both were amazing.",
            "I prefer low-budget trips focused on local food. Any ideas?",
            "Tell me more about Tbilisi. Is it worth visiting?",
            "How does Tbilisi compare to Chiang Mai?",
        ]
        for msg in turns:
            print(f"\nAlice : {msg}")
            reply = chat("thread_alice", msg, agent)
            print(f"Agent : {reply}")

        # Bob's session (separate memory)
        print("\n" + "═"*65)
        print("SESSION: Bob (thread_bob)")
        print("═"*65)

        turns = [
            "Hey, I'm Bob. I'm planning my first solo trip ever!",
            "I have a medium budget and I love history and bazaars.",
            "What about Paris? I've always dreamed of going.",
        ]
        for msg in turns:
            print(f"\nBob   : {msg}")
            reply = chat("thread_bob", msg, agent)
            print(f"Agent : {reply}")

        # Alice returns (memory should persist)
        print("\n" + "═"*65)
        print("SESSION: Alice returns (memory test)")
        print("═"*65)

        msg = "Hey, I'm back! Do you remember where I've been and what I like?"
        print(f"\nAlice : {msg}")
        reply = chat("thread_alice", msg, agent)
        print(f"Agent : {reply}")

        print(f"\n\nCheckpoint files saved in: checkpoints/")
        print("  thread_alice.md  — Alice's full conversation history")
        print("  thread_bob.md    — Bob's full conversation history")