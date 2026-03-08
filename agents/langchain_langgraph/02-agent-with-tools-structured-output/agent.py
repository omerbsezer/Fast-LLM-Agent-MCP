
from pydantic import BaseModel, Field
from typing import Literal
from langchain_aws import ChatBedrockConverse
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from dotenv import load_dotenv
load_dotenv()

# Structured output schema
class MediaItem(BaseModel):
    """A single book or movie recommendation."""
    title: str           = Field(description="Title of the book or movie")
    creator: str         = Field(description="Author (book) or director (movie)")
    year: int            = Field(description="Year of release or publication")
    media_type: Literal["book", "movie"] = Field(description="Type of media")
    reason: str          = Field(description="Why this matches the user's mood, 1-2 sentences")
    mood_tags: list[str] = Field(description="Mood tags, lowercase, 1-3 words each")


class RecommendationResult(BaseModel):
    """Structured recommendation response returned by the agent."""
    detected_mood: str               = Field(description="The mood detected from the user's message")
    confidence: float                = Field(description="Confidence score 0.0-1.0", ge=0.0, le=1.0)
    recommendations: list[MediaItem] = Field(description="List of 2-4 recommendations")
    summary: str                     = Field(description="One-sentence summary of the recommendations")


# Mock catalogue tools
BOOK_DB = [
    {"title": "The Midnight Library",   "creator": "Matt Haig",       "year": 2020, "moods": ["melancholic", "hopeful", "reflective"]},
    {"title": "Project Hail Mary",      "creator": "Andy Weir",       "year": 2021, "moods": ["adventurous", "curious", "uplifting"]},
    {"title": "Anxious People",         "creator": "Fredrik Backman", "year": 2019, "moods": ["funny", "warm", "melancholic"]},
    {"title": "The Hitchhiker's Guide", "creator": "Douglas Adams",   "year": 1979, "moods": ["funny", "adventurous", "curious"]},
    {"title": "Atomic Habits",          "creator": "James Clear",     "year": 2018, "moods": ["motivated", "focused", "uplifting"]},
]

MOVIE_DB = [
    {"title": "Everything Everywhere All at Once", "creator": "Daniels",           "year": 2022, "moods": ["adventurous", "funny", "melancholic", "reflective"]},
    {"title": "Soul",                              "creator": "Pete Docter",        "year": 2020, "moods": ["reflective", "hopeful", "warm"]},
    {"title": "The Grand Budapest Hotel",          "creator": "Wes Anderson",       "year": 2014, "moods": ["funny", "adventurous", "warm"]},
    {"title": "Interstellar",                      "creator": "Christopher Nolan",  "year": 2014, "moods": ["curious", "adventurous", "melancholic"]},
    {"title": "Amelie",                            "creator": "Jean-Pierre Jeunet", "year": 2001, "moods": ["warm", "hopeful", "funny"]},
]


@tool
def search_books(mood: str) -> list[dict]:
    """Search the book catalogue by mood keyword."""
    mood = mood.lower()
    return [b for b in BOOK_DB if any(mood in m for m in b["moods"])]


@tool
def search_movies(mood: str) -> list[dict]:
    """Search the movie catalogue by mood keyword."""
    mood = mood.lower()
    return [m for m in MOVIE_DB if any(mood in t for t in m["moods"])]


@tool
def get_trending(media_type: Literal["book", "movie"]) -> list[dict]:
    """Return the 3 most recently released items for the given media type."""
    db = BOOK_DB if media_type == "book" else MOVIE_DB
    return sorted(db, key=lambda x: x["year"], reverse=True)[:3]

llm = ChatBedrockConverse(
    model="us.amazon.nova-pro-v1:0",
    temperature=0,
    # Reads AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION from .env
)

tools = [search_books, search_movies, get_trending]

agent = create_agent(
    model=llm,
    tools=tools,
    response_format=ToolStrategy(RecommendationResult),
    system_prompt=(
        "You are a creative media concierge. "
        "Detect the user's mood, search for matching books and movies using your tools, "
        "then return a fully structured recommendation."
    ),
)

def recommend(user_message: str) -> RecommendationResult:
    result = agent.invoke({"messages": [{"role": "user", "content": user_message}]})
    return result["structured_response"]

if __name__ == "__main__":
    queries = [
        "I'm feeling a bit low today and want something that lifts my spirits.",
        "I'm in a curious and adventurous mood — blow my mind!",
        "I want something funny and light after a long week.",
    ]

    for query in queries:
        print(f"\n{'='*65}")
        print(f"USER    : {query}")
        rec: RecommendationResult = recommend(query)
        print(f"MOOD    : {rec.detected_mood}  (confidence: {rec.confidence:.0%})")
        print(f"SUMMARY : {rec.summary}")
        for item in rec.recommendations:
            icon = "book" if item.media_type == "book" else "film"
            print(f"\n  [{icon}]  {item.title} ({item.year}) — {item.creator}")
            print(f"          {item.reason}")
            print(f"          tags: {', '.join(item.mood_tags)}")